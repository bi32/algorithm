from collections import deque, defaultdict
import heapq


class MaxFlow:
    """最大流算法基类"""
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(lambda: defaultdict(int))
    
    def add_edge(self, u, v, capacity):
        """添加边"""
        self.graph[u][v] += capacity


class FordFulkerson(MaxFlow):
    """
    Ford-Fulkerson算法
    使用DFS寻找增广路径
    时间复杂度：O(E * max_flow)
    """
    
    def dfs(self, source, sink, parent):
        """DFS寻找从source到sink的路径"""
        visited = set([source])
        stack = [source]
        
        while stack:
            u = stack.pop()
            
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    
                    if v == sink:
                        return True
                    
                    stack.append(v)
        
        return False
    
    def max_flow(self, source, sink):
        """计算最大流"""
        parent = {}
        max_flow_value = 0
        
        # 创建残余图的副本
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v in self.graph[u]:
                residual[u][v] = self.graph[u][v]
        
        # 保存原图，使用残余图
        original_graph = self.graph
        self.graph = residual
        
        while self.dfs(source, sink, parent):
            # 找到路径的最小容量
            path_flow = float('inf')
            s = sink
            
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            
            # 更新残余图
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            
            max_flow_value += path_flow
            parent.clear()
        
        # 恢复原图
        self.graph = original_graph
        return max_flow_value


class EdmondsKarp(MaxFlow):
    """
    Edmonds-Karp算法
    使用BFS寻找最短增广路径
    时间复杂度：O(V * E²)
    """
    
    def bfs(self, source, sink, parent):
        """BFS寻找最短增广路径"""
        visited = {source}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    
                    if v == sink:
                        return True
                    
                    queue.append(v)
        
        return False
    
    def max_flow(self, source, sink):
        """计算最大流"""
        parent = {}
        max_flow_value = 0
        
        # 创建残余图
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v in self.graph[u]:
                residual[u][v] = self.graph[u][v]
        
        original_graph = self.graph
        self.graph = residual
        
        while self.bfs(source, sink, parent):
            # 找到路径的最小容量
            path_flow = float('inf')
            s = sink
            
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            
            # 更新残余图
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            
            max_flow_value += path_flow
            parent.clear()
        
        # 找出最小割
        visited = set()
        queue = deque([source])
        visited.add(source)
        
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    queue.append(v)
        
        min_cut = []
        for u in visited:
            for v in original_graph[u]:
                if v not in visited:
                    min_cut.append((u, v))
        
        self.graph = original_graph
        return max_flow_value, min_cut


class Dinic(MaxFlow):
    """
    Dinic算法
    使用层次图和阻塞流
    时间复杂度：O(V² * E)
    """
    
    def bfs_level_graph(self, source, sink):
        """构建层次图"""
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if self.level[v] < 0 and self.graph[u][v] > 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        
        return self.level[sink] >= 0
    
    def dfs_blocking_flow(self, u, sink, flow):
        """使用DFS寻找阻塞流"""
        if u == sink:
            return flow
        
        for v in list(self.graph[u].keys()):
            if self.level[v] == self.level[u] + 1 and self.graph[u][v] > 0:
                min_flow = min(flow, self.graph[u][v])
                result = self.dfs_blocking_flow(v, sink, min_flow)
                
                if result > 0:
                    self.graph[u][v] -= result
                    self.graph[v][u] += result
                    return result
        
        return 0
    
    def max_flow(self, source, sink):
        """计算最大流"""
        # 创建残余图
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v in self.graph[u]:
                residual[u][v] = self.graph[u][v]
        
        original_graph = self.graph
        self.graph = residual
        
        max_flow_value = 0
        
        while self.bfs_level_graph(source, sink):
            while True:
                flow = self.dfs_blocking_flow(source, sink, float('inf'))
                if flow == 0:
                    break
                max_flow_value += flow
        
        self.graph = original_graph
        return max_flow_value


class PushRelabel(MaxFlow):
    """
    Push-Relabel算法（预流推进）
    时间复杂度：O(V²√E) with gap heuristic
    """
    
    def __init__(self, n):
        super().__init__(n)
        self.height = [0] * n
        self.excess = [0] * n
    
    def push(self, u, v):
        """推送操作"""
        flow = min(self.excess[u], self.graph[u][v])
        self.graph[u][v] -= flow
        self.graph[v][u] += flow
        self.excess[u] -= flow
        self.excess[v] += flow
    
    def relabel(self, u):
        """重标号操作"""
        min_height = float('inf')
        
        for v in self.graph[u]:
            if self.graph[u][v] > 0:
                min_height = min(min_height, self.height[v])
        
        if min_height < float('inf'):
            self.height[u] = min_height + 1
    
    def max_flow(self, source, sink):
        """计算最大流"""
        # 创建残余图
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v in self.graph[u]:
                residual[u][v] = self.graph[u][v]
        
        original_graph = self.graph
        self.graph = residual
        
        # 初始化
        self.height[source] = self.n
        
        # 从源点推送流
        for v in self.graph[source]:
            if self.graph[source][v] > 0:
                flow = self.graph[source][v]
                self.graph[source][v] = 0
                self.graph[v][source] = flow
                self.excess[v] = flow
        
        # 活跃顶点列表
        active = []
        for v in range(self.n):
            if v != source and v != sink and self.excess[v] > 0:
                active.append(v)
        
        while active:
            u = active.pop(0)
            
            if self.excess[u] == 0:
                continue
            
            # 尝试推送
            pushed = False
            for v in self.graph[u]:
                if self.graph[u][v] > 0 and self.height[u] == self.height[v] + 1:
                    self.push(u, v)
                    
                    if v != source and v != sink and self.excess[v] > 0 and v not in active:
                        active.append(v)
                    
                    if self.excess[u] == 0:
                        pushed = True
                        break
            
            # 如果无法推送，重标号
            if not pushed and self.excess[u] > 0:
                self.relabel(u)
                active.append(u)
        
        max_flow_value = self.excess[sink]
        self.graph = original_graph
        return max_flow_value


class MinCostMaxFlow:
    """
    最小费用最大流
    使用SPFA（Bellman-Ford的队列优化）寻找最小费用增广路径
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(lambda: defaultdict(lambda: {'capacity': 0, 'cost': 0}))
    
    def add_edge(self, u, v, capacity, cost):
        """添加带费用的边"""
        self.graph[u][v]['capacity'] += capacity
        self.graph[u][v]['cost'] = cost
        self.graph[v][u]['cost'] = -cost  # 反向边费用为负
    
    def spfa(self, source, sink, parent, dist):
        """SPFA算法寻找最小费用路径"""
        dist[:] = [float('inf')] * self.n
        dist[source] = 0
        
        in_queue = [False] * self.n
        in_queue[source] = True
        
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            for v in self.graph[u]:
                if self.graph[u][v]['capacity'] > 0:
                    new_dist = dist[u] + self.graph[u][v]['cost']
                    
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        parent[v] = u
                        
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
        
        return dist[sink] < float('inf')
    
    def min_cost_max_flow(self, source, sink):
        """计算最小费用最大流"""
        parent = [-1] * self.n
        dist = [float('inf')] * self.n
        
        max_flow = 0
        min_cost = 0
        
        while self.spfa(source, sink, parent, dist):
            # 找到路径的最小容量
            path_flow = float('inf')
            v = sink
            
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.graph[u][v]['capacity'])
                v = u
            
            # 更新流量和费用
            max_flow += path_flow
            min_cost += path_flow * dist[sink]
            
            # 更新残余网络
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v]['capacity'] -= path_flow
                self.graph[v][u]['capacity'] += path_flow
                v = u
        
        return max_flow, min_cost


class BipartiteMatching:
    """
    二分图最大匹配
    使用最大流算法求解
    """
    
    def __init__(self, n1, n2):
        """n1: 左侧节点数, n2: 右侧节点数"""
        self.n1 = n1
        self.n2 = n2
        # 0: 源点, 1..n1: 左侧, n1+1..n1+n2: 右侧, n1+n2+1: 汇点
        self.flow = EdmondsKarp(n1 + n2 + 2)
        self.source = 0
        self.sink = n1 + n2 + 1
        
        # 连接源点到左侧
        for i in range(1, n1 + 1):
            self.flow.add_edge(self.source, i, 1)
        
        # 连接右侧到汇点
        for i in range(n1 + 1, n1 + n2 + 1):
            self.flow.add_edge(i, self.sink, 1)
    
    def add_edge(self, u, v):
        """添加匹配边（u在左侧，v在右侧）"""
        self.flow.add_edge(u + 1, self.n1 + v + 1, 1)
    
    def max_matching(self):
        """求最大匹配"""
        max_match, _ = self.flow.max_flow(self.source, self.sink)
        
        # 找出匹配的边
        matching = []
        for u in range(1, self.n1 + 1):
            for v in range(self.n1 + 1, self.n1 + self.n2 + 1):
                if u in self.flow.graph and v in self.flow.graph[u]:
                    # 检查残余图中的反向边
                    if v in self.flow.graph and u in self.flow.graph[v]:
                        if self.flow.graph[v][u] > 0:
                            matching.append((u - 1, v - self.n1 - 1))
        
        return max_match, matching


class HungarianAlgorithm:
    """
    匈牙利算法（求二分图最大匹配）
    时间复杂度：O(VE)
    """
    
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.graph = [[] for _ in range(n1)]
        self.match = [-1] * n2
    
    def add_edge(self, u, v):
        """添加边"""
        self.graph[u].append(v)
    
    def dfs(self, u, visited):
        """DFS寻找增广路径"""
        for v in self.graph[u]:
            if not visited[v]:
                visited[v] = True
                
                if self.match[v] == -1 or self.dfs(self.match[v], visited):
                    self.match[v] = u
                    return True
        
        return False
    
    def max_matching(self):
        """求最大匹配"""
        result = 0
        
        for u in range(self.n1):
            visited = [False] * self.n2
            if self.dfs(u, visited):
                result += 1
        
        matching = []
        for v in range(self.n2):
            if self.match[v] != -1:
                matching.append((self.match[v], v))
        
        return result, matching


def test_network_flow():
    """测试网络流算法"""
    print("=== 网络流算法测试 ===\n")
    
    # 测试最大流
    print("1. 最大流算法比较:")
    
    # 创建测试图
    algorithms = [
        ("Ford-Fulkerson", FordFulkerson(6)),
        ("Edmonds-Karp", EdmondsKarp(6)),
        ("Dinic", Dinic(6)),
        ("Push-Relabel", PushRelabel(6))
    ]
    
    # 添加边
    edges = [
        (0, 1, 16), (0, 2, 13),
        (1, 2, 10), (1, 3, 12),
        (2, 1, 4), (2, 4, 14),
        (3, 2, 9), (3, 5, 20),
        (4, 3, 7), (4, 5, 4)
    ]
    
    for name, algo in algorithms:
        for u, v, cap in edges:
            algo.add_edge(u, v, cap)
        
        if name == "Edmonds-Karp":
            max_flow, min_cut = algo.max_flow(0, 5)
            print(f"   {name}: 最大流 = {max_flow}")
            print(f"   最小割边: {min_cut}")
        else:
            max_flow = algo.max_flow(0, 5)
            print(f"   {name}: 最大流 = {max_flow}")
    
    # 测试最小费用最大流
    print("\n2. 最小费用最大流:")
    mcmf = MinCostMaxFlow(4)
    
    # 添加带费用的边
    mcmf.add_edge(0, 1, 2, 4)
    mcmf.add_edge(0, 2, 2, 2)
    mcmf.add_edge(1, 2, 1, 2)
    mcmf.add_edge(1, 3, 2, 3)
    mcmf.add_edge(2, 3, 2, 1)
    
    max_flow, min_cost = mcmf.min_cost_max_flow(0, 3)
    print(f"   最大流: {max_flow}")
    print(f"   最小费用: {min_cost}")
    
    # 测试二分图匹配
    print("\n3. 二分图最大匹配:")
    
    # 使用最大流
    bm_flow = BipartiteMatching(4, 4)
    edges_bipartite = [
        (0, 0), (0, 1),
        (1, 1), (1, 2),
        (2, 2), (2, 3),
        (3, 3)
    ]
    
    for u, v in edges_bipartite:
        bm_flow.add_edge(u, v)
    
    max_match1, matching1 = bm_flow.max_matching()
    print(f"   最大流方法: 最大匹配数 = {max_match1}")
    print(f"   匹配边: {matching1}")
    
    # 使用匈牙利算法
    hungarian = HungarianAlgorithm(4, 4)
    for u, v in edges_bipartite:
        hungarian.add_edge(u, v)
    
    max_match2, matching2 = hungarian.max_matching()
    print(f"   匈牙利算法: 最大匹配数 = {max_match2}")
    print(f"   匹配边: {matching2}")
    
    # 应用场景
    print("\n=== 网络流应用场景 ===")
    print("• 最大流：网络带宽分配、项目选择")
    print("• 最小割：图像分割、社交网络分析")
    print("• 最小费用流：物流运输、任务分配")
    print("• 二分图匹配：工作分配、稳定婚姻问题")
    
    # 复杂度分析
    print("\n=== 算法复杂度 ===")
    print("┌─────────────────┬──────────────┐")
    print("│ 算法             │ 时间复杂度    │")
    print("├─────────────────┼──────────────┤")
    print("│ Ford-Fulkerson  │ O(E·f)       │")
    print("│ Edmonds-Karp    │ O(VE²)       │")
    print("│ Dinic           │ O(V²E)       │")
    print("│ Push-Relabel    │ O(V²√E)      │")
    print("│ 匈牙利算法       │ O(VE)        │")
    print("└─────────────────┴──────────────┘")


if __name__ == '__main__':
    test_network_flow()