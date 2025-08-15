from collections import defaultdict, deque
import heapq


class TarjanSCC:
    """
    Tarjan算法求强连通分量
    时间复杂度：O(V + E)
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.index = 0
        self.stack = []
        self.indices = [-1] * n
        self.lowlinks = [-1] * n
        self.on_stack = [False] * n
        self.scc_count = 0
        self.scc_id = [-1] * n
    
    def add_edge(self, u, v):
        """添加有向边"""
        self.graph[u].append(v)
    
    def tarjan(self, v):
        """Tarjan算法核心"""
        self.indices[v] = self.index
        self.lowlinks[v] = self.index
        self.index += 1
        self.stack.append(v)
        self.on_stack[v] = True
        
        # 遍历邻接点
        for w in self.graph[v]:
            if self.indices[w] == -1:
                self.tarjan(w)
                self.lowlinks[v] = min(self.lowlinks[v], self.lowlinks[w])
            elif self.on_stack[w]:
                self.lowlinks[v] = min(self.lowlinks[v], self.indices[w])
        
        # 如果v是强连通分量的根
        if self.lowlinks[v] == self.indices[v]:
            scc = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                scc.append(w)
                self.scc_id[w] = self.scc_count
                if w == v:
                    break
            self.scc_count += 1
            return scc
        return None
    
    def find_sccs(self):
        """找出所有强连通分量"""
        sccs = []
        for v in range(self.n):
            if self.indices[v] == -1:
                scc = self.tarjan(v)
                if scc:
                    sccs.append(scc)
        return sccs


class KosarajuSCC:
    """
    Kosaraju算法求强连通分量
    时间复杂度：O(V + E)
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
    
    def add_edge(self, u, v):
        """添加有向边"""
        self.graph[u].append(v)
        self.reverse_graph[v].append(u)
    
    def dfs_first(self, v, visited, stack):
        """第一次DFS，记录完成时间"""
        visited[v] = True
        for u in self.graph[v]:
            if not visited[u]:
                self.dfs_first(u, visited, stack)
        stack.append(v)
    
    def dfs_second(self, v, visited, scc):
        """第二次DFS，在反向图上"""
        visited[v] = True
        scc.append(v)
        for u in self.reverse_graph[v]:
            if not visited[u]:
                self.dfs_second(u, visited, scc)
    
    def find_sccs(self):
        """找出所有强连通分量"""
        stack = []
        visited = [False] * self.n
        
        # 第一次DFS
        for i in range(self.n):
            if not visited[i]:
                self.dfs_first(i, visited, stack)
        
        # 第二次DFS（在反向图上）
        visited = [False] * self.n
        sccs = []
        
        while stack:
            v = stack.pop()
            if not visited[v]:
                scc = []
                self.dfs_second(v, visited, scc)
                sccs.append(scc)
        
        return sccs


class ArticulationPoints:
    """
    寻找割点（关节点）
    时间复杂度：O(V + E)
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.visited = [False] * n
        self.disc = [0] * n
        self.low = [0] * n
        self.parent = [-1] * n
        self.ap = [False] * n
        self.time = 0
    
    def add_edge(self, u, v):
        """添加无向边"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def ap_util(self, u):
        """寻找割点的辅助函数"""
        children = 0
        self.visited[u] = True
        self.disc[u] = self.low[u] = self.time
        self.time += 1
        
        for v in self.graph[u]:
            if not self.visited[v]:
                self.parent[v] = u
                children += 1
                self.ap_util(v)
                
                self.low[u] = min(self.low[u], self.low[v])
                
                # u是割点的条件
                if self.parent[u] == -1 and children > 1:
                    self.ap[u] = True
                
                if self.parent[u] != -1 and self.low[v] >= self.disc[u]:
                    self.ap[u] = True
            
            elif v != self.parent[u]:
                self.low[u] = min(self.low[u], self.disc[v])
    
    def find_articulation_points(self):
        """找出所有割点"""
        for i in range(self.n):
            if not self.visited[i]:
                self.ap_util(i)
        
        return [i for i in range(self.n) if self.ap[i]]


class Bridges:
    """
    寻找桥（割边）
    时间复杂度：O(V + E)
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.visited = [False] * n
        self.disc = [0] * n
        self.low = [0] * n
        self.parent = [-1] * n
        self.bridges = []
        self.time = 0
    
    def add_edge(self, u, v):
        """添加无向边"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def bridge_util(self, u):
        """寻找桥的辅助函数"""
        self.visited[u] = True
        self.disc[u] = self.low[u] = self.time
        self.time += 1
        
        for v in self.graph[u]:
            if not self.visited[v]:
                self.parent[v] = u
                self.bridge_util(v)
                
                self.low[u] = min(self.low[u], self.low[v])
                
                # 如果v的最低点不能到达u或u的祖先，则u-v是桥
                if self.low[v] > self.disc[u]:
                    self.bridges.append((u, v))
            
            elif v != self.parent[u]:
                self.low[u] = min(self.low[u], self.disc[v])
    
    def find_bridges(self):
        """找出所有桥"""
        for i in range(self.n):
            if not self.visited[i]:
                self.bridge_util(i)
        
        return self.bridges


class EulerianPath:
    """
    欧拉路径和欧拉回路
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.in_degree = [0] * n
        self.out_degree = [0] * n
    
    def add_edge(self, u, v, directed=True):
        """添加边"""
        self.graph[u].append(v)
        self.out_degree[u] += 1
        self.in_degree[v] += 1
        
        if not directed:
            self.graph[v].append(u)
            self.out_degree[v] += 1
            self.in_degree[u] += 1
    
    def has_eulerian_path(self):
        """判断是否存在欧拉路径"""
        start_nodes = 0
        end_nodes = 0
        
        for i in range(self.n):
            if self.out_degree[i] - self.in_degree[i] == 1:
                start_nodes += 1
            elif self.in_degree[i] - self.out_degree[i] == 1:
                end_nodes += 1
            elif self.in_degree[i] != self.out_degree[i]:
                return False
        
        return (start_nodes == 0 and end_nodes == 0) or \
               (start_nodes == 1 and end_nodes == 1)
    
    def has_eulerian_circuit(self):
        """判断是否存在欧拉回路"""
        for i in range(self.n):
            if self.in_degree[i] != self.out_degree[i]:
                return False
        return True
    
    def find_eulerian_path(self):
        """Hierholzer算法寻找欧拉路径"""
        if not self.has_eulerian_path():
            return None
        
        # 找起点
        start = 0
        for i in range(self.n):
            if self.out_degree[i] - self.in_degree[i] == 1:
                start = i
                break
            if self.out_degree[i] > 0:
                start = i
        
        # 复制图
        graph_copy = defaultdict(list)
        for u in self.graph:
            graph_copy[u] = self.graph[u].copy()
        
        stack = [start]
        path = []
        
        while stack:
            v = stack[-1]
            if graph_copy[v]:
                u = graph_copy[v].pop()
                stack.append(u)
            else:
                path.append(stack.pop())
        
        return path[::-1]


class HamiltonianPath:
    """
    哈密顿路径（使用回溯）
    时间复杂度：O(n!)
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = [[False] * n for _ in range(n)]
    
    def add_edge(self, u, v):
        """添加无向边"""
        self.graph[u][v] = True
        self.graph[v][u] = True
    
    def is_safe(self, v, path, pos):
        """检查是否可以将v添加到路径"""
        # 检查是否与前一个顶点相邻
        if not self.graph[path[pos - 1]][v]:
            return False
        
        # 检查是否已经在路径中
        if v in path[:pos]:
            return False
        
        return True
    
    def hamiltonian_path_util(self, path, pos):
        """哈密顿路径回溯辅助函数"""
        # 基础情况：所有顶点都在路径中
        if pos == self.n:
            return True
        
        # 尝试不同的顶点
        for v in range(self.n):
            if self.is_safe(v, path, pos):
                path[pos] = v
                
                if self.hamiltonian_path_util(path, pos + 1):
                    return True
                
                # 回溯
                path[pos] = -1
        
        return False
    
    def find_hamiltonian_path(self):
        """寻找哈密顿路径"""
        path = [-1] * self.n
        
        # 尝试从每个顶点开始
        for start in range(self.n):
            path[0] = start
            if self.hamiltonian_path_util(path, 1):
                return path
        
        return None
    
    def find_hamiltonian_circuit(self):
        """寻找哈密顿回路"""
        path = self.find_hamiltonian_path()
        if path and self.graph[path[-1]][path[0]]:
            return path + [path[0]]
        return None


class BipartiteGraph:
    """
    二分图相关算法
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.color = [-1] * n
    
    def add_edge(self, u, v):
        """添加无向边"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def is_bipartite_util(self, src):
        """BFS检查二分图"""
        queue = deque([src])
        self.color[src] = 0
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if self.color[v] == -1:
                    self.color[v] = 1 - self.color[u]
                    queue.append(v)
                elif self.color[v] == self.color[u]:
                    return False
        
        return True
    
    def is_bipartite(self):
        """检查是否是二分图"""
        for i in range(self.n):
            if self.color[i] == -1:
                if not self.is_bipartite_util(i):
                    return False
        return True
    
    def get_partitions(self):
        """获取二分图的两个分区"""
        if not self.is_bipartite():
            return None, None
        
        partition_0 = [i for i in range(self.n) if self.color[i] == 0]
        partition_1 = [i for i in range(self.n) if self.color[i] == 1]
        
        return partition_0, partition_1


class GraphColoring:
    """
    图着色算法
    """
    
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        """添加无向边"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def greedy_coloring(self):
        """贪心图着色"""
        result = [-1] * self.n
        result[0] = 0
        
        # 临时数组存储可用颜色
        available = [True] * self.n
        
        for u in range(1, self.n):
            # 标记邻接顶点的颜色为不可用
            for v in self.graph[u]:
                if result[v] != -1:
                    available[result[v]] = False
            
            # 找到第一个可用颜色
            for color in range(self.n):
                if available[color]:
                    result[u] = color
                    break
            
            # 重置可用颜色
            for v in self.graph[u]:
                if result[v] != -1:
                    available[result[v]] = True
        
        return result, max(result) + 1
    
    def is_safe_color(self, v, color, colors):
        """检查颜色是否安全"""
        for u in self.graph[v]:
            if colors[u] == color:
                return False
        return True
    
    def graph_coloring_util(self, m, colors, v):
        """图着色回溯"""
        if v == self.n:
            return True
        
        for c in range(m):
            if self.is_safe_color(v, c, colors):
                colors[v] = c
                
                if self.graph_coloring_util(m, colors, v + 1):
                    return True
                
                colors[v] = -1
        
        return False
    
    def graph_coloring(self, m):
        """m-着色问题"""
        colors = [-1] * self.n
        
        if self.graph_coloring_util(m, colors, 0):
            return colors
        return None


class MaxFlow2:
    """
    最大流算法集合（补充）
    """
    
    @staticmethod
    def ford_fulkerson_dfs(graph, source, sink):
        """
        Ford-Fulkerson算法（DFS版本）
        时间复杂度：O(E * max_flow)
        """
        def dfs(graph, source, sink, parent):
            visited = set([source])
            stack = [source]
            
            while stack:
                u = stack.pop()
                
                for v in range(len(graph)):
                    if v not in visited and graph[u][v] > 0:
                        visited.add(v)
                        parent[v] = u
                        if v == sink:
                            return True
                        stack.append(v)
            
            return False
        
        n = len(graph)
        # 创建残余图
        residual = [row[:] for row in graph]
        parent = [-1] * n
        max_flow = 0
        
        while dfs(residual, source, sink, parent):
            # 找到路径的最小容量
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual[parent[s]][s])
                s = parent[s]
            
            # 更新残余图
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = parent[v]
            
            max_flow += path_flow
            parent = [-1] * n
        
        return max_flow


def test_advanced_graph():
    """测试高级图算法"""
    print("=== 高级图算法测试 ===\n")
    
    # 测试强连通分量
    print("1. 强连通分量:")
    
    # Tarjan算法
    tarjan = TarjanSCC(8)
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), 
             (3, 4), (4, 5), (5, 3), (6, 7)]
    for u, v in edges:
        tarjan.add_edge(u, v)
    
    sccs_tarjan = tarjan.find_sccs()
    print(f"   Tarjan算法: {sccs_tarjan}")
    
    # Kosaraju算法
    kosaraju = KosarajuSCC(8)
    for u, v in edges:
        kosaraju.add_edge(u, v)
    
    sccs_kosaraju = kosaraju.find_sccs()
    print(f"   Kosaraju算法: {sccs_kosaraju}")
    
    # 测试割点和桥
    print("\n2. 割点和桥:")
    
    # 割点
    ap = ArticulationPoints(5)
    edges_ap = [(0, 1), (1, 2), (2, 3), (1, 3), (3, 4)]
    for u, v in edges_ap:
        ap.add_edge(u, v)
    
    articulation_points = ap.find_articulation_points()
    print(f"   割点: {articulation_points}")
    
    # 桥
    bridge = Bridges(5)
    for u, v in edges_ap:
        bridge.add_edge(u, v)
    
    bridges = bridge.find_bridges()
    print(f"   桥: {bridges}")
    
    # 测试欧拉路径
    print("\n3. 欧拉路径:")
    euler = EulerianPath(5)
    euler_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    for u, v in euler_edges:
        euler.add_edge(u, v, directed=False)
    
    print(f"   存在欧拉路径: {euler.has_eulerian_path()}")
    print(f"   存在欧拉回路: {euler.has_eulerian_circuit()}")
    
    path = euler.find_eulerian_path()
    if path:
        print(f"   欧拉路径: {path}")
    
    # 测试哈密顿路径
    print("\n4. 哈密顿路径:")
    hamilton = HamiltonianPath(4)
    hamilton_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    for u, v in hamilton_edges:
        hamilton.add_edge(u, v)
    
    ham_path = hamilton.find_hamiltonian_path()
    ham_circuit = hamilton.find_hamiltonian_circuit()
    print(f"   哈密顿路径: {ham_path}")
    print(f"   哈密顿回路: {ham_circuit}")
    
    # 测试二分图
    print("\n5. 二分图:")
    bipartite = BipartiteGraph(6)
    bi_edges = [(0, 1), (0, 3), (2, 1), (2, 3), (4, 5)]
    for u, v in bi_edges:
        bipartite.add_edge(u, v)
    
    is_bip = bipartite.is_bipartite()
    print(f"   是二分图: {is_bip}")
    
    if is_bip:
        p0, p1 = bipartite.get_partitions()
        print(f"   分区1: {p0}")
        print(f"   分区2: {p1}")
    
    # 测试图着色
    print("\n6. 图着色:")
    coloring = GraphColoring(5)
    color_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
    for u, v in color_edges:
        coloring.add_edge(u, v)
    
    colors, num_colors = coloring.greedy_coloring()
    print(f"   贪心着色: {colors}")
    print(f"   使用颜色数: {num_colors}")
    
    m_colors = coloring.graph_coloring(3)
    print(f"   3-着色: {m_colors}")
    
    # 复杂度分析
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │")
    print("├──────────────────┼──────────────┤")
    print("│ Tarjan SCC       │ O(V + E)     │")
    print("│ Kosaraju SCC     │ O(V + E)     │")
    print("│ 割点/桥          │ O(V + E)     │")
    print("│ 欧拉路径         │ O(E)         │")
    print("│ 哈密顿路径       │ O(n!)        │")
    print("│ 二分图检测       │ O(V + E)     │")
    print("│ 贪心着色         │ O(V + E)     │")
    print("└──────────────────┴──────────────┘")


if __name__ == '__main__':
    test_advanced_graph()