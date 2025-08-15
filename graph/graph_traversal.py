from collections import deque, defaultdict


class Graph:
    """
    图的基本实现，支持有向图和无向图
    """
    
    def __init__(self, directed=True):
        self.graph = defaultdict(list)
        self.directed = directed
        self.vertices = set()
    
    def add_edge(self, u, v):
        """添加边"""
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)
        
        if not self.directed:
            self.graph[v].append(u)
    
    def add_vertex(self, v):
        """添加顶点"""
        self.vertices.add(v)
        if v not in self.graph:
            self.graph[v] = []
    
    def bfs(self, start):
        """
        广度优先搜索
        时间复杂度：O(V + E)
        空间复杂度：O(V)
        """
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def bfs_shortest_path(self, start, end):
        """
        使用BFS找最短路径（无权图）
        """
        if start == end:
            return [start]
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            vertex, path = queue.popleft()
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    if neighbor == end:
                        return new_path
                    
                    queue.append((neighbor, new_path))
        
        return None  # 无路径
    
    def dfs(self, start):
        """
        深度优先搜索（迭代版本）
        时间复杂度：O(V + E)
        空间复杂度：O(V)
        """
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            vertex = stack.pop()
            
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # 为了保持顺序，反向添加邻居
                for neighbor in reversed(self.graph[vertex]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    def dfs_recursive(self, start, visited=None):
        """
        深度优先搜索（递归版本）
        """
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                result.extend(self.dfs_recursive(neighbor, visited))
        
        return result
    
    def dfs_all_paths(self, start, end, path=None):
        """
        找出所有从start到end的路径
        """
        if path is None:
            path = []
        
        path = path + [start]
        
        if start == end:
            return [path]
        
        paths = []
        for neighbor in self.graph[start]:
            if neighbor not in path:  # 避免循环
                new_paths = self.dfs_all_paths(neighbor, end, path)
                paths.extend(new_paths)
        
        return paths
    
    def has_cycle_directed(self):
        """
        检测有向图中是否存在环
        使用三色标记法
        """
        WHITE = 0  # 未访问
        GRAY = 1   # 正在访问
        BLACK = 2  # 访问完成
        
        color = {v: WHITE for v in self.vertices}
        
        def visit(v):
            if color[v] == GRAY:
                return True  # 发现环
            
            if color[v] == BLACK:
                return False  # 已访问过
            
            color[v] = GRAY
            
            for neighbor in self.graph[v]:
                if visit(neighbor):
                    return True
            
            color[v] = BLACK
            return False
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                if visit(vertex):
                    return True
        
        return False
    
    def has_cycle_undirected(self):
        """
        检测无向图中是否存在环
        使用并查集或DFS
        """
        visited = set()
        
        def dfs(v, parent):
            visited.add(v)
            
            for neighbor in self.graph[v]:
                if neighbor not in visited:
                    if dfs(neighbor, v):
                        return True
                elif parent != neighbor:
                    return True  # 发现环
            
            return False
        
        for vertex in self.vertices:
            if vertex not in visited:
                if dfs(vertex, None):
                    return True
        
        return False
    
    def topological_sort_dfs(self):
        """
        拓扑排序（DFS方法）
        时间复杂度：O(V + E)
        """
        if not self.directed:
            raise ValueError("拓扑排序只适用于有向图")
        
        visited = set()
        stack = []
        
        def dfs(v):
            visited.add(v)
            
            for neighbor in self.graph[v]:
                if neighbor not in visited:
                    dfs(neighbor)
            
            stack.append(v)
        
        for vertex in self.vertices:
            if vertex not in visited:
                dfs(vertex)
        
        return stack[::-1]  # 反转得到拓扑序
    
    def topological_sort_kahn(self):
        """
        拓扑排序（Kahn算法，基于入度）
        时间复杂度：O(V + E)
        """
        if not self.directed:
            raise ValueError("拓扑排序只适用于有向图")
        
        # 计算入度
        in_degree = {v: 0 for v in self.vertices}
        for v in self.vertices:
            for neighbor in self.graph[v]:
                in_degree[neighbor] += 1
        
        # 找出所有入度为0的顶点
        queue = deque([v for v in self.vertices if in_degree[v] == 0])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            # 减少邻居的入度
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 如果结果中的顶点数不等于总顶点数，说明有环
        if len(result) != len(self.vertices):
            return None  # 图中有环
        
        return result
    
    def connected_components(self):
        """
        找出无向图的连通分量
        """
        if self.directed:
            raise ValueError("连通分量只适用于无向图")
        
        visited = set()
        components = []
        
        def dfs(v, component):
            visited.add(v)
            component.append(v)
            
            for neighbor in self.graph[v]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for vertex in self.vertices:
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(component)
        
        return components
    
    def strongly_connected_components(self):
        """
        Kosaraju算法找强连通分量
        时间复杂度：O(V + E)
        """
        if not self.directed:
            raise ValueError("强连通分量只适用于有向图")
        
        # 第一次DFS，记录完成时间
        visited = set()
        finish_stack = []
        
        def dfs1(v):
            visited.add(v)
            for neighbor in self.graph[v]:
                if neighbor not in visited:
                    dfs1(neighbor)
            finish_stack.append(v)
        
        for vertex in self.vertices:
            if vertex not in visited:
                dfs1(vertex)
        
        # 构建反向图
        reverse_graph = defaultdict(list)
        for v in self.vertices:
            for neighbor in self.graph[v]:
                reverse_graph[neighbor].append(v)
        
        # 第二次DFS，在反向图上按完成时间逆序遍历
        visited.clear()
        sccs = []
        
        def dfs2(v, scc):
            visited.add(v)
            scc.append(v)
            for neighbor in reverse_graph[v]:
                if neighbor not in visited:
                    dfs2(neighbor, scc)
        
        while finish_stack:
            vertex = finish_stack.pop()
            if vertex not in visited:
                scc = []
                dfs2(vertex, scc)
                sccs.append(scc)
        
        return sccs
    
    def bridges(self):
        """
        找出所有桥（割边）
        Tarjan算法
        """
        if self.directed:
            raise ValueError("桥的查找只适用于无向图")
        
        visited = set()
        disc = {}  # 发现时间
        low = {}   # 最低可达时间
        parent = {}
        bridges = []
        time = [0]
        
        def dfs(u):
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in self.graph[u]:
                if v not in visited:
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # 如果low[v] > disc[u]，则(u,v)是桥
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        for vertex in self.vertices:
            if vertex not in visited:
                dfs(vertex)
        
        return bridges
    
    def articulation_points(self):
        """
        找出所有割点（关节点）
        """
        if self.directed:
            raise ValueError("割点查找只适用于无向图")
        
        visited = set()
        disc = {}
        low = {}
        parent = {}
        articulation = set()
        time = [0]
        
        def dfs(u):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in self.graph[u]:
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # 根节点的特殊情况
                    if u not in parent and children > 1:
                        articulation.add(u)
                    
                    # 非根节点
                    if u in parent and low[v] >= disc[u]:
                        articulation.add(u)
                
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        for vertex in self.vertices:
            if vertex not in visited:
                parent.clear()
                dfs(vertex)
        
        return list(articulation)
    
    def is_bipartite(self):
        """
        检查图是否为二分图
        使用着色法
        """
        color = {}
        
        def bfs(start):
            queue = deque([start])
            color[start] = 0
            
            while queue:
                u = queue.popleft()
                
                for v in self.graph[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
            
            return True
        
        for vertex in self.vertices:
            if vertex not in color:
                if not bfs(vertex):
                    return False
        
        return True


def test_graph_traversal():
    """测试图遍历算法"""
    print("=== 图遍历算法测试 ===\n")
    
    # 创建测试图
    g = Graph(directed=True)
    edges = [
        (0, 1), (0, 2), (1, 2), (2, 0),
        (2, 3), (3, 3)
    ]
    
    for u, v in edges:
        g.add_edge(u, v)
    
    print("图的边:", edges)
    
    # BFS测试
    print("\nBFS遍历 (从0开始):", g.bfs(0))
    
    # DFS测试
    print("DFS遍历 (从0开始):", g.dfs(0))
    print("DFS递归 (从0开始):", g.dfs_recursive(0))
    
    # 环检测
    print("\n有向图环检测:", g.has_cycle_directed())
    
    # 创建DAG测试拓扑排序
    dag = Graph(directed=True)
    dag_edges = [
        (5, 2), (5, 0), (4, 0), (4, 1),
        (2, 3), (3, 1)
    ]
    
    for u, v in dag_edges:
        dag.add_edge(u, v)
    
    print("\nDAG的边:", dag_edges)
    print("拓扑排序 (DFS):", dag.topological_sort_dfs())
    print("拓扑排序 (Kahn):", dag.topological_sort_kahn())
    
    # 强连通分量测试
    scc_graph = Graph(directed=True)
    scc_edges = [
        (0, 1), (1, 2), (2, 0),  # SCC 1
        (2, 3), (3, 4), (4, 3),  # 连接 + SCC 2
        (5, 6), (6, 5)           # SCC 3
    ]
    
    for u, v in scc_edges:
        scc_graph.add_edge(u, v)
    
    print("\n强连通分量测试:")
    print("图的边:", scc_edges)
    print("强连通分量:", scc_graph.strongly_connected_components())
    
    # 无向图测试
    ug = Graph(directed=False)
    ug_edges = [
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6)
    ]
    
    for u, v in ug_edges:
        ug.add_edge(u, v)
    
    print("\n无向图测试:")
    print("边:", ug_edges)
    print("连通分量:", ug.connected_components())
    
    # 路径查找
    print("\n路径查找:")
    path_graph = Graph(directed=False)
    path_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    
    for u, v in path_edges:
        path_graph.add_edge(u, v)
    
    print("最短路径 0->4:", path_graph.bfs_shortest_path(0, 4))
    print("所有路径 0->4:", path_graph.dfs_all_paths(0, 4))
    
    # 二分图测试
    bi_graph = Graph(directed=False)
    bi_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    for u, v in bi_edges:
        bi_graph.add_edge(u, v)
    
    print("\n二分图测试:")
    print("边:", bi_edges)
    print("是否二分图:", bi_graph.is_bipartite())


if __name__ == '__main__':
    test_graph_traversal()