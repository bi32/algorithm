class UnionFind:
    """
    并查集数据结构
    用于Kruskal算法
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x):
        """查找根节点（路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """合并两个集合（按秩合并）"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        """检查是否连通"""
        return self.find(x) == self.find(y)


def kruskal(n, edges):
    """
    Kruskal最小生成树算法
    
    时间复杂度：O(E log E)，主要是排序
    空间复杂度：O(V)
    
    参数:
        n: 顶点数
        edges: 边列表，每条边为(u, v, weight)
    
    返回:
        mst_weight: 最小生成树的总权重
        mst_edges: 最小生成树的边
    """
    # 按权重排序
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst_edges = []
    mst_weight = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            mst_weight += weight
            
            # 已经有n-1条边，形成了树
            if len(mst_edges) == n - 1:
                break
    
    # 检查是否连通
    if uf.count != 1:
        return None, []  # 图不连通
    
    return mst_weight, mst_edges


def prim(n, adj_list):
    """
    Prim最小生成树算法
    
    时间复杂度：O(E log V) 使用堆
    空间复杂度：O(V)
    
    参数:
        n: 顶点数
        adj_list: 邻接表，adj_list[u] = [(v, weight), ...]
    
    返回:
        mst_weight: 最小生成树的总权重
        mst_edges: 最小生成树的边
    """
    import heapq
    
    visited = [False] * n
    min_heap = [(0, 0, -1)]  # (weight, vertex, parent)
    mst_edges = []
    mst_weight = 0
    
    while min_heap:
        weight, u, parent = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        
        visited[u] = True
        
        if parent != -1:
            mst_edges.append((parent, u, weight))
            mst_weight += weight
        
        # 添加所有邻边到堆
        for v, w in adj_list[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v, u))
    
    # 检查是否所有顶点都被访问
    if not all(visited):
        return None, []  # 图不连通
    
    return mst_weight, mst_edges


def prim_dense(n, graph):
    """
    Prim算法（稠密图优化版本）
    使用数组而不是堆
    
    时间复杂度：O(V²)
    适合稠密图
    
    参数:
        n: 顶点数
        graph: 邻接矩阵，graph[i][j]为边权重，INF表示无边
    """
    INF = float('inf')
    visited = [False] * n
    min_cost = [INF] * n
    parent = [-1] * n
    
    # 从顶点0开始
    min_cost[0] = 0
    mst_weight = 0
    mst_edges = []
    
    for _ in range(n):
        # 找到未访问的最小权重顶点
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or min_cost[v] < min_cost[u]):
                u = v
        
        if min_cost[u] == INF:
            return None, []  # 图不连通
        
        visited[u] = True
        mst_weight += min_cost[u]
        
        if parent[u] != -1:
            mst_edges.append((parent[u], u, min_cost[u]))
        
        # 更新邻接顶点的最小权重
        for v in range(n):
            if not visited[v] and graph[u][v] < min_cost[v]:
                min_cost[v] = graph[u][v]
                parent[v] = u
    
    return mst_weight, mst_edges


def boruvka(n, edges):
    """
    Borůvka最小生成树算法
    并行友好的算法
    
    时间复杂度：O(E log V)
    """
    uf = UnionFind(n)
    mst_edges = []
    mst_weight = 0
    
    # 将边按照端点组织
    edge_list = edges.copy()
    
    while uf.count > 1:
        # 为每个组件找到最小边
        min_edge = {}
        
        for u, v, weight in edge_list:
            root_u = uf.find(u)
            root_v = uf.find(v)
            
            if root_u != root_v:
                # 更新组件的最小边
                if root_u not in min_edge or weight < min_edge[root_u][2]:
                    min_edge[root_u] = (u, v, weight)
                if root_v not in min_edge or weight < min_edge[root_v][2]:
                    min_edge[root_v] = (u, v, weight)
        
        if not min_edge:
            break
        
        # 添加所有最小边
        for u, v, weight in set(min_edge.values()):
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                mst_weight += weight
    
    if uf.count != 1:
        return None, []
    
    return mst_weight, mst_edges


def second_best_mst(n, edges):
    """
    次小生成树
    找到权重第二小的生成树
    """
    # 先找到MST
    mst_weight, mst_edges = kruskal(n, edges)
    
    if mst_weight is None:
        return None, []
    
    mst_edge_set = set(mst_edges)
    second_best = float('inf')
    second_best_edges = []
    
    # 对每条MST边，尝试替换
    for remove_edge in mst_edges:
        # 创建新的边列表，不包含当前边
        temp_edges = [e for e in edges if e != remove_edge]
        
        # 运行Kruskal
        temp_weight, temp_mst = kruskal(n, temp_edges)
        
        if temp_weight is not None and temp_weight > mst_weight:
            if temp_weight < second_best:
                second_best = temp_weight
                second_best_edges = temp_mst
    
    return second_best if second_best != float('inf') else None, second_best_edges


def minimum_bottleneck_spanning_tree(n, edges):
    """
    最小瓶颈生成树
    最小化生成树中的最大边权重
    """
    # Kruskal算法自然产生MBST
    return kruskal(n, edges)


def degree_constrained_mst(n, edges, max_degree):
    """
    度约束最小生成树
    每个顶点的度不超过max_degree
    """
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    degree = [0] * n
    mst_edges = []
    mst_weight = 0
    
    for u, v, weight in edges:
        # 检查度约束
        if degree[u] < max_degree and degree[v] < max_degree:
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                mst_weight += weight
                degree[u] += 1
                degree[v] += 1
                
                if len(mst_edges) == n - 1:
                    break
    
    if uf.count != 1:
        return None, []
    
    return mst_weight, mst_edges


class SteinerTree:
    """
    Steiner树问题
    连接指定终端节点的最小树（可以包含非终端节点）
    """
    
    @staticmethod
    def approximate(n, edges, terminals):
        """
        2-近似算法
        """
        # 构建完整图的MST
        mst_weight, mst_edges = kruskal(n, edges)
        
        if mst_weight is None:
            return None, []
        
        # 构建MST的邻接表
        adj = [[] for _ in range(n)]
        for u, v, w in mst_edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # 找到包含所有终端的最小子树
        # 这里使用简化方法：保留连接终端的路径
        terminal_set = set(terminals)
        steiner_edges = []
        steiner_weight = 0
        
        # DFS找到连接终端的边
        visited = [False] * n
        
        def dfs(u, parent=-1):
            visited[u] = True
            is_useful = u in terminal_set
            
            for v, w in adj[u]:
                if v != parent and not visited[v]:
                    child_useful = dfs(v, u)
                    if child_useful:
                        steiner_edges.append((u, v, w))
                        steiner_weight_ref[0] += w
                        is_useful = True
            
            return is_useful
        
        steiner_weight_ref = [0]
        dfs(terminals[0])
        
        return steiner_weight_ref[0], steiner_edges


def test_mst():
    """测试最小生成树算法"""
    print("=== 最小生成树算法测试 ===\n")
    
    # 创建测试图
    n = 6
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 1),
        (1, 3, 2), (2, 3, 4), (3, 4, 2),
        (4, 5, 6), (3, 5, 3), (2, 4, 5)
    ]
    
    print(f"顶点数: {n}")
    print(f"边: {edges}\n")
    
    # Kruskal算法
    print("1. Kruskal算法:")
    weight, mst = kruskal(n, edges)
    print(f"   最小生成树权重: {weight}")
    print(f"   MST边: {mst}")
    
    # Prim算法
    print("\n2. Prim算法:")
    # 构建邻接表
    adj_list = [[] for _ in range(n)]
    for u, v, w in edges:
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
    
    weight2, mst2 = prim(n, adj_list)
    print(f"   最小生成树权重: {weight2}")
    print(f"   MST边: {mst2}")
    
    # Borůvka算法
    print("\n3. Borůvka算法:")
    weight3, mst3 = boruvka(n, edges)
    print(f"   最小生成树权重: {weight3}")
    
    # 次小生成树
    print("\n4. 次小生成树:")
    second_weight, second_mst = second_best_mst(n, edges)
    if second_weight:
        print(f"   次小生成树权重: {second_weight}")
    
    # 度约束MST
    print("\n5. 度约束MST (最大度=2):")
    dc_weight, dc_mst = degree_constrained_mst(n, edges, 2)
    if dc_weight:
        print(f"   权重: {dc_weight}")
        print(f"   边: {dc_mst}")
    
    # 测试不连通图
    print("\n6. 不连通图测试:")
    edges_disconnected = [(0, 1, 1), (2, 3, 2)]
    weight_d, mst_d = kruskal(4, edges_disconnected)
    print(f"   结果: {'不连通' if weight_d is None else '连通'}")
    
    # 性能分析
    print("\n=== 算法复杂度分析 ===")
    print("┌─────────────┬────────────┬───────────┐")
    print("│ 算法         │ 时间复杂度  │ 适用场景   │")
    print("├─────────────┼────────────┼───────────┤")
    print("│ Kruskal     │ O(E log E) │ 稀疏图     │")
    print("│ Prim(堆)    │ O(E log V) │ 稀疏图     │")
    print("│ Prim(数组)  │ O(V²)      │ 稠密图     │")
    print("│ Borůvka     │ O(E log V) │ 并行计算   │")
    print("└─────────────┴────────────┴───────────┘")
    
    # 应用场景
    print("\n=== MST应用场景 ===")
    print("• 网络设计：最小成本连接所有节点")
    print("• 聚类分析：单链接聚类")
    print("• 图像分割：基于图的分割算法")
    print("• 近似算法：旅行商问题的近似解")


if __name__ == '__main__':
    test_mst()