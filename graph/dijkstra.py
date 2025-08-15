import heapq
from collections import defaultdict


def dijkstra(graph, start):
    """
    Dijkstra最短路径算法
    时间复杂度：O((V + E) log V)，使用堆优化
    空间复杂度：O(V)
    
    参数:
        graph: 邻接表表示的图，graph[u] = [(v, weight), ...]
        start: 起始节点
    
    返回:
        distances: 从起始节点到所有其他节点的最短距离
        parents: 最短路径树的父节点记录
    """
    # 初始化距离字典
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # 父节点记录，用于重建路径
    parents = {node: None for node in graph}
    
    # 优先队列：(距离, 节点)
    pq = [(0, start)]
    
    # 已访问节点集合
    visited = set()
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        # 如果已访问过，跳过
        if current in visited:
            continue
        
        visited.add(current)
        
        # 如果当前距离大于已知最短距离，跳过
        if current_distance > distances[current]:
            continue
        
        # 遍历邻居节点
        for neighbor, weight in graph[current]:
            distance = current_distance + weight
            
            # 如果找到更短的路径，更新
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    return distances, parents


def dijkstra_with_path(graph, start, end):
    """
    Dijkstra算法，返回具体路径
    """
    distances, parents = dijkstra(graph, start)
    
    if distances[end] == float('inf'):
        return None, float('inf')
    
    # 重建路径
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents[current]
    
    path.reverse()
    return path, distances[end]


class Graph:
    """图类，支持Dijkstra算法"""
    
    def __init__(self, directed=False):
        """
        初始化图
        directed: 是否为有向图
        """
        self.graph = defaultdict(list)
        self.directed = directed
        self.nodes = set()
    
    def add_edge(self, u, v, weight=1):
        """添加边"""
        self.graph[u].append((v, weight))
        self.nodes.add(u)
        self.nodes.add(v)
        
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def dijkstra(self, start):
        """运行Dijkstra算法"""
        return dijkstra(self.graph, start)
    
    def shortest_path(self, start, end):
        """获取最短路径"""
        return dijkstra_with_path(self.graph, start, end)
    
    def all_pairs_shortest_paths(self):
        """
        计算所有节点对之间的最短路径
        使用多次Dijkstra算法
        """
        all_paths = {}
        
        for node in self.nodes:
            distances, _ = self.dijkstra(node)
            all_paths[node] = distances
        
        return all_paths
    
    def print_graph(self):
        """打印图的结构"""
        print("图结构:")
        for node in sorted(self.nodes):
            edges = self.graph[node]
            if edges:
                edge_str = ", ".join(f"{v}(w={w})" for v, w in edges)
                print(f"  {node} -> {edge_str}")
            else:
                print(f"  {node} -> (无出边)")


def dijkstra_2d_grid(grid, start, end):
    """
    在2D网格上运行Dijkstra算法
    grid[i][j] 表示通过该格子的代价
    """
    rows, cols = len(grid), len(grid[0])
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = grid[start[0]][start[1]]
    
    # 优先队列：(距离, 行, 列)
    pq = [(grid[start[0]][start[1]], start[0], start[1])]
    
    # 四个方向：上、右、下、左
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # 父节点记录
    parents = {}
    
    while pq:
        curr_dist, row, col = heapq.heappop(pq)
        
        if (row, col) == end:
            break
        
        if curr_dist > distances[row][col]:
            continue
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_dist = curr_dist + grid[new_row][new_col]
                
                if new_dist < distances[new_row][new_col]:
                    distances[new_row][new_col] = new_dist
                    parents[(new_row, new_col)] = (row, col)
                    heapq.heappush(pq, (new_dist, new_row, new_col))
    
    # 重建路径
    if distances[end[0]][end[1]] == float('inf'):
        return None, float('inf')
    
    path = []
    current = end
    while current in parents:
        path.append(current)
        current = parents[current]
    path.append(start)
    path.reverse()
    
    return path, distances[end[0]][end[1]]


def dijkstra_with_constraints(graph, start, end, max_stops=None):
    """
    带约束的Dijkstra算法
    max_stops: 最大中转次数限制
    """
    # 状态：(距离, 节点, 经过的节点数)
    pq = [(0, start, 0)]
    
    # 记录到达每个节点的最短距离和对应的步数
    best = {}
    
    while pq:
        dist, node, stops = heapq.heappop(pq)
        
        # 如果超过最大中转次数，跳过
        if max_stops is not None and stops > max_stops:
            continue
        
        # 如果已有更优解，跳过
        if node in best and best[node] <= (dist, stops):
            continue
        
        best[node] = (dist, stops)
        
        # 如果到达终点，返回结果
        if node == end:
            return dist
        
        # 扩展邻居
        for neighbor, weight in graph[node]:
            if neighbor not in best:
                heapq.heappush(pq, (dist + weight, neighbor, stops + 1))
    
    return float('inf')


def test_dijkstra():
    """测试Dijkstra算法"""
    print("=== Dijkstra算法测试 ===\n")
    
    # 创建示例图
    g = Graph(directed=True)
    
    # 添加边
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 1),
        ('B', 'D', 5),
        ('C', 'D', 8),
        ('C', 'E', 10),
        ('D', 'E', 2),
        ('D', 'F', 6),
        ('E', 'F', 3)
    ]
    
    for u, v, w in edges:
        g.add_edge(u, v, w)
    
    g.print_graph()
    
    # 测试单源最短路径
    print("\n从节点A出发的最短路径:")
    distances, parents = g.dijkstra('A')
    
    for node in sorted(g.nodes):
        if distances[node] == float('inf'):
            print(f"  A -> {node}: 不可达")
        else:
            print(f"  A -> {node}: 距离 = {distances[node]}")
    
    # 测试具体路径
    print("\n具体路径测试:")
    test_pairs = [('A', 'F'), ('A', 'E'), ('B', 'E')]
    
    for start, end in test_pairs:
        path, distance = g.shortest_path(start, end)
        if path:
            path_str = " -> ".join(path)
            print(f"  {start} 到 {end}: {path_str} (距离: {distance})")
        else:
            print(f"  {start} 到 {end}: 不可达")
    
    # 测试2D网格
    print("\n=== 2D网格最短路径 ===")
    grid = [
        [1, 3, 1, 2],
        [2, 8, 9, 1],
        [5, 3, 2, 1],
        [1, 2, 3, 4]
    ]
    
    print("网格代价矩阵:")
    for row in grid:
        print("  ", row)
    
    start = (0, 0)
    end = (3, 3)
    path, cost = dijkstra_2d_grid(grid, start, end)
    
    if path:
        print(f"\n从 {start} 到 {end} 的最短路径:")
        print(f"  路径: {path}")
        print(f"  总代价: {cost}")
        
        # 可视化路径
        print("\n路径可视化 (*表示路径):")
        for i in range(len(grid)):
            row_str = "  "
            for j in range(len(grid[0])):
                if (i, j) in path:
                    row_str += "*" + str(grid[i][j]) + " "
                else:
                    row_str += " " + str(grid[i][j]) + " "
            print(row_str)
    
    # 测试带约束的Dijkstra
    print("\n=== 带约束的Dijkstra ===")
    print("限制最大中转次数为2:")
    
    g2 = Graph(directed=False)
    edges2 = [
        ('A', 'B', 1),
        ('A', 'C', 4),
        ('B', 'C', 2),
        ('B', 'D', 5),
        ('C', 'D', 1)
    ]
    
    for u, v, w in edges2:
        g2.add_edge(u, v, w)
    
    for max_stops in [None, 1, 2, 3]:
        dist = dijkstra_with_constraints(g2.graph, 'A', 'D', max_stops)
        if max_stops is None:
            print(f"  无限制: 最短距离 = {dist}")
        else:
            print(f"  最多{max_stops}次中转: 最短距离 = {dist}")
    
    # 性能分析
    print("\n=== 算法复杂度分析 ===")
    print("时间复杂度:")
    print("  - 使用二叉堆: O((V + E) log V)")
    print("  - 使用斐波那契堆: O(E + V log V)")
    print("  - 朴素实现: O(V²)")
    print("\n空间复杂度: O(V)")
    print("\n适用场景:")
    print("  - 单源最短路径")
    print("  - 非负权重图")
    print("  - 稀疏图效果更好")


if __name__ == '__main__':
    test_dijkstra()