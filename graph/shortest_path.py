import heapq
from collections import defaultdict, deque
import math


def floyd_warshall(n, edges):
    """
    Floyd-Warshall算法
    求所有点对之间的最短路径
    时间复杂度：O(V³)
    空间复杂度：O(V²)
    """
    # 初始化距离矩阵
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    # 对角线设为0
    for i in range(n):
        dist[i][i] = 0
    
    # 填充边的权重
    for u, v, w in edges:
        dist[u][v] = w
        next_node[u][v] = v
    
    # Floyd-Warshall核心算法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # 检测负环
    for i in range(n):
        if dist[i][i] < 0:
            return None, None  # 存在负环
    
    # 构建路径
    def get_path(i, j):
        if dist[i][j] == float('inf'):
            return None
        
        path = [i]
        while i != j:
            i = next_node[i][j]
            if i is None:
                return None
            path.append(i)
        return path
    
    return dist, get_path


def bellman_ford(n, edges, source):
    """
    Bellman-Ford算法
    单源最短路径，可以处理负权边
    时间复杂度：O(VE)
    空间复杂度：O(V)
    """
    # 初始化距离
    dist = [float('inf')] * n
    dist[source] = 0
    parent = [-1] * n
    
    # 松弛操作 V-1 次
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        
        # 如果没有更新，提前终止
        if not updated:
            break
    
    # 检测负环
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None, None  # 存在负环
    
    # 构建路径
    def get_path(target):
        if dist[target] == float('inf'):
            return None
        
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = parent[current]
        
        return list(reversed(path))
    
    return dist, get_path


def spfa(n, graph, source):
    """
    SPFA算法（Bellman-Ford的队列优化）
    时间复杂度：平均O(E)，最坏O(VE)
    """
    dist = [float('inf')] * n
    dist[source] = 0
    parent = [-1] * n
    
    in_queue = [False] * n
    queue = deque([source])
    in_queue[source] = True
    
    # 记录入队次数，用于检测负环
    count = [0] * n
    count[source] = 1
    
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    
                    # 如果某个节点入队超过n次，说明存在负环
                    if count[v] > n:
                        return None, None
    
    return dist, parent


def johnson(n, edges):
    """
    Johnson算法
    使用重新加权的技术求所有点对最短路径
    时间复杂度：O(V²log V + VE)
    """
    # 添加虚拟源点
    virtual_source = n
    extended_edges = edges.copy()
    
    # 从虚拟源点到所有其他点添加权重为0的边
    for i in range(n):
        extended_edges.append((virtual_source, i, 0))
    
    # 使用Bellman-Ford计算从虚拟源点出发的最短路径
    dist_from_virtual, _ = bellman_ford(n + 1, extended_edges, virtual_source)
    
    if dist_from_virtual is None:
        return None  # 存在负环
    
    # 重新加权
    reweighted_edges = []
    graph = defaultdict(list)
    
    for u, v, w in edges:
        new_weight = w + dist_from_virtual[u] - dist_from_virtual[v]
        reweighted_edges.append((u, v, new_weight))
        graph[u].append((v, new_weight))
    
    # 对每个顶点运行Dijkstra算法
    all_pairs_dist = [[float('inf')] * n for _ in range(n)]
    
    for source in range(n):
        # Dijkstra算法
        dist = [float('inf')] * n
        dist[source] = 0
        pq = [(0, source)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        
        # 恢复原始权重
        for target in range(n):
            if dist[target] != float('inf'):
                all_pairs_dist[source][target] = (
                    dist[target] - dist_from_virtual[source] + dist_from_virtual[target]
                )
    
    return all_pairs_dist


class AStar:
    """
    A*搜索算法
    用于在图中寻找最短路径
    """
    
    def __init__(self, graph, heuristic):
        """
        graph: 邻接表表示的图
        heuristic: 启发函数 h(n)
        """
        self.graph = graph
        self.heuristic = heuristic
    
    def search(self, start, goal):
        """
        A*搜索
        时间复杂度：O(b^d)，其中b是分支因子，d是深度
        空间复杂度：O(b^d)
        """
        # f(n) = g(n) + h(n)
        open_set = [(self.heuristic(start, goal), 0, start, [])]
        closed_set = set()
        g_score = {start: 0}
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path + [current], g
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor, cost in self.graph[current]:
                if neighbor in closed_set:
                    continue
                
                tentative_g = g + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(
                        open_set,
                        (f_score, tentative_g, neighbor, path + [current])
                    )
        
        return None, float('inf')


class GridAStar:
    """
    网格上的A*算法实现
    常用于路径规划
    """
    
    def __init__(self, grid):
        """grid: 2D数组，0表示可通过，1表示障碍物"""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def heuristic(self, a, b):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        """获取相邻的可达节点"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        # 如果需要8方向移动，添加对角线
        # directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            new_x, new_y = node[0] + dx, node[1] + dy
            
            if (0 <= new_x < self.rows and 
                0 <= new_y < self.cols and 
                self.grid[new_x][new_y] == 0):
                
                # 对角线移动的代价更高
                cost = 1.414 if abs(dx) + abs(dy) == 2 else 1
                neighbors.append(((new_x, new_y), cost))
        
        return neighbors
    
    def search(self, start, goal):
        """
        在网格上执行A*搜索
        """
        if (self.grid[start[0]][start[1]] == 1 or 
            self.grid[goal[0]][goal[1]] == 1):
            return None, float('inf')
        
        open_set = [(self.heuristic(start, goal), 0, start, [])]
        closed_set = set()
        g_score = {start: 0}
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path + [current], g
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor, cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(
                        open_set,
                        (f_score, tentative_g, neighbor, path + [current])
                    )
        
        return None, float('inf')


def bidirectional_search(graph, start, goal):
    """
    双向搜索
    从起点和终点同时搜索
    时间复杂度：O(b^(d/2))
    """
    if start == goal:
        return [start], 0
    
    # 前向搜索
    forward_visited = {start: (None, 0)}
    forward_queue = deque([(start, 0)])
    
    # 后向搜索
    backward_visited = {goal: (None, 0)}
    backward_queue = deque([(goal, 0)])
    
    def reconstruct_path(meeting_point):
        # 构建前向路径
        path = []
        current = meeting_point
        while current is not None:
            path.append(current)
            current = forward_visited[current][0]
        path.reverse()
        
        # 构建后向路径
        current = backward_visited[meeting_point][0]
        while current is not None:
            path.append(current)
            current = backward_visited[current][0]
        
        return path
    
    while forward_queue and backward_queue:
        # 前向扩展
        if forward_queue:
            current, dist = forward_queue.popleft()
            
            for neighbor, weight in graph[current]:
                if neighbor in backward_visited:
                    # 找到相遇点
                    total_dist = dist + weight + backward_visited[neighbor][1]
                    return reconstruct_path(current), total_dist
                
                if neighbor not in forward_visited:
                    forward_visited[neighbor] = (current, dist + weight)
                    forward_queue.append((neighbor, dist + weight))
        
        # 后向扩展
        if backward_queue:
            current, dist = backward_queue.popleft()
            
            # 需要反向图
            for node in graph:
                for neighbor, weight in graph[node]:
                    if neighbor == current:
                        if node in forward_visited:
                            # 找到相遇点
                            total_dist = forward_visited[node][1] + weight + dist
                            return reconstruct_path(node), total_dist
                        
                        if node not in backward_visited:
                            backward_visited[node] = (current, dist + weight)
                            backward_queue.append((node, dist + weight))
    
    return None, float('inf')


def test_shortest_path():
    """测试最短路径算法"""
    print("=== 最短路径算法测试 ===\n")
    
    # 测试Floyd-Warshall
    print("1. Floyd-Warshall算法:")
    n = 4
    edges = [
        (0, 1, 5), (0, 3, 10),
        (1, 2, 3),
        (2, 3, 1)
    ]
    
    dist, get_path = floyd_warshall(n, edges)
    if dist:
        print("   距离矩阵:")
        for row in dist:
            print(f"   {[d if d != float('inf') else '∞' for d in row]}")
        
        print(f"   从0到3的路径: {get_path(0, 3)}")
    
    # 测试Bellman-Ford
    print("\n2. Bellman-Ford算法:")
    edges_bf = [
        (0, 1, 5), (0, 3, 10),
        (1, 2, 3),
        (2, 3, 1),
        (3, 1, -7)  # 负权边
    ]
    
    dist, get_path = bellman_ford(n, edges_bf, 0)
    if dist:
        print(f"   从0出发的最短距离: {dist}")
        print(f"   从0到3的路径: {get_path(3)}")
    
    # 测试Johnson算法
    print("\n3. Johnson算法:")
    dist_matrix = johnson(n, edges)
    if dist_matrix:
        print("   所有点对最短路径:")
        for i, row in enumerate(dist_matrix):
            print(f"   从{i}: {[d if d != float('inf') else '∞' for d in row]}")
    
    # 测试A*算法（图搜索）
    print("\n4. A*算法（图搜索）:")
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2), ('F', 6)],
        'E': [('F', 3)],
        'F': []
    }
    
    # 简单的启发函数（假设的直线距离）
    h_values = {
        'A': 10, 'B': 8, 'C': 7,
        'D': 6, 'E': 3, 'F': 0
    }
    
    def heuristic(node, goal):
        return h_values.get(node, 0)
    
    astar = AStar(graph, heuristic)
    path, cost = astar.search('A', 'F')
    print(f"   从A到F的路径: {path}")
    print(f"   总代价: {cost}")
    
    # 测试网格A*算法
    print("\n5. A*算法（网格搜索）:")
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    grid_astar = GridAStar(grid)
    start = (0, 0)
    goal = (4, 4)
    path, cost = grid_astar.search(start, goal)
    
    if path:
        print(f"   从{start}到{goal}的路径:")
        # 可视化路径
        path_set = set(path)
        for i in range(len(grid)):
            row_str = "   "
            for j in range(len(grid[0])):
                if (i, j) == start:
                    row_str += "S "
                elif (i, j) == goal:
                    row_str += "G "
                elif (i, j) in path_set:
                    row_str += "* "
                elif grid[i][j] == 1:
                    row_str += "# "
                else:
                    row_str += ". "
            print(row_str)
        print(f"   路径长度: {cost:.2f}")
    
    # 复杂度分析
    print("\n=== 算法复杂度分析 ===")
    print("┌──────────────────┬──────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │ 空间复杂度    │")
    print("├──────────────────┼──────────────┼──────────────┤")
    print("│ Floyd-Warshall   │ O(V³)        │ O(V²)        │")
    print("│ Bellman-Ford     │ O(VE)        │ O(V)         │")
    print("│ SPFA            │ O(kE)        │ O(V)         │")
    print("│ Johnson         │ O(V²logV+VE) │ O(V²)        │")
    print("│ A*              │ O(b^d)       │ O(b^d)       │")
    print("│ 双向搜索         │ O(b^(d/2))   │ O(b^(d/2))   │")
    print("└──────────────────┴──────────────┴──────────────┘")
    
    print("\n=== 算法选择指南 ===")
    print("• 稠密图所有点对：Floyd-Warshall")
    print("• 稀疏图所有点对：Johnson")
    print("• 有负权边单源：Bellman-Ford/SPFA")
    print("• 无负权边单源：Dijkstra")
    print("• 有启发信息：A*")
    print("• 大规模图：双向搜索")


if __name__ == '__main__':
    test_shortest_path()