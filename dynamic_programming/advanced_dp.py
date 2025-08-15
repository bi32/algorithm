from collections import defaultdict, deque
import math


class TreeDP:
    """树形动态规划"""
    
    @staticmethod
    def max_independent_set(tree, root):
        """
        最大独立集问题
        在树中选择一些节点，使得没有两个节点相邻
        时间复杂度：O(n)
        """
        def dfs(node, parent):
            # dp[node][0]: 不选择node时的最大值
            # dp[node][1]: 选择node时的最大值
            include = 1  # 选择当前节点
            exclude = 0  # 不选择当前节点
            
            for child in tree[node]:
                if child != parent:
                    child_inc, child_exc = dfs(child, node)
                    include += child_exc  # 选了当前节点，子节点不能选
                    exclude += max(child_inc, child_exc)  # 不选当前节点，子节点可选可不选
            
            return include, exclude
        
        inc, exc = dfs(root, -1)
        return max(inc, exc)
    
    @staticmethod
    def tree_diameter(tree, n):
        """
        树的直径（最长路径）
        时间复杂度：O(n)
        """
        if n == 0:
            return 0
        
        diameter = [0]
        
        def dfs(node, parent):
            first_max = second_max = 0
            
            for child in tree[node]:
                if child != parent:
                    child_height = dfs(child, node)
                    
                    if child_height > first_max:
                        second_max = first_max
                        first_max = child_height
                    elif child_height > second_max:
                        second_max = child_height
            
            diameter[0] = max(diameter[0], first_max + second_max)
            return first_max + 1
        
        dfs(0, -1)
        return diameter[0]
    
    @staticmethod
    def tree_centroid(tree, n):
        """
        找树的重心
        重心：删除后最大子树最小的节点
        时间复杂度：O(n)
        """
        size = [0] * n
        centroid = []
        min_max_subtree = n
        
        def dfs(node, parent):
            nonlocal min_max_subtree
            size[node] = 1
            max_subtree = 0
            
            for child in tree[node]:
                if child != parent:
                    dfs(child, node)
                    size[node] += size[child]
                    max_subtree = max(max_subtree, size[child])
            
            # 考虑上方的子树
            max_subtree = max(max_subtree, n - size[node])
            
            if max_subtree < min_max_subtree:
                min_max_subtree = max_subtree
                centroid.clear()
                centroid.append(node)
            elif max_subtree == min_max_subtree:
                centroid.append(node)
        
        dfs(0, -1)
        return centroid
    
    @staticmethod
    def tree_dp_paths(tree, root, values):
        """
        树上路径DP
        求树上所有路径的最大权值和
        """
        max_path_sum = [float('-inf')]
        
        def dfs(node, parent):
            # 以node为端点的最大路径和
            max_ending_here = values[node]
            
            for child in tree[node]:
                if child != parent:
                    child_max = dfs(child, node)
                    # 更新全局最大值（node作为路径的转折点）
                    max_path_sum[0] = max(max_path_sum[0], 
                                         max_ending_here + max(0, child_max))
                    # 更新以node为端点的最大路径
                    max_ending_here = max(max_ending_here, 
                                         values[node] + child_max)
            
            max_path_sum[0] = max(max_path_sum[0], max_ending_here)
            return max_ending_here
        
        dfs(root, -1)
        return max_path_sum[0]


class BitDP:
    """状态压缩动态规划"""
    
    @staticmethod
    def traveling_salesman(dist):
        """
        旅行商问题（TSP）
        使用状态压缩DP求解
        时间复杂度：O(n² * 2^n)
        空间复杂度：O(n * 2^n)
        """
        n = len(dist)
        # dp[mask][i]: 访问了mask中的城市，当前在城市i的最小代价
        dp = [[float('inf')] * n for _ in range(1 << n)]
        dp[1][0] = 0  # 从城市0开始
        
        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    
                    new_mask = mask | (1 << v)
                    dp[new_mask][v] = min(dp[new_mask][v], 
                                         dp[mask][u] + dist[u][v])
        
        # 找到访问所有城市后回到起点的最小代价
        ans = float('inf')
        full_mask = (1 << n) - 1
        for i in range(1, n):
            ans = min(ans, dp[full_mask][i] + dist[i][0])
        
        return ans
    
    @staticmethod
    def assignment_problem(cost):
        """
        分配问题
        n个人分配n个任务，每人一个任务
        时间复杂度：O(n * 2^n)
        """
        n = len(cost)
        # dp[mask]: 前i个人分配了mask中的任务的最小代价
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # 计算已分配的人数
            person = bin(mask).count('1')
            if person >= n:
                continue
            
            # 尝试给第person个人分配任务
            for task in range(n):
                if not (mask & (1 << task)):
                    new_mask = mask | (1 << task)
                    dp[new_mask] = min(dp[new_mask], 
                                      dp[mask] + cost[person][task])
        
        return dp[(1 << n) - 1]
    
    @staticmethod
    def subset_sum_count(nums, target):
        """
        子集和计数
        有多少个子集的和等于target
        使用状态压缩枚举所有子集
        """
        n = len(nums)
        count = 0
        
        for mask in range(1 << n):
            subset_sum = 0
            for i in range(n):
                if mask & (1 << i):
                    subset_sum += nums[i]
            
            if subset_sum == target:
                count += 1
        
        return count
    
    @staticmethod
    def max_weight_independent_set_graph(weights, edges, n):
        """
        图的最大权独立集（小规模）
        状态压缩DP
        时间复杂度：O(2^n * n)
        """
        # 构建邻接矩阵
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        
        max_weight = 0
        
        for mask in range(1 << n):
            # 检查是否是独立集
            is_independent = True
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                for j in range(i + 1, n):
                    if not (mask & (1 << j)):
                        continue
                    if adj[i][j]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            
            if is_independent:
                weight = sum(weights[i] for i in range(n) 
                           if mask & (1 << i))
                max_weight = max(max_weight, weight)
        
        return max_weight


class DigitDP:
    """数位动态规划"""
    
    @staticmethod
    def count_numbers_with_digit(left, right, digit):
        """
        统计[left, right]范围内包含数字digit的个数
        """
        def count_up_to(n):
            if n < 0:
                return 0
            
            s = str(n)
            length = len(s)
            
            # dp[pos][has_digit][tight]
            # pos: 当前位置
            # has_digit: 是否已经包含digit
            # tight: 是否贴着上界
            memo = {}
            
            def dp(pos, has_digit, tight):
                if pos == length:
                    return 1 if has_digit else 0
                
                if (pos, has_digit, tight) in memo:
                    return memo[(pos, has_digit, tight)]
                
                limit = int(s[pos]) if tight else 9
                result = 0
                
                for d in range(0, limit + 1):
                    new_has = has_digit or (d == digit)
                    new_tight = tight and (d == limit)
                    result += dp(pos + 1, new_has, new_tight)
                
                memo[(pos, has_digit, tight)] = result
                return result
            
            return dp(0, False, True)
        
        return count_up_to(right) - count_up_to(left - 1)
    
    @staticmethod
    def count_numbers_divisible_by_k(left, right, k):
        """
        统计[left, right]范围内能被k整除的数的个数
        """
        def count_up_to(n):
            if n < 0:
                return 0
            
            s = str(n)
            length = len(s)
            
            # dp[pos][remainder][tight]
            memo = {}
            
            def dp(pos, remainder, tight):
                if pos == length:
                    return 1 if remainder == 0 else 0
                
                if (pos, remainder, tight) in memo:
                    return memo[(pos, remainder, tight)]
                
                limit = int(s[pos]) if tight else 9
                result = 0
                
                for d in range(0, limit + 1):
                    new_remainder = (remainder * 10 + d) % k
                    new_tight = tight and (d == limit)
                    result += dp(pos + 1, new_remainder, new_tight)
                
                memo[(pos, remainder, tight)] = result
                return result
            
            return dp(0, 0, True)
        
        return count_up_to(right) - count_up_to(left - 1)
    
    @staticmethod
    def sum_of_digits_in_range(left, right):
        """
        计算[left, right]范围内所有数的数位和
        """
        def sum_up_to(n):
            if n < 0:
                return 0
            
            s = str(n)
            length = len(s)
            
            # dp返回(count, sum)
            memo = {}
            
            def dp(pos, digit_sum, tight):
                if pos == length:
                    return (1, digit_sum)
                
                if (pos, digit_sum, tight) in memo:
                    return memo[(pos, digit_sum, tight)]
                
                limit = int(s[pos]) if tight else 9
                count = 0
                total = 0
                
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    cnt, sm = dp(pos + 1, digit_sum + d, new_tight)
                    count += cnt
                    total += sm
                
                memo[(pos, digit_sum, tight)] = (count, total)
                return (count, total)
            
            _, total_sum = dp(0, 0, True)
            return total_sum
        
        return sum_up_to(right) - sum_up_to(left - 1)


class ProbabilityDP:
    """期望/概率动态规划"""
    
    @staticmethod
    def dice_sum_probability(n, target):
        """
        n个骰子，求和为target的概率
        """
        # dp[i][j]: i个骰子和为j的方案数
        dp = [[0] * (6 * n + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(i, 6 * i + 1):
                for k in range(1, 7):
                    if j >= k:
                        dp[i][j] += dp[i-1][j-k]
        
        if target < n or target > 6 * n:
            return 0.0
        
        return dp[n][target] / (6 ** n)
    
    @staticmethod
    def expected_value_game(n, p_win, win_value, lose_value):
        """
        游戏期望值
        每局获胜概率p_win，赢得win_value，输掉lose_value
        求n局后的期望收益
        """
        # dp[i]: i局后的期望收益
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = dp[i-1] + p_win * win_value + (1 - p_win) * lose_value
        
        return dp[n]
    
    @staticmethod
    def random_walk_probability(steps, target):
        """
        随机游走
        每步向左或向右概率各0.5
        求steps步后到达target的概率
        """
        # dp[i][j]: i步后在位置j的概率
        max_pos = steps + abs(target) + 1
        dp = defaultdict(lambda: defaultdict(float))
        dp[0][0] = 1.0
        
        for i in range(steps):
            for pos in range(-max_pos, max_pos + 1):
                if dp[i][pos] > 0:
                    dp[i+1][pos-1] += dp[i][pos] * 0.5
                    dp[i+1][pos+1] += dp[i][pos] * 0.5
        
        return dp[steps][target]
    
    @staticmethod
    def markov_chain_steady_state(transition_matrix):
        """
        马尔可夫链稳态分布
        使用幂迭代法
        """
        n = len(transition_matrix)
        state = [1.0 / n] * n
        
        for _ in range(1000):  # 迭代足够多次
            new_state = [0] * n
            for j in range(n):
                for i in range(n):
                    new_state[j] += state[i] * transition_matrix[i][j]
            state = new_state
        
        return state


class OptimizationDP:
    """优化动态规划"""
    
    @staticmethod
    def convex_hull_optimization(lines, queries):
        """
        凸包优化（CHT）
        用于优化形如 dp[i] = min(dp[j] + a[j] * b[i]) 的转移
        """
        def bad(l1, l2, l3):
            """检查l2是否应该被删除"""
            # (y2-y1)/(x2-x1) >= (y3-y2)/(x3-x2)
            return ((l2[1] - l1[1]) * (l3[0] - l2[0]) >= 
                   (l3[1] - l2[1]) * (l2[0] - l1[0]))
        
        hull = []
        for line in lines:
            while len(hull) >= 2 and bad(hull[-2], hull[-1], line):
                hull.pop()
            hull.append(line)
        
        def query(x):
            """查询x处的最小值"""
            left, right = 0, len(hull) - 1
            while left < right:
                mid = (left + right) // 2
                if hull[mid][0] * x + hull[mid][1] > hull[mid+1][0] * x + hull[mid+1][1]:
                    left = mid + 1
                else:
                    right = mid
            return hull[left][0] * x + hull[left][1]
        
        return [query(q) for q in queries]
    
    @staticmethod
    def slope_optimization(costs, k):
        """
        斜率优化DP
        将数组分成k段，最小化每段的代价和
        """
        n = len(costs)
        if k >= n:
            return sum(costs)
        
        # 前缀和
        prefix = [0]
        for c in costs:
            prefix.append(prefix[-1] + c)
        
        # dp[i][j]: 前i个元素分成j段的最小代价
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for j in range(1, k + 1):
            for i in range(j, n + 1):
                for l in range(j - 1, i):
                    # [l+1, i]作为一段
                    segment_cost = (prefix[i] - prefix[l]) ** 2
                    dp[i][j] = min(dp[i][j], dp[l][j-1] + segment_cost)
        
        return dp[n][k]


def test_advanced_dp():
    """测试高级动态规划算法"""
    print("=== 高级动态规划测试 ===\n")
    
    # 测试树形DP
    print("1. 树形动态规划:")
    tree = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0],
        3: [1],
        4: [1]
    }
    
    print(f"   最大独立集: {TreeDP.max_independent_set(tree, 0)}")
    print(f"   树的直径: {TreeDP.tree_diameter(tree, 5)}")
    print(f"   树的重心: {TreeDP.tree_centroid(tree, 5)}")
    
    values = [1, -2, 3, 4, -1]
    print(f"   最大路径和: {TreeDP.tree_dp_paths(tree, 0, values)}")
    
    # 测试状态压缩DP
    print("\n2. 状态压缩DP:")
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    print(f"   TSP最短路径: {BitDP.traveling_salesman(dist)}")
    
    cost = [
        [9, 2, 7],
        [6, 4, 3],
        [5, 8, 1]
    ]
    print(f"   分配问题最小代价: {BitDP.assignment_problem(cost)}")
    
    nums = [1, 2, 3, 4]
    target = 5
    print(f"   和为{target}的子集数: {BitDP.subset_sum_count(nums, target)}")
    
    # 测试数位DP
    print("\n3. 数位DP:")
    left, right = 10, 100
    digit = 7
    print(f"   [{left}, {right}]中包含{digit}的数: "
          f"{DigitDP.count_numbers_with_digit(left, right, digit)}")
    
    k = 3
    print(f"   [{left}, {right}]中能被{k}整除的数: "
          f"{DigitDP.count_numbers_divisible_by_k(left, right, k)}")
    
    print(f"   [{left}, {right}]所有数的数位和: "
          f"{DigitDP.sum_of_digits_in_range(left, right)}")
    
    # 测试概率DP
    print("\n4. 概率/期望DP:")
    n_dice = 3
    target_sum = 10
    prob = ProbabilityDP.dice_sum_probability(n_dice, target_sum)
    print(f"   {n_dice}个骰子和为{target_sum}的概率: {prob:.4f}")
    
    games = 10
    p_win = 0.6
    win_val = 10
    lose_val = -5
    expected = ProbabilityDP.expected_value_game(games, p_win, win_val, lose_val)
    print(f"   {games}局游戏的期望收益: {expected:.2f}")
    
    steps = 4
    target_pos = 2
    walk_prob = ProbabilityDP.random_walk_probability(steps, target_pos)
    print(f"   {steps}步后到达位置{target_pos}的概率: {walk_prob:.4f}")
    
    # 马尔可夫链
    transition = [
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ]
    steady = ProbabilityDP.markov_chain_steady_state(transition)
    print(f"   马尔可夫链稳态分布: {[f'{p:.3f}' for p in steady]}")
    
    # 测试优化DP
    print("\n5. 优化DP:")
    lines = [(1, 0), (2, -2), (3, -6)]  # y = ax + b形式
    queries = [1, 2, 3]
    results = OptimizationDP.convex_hull_optimization(lines, queries)
    print(f"   凸包优化查询结果: {results}")
    
    # 复杂度分析
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │")
    print("├──────────────────┼──────────────┤")
    print("│ 树形DP           │ O(n)         │")
    print("│ TSP(状态压缩)     │ O(n²·2^n)    │")
    print("│ 数位DP           │ O(d·S·2)     │")
    print("│ 概率DP           │ O(n·m)       │")
    print("│ 凸包优化         │ O(n log n)   │")
    print("└──────────────────┴──────────────┘")


if __name__ == '__main__':
    test_advanced_dp()