def knapsack_01(weights, values, capacity):
    """
    0-1背包问题
    每个物品只能选择一次
    
    时间复杂度：O(n * W)
    空间复杂度：O(n * W)
    
    参数:
        weights: 物品重量列表
        values: 物品价值列表
        capacity: 背包容量
    
    返回:
        max_value: 最大价值
        selected_items: 选中的物品索引列表
    """
    n = len(weights)
    
    # dp[i][w] 表示前i个物品，容量为w时的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充DP表
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 不选第i个物品
            dp[i][w] = dp[i-1][w]
            
            # 如果可以选第i个物品
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
    
    # 回溯找出选中的物品
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
    
    selected_items.reverse()
    return dp[n][capacity], selected_items


def knapsack_01_optimized(weights, values, capacity):
    """
    0-1背包问题（空间优化版）
    使用一维数组
    
    时间复杂度：O(n * W)
    空间复杂度：O(W)
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 从后向前遍历，避免重复使用物品
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_unbounded(weights, values, capacity):
    """
    完全背包问题
    每个物品可以选择无限次
    
    时间复杂度：O(n * W)
    空间复杂度：O(W)
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 从前向后遍历，允许重复使用物品
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_bounded(weights, values, quantities, capacity):
    """
    多重背包问题
    每个物品有限定的数量
    
    时间复杂度：O(n * W * max(quantities))
    空间复杂度：O(W)
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 多重背包转化为0-1背包
        for w in range(capacity, weights[i] - 1, -1):
            for k in range(1, min(quantities[i], w // weights[i]) + 1):
                if w >= k * weights[i]:
                    dp[w] = max(dp[w], dp[w - k * weights[i]] + k * values[i])
    
    return dp[capacity]


def knapsack_bounded_binary(weights, values, quantities, capacity):
    """
    多重背包问题（二进制优化）
    使用二进制拆分优化
    
    时间复杂度：O(n * W * log(max(quantities)))
    空间复杂度：O(W)
    """
    # 二进制拆分物品
    new_weights = []
    new_values = []
    
    for i in range(len(weights)):
        k = 1
        quantity = quantities[i]
        
        while k <= quantity:
            new_weights.append(k * weights[i])
            new_values.append(k * values[i])
            quantity -= k
            k *= 2
        
        if quantity > 0:
            new_weights.append(quantity * weights[i])
            new_values.append(quantity * values[i])
    
    # 转化为0-1背包问题
    return knapsack_01_optimized(new_weights, new_values, capacity)


def fractional_knapsack(weights, values, capacity):
    """
    分数背包问题（贪心算法）
    物品可以分割
    
    时间复杂度：O(n log n)
    空间复杂度：O(n)
    """
    n = len(weights)
    
    # 计算单位价值并排序
    items = [(values[i] / weights[i], weights[i], values[i], i) 
             for i in range(n)]
    items.sort(reverse=True)
    
    total_value = 0
    selected = []
    remaining_capacity = capacity
    
    for unit_value, weight, value, index in items:
        if remaining_capacity >= weight:
            # 完全装入
            total_value += value
            remaining_capacity -= weight
            selected.append((index, weight, value))
        elif remaining_capacity > 0:
            # 部分装入
            fraction = remaining_capacity / weight
            total_value += value * fraction
            selected.append((index, remaining_capacity, value * fraction))
            remaining_capacity = 0
        
        if remaining_capacity == 0:
            break
    
    return total_value, selected


def knapsack_2d(weights, volumes, values, weight_capacity, volume_capacity):
    """
    二维背包问题
    同时考虑重量和体积限制
    
    时间复杂度：O(n * W * V)
    空间复杂度：O(W * V)
    """
    n = len(weights)
    
    # dp[w][v] 表示重量限制为w，体积限制为v时的最大价值
    dp = [[0] * (volume_capacity + 1) for _ in range(weight_capacity + 1)]
    
    for i in range(n):
        # 从后向前遍历
        for w in range(weight_capacity, weights[i] - 1, -1):
            for v in range(volume_capacity, volumes[i] - 1, -1):
                dp[w][v] = max(dp[w][v], 
                              dp[w - weights[i]][v - volumes[i]] + values[i])
    
    return dp[weight_capacity][volume_capacity]


def knapsack_with_dependency(weights, values, dependencies, capacity):
    """
    有依赖的背包问题
    某些物品的选择依赖于其他物品
    
    dependencies[i] = j 表示选择物品i必须先选择物品j
    dependencies[i] = -1 表示物品i没有依赖
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    # 拓扑排序处理依赖关系
    def can_select(item, selected):
        if dependencies[item] == -1:
            return True
        return dependencies[item] in selected
    
    # 使用位掩码表示选中的物品集合
    for mask in range(1 << n):
        selected = []
        total_weight = 0
        total_value = 0
        valid = True
        
        for i in range(n):
            if mask & (1 << i):
                if not can_select(i, selected):
                    valid = False
                    break
                selected.append(i)
                total_weight += weights[i]
                total_value += values[i]
        
        if valid and total_weight <= capacity:
            dp[total_weight] = max(dp[total_weight], total_value)
    
    return max(dp)


def print_knapsack_solution(weights, values, capacity, max_value, selected_items):
    """打印背包问题解决方案"""
    print(f"背包容量: {capacity}")
    print(f"最大价值: {max_value}")
    print("\n选中的物品:")
    
    total_weight = 0
    total_value = 0
    
    for i in selected_items:
        print(f"  物品{i+1}: 重量={weights[i]}, 价值={values[i]}")
        total_weight += weights[i]
        total_value += values[i]
    
    print(f"\n总重量: {total_weight}/{capacity}")
    print(f"总价值: {total_value}")
    print(f"空间利用率: {total_weight/capacity*100:.1f}%")


def test_knapsack():
    """测试背包问题"""
    print("=== 背包问题测试 ===\n")
    
    # 0-1背包测试
    print("1. 0-1背包问题:")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    max_value, selected = knapsack_01(weights, values, capacity)
    print_knapsack_solution(weights, values, capacity, max_value, selected)
    
    # 验证优化版本
    optimized_value = knapsack_01_optimized(weights, values, capacity)
    print(f"优化版本结果: {optimized_value}")
    
    # 完全背包测试
    print("\n2. 完全背包问题:")
    weights2 = [2, 3, 4]
    values2 = [3, 4, 5]
    capacity2 = 10
    
    max_value2 = knapsack_unbounded(weights2, values2, capacity2)
    print(f"背包容量: {capacity2}")
    print(f"物品(可重复): 重量={weights2}, 价值={values2}")
    print(f"最大价值: {max_value2}")
    
    # 多重背包测试
    print("\n3. 多重背包问题:")
    weights3 = [2, 3, 4]
    values3 = [3, 4, 5]
    quantities3 = [2, 1, 3]
    capacity3 = 10
    
    max_value3 = knapsack_bounded(weights3, values3, quantities3, capacity3)
    max_value3_binary = knapsack_bounded_binary(weights3, values3, quantities3, capacity3)
    
    print(f"背包容量: {capacity3}")
    print(f"物品: 重量={weights3}, 价值={values3}, 数量={quantities3}")
    print(f"最大价值(普通): {max_value3}")
    print(f"最大价值(二进制优化): {max_value3_binary}")
    
    # 分数背包测试
    print("\n4. 分数背包问题:")
    weights4 = [10, 20, 30]
    values4 = [60, 100, 120]
    capacity4 = 50
    
    max_value4, selected4 = fractional_knapsack(weights4, values4, capacity4)
    
    print(f"背包容量: {capacity4}")
    print(f"最大价值: {max_value4}")
    print("选中的物品:")
    for idx, weight, value in selected4:
        if weight == weights4[idx]:
            print(f"  物品{idx+1}: 完全装入 (重量={weight}, 价值={value:.1f})")
        else:
            fraction = weight / weights4[idx]
            print(f"  物品{idx+1}: 装入{fraction*100:.1f}% (重量={weight}, 价值={value:.1f})")
    
    # 二维背包测试
    print("\n5. 二维背包问题:")
    weights5 = [2, 3, 4]
    volumes5 = [3, 4, 5]
    values5 = [3, 4, 5]
    weight_capacity5 = 7
    volume_capacity5 = 10
    
    max_value5 = knapsack_2d(weights5, volumes5, values5, 
                             weight_capacity5, volume_capacity5)
    
    print(f"重量限制: {weight_capacity5}, 体积限制: {volume_capacity5}")
    print(f"物品: 重量={weights5}, 体积={volumes5}, 价值={values5}")
    print(f"最大价值: {max_value5}")
    
    # 算法比较
    print("\n=== 背包问题变体比较 ===")
    print("┌──────────────┬─────────────┬──────────┬────────────┐")
    print("│ 问题类型      │ 时间复杂度   │ 空间复杂度│ 特点        │")
    print("├──────────────┼─────────────┼──────────┼────────────┤")
    print("│ 0-1背包      │ O(nW)       │ O(W)     │ 每个物品一次 │")
    print("│ 完全背包      │ O(nW)       │ O(W)     │ 物品无限    │")
    print("│ 多重背包      │ O(nW∑k)     │ O(W)     │ 物品有限    │")
    print("│ 分数背包      │ O(nlogn)    │ O(n)     │ 物品可分割  │")
    print("│ 二维背包      │ O(nWV)      │ O(WV)    │ 多维约束    │")
    print("└──────────────┴─────────────┴──────────┴────────────┘")


if __name__ == '__main__':
    test_knapsack()