import random
import time


def max_subarray_brute_force(arr):
    """
    暴力法求解最大子数组
    时间复杂度：O(n²)
    空间复杂度：O(1)
    """
    n = len(arr)
    max_sum = float('-inf')
    start = end = 0
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += arr[j]
            if current_sum > max_sum:
                max_sum = current_sum
                start = i
                end = j
    
    return max_sum, start, end


def max_crossing_subarray(arr, low, mid, high):
    """
    找出跨越中点的最大子数组
    """
    # 左半部分的最大和
    left_sum = float('-inf')
    sum_val = 0
    max_left = mid
    
    for i in range(mid, low - 1, -1):
        sum_val += arr[i]
        if sum_val > left_sum:
            left_sum = sum_val
            max_left = i
    
    # 右半部分的最大和
    right_sum = float('-inf')
    sum_val = 0
    max_right = mid + 1
    
    for j in range(mid + 1, high + 1):
        sum_val += arr[j]
        if sum_val > right_sum:
            right_sum = sum_val
            max_right = j
    
    return left_sum + right_sum, max_left, max_right


def max_subarray_divide_conquer(arr, low=None, high=None):
    """
    分治法求解最大子数组
    时间复杂度：O(n log n)
    空间复杂度：O(log n) - 递归栈
    """
    if low is None:
        low = 0
        high = len(arr) - 1
    
    # 基本情况：只有一个元素
    if low == high:
        return arr[low], low, high
    
    mid = (low + high) // 2
    
    # 递归求解左半部分
    left_sum, left_low, left_high = max_subarray_divide_conquer(arr, low, mid)
    
    # 递归求解右半部分
    right_sum, right_low, right_high = max_subarray_divide_conquer(arr, mid + 1, high)
    
    # 求解跨越中点的最大子数组
    cross_sum, cross_low, cross_high = max_crossing_subarray(arr, low, mid, high)
    
    # 返回三者中的最大值
    if left_sum >= right_sum and left_sum >= cross_sum:
        return left_sum, left_low, left_high
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return right_sum, right_low, right_high
    else:
        return cross_sum, cross_low, cross_high


def max_subarray_kadane(arr):
    """
    Kadane算法（动态规划）
    时间复杂度：O(n)
    空间复杂度：O(1)
    """
    max_sum = float('-inf')
    current_sum = 0
    start = end = 0
    temp_start = 0
    
    for i in range(len(arr)):
        current_sum += arr[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
        
        if current_sum < 0:
            current_sum = 0
            temp_start = i + 1
    
    return max_sum, start, end


def max_subarray_dp(arr):
    """
    动态规划解法（显式DP数组）
    时间复杂度：O(n)
    空间复杂度：O(n)
    """
    n = len(arr)
    if n == 0:
        return 0, -1, -1
    
    # dp[i] 表示以i结尾的最大子数组和
    dp = [0] * n
    dp[0] = arr[0]
    max_sum = arr[0]
    max_end = 0
    
    # 记录每个位置的起始索引
    start_indices = [0] * n
    
    for i in range(1, n):
        if dp[i-1] > 0:
            dp[i] = dp[i-1] + arr[i]
            start_indices[i] = start_indices[i-1]
        else:
            dp[i] = arr[i]
            start_indices[i] = i
        
        if dp[i] > max_sum:
            max_sum = dp[i]
            max_end = i
    
    return max_sum, start_indices[max_end], max_end


def max_subarray_linear_time(arr):
    """
    线性时间算法（优化版）
    同时返回最大子数组和、起始位置、结束位置
    """
    if not arr:
        return 0, -1, -1
    
    max_sum = arr[0]
    max_start = max_end = 0
    current_sum = arr[0]
    current_start = 0
    
    for i in range(1, len(arr)):
        if current_sum < 0:
            current_sum = arr[i]
            current_start = i
        else:
            current_sum += arr[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            max_start = current_start
            max_end = i
    
    return max_sum, max_start, max_end


def max_subarray_circular(arr):
    """
    环形数组的最大子数组和
    考虑数组首尾相连的情况
    """
    n = len(arr)
    if n == 0:
        return 0
    
    # 情况1：最大子数组不跨越边界（普通情况）
    max_kadane = kadane_max(arr)
    
    # 情况2：最大子数组跨越边界
    # 等价于找最小子数组，然后用总和减去它
    total_sum = sum(arr)
    min_kadane = kadane_min(arr)
    
    # 如果所有元素都是负数，min_kadane会等于total_sum
    if min_kadane == total_sum:
        return max_kadane
    
    return max(max_kadane, total_sum - min_kadane)


def kadane_max(arr):
    """辅助函数：找最大子数组和"""
    max_sum = arr[0]
    current_sum = arr[0]
    
    for i in range(1, len(arr)):
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def kadane_min(arr):
    """辅助函数：找最小子数组和"""
    min_sum = arr[0]
    current_sum = arr[0]
    
    for i in range(1, len(arr)):
        current_sum = min(arr[i], current_sum + arr[i])
        min_sum = min(min_sum, current_sum)
    
    return min_sum


def max_submatrix_sum(matrix):
    """
    二维最大子矩阵和
    时间复杂度：O(n³)
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')
    result = {'sum': max_sum, 'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    for top in range(rows):
        temp = [0] * cols
        
        for bottom in range(top, rows):
            # 将第top行到第bottom行压缩成一维数组
            for i in range(cols):
                temp[i] += matrix[bottom][i]
            
            # 在一维数组上应用Kadane算法
            current_sum, start, end = max_subarray_kadane(temp)
            
            if current_sum > max_sum:
                max_sum = current_sum
                result = {
                    'sum': max_sum,
                    'top': top,
                    'bottom': bottom,
                    'left': start,
                    'right': end
                }
    
    return result


def benchmark_algorithms(arr):
    """性能测试不同算法"""
    algorithms = [
        ("暴力法", max_subarray_brute_force),
        ("分治法", lambda x: max_subarray_divide_conquer(x)),
        ("Kadane算法", max_subarray_kadane),
        ("动态规划", max_subarray_dp),
        ("线性优化", max_subarray_linear_time)
    ]
    
    print(f"数组长度: {len(arr)}")
    print("-" * 50)
    
    for name, func in algorithms:
        start_time = time.time()
        result = func(arr)
        end_time = time.time()
        
        print(f"{name:12s}: {end_time - start_time:.6f}秒")
        if len(arr) <= 20:
            print(f"  结果: 最大和={result[0]}, 区间=[{result[1]}, {result[2]}]")


def test_max_subarray():
    """测试最大子数组算法"""
    print("=== 最大子数组问题 ===\n")
    
    # 测试用例1：普通情况
    arr1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print("测试数组1:", arr1)
    
    for name, func in [
        ("暴力法", max_subarray_brute_force),
        ("分治法", lambda x: max_subarray_divide_conquer(x)),
        ("Kadane", max_subarray_kadane),
        ("动态规划", max_subarray_dp)
    ]:
        result = func(arr1)
        print(f"{name}: 最大和={result[0]}, 子数组={arr1[result[1]:result[2]+1]}")
    
    # 测试用例2：全负数
    print("\n测试数组2（全负数）:")
    arr2 = [-5, -2, -8, -1, -4]
    result = max_subarray_kadane(arr2)
    print(f"最大和={result[0]}, 子数组={arr2[result[1]:result[2]+1]}")
    
    # 测试用例3：环形数组
    print("\n测试环形数组:")
    arr3 = [5, -3, 5]
    print(f"普通数组: {arr3}")
    print(f"环形最大和: {max_subarray_circular(arr3)}")
    
    # 测试用例4：二维矩阵
    print("\n测试二维最大子矩阵:")
    matrix = [
        [1, 2, -1, -4, -20],
        [-8, -3, 4, 2, 1],
        [3, 8, 10, 1, 3],
        [-4, -1, 1, 7, -6]
    ]
    
    result = max_submatrix_sum(matrix)
    print(f"最大子矩阵和: {result['sum']}")
    print(f"位置: 行[{result['top']}, {result['bottom']}], 列[{result['left']}, {result['right']}]")
    
    # 性能测试
    print("\n=== 性能测试 ===")
    for size in [100, 1000]:
        print(f"\n测试规模 n={size}:")
        test_arr = [random.randint(-100, 100) for _ in range(size)]
        benchmark_algorithms(test_arr)
    
    # 算法复杂度分析
    print("\n=== 复杂度分析 ===")
    print("┌─────────────┬──────────┬──────────┐")
    print("│ 算法         │ 时间复杂度 │ 空间复杂度 │")
    print("├─────────────┼──────────┼──────────┤")
    print("│ 暴力法       │ O(n²)    │ O(1)     │")
    print("│ 分治法       │ O(nlogn) │ O(logn)  │")
    print("│ Kadane算法   │ O(n)     │ O(1)     │")
    print("│ 动态规划     │ O(n)     │ O(n)     │")
    print("│ 二维扩展     │ O(n³)    │ O(n)     │")
    print("└─────────────┴──────────┴──────────┘")


if __name__ == '__main__':
    test_max_subarray()