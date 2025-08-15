import math
from collections import defaultdict, deque
import heapq


class MatrixAlgorithms:
    """矩阵算法集合"""
    
    @staticmethod
    def matrix_multiply(A, B):
        """
        矩阵乘法
        时间复杂度：O(n³)
        """
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        
        C = [[0] * cols_B for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i][j] += A[i][k] * B[k][j]
        
        return C
    
    @staticmethod
    def matrix_power(matrix, n):
        """
        矩阵快速幂
        时间复杂度：O(m³ log n)，m为矩阵大小
        """
        size = len(matrix)
        result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        
        while n > 0:
            if n & 1:
                result = MatrixAlgorithms.matrix_multiply(result, matrix)
            matrix = MatrixAlgorithms.matrix_multiply(matrix, matrix)
            n >>= 1
        
        return result
    
    @staticmethod
    def spiral_order(matrix):
        """
        螺旋遍历矩阵
        时间复杂度：O(mn)
        """
        if not matrix:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # 向右
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1
            
            # 向下
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1
            
            # 向左
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])
                bottom -= 1
            
            # 向上
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1
        
        return result
    
    @staticmethod
    def rotate_matrix_90(matrix):
        """
        原地旋转矩阵90度
        时间复杂度：O(n²)
        空间复杂度：O(1)
        """
        n = len(matrix)
        
        # 转置
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # 反转每行
        for i in range(n):
            matrix[i].reverse()
        
        return matrix


class IntervalAlgorithms:
    """区间算法"""
    
    @staticmethod
    def merge_intervals(intervals):
        """
        合并区间
        时间复杂度：O(n log n)
        """
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for interval in intervals[1:]:
            if interval[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], interval[1])
            else:
                merged.append(interval)
        
        return merged
    
    @staticmethod
    def insert_interval(intervals, new_interval):
        """
        插入区间
        时间复杂度：O(n)
        """
        result = []
        i = 0
        n = len(intervals)
        
        # 添加所有在新区间之前的区间
        while i < n and intervals[i][1] < new_interval[0]:
            result.append(intervals[i])
            i += 1
        
        # 合并重叠区间
        while i < n and intervals[i][0] <= new_interval[1]:
            new_interval[0] = min(new_interval[0], intervals[i][0])
            new_interval[1] = max(new_interval[1], intervals[i][1])
            i += 1
        
        result.append(new_interval)
        
        # 添加剩余区间
        while i < n:
            result.append(intervals[i])
            i += 1
        
        return result
    
    @staticmethod
    def interval_scheduling(intervals):
        """
        区间调度（最多不重叠区间）
        贪心算法
        时间复杂度：O(n log n)
        """
        if not intervals:
            return 0
        
        # 按结束时间排序
        intervals.sort(key=lambda x: x[1])
        
        count = 1
        end = intervals[0][1]
        
        for i in range(1, len(intervals)):
            if intervals[i][0] >= end:
                count += 1
                end = intervals[i][1]
        
        return count


class MonotonicStack:
    """单调栈算法"""
    
    @staticmethod
    def next_greater_element(nums):
        """
        下一个更大元素
        时间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                result[idx] = nums[i]
            stack.append(i)
        
        return result
    
    @staticmethod
    def daily_temperatures(temperatures):
        """
        每日温度（等待更高温度的天数）
        时间复杂度：O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []
        
        for i in range(n):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                idx = stack.pop()
                result[idx] = i - idx
            stack.append(i)
        
        return result
    
    @staticmethod
    def largest_rectangle_histogram(heights):
        """
        柱状图中最大矩形
        时间复杂度：O(n)
        """
        stack = []
        max_area = 0
        heights.append(0)  # 哨兵
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height_idx = stack.pop()
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, heights[height_idx] * width)
            stack.append(i)
        
        return max_area
    
    @staticmethod
    def trap_rain_water(heights):
        """
        接雨水
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not heights:
            return 0
        
        left, right = 0, len(heights) - 1
        left_max = right_max = 0
        water = 0
        
        while left < right:
            if heights[left] < heights[right]:
                if heights[left] >= left_max:
                    left_max = heights[left]
                else:
                    water += left_max - heights[left]
                left += 1
            else:
                if heights[right] >= right_max:
                    right_max = heights[right]
                else:
                    water += right_max - heights[right]
                right -= 1
        
        return water


class TwoPointers:
    """双指针算法"""
    
    @staticmethod
    def three_sum(nums):
        """
        三数之和
        时间复杂度：O(n²)
        """
        nums.sort()
        n = len(nums)
        result = []
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left, right = i + 1, n - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        
        return result
    
    @staticmethod
    def container_with_most_water(heights):
        """
        盛水最多的容器
        时间复杂度：O(n)
        """
        left, right = 0, len(heights) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            height = min(heights[left], heights[right])
            max_area = max(max_area, width * height)
            
            if heights[left] < heights[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    @staticmethod
    def remove_duplicates(nums):
        """
        删除有序数组中的重复项
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not nums:
            return 0
        
        slow = 0
        
        for fast in range(1, len(nums)):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
        
        return slow + 1


class BitManipulation:
    """位运算技巧"""
    
    @staticmethod
    def single_number(nums):
        """
        只出现一次的数字（其他都出现两次）
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    @staticmethod
    def count_bits(n):
        """
        计算0到n每个数的二进制中1的个数
        时间复杂度：O(n)
        """
        result = [0] * (n + 1)
        for i in range(1, n + 1):
            result[i] = result[i >> 1] + (i & 1)
        return result
    
    @staticmethod
    def hamming_distance(x, y):
        """
        汉明距离
        """
        xor = x ^ y
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1  # 清除最低位的1
        return distance
    
    @staticmethod
    def power_of_two(n):
        """
        判断是否是2的幂
        """
        return n > 0 and (n & (n - 1)) == 0
    
    @staticmethod
    def reverse_bits(n):
        """
        翻转二进制位
        """
        result = 0
        for _ in range(32):
            result = (result << 1) | (n & 1)
            n >>= 1
        return result


class MathAlgorithms:
    """数学算法"""
    
    @staticmethod
    def sqrt_newton(x):
        """
        牛顿法求平方根
        时间复杂度：O(log n)
        """
        if x < 0:
            return None
        if x == 0:
            return 0
        
        guess = x
        epsilon = 1e-10
        
        while abs(guess * guess - x) > epsilon:
            guess = (guess + x / guess) / 2
        
        return guess
    
    @staticmethod
    def pow_fast(x, n):
        """
        快速幂（支持负数）
        时间复杂度：O(log n)
        """
        if n == 0:
            return 1
        
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1
        while n:
            if n & 1:
                result *= x
            x *= x
            n >>= 1
        
        return result
    
    @staticmethod
    def catalan_number(n):
        """
        卡特兰数
        C(n) = C(0)C(n-1) + C(1)C(n-2) + ... + C(n-1)C(0)
        """
        if n <= 1:
            return 1
        
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] += dp[j] * dp[i - 1 - j]
        
        return dp[n]
    
    @staticmethod
    def josephus_problem(n, k):
        """
        约瑟夫环问题
        n个人围成圈，每次数k个人淘汰
        返回最后幸存者的位置（0-indexed）
        """
        if n == 1:
            return 0
        return (MathAlgorithms.josephus_problem(n - 1, k) + k) % n
    
    @staticmethod
    def gray_code(n):
        """
        格雷码生成
        """
        result = []
        for i in range(1 << n):
            result.append(i ^ (i >> 1))
        return result


class GameTheory:
    """博弈论算法"""
    
    @staticmethod
    def nim_game(piles):
        """
        Nim游戏
        返回先手是否必胜
        """
        xor_sum = 0
        for pile in piles:
            xor_sum ^= pile
        return xor_sum != 0
    
    @staticmethod
    def can_win_nim(n):
        """
        简单Nim游戏（每次取1-3个）
        """
        return n % 4 != 0
    
    @staticmethod
    def stone_game_dp(piles):
        """
        石子游戏（动态规划）
        两人轮流从两端取石子
        """
        n = len(piles)
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = piles[i]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = max(piles[i] - dp[i + 1][j],
                             piles[j] - dp[i][j - 1])
        
        return dp[0][n - 1] > 0


class StringAlgorithms2:
    """字符串算法补充"""
    
    @staticmethod
    def longest_palindrome_expand(s):
        """
        最长回文子串（中心扩展）
        时间复杂度：O(n²)
        空间复杂度：O(1)
        """
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        if not s:
            return ""
        
        start = end = 0
        
        for i in range(len(s)):
            len1 = expand_around_center(i, i)      # 奇数长度
            len2 = expand_around_center(i, i + 1)  # 偶数长度
            max_len = max(len1, len2)
            
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        
        return s[start:end + 1]
    
    @staticmethod
    def word_break(s, word_dict):
        """
        单词拆分
        时间复杂度：O(n²)
        """
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_dict:
                    dp[i] = True
                    break
        
        return dp[n]
    
    @staticmethod
    def min_edit_distance(word1, word2):
        """
        最小编辑距离
        时间复杂度：O(mn)
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],    # 删除
                                      dp[i][j - 1],      # 插入
                                      dp[i - 1][j - 1])  # 替换
        
        return dp[m][n]


def test_competitive_algorithms():
    """测试竞赛算法"""
    print("=== 算法竞赛常见算法测试 ===\n")
    
    # 测试矩阵算法
    print("1. 矩阵算法:")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    spiral = MatrixAlgorithms.spiral_order(matrix)
    print(f"   螺旋遍历: {spiral}")
    
    # 矩阵快速幂（斐波那契）
    fib_matrix = [[1, 1], [1, 0]]
    result = MatrixAlgorithms.matrix_power(fib_matrix, 10)
    print(f"   第10个斐波那契数: {result[0][1]}")
    
    # 测试区间算法
    print("\n2. 区间算法:")
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    merged = IntervalAlgorithms.merge_intervals(intervals)
    print(f"   合并区间: {merged}")
    
    # 测试单调栈
    print("\n3. 单调栈:")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    days = MonotonicStack.daily_temperatures(temps)
    print(f"   温度: {temps}")
    print(f"   等待天数: {days}")
    
    heights = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    water = MonotonicStack.trap_rain_water(heights)
    print(f"   接雨水: {water}")
    
    # 测试双指针
    print("\n4. 双指针:")
    nums = [-1, 0, 1, 2, -1, -4]
    three_sum_result = TwoPointers.three_sum(nums)
    print(f"   三数之和为0: {three_sum_result}")
    
    # 测试位运算
    print("\n5. 位运算:")
    print(f"   0-5的二进制1个数: {BitManipulation.count_bits(5)}")
    print(f"   8是2的幂: {BitManipulation.power_of_two(8)}")
    print(f"   汉明距离(1,4): {BitManipulation.hamming_distance(1, 4)}")
    
    # 测试数学算法
    print("\n6. 数学算法:")
    print(f"   sqrt(10): {MathAlgorithms.sqrt_newton(10):.4f}")
    print(f"   第5个卡特兰数: {MathAlgorithms.catalan_number(5)}")
    print(f"   约瑟夫环(n=5,k=2): {MathAlgorithms.josephus_problem(5, 2)}")
    print(f"   格雷码(n=3): {MathAlgorithms.gray_code(3)}")
    
    # 测试博弈论
    print("\n7. 博弈论:")
    piles = [3, 7, 2, 3]
    print(f"   Nim游戏{piles}先手必胜: {GameTheory.nim_game(piles)}")
    print(f"   简单Nim(n=4)先手必胜: {GameTheory.can_win_nim(4)}")
    
    # 测试字符串算法
    print("\n8. 字符串算法:")
    s = "babad"
    palindrome = StringAlgorithms2.longest_palindrome_expand(s)
    print(f"   '{s}'的最长回文: {palindrome}")
    
    word_dict = {"leet", "code"}
    can_break = StringAlgorithms2.word_break("leetcode", word_dict)
    print(f"   'leetcode'能否拆分: {can_break}")
    
    # 复杂度总结
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │")
    print("├──────────────────┼──────────────┤")
    print("│ 矩阵快速幂        │ O(m³log n)   │")
    print("│ 区间合并         │ O(n log n)   │")
    print("│ 单调栈           │ O(n)         │")
    print("│ 双指针           │ O(n)         │")
    print("│ 位运算           │ O(1)         │")
    print("│ 卡特兰数         │ O(n²)        │")
    print("│ 最长回文(中心)    │ O(n²)        │")
    print("└──────────────────┴──────────────┘")


if __name__ == '__main__':
    test_competitive_algorithms()