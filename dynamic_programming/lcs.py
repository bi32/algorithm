def longest_common_subsequence(text1, text2):
    """
    最长公共子序列（LCS）
    时间复杂度：O(mn)
    空间复杂度：O(mn)
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的LCS长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def lcs_with_sequence(text1, text2):
    """
    返回LCS的长度和具体序列
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 回溯构建LCS
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return dp[m][n], ''.join(lcs)


def lcs_space_optimized(text1, text2):
    """
    空间优化的LCS（只需要O(min(m,n))空间）
    """
    # 确保text1是较短的字符串
    if len(text1) > len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    
    # 只需要两行
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text2[i-1] == text1[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        
        prev, curr = curr, prev
    
    return prev[m]


def longest_common_substring(text1, text2):
    """
    最长公共子串（连续）
    时间复杂度：O(mn)
    空间复杂度：O(mn)
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] 表示以text1[i-1]和text2[j-1]结尾的最长公共子串长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    max_length = 0
    ending_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_pos = i
            else:
                dp[i][j] = 0
    
    # 提取子串
    substring = text1[ending_pos - max_length:ending_pos] if max_length > 0 else ""
    
    return max_length, substring


def edit_distance(word1, word2):
    """
    编辑距离（Levenshtein距离）
    可以进行插入、删除、替换操作
    时间复杂度：O(mn)
    空间复杂度：O(mn)
    """
    m, n = len(word1), len(word2)
    
    # dp[i][j] 表示word1[0:i]转换到word2[0:j]的最小操作数
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i  # 删除i个字符
    for j in range(n + 1):
        dp[0][j] = j  # 插入j个字符
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # 删除
                    dp[i][j-1],      # 插入
                    dp[i-1][j-1]     # 替换
                )
    
    return dp[m][n]


def edit_distance_with_operations(word1, word2):
    """
    返回编辑距离和具体的操作序列
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # 回溯找出操作序列
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i == 0:
            operations.append(f"插入 '{word2[j-1]}' 在位置 {i}")
            j -= 1
        elif j == 0:
            operations.append(f"删除 '{word1[i-1]}' 在位置 {i-1}")
            i -= 1
        elif word1[i-1] == word2[j-1]:
            i -= 1
            j -= 1
        else:
            min_val = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            if dp[i-1][j-1] == min_val:
                operations.append(f"替换 '{word1[i-1]}' 为 '{word2[j-1]}' 在位置 {i-1}")
                i -= 1
                j -= 1
            elif dp[i-1][j] == min_val:
                operations.append(f"删除 '{word1[i-1]}' 在位置 {i-1}")
                i -= 1
            else:
                operations.append(f"插入 '{word2[j-1]}' 在位置 {i}")
                j -= 1
    
    operations.reverse()
    return dp[m][n], operations


def longest_palindromic_subsequence(s):
    """
    最长回文子序列
    时间复杂度：O(n²)
    空间复杂度：O(n²)
    """
    n = len(s)
    
    # dp[i][j] 表示s[i:j+1]的最长回文子序列长度
    dp = [[0] * n for _ in range(n)]
    
    # 单个字符是回文
    for i in range(n):
        dp[i][i] = 1
    
    # 从长度2开始
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    return dp[0][n-1]


def longest_palindromic_substring(s):
    """
    最长回文子串
    时间复杂度：O(n²)
    空间复杂度：O(n²)
    """
    if not s:
        return ""
    
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    
    start = 0
    max_length = 1
    
    # 单个字符是回文
    for i in range(n):
        dp[i][i] = True
    
    # 检查长度为2的子串
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2
    
    # 检查长度大于2的子串
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_length = length
    
    return s[start:start + max_length]


def matrix_chain_multiplication(dims):
    """
    矩阵链乘法
    dims[i] 表示第i个矩阵的行数，dims[i+1]表示列数
    时间复杂度：O(n³)
    空间复杂度：O(n²)
    """
    n = len(dims) - 1  # 矩阵数量
    
    # dp[i][j] 表示计算矩阵i到j的最小乘法次数
    dp = [[0] * n for _ in range(n)]
    
    # 括号位置记录
    split = [[0] * n for _ in range(n)]
    
    # 链长度从2开始
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    
    return dp[0][n-1], split


def print_optimal_parenthesization(split, i, j):
    """打印最优括号化方案"""
    if i == j:
        print(f"A{i+1}", end="")
    else:
        print("(", end="")
        print_optimal_parenthesization(split, i, split[i][j])
        print_optimal_parenthesization(split, split[i][j] + 1, j)
        print(")", end="")


def optimal_bst(keys, freq):
    """
    最优二叉搜索树
    keys: 键的列表（已排序）
    freq: 对应的访问频率
    时间复杂度：O(n³)
    """
    n = len(keys)
    
    # dp[i][j] 表示keys[i:j+1]构成的最优BST的期望搜索代价
    dp = [[0] * n for _ in range(n)]
    
    # 累积频率和
    sum_freq = [[0] * n for _ in range(n)]
    
    # 单个节点的情况
    for i in range(n):
        dp[i][i] = freq[i]
        sum_freq[i][i] = freq[i]
    
    # 计算累积频率
    for i in range(n):
        for j in range(i + 1, n):
            sum_freq[i][j] = sum_freq[i][j-1] + freq[j]
    
    # 从长度2开始
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            # 尝试每个节点作为根
            for r in range(i, j + 1):
                left_cost = dp[i][r-1] if r > i else 0
                right_cost = dp[r+1][j] if r < j else 0
                
                cost = left_cost + right_cost + sum_freq[i][j]
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
    
    return dp[0][n-1]


def test_dp_algorithms():
    """测试动态规划算法"""
    print("=== 动态规划算法测试 ===\n")
    
    # LCS测试
    print("1. 最长公共子序列(LCS):")
    text1 = "ABCDGH"
    text2 = "AEDFHR"
    length, lcs = lcs_with_sequence(text1, text2)
    print(f"  文本1: {text1}")
    print(f"  文本2: {text2}")
    print(f"  LCS长度: {length}")
    print(f"  LCS序列: {lcs}")
    
    # 最长公共子串测试
    print("\n2. 最长公共子串:")
    s1 = "GeeksforGeeks"
    s2 = "GeeksQuiz"
    length, substring = longest_common_substring(s1, s2)
    print(f"  字符串1: {s1}")
    print(f"  字符串2: {s2}")
    print(f"  最长公共子串: '{substring}' (长度: {length})")
    
    # 编辑距离测试
    print("\n3. 编辑距离:")
    word1 = "saturday"
    word2 = "sunday"
    distance, operations = edit_distance_with_operations(word1, word2)
    print(f"  从 '{word1}' 到 '{word2}'")
    print(f"  编辑距离: {distance}")
    print(f"  操作序列:")
    for op in operations:
        print(f"    - {op}")
    
    # 回文子序列测试
    print("\n4. 最长回文子序列:")
    s = "BBABCBCAB"
    print(f"  字符串: {s}")
    print(f"  最长回文子序列长度: {longest_palindromic_subsequence(s)}")
    
    # 回文子串测试
    print("\n5. 最长回文子串:")
    s = "babad"
    print(f"  字符串: {s}")
    print(f"  最长回文子串: '{longest_palindromic_substring(s)}'")
    
    # 矩阵链乘法测试
    print("\n6. 矩阵链乘法:")
    dims = [30, 35, 15, 5, 10, 20, 25]
    min_cost, split = matrix_chain_multiplication(dims)
    print(f"  矩阵维度: {dims}")
    print(f"  最小乘法次数: {min_cost}")
    print(f"  最优括号化: ", end="")
    print_optimal_parenthesization(split, 0, len(dims) - 2)
    print()
    
    # 最优BST测试
    print("\n7. 最优二叉搜索树:")
    keys = [10, 12, 20]
    freq = [34, 8, 50]
    print(f"  键: {keys}")
    print(f"  频率: {freq}")
    print(f"  最小搜索代价: {optimal_bst(keys, freq)}")
    
    # 复杂度分析
    print("\n=== 复杂度分析 ===")
    print("┌─────────────────────┬──────────┬──────────┐")
    print("│ 算法                 │ 时间复杂度 │ 空间复杂度 │")
    print("├─────────────────────┼──────────┼──────────┤")
    print("│ LCS                 │ O(mn)    │ O(mn)    │")
    print("│ 编辑距离             │ O(mn)    │ O(mn)    │")
    print("│ 最长回文子序列        │ O(n²)    │ O(n²)    │")
    print("│ 矩阵链乘法           │ O(n³)    │ O(n²)    │")
    print("│ 最优BST             │ O(n³)    │ O(n²)    │")
    print("└─────────────────────┴──────────┴──────────┘")


if __name__ == '__main__':
    test_dp_algorithms()