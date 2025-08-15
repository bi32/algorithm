def compute_lps_array(pattern):
    """
    计算最长真前后缀数组（Longest Proper Prefix Suffix）
    lps[i] = 模式串pattern[0..i]的最长真前后缀长度
    
    时间复杂度：O(m)，m为模式串长度
    空间复杂度：O(m)
    """
    m = len(pattern)
    lps = [0] * m
    length = 0  # 最长真前后缀的长度
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # 回退到前一个可能的匹配位置
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps


def kmp_search(text, pattern):
    """
    KMP字符串匹配算法
    
    时间复杂度：O(n + m)
    空间复杂度：O(m)
    
    参数:
        text: 文本串
        pattern: 模式串
    
    返回:
        所有匹配位置的列表
    """
    if not pattern:
        return []
    
    n = len(text)
    m = len(pattern)
    
    # 计算LPS数组
    lps = compute_lps_array(pattern)
    
    matches = []
    i = 0  # text的索引
    j = 0  # pattern的索引
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            # 找到匹配
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches


def kmp_count(text, pattern):
    """
    统计模式串在文本中出现的次数
    """
    return len(kmp_search(text, pattern))


def kmp_search_first(text, pattern):
    """
    找到第一个匹配的位置
    
    时间复杂度：O(n + m)
    空间复杂度：O(m)
    """
    if not pattern:
        return 0
    
    n = len(text)
    m = len(pattern)
    
    lps = compute_lps_array(pattern)
    
    i = j = 0
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            return i - j
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return -1


def kmp_with_wildcards(text, pattern, wildcard='?'):
    """
    支持通配符的KMP匹配
    wildcard可以匹配任意单个字符
    """
    n = len(text)
    m = len(pattern)
    
    if m == 0:
        return []
    
    # 修改LPS计算以支持通配符
    def compute_lps_with_wildcard(pattern):
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length] or \
               pattern[i] == wildcard or pattern[length] == wildcard:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    lps = compute_lps_with_wildcard(pattern)
    matches = []
    i = j = 0
    
    while i < n:
        if text[i] == pattern[j] or pattern[j] == wildcard:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j] and pattern[j] != wildcard:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches


def find_all_palindromes_kmp(text):
    """
    使用KMP思想找出所有回文子串的中心
    """
    n = len(text)
    palindromes = []
    
    # 处理奇数长度回文
    for center in range(n):
        # 构建以center为中心的最长回文
        left = right = center
        while left >= 0 and right < n and text[left] == text[right]:
            palindromes.append((left, right))
            left -= 1
            right += 1
    
    # 处理偶数长度回文
    for center in range(n - 1):
        if text[center] == text[center + 1]:
            left = center
            right = center + 1
            while left >= 0 and right < n and text[left] == text[right]:
                palindromes.append((left, right))
                left -= 1
                right += 1
    
    return palindromes


def circular_pattern_match(text, pattern):
    """
    环形字符串匹配
    检查pattern的任意旋转是否在text中
    """
    if len(pattern) == 0:
        return True
    
    # 将pattern复制一份接在后面
    doubled_pattern = pattern + pattern
    
    # 在doubled_pattern中搜索text
    # 如果找到，说明存在某个旋转匹配
    for i in range(len(pattern)):
        rotated = doubled_pattern[i:i + len(pattern)]
        if kmp_search_first(text, rotated) != -1:
            return True, i  # 返回True和旋转的位置
    
    return False, -1


def longest_prefix_suffix(s):
    """
    找出字符串的最长真前后缀
    """
    n = len(s)
    if n == 0:
        return ""
    
    lps = compute_lps_array(s)
    max_length = lps[n - 1]
    
    if max_length == 0:
        return ""
    
    return s[:max_length]


def pattern_period(pattern):
    """
    计算模式串的周期
    如果pattern = s^k (s重复k次)，返回len(s)
    """
    n = len(pattern)
    if n == 0:
        return 0
    
    lps = compute_lps_array(pattern)
    
    # 周期长度
    period = n - lps[n - 1]
    
    # 检查是否真的是周期
    if n % period == 0:
        return period
    else:
        return n


def visualize_kmp_matching(text, pattern):
    """
    可视化KMP匹配过程
    """
    if not pattern:
        return
    
    n = len(text)
    m = len(pattern)
    lps = compute_lps_array(pattern)
    
    print(f"文本: {text}")
    print(f"模式: {pattern}")
    print(f"LPS数组: {lps}")
    print("\n匹配过程:")
    print("-" * 50)
    
    i = j = 0
    step = 0
    
    while i < n:
        step += 1
        print(f"\n步骤 {step}:")
        print(f"文本: {text}")
        print(f"      {' ' * (i - j)}{pattern}")
        print(f"      {' ' * i}^")
        
        if text[i] == pattern[j]:
            print(f"匹配: text[{i}]='{text[i]}' == pattern[{j}]='{pattern[j]}'")
            i += 1
            j += 1
        else:
            print(f"失配: text[{i}]='{text[i]}' != pattern[{j}]='{pattern[j]}'")
        
        if j == m:
            print(f"*** 找到匹配，位置: {i - j} ***")
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                print(f"使用LPS回退: j = lps[{j - 1}] = {lps[j - 1]}")
                j = lps[j - 1]
            else:
                print("j = 0, i向前移动")
                i += 1


def test_kmp():
    """测试KMP算法"""
    print("=== KMP字符串匹配算法 ===\n")
    
    # 基本测试
    text1 = "ABABDABACDABABCABAB"
    pattern1 = "ABABCABAB"
    
    print(f"文本: {text1}")
    print(f"模式: {pattern1}")
    
    lps = compute_lps_array(pattern1)
    print(f"LPS数组: {lps}")
    
    matches = kmp_search(text1, pattern1)
    print(f"匹配位置: {matches}")
    
    for pos in matches:
        print(f"  位置 {pos}: {text1[pos:pos+len(pattern1)]}")
    
    # 测试多个匹配
    print("\n多个匹配测试:")
    text2 = "aaaaaaa"
    pattern2 = "aaa"
    matches2 = kmp_search(text2, pattern2)
    print(f"文本: {text2}")
    print(f"模式: {pattern2}")
    print(f"匹配位置: {matches2}")
    
    # 测试通配符匹配
    print("\n通配符匹配测试:")
    text3 = "abcdefg"
    pattern3 = "a?c"
    matches3 = kmp_with_wildcards(text3, pattern3)
    print(f"文本: {text3}")
    print(f"模式: {pattern3} (?匹配任意字符)")
    print(f"匹配位置: {matches3}")
    
    # 测试回文
    print("\n回文查找测试:")
    text4 = "abacabad"
    palindromes = find_all_palindromes_kmp(text4)
    print(f"文本: {text4}")
    print("回文子串:")
    for start, end in palindromes:
        if end - start >= 2:  # 只显示长度>=3的回文
            print(f"  [{start}, {end}]: {text4[start:end+1]}")
    
    # 测试周期
    print("\n周期检测测试:")
    patterns = ["abcabc", "ababa", "abcd", "aaaa"]
    for p in patterns:
        period = pattern_period(p)
        print(f"模式 '{p}' 的周期: {period}")
    
    # 可视化匹配过程
    print("\n=== KMP匹配过程可视化 ===")
    visualize_kmp_matching("ABACABAB", "ABAB")
    
    # 性能测试
    print("\n=== 性能测试 ===")
    import time
    
    # 生成测试数据
    long_text = "a" * 10000 + "b"
    long_pattern = "a" * 100 + "b"
    
    start = time.time()
    result = kmp_search(long_text, long_pattern)
    kmp_time = time.time() - start
    
    # 对比暴力算法
    def brute_force(text, pattern):
        matches = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i+len(pattern)] == pattern:
                matches.append(i)
        return matches
    
    start = time.time()
    result2 = brute_force(long_text, long_pattern)
    brute_time = time.time() - start
    
    print(f"文本长度: {len(long_text)}, 模式长度: {len(long_pattern)}")
    print(f"KMP算法: {kmp_time:.6f}秒")
    print(f"暴力算法: {brute_time:.6f}秒")
    print(f"加速比: {brute_time/kmp_time:.2f}x")


if __name__ == '__main__':
    test_kmp()