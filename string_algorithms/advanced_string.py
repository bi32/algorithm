def boyer_moore(text, pattern):
    """
    Boyer-Moore字符串匹配算法
    时间复杂度：平均O(n/m)，最坏O(nm)
    """
    def build_bad_char_table(pattern):
        """构建坏字符表"""
        table = {}
        for i in range(len(pattern)):
            table[pattern[i]] = i
        return table
    
    def build_good_suffix_table(pattern):
        """构建好后缀表"""
        m = len(pattern)
        good_suffix = [0] * m
        
        # 构建suffix数组
        suffix = [0] * m
        suffix[m - 1] = m
        
        for i in range(m - 2, -1, -1):
            j = i
            while j >= 0 and pattern[j] == pattern[m - 1 - i + j]:
                j -= 1
            suffix[i] = i - j
        
        # Case 1: 好后缀在模式中重复出现
        for i in range(m):
            good_suffix[i] = m
        
        j = 0
        for i in range(m - 1, -1, -1):
            if suffix[i] == i + 1:
                while j < m - 1 - i:
                    if good_suffix[j] == m:
                        good_suffix[j] = m - 1 - i
                    j += 1
        
        # Case 2: 好后缀的前缀匹配模式的后缀
        for i in range(m - 1):
            good_suffix[m - 1 - suffix[i]] = m - 1 - i
        
        return good_suffix
    
    n = len(text)
    m = len(pattern)
    
    if m == 0:
        return []
    
    bad_char = build_bad_char_table(pattern)
    good_suffix = build_good_suffix_table(pattern)
    
    matches = []
    i = 0
    
    while i <= n - m:
        j = m - 1
        
        # 从右向左比较
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        
        if j < 0:
            # 找到匹配
            matches.append(i)
            
            # 移动到下一个可能的位置
            if i + m < n:
                i += good_suffix[0]
            else:
                i += 1
        else:
            # 计算移动距离
            bad_char_shift = j - bad_char.get(text[i + j], -1)
            good_suffix_shift = good_suffix[j]
            i += max(bad_char_shift, good_suffix_shift)
    
    return matches


def rabin_karp(text, pattern, prime=101):
    """
    Rabin-Karp字符串匹配算法（滚动哈希）
    时间复杂度：平均O(n+m)，最坏O(nm)
    """
    n = len(text)
    m = len(pattern)
    
    if m == 0 or m > n:
        return []
    
    # 计算模式的哈希值
    pattern_hash = 0
    text_hash = 0
    h = 1
    
    # h = pow(256, m-1) % prime
    for i in range(m - 1):
        h = (h * 256) % prime
    
    # 计算初始哈希值
    for i in range(m):
        pattern_hash = (256 * pattern_hash + ord(pattern[i])) % prime
        text_hash = (256 * text_hash + ord(text[i])) % prime
    
    matches = []
    
    for i in range(n - m + 1):
        # 检查哈希值是否匹配
        if pattern_hash == text_hash:
            # 哈希匹配，验证实际字符串
            if text[i:i + m] == pattern:
                matches.append(i)
        
        # 计算下一个窗口的哈希值
        if i < n - m:
            text_hash = (256 * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            
            # 处理负数
            if text_hash < 0:
                text_hash += prime
    
    return matches


def z_algorithm(s):
    """
    Z算法
    计算Z数组：z[i]表示s[i:]与s的最长公共前缀长度
    时间复杂度：O(n)
    """
    n = len(s)
    z = [0] * n
    z[0] = n
    
    l, r = 0, 0
    
    for i in range(1, n):
        if i > r:
            # i在Z-box外
            l, r = i, i
            while r < n and s[r - l] == s[r]:
                r += 1
            z[i] = r - l
            r -= 1
        else:
            # i在Z-box内
            k = i - l
            if z[k] < r - i + 1:
                z[i] = z[k]
            else:
                l = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1
    
    return z


def z_pattern_search(text, pattern):
    """
    使用Z算法进行模式匹配
    时间复杂度：O(n + m)
    """
    # 构造组合字符串
    combined = pattern + "$" + text
    z = z_algorithm(combined)
    
    matches = []
    pattern_len = len(pattern)
    
    for i in range(len(z)):
        if z[i] == pattern_len:
            matches.append(i - pattern_len - 1)
    
    return matches


def manacher(s):
    """
    Manacher算法
    找出所有回文子串
    时间复杂度：O(n)
    """
    # 预处理字符串，插入分隔符
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    p = [0] * n
    center = right = 0
    
    for i in range(1, n - 1):
        # 利用对称性
        mirror = 2 * center - i
        
        if i < right:
            p[i] = min(right - i, p[mirror])
        
        # 尝试扩展回文
        while processed[i + p[i] + 1] == processed[i - p[i] - 1]:
            p[i] += 1
        
        # 更新center和right
        if i + p[i] > right:
            center, right = i, i + p[i]
    
    # 提取所有回文子串
    palindromes = []
    for i in range(1, n - 1):
        length = p[i]
        if length > 0:
            start = (i - length) // 2
            end = start + length
            palindromes.append((start, end, s[start:end]))
    
    return palindromes


def longest_palindrome_manacher(s):
    """
    使用Manacher算法找最长回文子串
    时间复杂度：O(n)
    """
    if not s:
        return ""
    
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    p = [0] * n
    center = right = 0
    max_len = 0
    max_center = 0
    
    for i in range(1, n - 1):
        mirror = 2 * center - i
        
        if i < right:
            p[i] = min(right - i, p[mirror])
        
        while processed[i + p[i] + 1] == processed[i - p[i] - 1]:
            p[i] += 1
        
        if i + p[i] > right:
            center, right = i, i + p[i]
        
        if p[i] > max_len:
            max_len = p[i]
            max_center = i
    
    start = (max_center - max_len) // 2
    return s[start:start + max_len]


def string_hash(s, mod=10**9 + 7, base=31):
    """
    字符串哈希
    用于快速比较子串
    """
    n = len(s)
    hash_values = [0] * (n + 1)
    power = [1] * (n + 1)
    
    for i in range(n):
        hash_values[i + 1] = (hash_values[i] * base + ord(s[i])) % mod
        power[i + 1] = (power[i] * base) % mod
    
    def get_hash(l, r):
        """获取s[l:r+1]的哈希值"""
        return (hash_values[r + 1] - hash_values[l] * power[r - l + 1]) % mod
    
    return get_hash


def longest_common_prefix_array(s):
    """
    LCP数组（最长公共前缀数组）
    配合后缀数组使用
    """
    n = len(s)
    
    # 构建后缀数组（简化版）
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    suffix_array = [i for _, i in suffixes]
    
    # 构建LCP数组
    lcp = [0] * n
    rank = [0] * n
    
    for i in range(n):
        rank[suffix_array[i]] = i
    
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    
    return suffix_array, lcp


def suffix_array_construction(s):
    """
    后缀数组构造（倍增算法）
    时间复杂度：O(n log n)
    """
    n = len(s)
    
    # 初始排名
    rank = [ord(c) for c in s]
    suffix_array = list(range(n))
    
    k = 1
    while k < n:
        # 按(rank[i], rank[i+k])排序
        def get_key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        
        suffix_array.sort(key=get_key)
        
        # 更新排名
        new_rank = [0] * n
        for i in range(1, n):
            prev = suffix_array[i - 1]
            curr = suffix_array[i]
            if get_key(prev) == get_key(curr):
                new_rank[curr] = new_rank[prev]
            else:
                new_rank[curr] = new_rank[prev] + 1
        
        rank = new_rank
        k *= 2
    
    return suffix_array


def lyndon_factorization(s):
    """
    Lyndon分解
    将字符串分解为Lyndon词的非递增序列
    时间复杂度：O(n)
    """
    n = len(s)
    k = 0
    factorization = []
    
    while k < n:
        i = k
        j = k + 1
        
        while j < n and s[i] <= s[j]:
            if s[i] < s[j]:
                i = k
            else:
                i += 1
            j += 1
        
        while k <= i:
            factorization.append(s[k:k + j - i])
            k += j - i
    
    return factorization


def min_rotation(s):
    """
    最小表示法
    找出字符串的最小循环移位
    时间复杂度：O(n)
    """
    s = s + s
    n = len(s) // 2
    i = 0
    j = 1
    
    while i < n and j < n:
        k = 0
        while k < n and s[i + k] == s[j + k]:
            k += 1
        
        if k == n:
            break
        
        if s[i + k] > s[j + k]:
            i = i + k + 1
            if i <= j:
                i = j + 1
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
    
    pos = min(i, j)
    return s[pos:pos + n]


def test_advanced_string():
    """测试高级字符串算法"""
    print("=== 高级字符串算法测试 ===\n")
    
    text = "abracadabra"
    pattern = "abra"
    
    # Boyer-Moore测试
    print("1. Boyer-Moore算法:")
    print(f"   文本: {text}")
    print(f"   模式: {pattern}")
    matches = boyer_moore(text, pattern)
    print(f"   匹配位置: {matches}")
    
    # Rabin-Karp测试
    print("\n2. Rabin-Karp算法:")
    matches = rabin_karp(text, pattern)
    print(f"   匹配位置: {matches}")
    
    # Z算法测试
    print("\n3. Z算法:")
    s = "aabxaayaab"
    z_array = z_algorithm(s)
    print(f"   字符串: {s}")
    print(f"   Z数组: {z_array}")
    
    matches = z_pattern_search(text, pattern)
    print(f"   模式匹配位置: {matches}")
    
    # Manacher算法测试
    print("\n4. Manacher算法:")
    s = "babad"
    longest = longest_palindrome_manacher(s)
    print(f"   字符串: {s}")
    print(f"   最长回文子串: {longest}")
    
    palindromes = manacher(s)
    print(f"   所有回文子串: {[p[2] for p in palindromes if len(p[2]) > 1]}")
    
    # 后缀数组测试
    print("\n5. 后缀数组:")
    s = "banana"
    sa = suffix_array_construction(s)
    print(f"   字符串: {s}")
    print(f"   后缀数组: {sa}")
    print(f"   后缀排序:")
    for i in sa:
        print(f"     {i}: {s[i:]}")
    
    # LCP数组测试
    sa, lcp = longest_common_prefix_array(s)
    print(f"   LCP数组: {lcp}")
    
    # Lyndon分解测试
    print("\n6. Lyndon分解:")
    s = "abcabcabc"
    factorization = lyndon_factorization(s)
    print(f"   字符串: {s}")
    print(f"   Lyndon分解: {factorization}")
    
    # 最小表示法测试
    print("\n7. 最小表示法:")
    s = "cbabc"
    min_rot = min_rotation(s)
    print(f"   字符串: {s}")
    print(f"   最小循环移位: {min_rot}")
    
    # 字符串哈希测试
    print("\n8. 字符串哈希:")
    s = "abcdefg"
    get_hash = string_hash(s)
    print(f"   字符串: {s}")
    print(f"   子串'bcd'的哈希: {get_hash(1, 3)}")
    print(f"   子串'def'的哈希: {get_hash(3, 5)}")
    
    # 复杂度分析
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┬──────────────┐")
    print("│ 算法              │ 平均复杂度    │ 最坏复杂度    │")
    print("├──────────────────┼──────────────┼──────────────┤")
    print("│ Boyer-Moore      │ O(n/m)       │ O(nm)        │")
    print("│ Rabin-Karp       │ O(n+m)       │ O(nm)        │")
    print("│ Z算法            │ O(n)         │ O(n)         │")
    print("│ Manacher         │ O(n)         │ O(n)         │")
    print("│ 后缀数组         │ O(n log n)   │ O(n log n)   │")
    print("│ Lyndon分解       │ O(n)         │ O(n)         │")
    print("└──────────────────┴──────────────┴──────────────┘")


if __name__ == '__main__':
    test_advanced_string()