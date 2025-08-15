class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0  # 用于统计词频


class Trie:
    """
    Trie树（前缀树/字典树）实现
    用于高效的字符串插入、搜索和前缀匹配
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word):
        """
        插入单词
        时间复杂度：O(m)，m为单词长度
        空间复杂度：O(m)
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
        
        node.is_end_of_word = True
        node.frequency += 1
    
    def search(self, word):
        """
        搜索单词
        时间复杂度：O(m)
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """
        检查是否存在以prefix为前缀的单词
        时间复杂度：O(m)
        """
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def delete(self, word):
        """
        删除单词
        时间复杂度：O(m)
        """
        def _delete(node, word, index):
            if index == len(word):
                # 到达单词末尾
                if not node.is_end_of_word:
                    return False  # 单词不存在
                
                node.is_end_of_word = False
                node.frequency = 0
                
                # 如果没有子节点，可以删除这个节点
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False  # 单词不存在
            
            child_node = node.children[char]
            should_delete_child = _delete(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                # 如果当前节点不是其他单词的结尾且没有其他子节点，也可以删除
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        if self.search(word):
            _delete(self.root, word, 0)
            self.word_count -= 1
            return True
        return False
    
    def get_all_words(self):
        """
        获取所有单词
        """
        words = []
        
        def dfs(node, path):
            if node.is_end_of_word:
                words.append(''.join(path))
            
            for char, child in node.children.items():
                path.append(char)
                dfs(child, path)
                path.pop()
        
        dfs(self.root, [])
        return words
    
    def get_words_with_prefix(self, prefix):
        """
        获取所有以prefix为前缀的单词
        """
        node = self.root
        
        # 先找到前缀的最后一个节点
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # 从该节点开始DFS收集所有单词
        words = []
        
        def dfs(node, path):
            if node.is_end_of_word:
                words.append(prefix + ''.join(path))
            
            for char, child in node.children.items():
                path.append(char)
                dfs(child, path)
                path.pop()
        
        dfs(node, [])
        return words
    
    def auto_complete(self, prefix, max_suggestions=5):
        """
        自动补全功能
        返回最多max_suggestions个建议
        """
        suggestions = self.get_words_with_prefix(prefix)
        
        # 按频率排序（如果有频率信息）
        def get_frequency(word):
            node = self.root
            for char in word:
                node = node.children[char]
            return node.frequency
        
        suggestions.sort(key=get_frequency, reverse=True)
        return suggestions[:max_suggestions]
    
    def longest_common_prefix(self):
        """
        找出所有单词的最长公共前缀
        """
        if not self.root.children:
            return ""
        
        prefix = []
        node = self.root
        
        while len(node.children) == 1 and not node.is_end_of_word:
            char, child = next(iter(node.children.items()))
            prefix.append(char)
            node = child
        
        return ''.join(prefix)


class CompressedTrie:
    """
    压缩Trie树（Radix Tree/Patricia Trie）
    将只有一个子节点的路径压缩成一条边
    """
    
    class Node:
        def __init__(self, key=""):
            self.key = key
            self.children = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.Node()
    
    def insert(self, word):
        """插入单词"""
        node = self.root
        i = 0
        
        while i < len(word):
            found = False
            
            for edge_label, child in node.children.items():
                # 找到公共前缀
                j = 0
                while j < len(edge_label) and i + j < len(word) and \
                      edge_label[j] == word[i + j]:
                    j += 1
                
                if j > 0:  # 有公共前缀
                    if j == len(edge_label):
                        # 边标签是word剩余部分的前缀
                        node = child
                        i += j
                        found = True
                        break
                    else:
                        # 需要分裂边
                        # 创建新的中间节点
                        common_prefix = edge_label[:j]
                        remaining_edge = edge_label[j:]
                        remaining_word = word[i + j:] if i + j < len(word) else ""
                        
                        # 创建中间节点
                        middle_node = self.Node()
                        node.children[common_prefix] = middle_node
                        
                        # 原来的子节点成为中间节点的子节点
                        middle_node.children[remaining_edge] = child
                        del node.children[edge_label]
                        
                        if remaining_word:
                            # 为剩余的单词部分创建新节点
                            new_node = self.Node()
                            new_node.is_end_of_word = True
                            middle_node.children[remaining_word] = new_node
                        else:
                            middle_node.is_end_of_word = True
                        
                        return
            
            if not found:
                # 没有找到公共前缀，创建新边
                remaining = word[i:]
                new_node = self.Node()
                new_node.is_end_of_word = True
                node.children[remaining] = new_node
                return
        
        node.is_end_of_word = True
    
    def search(self, word):
        """搜索单词"""
        node = self.root
        i = 0
        
        while i < len(word):
            found = False
            
            for edge_label, child in node.children.items():
                if word[i:].startswith(edge_label):
                    node = child
                    i += len(edge_label)
                    found = True
                    break
            
            if not found:
                return False
        
        return node.is_end_of_word


class SuffixTrie:
    """
    后缀Trie树
    用于字符串匹配和后缀相关操作
    """
    
    def __init__(self, text):
        self.trie = Trie()
        self.text = text
        self._build_suffix_trie()
    
    def _build_suffix_trie(self):
        """构建后缀Trie"""
        n = len(self.text)
        
        for i in range(n):
            # 插入所有后缀
            suffix = self.text[i:] + '$'  # 添加结束符
            self.trie.insert(suffix)
    
    def search_pattern(self, pattern):
        """
        搜索模式在文本中的所有出现位置
        """
        positions = []
        suffixes = self.trie.get_words_with_prefix(pattern)
        
        for suffix in suffixes:
            # 计算位置
            pos = len(self.text) - (len(suffix) - 1)  # 减去'$'
            positions.append(pos)
        
        return sorted(positions)
    
    def longest_repeated_substring(self):
        """
        找出最长重复子串
        """
        def dfs(node, depth, path):
            # 如果有多个子节点，说明有重复
            if len(node.children) > 1:
                nonlocal max_length, longest_substring
                if depth > max_length:
                    max_length = depth
                    longest_substring = ''.join(path)
            
            for char, child in node.children.items():
                if char != '$':  # 忽略结束符
                    path.append(char)
                    dfs(child, depth + 1, path)
                    path.pop()
        
        max_length = 0
        longest_substring = ""
        dfs(self.trie.root, 0, [])
        
        return longest_substring


class ACAutomaton:
    """
    AC自动机（Aho-Corasick算法）
    用于多模式字符串匹配
    """
    
    class Node:
        def __init__(self):
            self.children = {}
            self.fail = None  # 失败指针
            self.output = []  # 输出列表
    
    def __init__(self):
        self.root = self.Node()
        self.root.fail = self.root
    
    def add_pattern(self, pattern):
        """添加模式串"""
        node = self.root
        
        for char in pattern:
            if char not in node.children:
                node.children[char] = self.Node()
            node = node.children[char]
        
        node.output.append(pattern)
    
    def build_failure_links(self):
        """构建失败链接"""
        from collections import deque
        
        queue = deque()
        
        # 第一层节点的失败指针指向根
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)
        
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # 找失败指针
                fail_node = current.fail
                
                while fail_node != self.root and char not in fail_node.children:
                    fail_node = fail_node.fail
                
                if char in fail_node.children and fail_node.children[char] != child:
                    child.fail = fail_node.children[char]
                else:
                    child.fail = self.root
                
                # 合并输出
                if child.fail.output:
                    child.output.extend(child.fail.output)
    
    def search(self, text):
        """
        在文本中搜索所有模式
        返回：[(位置, 模式), ...]
        """
        matches = []
        node = self.root
        
        for i, char in enumerate(text):
            while node != self.root and char not in node.children:
                node = node.fail
            
            if char in node.children:
                node = node.children[char]
            
            if node.output:
                for pattern in node.output:
                    matches.append((i - len(pattern) + 1, pattern))
        
        return matches


def test_trie():
    """测试Trie树"""
    print("=== Trie树测试 ===\n")
    
    # 基本Trie测试
    trie = Trie()
    words = ["apple", "app", "application", "apply", "banana", "band", "can", "candy"]
    
    for word in words:
        trie.insert(word)
    
    print("插入单词:", words)
    print(f"单词总数: {trie.word_count}")
    
    # 搜索测试
    print("\n搜索测试:")
    test_words = ["app", "apple", "appl", "ban"]
    for word in test_words:
        print(f"  '{word}' 存在: {trie.search(word)}")
    
    # 前缀测试
    print("\n前缀测试:")
    prefixes = ["app", "ban", "ca"]
    for prefix in prefixes:
        print(f"  前缀 '{prefix}': {trie.get_words_with_prefix(prefix)}")
    
    # 自动补全
    print("\n自动补全:")
    print(f"  'app' -> {trie.auto_complete('app', 3)}")
    
    # 删除测试
    print("\n删除 'app':")
    trie.delete("app")
    print(f"  'app' 存在: {trie.search('app')}")
    print(f"  'apple' 存在: {trie.search('apple')}")
    
    # 压缩Trie测试
    print("\n=== 压缩Trie测试 ===")
    comp_trie = CompressedTrie()
    comp_words = ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon"]
    
    for word in comp_words:
        comp_trie.insert(word)
    
    print("插入单词:", comp_words)
    print("搜索 'romulus':", comp_trie.search("romulus"))
    print("搜索 'roman':", comp_trie.search("roman"))
    
    # 后缀Trie测试
    print("\n=== 后缀Trie测试 ===")
    text = "banana"
    suffix_trie = SuffixTrie(text)
    
    print(f"文本: {text}")
    print(f"搜索 'ana': 位置 {suffix_trie.search_pattern('ana')}")
    print(f"搜索 'na': 位置 {suffix_trie.search_pattern('na')}")
    print(f"最长重复子串: '{suffix_trie.longest_repeated_substring()}'")
    
    # AC自动机测试
    print("\n=== AC自动机测试 ===")
    ac = ACAutomaton()
    patterns = ["he", "she", "his", "hers"]
    
    for pattern in patterns:
        ac.add_pattern(pattern)
    
    ac.build_failure_links()
    
    text = "ahishers"
    matches = ac.search(text)
    
    print(f"模式: {patterns}")
    print(f"文本: {text}")
    print("匹配结果:")
    for pos, pattern in matches:
        print(f"  位置 {pos}: '{pattern}'")
    
    # 性能分析
    print("\n=== 复杂度分析 ===")
    print("┌────────────┬────────────┬────────────┬──────────────┐")
    print("│ 操作        │ Trie       │ 压缩Trie   │ AC自动机     │")
    print("├────────────┼────────────┼────────────┼──────────────┤")
    print("│ 插入        │ O(m)       │ O(m)       │ O(m)         │")
    print("│ 搜索        │ O(m)       │ O(m)       │ O(n+z)       │")
    print("│ 空间        │ O(ALPHABET*N)│ O(N)      │ O(ALPHABET*N)│")
    print("└────────────┴────────────┴────────────┴──────────────┘")
    print("m: 单词长度, n: 文本长度, z: 匹配数, N: 节点数")


if __name__ == '__main__':
    test_trie()