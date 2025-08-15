import random
import math
from collections import deque, defaultdict


class UnionFind:
    """
    并查集（Union-Find/Disjoint Set）
    带路径压缩和按秩合并优化
    """
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n  # 连通分量数
    
    def find(self, x):
        """查找根节点（路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """合并两个集合（按秩合并）"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        """检查是否连通"""
        return self.find(x) == self.find(y)
    
    def get_size(self, x):
        """获取集合大小"""
        return self.size[self.find(x)]


class FenwickTree:
    """
    树状数组（Fenwick Tree/Binary Indexed Tree）
    用于高效计算前缀和
    """
    
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        """更新第i个元素，增加delta"""
        i += 1  # 树状数组从1开始
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        """查询前i个元素的和"""
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s
    
    def range_query(self, left, right):
        """查询[left, right]区间和"""
        if left > 0:
            return self.query(right) - self.query(left - 1)
        return self.query(right)


class MonotonicQueue:
    """
    单调队列
    用于滑动窗口最大/最小值
    """
    
    def __init__(self, mode='max'):
        self.queue = deque()
        self.mode = mode  # 'max' or 'min'
    
    def push(self, value):
        """添加元素"""
        if self.mode == 'max':
            while self.queue and self.queue[-1] < value:
                self.queue.pop()
        else:  # min
            while self.queue and self.queue[-1] > value:
                self.queue.pop()
        self.queue.append(value)
    
    def pop(self, value):
        """移除元素（如果是队首）"""
        if self.queue and self.queue[0] == value:
            self.queue.popleft()
    
    def get_optimal(self):
        """获取最大/最小值"""
        return self.queue[0] if self.queue else None


class SlidingWindow:
    """
    滑动窗口算法集合
    """
    
    @staticmethod
    def max_sliding_window(nums, k):
        """
        滑动窗口最大值
        时间复杂度：O(n)
        """
        if not nums or k <= 0:
            return []
        
        n = len(nums)
        if k >= n:
            return [max(nums)]
        
        mq = MonotonicQueue('max')
        result = []
        
        # 初始化窗口
        for i in range(k):
            mq.push(nums[i])
        result.append(mq.get_optimal())
        
        # 滑动窗口
        for i in range(k, n):
            mq.pop(nums[i - k])
            mq.push(nums[i])
            result.append(mq.get_optimal())
        
        return result
    
    @staticmethod
    def longest_substring_without_repeat(s):
        """
        最长无重复子串
        时间复杂度：O(n)
        """
        char_index = {}
        max_length = 0
        start = 0
        
        for end, char in enumerate(s):
            if char in char_index and char_index[char] >= start:
                start = char_index[char] + 1
            
            char_index[char] = end
            max_length = max(max_length, end - start + 1)
        
        return max_length
    
    @staticmethod
    def min_window_substring(s, t):
        """
        最小覆盖子串
        时间复杂度：O(n)
        """
        if not s or not t:
            return ""
        
        need = defaultdict(int)
        for char in t:
            need[char] += 1
        
        window = defaultdict(int)
        left = right = 0
        valid = 0
        start = 0
        min_len = float('inf')
        
        while right < len(s):
            char = s[right]
            right += 1
            
            if char in need:
                window[char] += 1
                if window[char] == need[char]:
                    valid += 1
            
            while valid == len(need):
                if right - left < min_len:
                    start = left
                    min_len = right - left
                
                char = s[left]
                left += 1
                
                if char in need:
                    if window[char] == need[char]:
                        valid -= 1
                    window[char] -= 1
        
        return "" if min_len == float('inf') else s[start:start + min_len]


class ReservoirSampling:
    """
    蓄水池采样
    从未知大小的流中随机采样k个元素
    """
    
    def __init__(self, k):
        self.k = k
        self.reservoir = []
        self.count = 0
    
    def add(self, value):
        """添加一个元素"""
        self.count += 1
        
        if len(self.reservoir) < self.k:
            self.reservoir.append(value)
        else:
            # 以k/count的概率替换
            j = random.randint(0, self.count - 1)
            if j < self.k:
                self.reservoir[j] = value
    
    def get_sample(self):
        """获取采样结果"""
        return self.reservoir.copy()


class MorrisTraversal:
    """
    Morris遍历
    O(1)空间复杂度的二叉树遍历
    """
    
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    @staticmethod
    def inorder(root):
        """
        Morris中序遍历
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        result = []
        current = root
        
        while current:
            if not current.left:
                result.append(current.val)
                current = current.right
            else:
                # 找到前驱节点
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # 建立线索
                    predecessor.right = current
                    current = current.left
                else:
                    # 恢复树结构
                    predecessor.right = None
                    result.append(current.val)
                    current = current.right
        
        return result
    
    @staticmethod
    def preorder(root):
        """
        Morris前序遍历
        """
        result = []
        current = root
        
        while current:
            if not current.left:
                result.append(current.val)
                current = current.right
            else:
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    result.append(current.val)
                    predecessor.right = current
                    current = current.left
                else:
                    predecessor.right = None
                    current = current.right
        
        return result


class Trie2:
    """
    字典树（增强版）
    支持前缀统计和模糊匹配
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
            self.count = 0  # 前缀计数
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def insert(self, word):
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word):
        """搜索单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        """搜索前缀"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def count_prefix(self, prefix):
        """统计前缀出现次数"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count
    
    def fuzzy_search(self, pattern):
        """
        模糊搜索（支持通配符.）
        . 可以匹配任意字符
        """
        def dfs(node, pattern, index):
            if index == len(pattern):
                return node.is_end
            
            char = pattern[index]
            if char == '.':
                # 匹配任意字符
                for child in node.children.values():
                    if dfs(child, pattern, index + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], pattern, index + 1)
        
        return dfs(self.root, pattern, 0)


class LRUCache:
    """
    LRU缓存
    使用双向链表+哈希表实现
    """
    
    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        # 使用伪头部和伪尾部节点
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """添加节点到头部"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """删除节点"""
        prev = node.prev
        next = node.next
        prev.next = next
        next.prev = prev
    
    def _move_to_head(self, node):
        """移动节点到头部"""
        self._remove_node(node)
        self._add_node(node)
    
    def get(self, key):
        """获取值"""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        """设置值"""
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            node = self.Node(key, value)
            self.cache[key] = node
            self._add_node(node)
            
            if len(self.cache) > self.capacity:
                # 删除最久未使用的节点
                tail = self.tail.prev
                self._remove_node(tail)
                del self.cache[tail.key]


class MedianFinder:
    """
    数据流中位数
    使用两个堆维护
    """
    
    def __init__(self):
        self.small = []  # 最大堆（存储较小的一半）
        self.large = []  # 最小堆（存储较大的一半）
    
    def add_num(self, num):
        """添加数字"""
        import heapq
        
        # 先加入小堆
        heapq.heappush(self.small, -num)
        
        # 平衡：将小堆的最大值移到大堆
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # 保持小堆元素数量>=大堆
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self):
        """找中位数"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0


class TopKFrequent:
    """
    Top K频繁元素算法
    """
    
    @staticmethod
    def bucket_sort_solution(nums, k):
        """
        桶排序解法
        时间复杂度：O(n)
        """
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        # 桶排序
        buckets = [[] for _ in range(len(nums) + 1)]
        for num, freq in count.items():
            buckets[freq].append(num)
        
        # 收集结果
        result = []
        for i in range(len(buckets) - 1, -1, -1):
            for num in buckets[i]:
                result.append(num)
                if len(result) == k:
                    return result
        
        return result
    
    @staticmethod
    def quick_select_solution(nums, k):
        """
        快速选择解法
        平均时间复杂度：O(n)
        """
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        unique = list(count.keys())
        
        def partition(left, right):
            pivot = count[unique[right]]
            i = left
            
            for j in range(left, right):
                if count[unique[j]] >= pivot:
                    unique[i], unique[j] = unique[j], unique[i]
                    i += 1
            
            unique[i], unique[right] = unique[right], unique[i]
            return i
        
        def quick_select(left, right, k):
            if left == right:
                return
            
            pivot_index = partition(left, right)
            
            if pivot_index == k - 1:
                return
            elif pivot_index < k - 1:
                quick_select(pivot_index + 1, right, k)
            else:
                quick_select(left, pivot_index - 1, k)
        
        quick_select(0, len(unique) - 1, k)
        return unique[:k]


class RangeSumQuery2D:
    """
    二维区域和查询
    使用二维前缀和
    """
    
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] + 
                                    self.prefix[i-1][j] + 
                                    self.prefix[i][j-1] - 
                                    self.prefix[i-1][j-1])
    
    def sum_region(self, row1, col1, row2, col2):
        """查询矩形区域和"""
        return (self.prefix[row2+1][col2+1] - 
                self.prefix[row1][col2+1] - 
                self.prefix[row2+1][col1] + 
                self.prefix[row1][col1])


def test_classic_algorithms():
    """测试经典算法"""
    print("=== 经典算法补充测试 ===\n")
    
    # 测试并查集
    print("1. 并查集:")
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    print(f"   0和2连通: {uf.connected(0, 2)}")
    print(f"   0和3连通: {uf.connected(0, 3)}")
    print(f"   连通分量数: {uf.count}")
    uf.union(2, 3)
    print(f"   合并2,3后连通分量数: {uf.count}")
    
    # 测试树状数组
    print("\n2. 树状数组:")
    ft = FenwickTree(10)
    nums = [1, 3, 5, 7, 9, 11]
    for i, num in enumerate(nums):
        ft.update(i, num)
    print(f"   数组: {nums}")
    print(f"   前3个元素和: {ft.query(2)}")
    print(f"   区间[1,4]和: {ft.range_query(1, 4)}")
    
    # 测试滑动窗口
    print("\n3. 滑动窗口:")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    max_window = SlidingWindow.max_sliding_window(nums, k)
    print(f"   数组: {nums}")
    print(f"   窗口大小{k}的最大值: {max_window}")
    
    s = "abcabcbb"
    longest = SlidingWindow.longest_substring_without_repeat(s)
    print(f"   字符串'{s}'最长无重复子串长度: {longest}")
    
    # 测试蓄水池采样
    print("\n4. 蓄水池采样:")
    rs = ReservoirSampling(3)
    stream = list(range(10))
    for val in stream:
        rs.add(val)
    sample = rs.get_sample()
    print(f"   数据流: {stream}")
    print(f"   采样结果(k=3): {sample}")
    
    # 测试字典树
    print("\n5. 增强字典树:")
    trie = Trie2()
    words = ["apple", "app", "apricot", "application"]
    for word in words:
        trie.insert(word)
    
    print(f"   插入单词: {words}")
    print(f"   搜索'app': {trie.search('app')}")
    print(f"   前缀'app'出现次数: {trie.count_prefix('app')}")
    print(f"   模糊搜索'ap.le': {trie.fuzzy_search('ap.le')}")
    
    # 测试LRU缓存
    print("\n6. LRU缓存:")
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"   get(1): {lru.get(1)}")
    lru.put(3, 3)  # 淘汰2
    print(f"   get(2): {lru.get(2)}")  # -1
    lru.put(4, 4)  # 淘汰1
    print(f"   get(1): {lru.get(1)}")  # -1
    print(f"   get(3): {lru.get(3)}")
    
    # 测试数据流中位数
    print("\n7. 数据流中位数:")
    mf = MedianFinder()
    stream = [2, 3, 4, 1, 5]
    for num in stream:
        mf.add_num(num)
        print(f"   添加{num}后，中位数: {mf.find_median()}")
    
    # 测试Top K频繁元素
    print("\n8. Top K频繁元素:")
    nums = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    k = 2
    top_k = TopKFrequent.bucket_sort_solution(nums, k)
    print(f"   数组: {nums}")
    print(f"   Top {k}频繁元素: {top_k}")
    
    # 测试二维前缀和
    print("\n9. 二维区域和:")
    matrix = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5]
    ]
    rsq = RangeSumQuery2D(matrix)
    print(f"   矩阵:")
    for row in matrix:
        print(f"     {row}")
    print(f"   区域[(1,1),(2,3)]和: {rsq.sum_region(1, 1, 2, 3)}")
    
    # 复杂度总结
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │ 空间复杂度    │")
    print("├──────────────────┼──────────────┼──────────────┤")
    print("│ 并查集           │ O(α(n))      │ O(n)         │")
    print("│ 树状数组         │ O(log n)     │ O(n)         │")
    print("│ 滑动窗口最值      │ O(n)         │ O(k)         │")
    print("│ 蓄水池采样       │ O(n)         │ O(k)         │")
    print("│ LRU缓存         │ O(1)         │ O(capacity)  │")
    print("│ 数据流中位数     │ O(log n)     │ O(n)         │")
    print("│ Morris遍历      │ O(n)         │ O(1)         │")
    print("└──────────────────┴──────────────┴──────────────┘")


if __name__ == '__main__':
    test_classic_algorithms()