import heapq
from collections import defaultdict, Counter


class HuffmanNode:
    """霍夫曼树节点"""
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoding:
    """
    霍夫曼编码实现
    用于数据压缩
    """
    
    def __init__(self):
        self.root = None
        self.codes = {}
        self.reverse_codes = {}
    
    def build_frequency_table(self, text):
        """构建字符频率表"""
        return Counter(text)
    
    def build_huffman_tree(self, freq_table):
        """
        构建霍夫曼树
        时间复杂度：O(n log n)
        """
        if not freq_table:
            return None
        
        # 创建最小堆
        heap = []
        for char, freq in freq_table.items():
            node = HuffmanNode(char=char, freq=freq)
            heapq.heappush(heap, node)
        
        # 特殊情况：只有一个字符
        if len(heap) == 1:
            node = heapq.heappop(heap)
            root = HuffmanNode(freq=node.freq)
            root.left = node
            return root
        
        # 构建霍夫曼树
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            parent = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right
            )
            
            heapq.heappush(heap, parent)
        
        return heap[0]
    
    def generate_codes(self, root, code=""):
        """生成霍夫曼编码"""
        if not root:
            return
        
        # 叶子节点
        if root.char is not None:
            self.codes[root.char] = code if code else "0"
            self.reverse_codes[code if code else "0"] = root.char
            return
        
        # 递归生成编码
        if root.left:
            self.generate_codes(root.left, code + "0")
        if root.right:
            self.generate_codes(root.right, code + "1")
    
    def encode(self, text):
        """
        编码文本
        时间复杂度：O(n)
        """
        if not text:
            return ""
        
        # 构建频率表
        freq_table = self.build_frequency_table(text)
        
        # 构建霍夫曼树
        self.root = self.build_huffman_tree(freq_table)
        
        # 生成编码
        self.codes.clear()
        self.reverse_codes.clear()
        self.generate_codes(self.root)
        
        # 编码文本
        encoded = []
        for char in text:
            encoded.append(self.codes[char])
        
        return ''.join(encoded)
    
    def decode(self, encoded_text):
        """
        解码文本
        时间复杂度：O(n)
        """
        if not encoded_text or not self.root:
            return ""
        
        decoded = []
        node = self.root
        
        for bit in encoded_text:
            if bit == '0':
                node = node.left
            else:
                node = node.right
            
            # 到达叶子节点
            if node.char is not None:
                decoded.append(node.char)
                node = self.root
        
        return ''.join(decoded)
    
    def calculate_compression_ratio(self, original, encoded):
        """计算压缩率"""
        original_bits = len(original) * 8  # 假设每个字符8位
        encoded_bits = len(encoded)
        
        if original_bits == 0:
            return 0
        
        compression_ratio = (1 - encoded_bits / original_bits) * 100
        return compression_ratio
    
    def print_codes(self):
        """打印编码表"""
        print("字符编码表:")
        for char, code in sorted(self.codes.items()):
            if char == ' ':
                print(f"  空格: {code}")
            elif char == '\n':
                print(f"  换行: {code}")
            else:
                print(f"  '{char}': {code}")


def activity_selection(activities):
    """
    活动选择问题
    贪心算法：总是选择最早结束的活动
    
    时间复杂度：O(n log n)
    
    参数:
        activities: [(start, end), ...] 活动列表
    
    返回:
        最大不冲突活动集合
    """
    if not activities:
        return []
    
    # 按结束时间排序
    activities = sorted(activities, key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end = activities[0][1]
    
    for i in range(1, len(activities)):
        start, end = activities[i]
        
        # 如果活动开始时间不早于上一个活动结束时间
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    
    return selected


def weighted_activity_selection(activities):
    """
    带权重的活动选择（动态规划）
    每个活动有权重，目标是最大化总权重
    
    时间复杂度：O(n²)
    """
    if not activities:
        return [], 0
    
    # activities: [(start, end, weight), ...]
    n = len(activities)
    
    # 按结束时间排序
    activities = sorted(activities, key=lambda x: x[1])
    
    # dp[i] = 包含活动i的最大权重
    dp = [0] * n
    parent = [-1] * n
    
    for i in range(n):
        dp[i] = activities[i][2]  # 自身权重
        
        # 找到不冲突的最近活动
        for j in range(i - 1, -1, -1):
            if activities[j][1] <= activities[i][0]:
                if dp[j] + activities[i][2] > dp[i]:
                    dp[i] = dp[j] + activities[i][2]
                    parent[i] = j
                break
    
    # 找到最大权重
    max_weight = max(dp)
    max_idx = dp.index(max_weight)
    
    # 回溯找出选中的活动
    selected = []
    idx = max_idx
    while idx != -1:
        selected.append(activities[idx])
        idx = parent[idx]
    
    selected.reverse()
    return selected, max_weight


def fractional_knapsack_greedy(items, capacity):
    """
    分数背包问题（贪心算法）
    物品可以分割
    
    时间复杂度：O(n log n)
    """
    if not items or capacity <= 0:
        return 0, []
    
    # items: [(weight, value), ...]
    # 计算单位价值并排序
    value_per_weight = []
    for i, (weight, value) in enumerate(items):
        value_per_weight.append((value / weight, weight, value, i))
    
    value_per_weight.sort(reverse=True)
    
    total_value = 0
    selected = []
    remaining = capacity
    
    for vpw, weight, value, idx in value_per_weight:
        if remaining >= weight:
            # 完全装入
            total_value += value
            selected.append((idx, weight, value))
            remaining -= weight
        elif remaining > 0:
            # 部分装入
            fraction = remaining / weight
            total_value += value * fraction
            selected.append((idx, remaining, value * fraction))
            remaining = 0
            break
    
    return total_value, selected


def job_scheduling(jobs):
    """
    作业调度问题
    最小化平均完成时间
    
    贪心策略：最短作业优先
    """
    if not jobs:
        return [], 0
    
    # jobs: [(duration, deadline), ...]
    n = len(jobs)
    
    # 按持续时间排序（最短作业优先）
    sorted_jobs = sorted(enumerate(jobs), key=lambda x: x[1][0])
    
    schedule = []
    current_time = 0
    total_completion_time = 0
    
    for idx, (duration, deadline) in sorted_jobs:
        current_time += duration
        total_completion_time += current_time
        schedule.append((idx, current_time))
    
    avg_completion_time = total_completion_time / n
    return schedule, avg_completion_time


def minimum_spanning_tree_kruskal(n, edges):
    """
    Kruskal最小生成树算法（贪心）
    """
    # 这里简化实现，实际在mst.py中有完整实现
    edges.sort(key=lambda x: x[2])
    
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True
    
    mst = []
    total_weight = 0
    
    for u, v, w in edges:
        if union(u, v):
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == n - 1:
                break
    
    return total_weight, mst


def coin_change_greedy(coins, amount):
    """
    硬币找零问题（贪心算法）
    注意：贪心算法不一定得到最优解
    """
    coins.sort(reverse=True)
    
    count = 0
    selected = []
    
    for coin in coins:
        if amount >= coin:
            num = amount // coin
            count += num
            selected.append((coin, num))
            amount %= coin
    
    if amount > 0:
        return -1, []  # 无法找零
    
    return count, selected


def interval_partitioning(intervals):
    """
    区间划分问题
    最少需要多少个资源来处理所有区间
    """
    if not intervals:
        return 0, []
    
    # 创建事件列表
    events = []
    for start, end in intervals:
        events.append((start, 'start'))
        events.append((end, 'end'))
    
    events.sort()
    
    max_resources = 0
    current_resources = 0
    
    for time, event_type in events:
        if event_type == 'start':
            current_resources += 1
            max_resources = max(max_resources, current_resources)
        else:
            current_resources -= 1
    
    return max_resources


def test_greedy_algorithms():
    """测试贪心算法"""
    print("=== 贪心算法测试 ===\n")
    
    # 霍夫曼编码测试
    print("1. 霍夫曼编码:")
    text = "this is an example for huffman encoding"
    huffman = HuffmanCoding()
    
    encoded = huffman.encode(text)
    decoded = huffman.decode(encoded)
    
    print(f"   原文: {text}")
    print(f"   编码长度: {len(encoded)} bits")
    print(f"   原始长度: {len(text) * 8} bits")
    print(f"   压缩率: {huffman.calculate_compression_ratio(text, encoded):.1f}%")
    print(f"   解码验证: {decoded == text}")
    
    huffman.print_codes()
    
    # 活动选择测试
    print("\n2. 活动选择问题:")
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), 
                  (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
    
    selected = activity_selection(activities)
    print(f"   活动: {activities}")
    print(f"   选中: {selected}")
    print(f"   最大活动数: {len(selected)}")
    
    # 带权重的活动选择
    print("\n3. 带权重活动选择:")
    weighted_activities = [(1, 4, 5), (3, 5, 1), (0, 6, 8), (5, 7, 2), (5, 9, 6)]
    selected_w, max_weight = weighted_activity_selection(weighted_activities)
    print(f"   活动(start, end, weight): {weighted_activities}")
    print(f"   选中: {selected_w}")
    print(f"   最大权重: {max_weight}")
    
    # 分数背包
    print("\n4. 分数背包问题:")
    items = [(10, 60), (20, 100), (30, 120)]  # (weight, value)
    capacity = 50
    
    max_value, selected_items = fractional_knapsack_greedy(items, capacity)
    print(f"   物品(重量, 价值): {items}")
    print(f"   背包容量: {capacity}")
    print(f"   最大价值: {max_value}")
    
    # 作业调度
    print("\n5. 作业调度:")
    jobs = [(3, 10), (1, 5), (2, 8), (4, 12)]  # (duration, deadline)
    schedule, avg_time = job_scheduling(jobs)
    print(f"   作业(时长, 截止): {jobs}")
    print(f"   调度顺序: {[s[0] for s in schedule]}")
    print(f"   平均完成时间: {avg_time:.2f}")
    
    # 硬币找零
    print("\n6. 硬币找零:")
    coins = [25, 10, 5, 1]
    amount = 67
    
    count, selected = coin_change_greedy(coins, amount)
    print(f"   硬币面值: {coins}")
    print(f"   找零金额: {amount}")
    print(f"   最少硬币数: {count}")
    print(f"   使用硬币: {selected}")
    
    # 区间划分
    print("\n7. 区间划分:")
    intervals = [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
    resources = interval_partitioning(intervals)
    print(f"   区间: {intervals}")
    print(f"   最少资源数: {resources}")
    
    # 算法特点
    print("\n=== 贪心算法特点 ===")
    print("优点:")
    print("  • 简单直观，易于实现")
    print("  • 时间复杂度通常较低")
    print("  • 对某些问题能得到最优解")
    print("\n缺点:")
    print("  • 不能保证所有问题都得到最优解")
    print("  • 需要证明贪心选择性质")
    print("  • 局部最优不一定导致全局最优")


if __name__ == '__main__':
    test_greedy_algorithms()