import random
import hashlib
from collections import defaultdict


class SkipListNode:
    """跳表节点"""
    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList:
    """
    跳表实现
    平均时间复杂度：O(log n) for search, insert, delete
    空间复杂度：O(n)
    """
    
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipListNode(None, None, max_level)
        self.size = 0
    
    def random_level(self):
        """随机生成节点层数"""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, key):
        """查找元素"""
        current = self.header
        
        # 从最高层开始查找
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        current = current.forward[0]
        
        if current and current.key == key:
            return current.value
        return None
    
    def insert(self, key, value):
        """插入元素"""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # 找到插入位置
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # 如果key已存在，更新值
        if current and current.key == key:
            current.value = value
            return
        
        # 创建新节点
        new_level = self.random_level()
        
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level
        
        new_node = SkipListNode(key, value, new_level)
        
        # 更新指针
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        
        self.size += 1
    
    def delete(self, key):
        """删除元素"""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # 找到删除位置
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # 如果找到key，删除节点
        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            
            # 更新层数
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
            
            self.size -= 1
            return True
        
        return False
    
    def display(self):
        """显示跳表结构"""
        print("Skip List:")
        for level in range(self.level, -1, -1):
            print(f"Level {level}: ", end="")
            node = self.header.forward[level]
            while node:
                print(f"{node.key}:{node.value} -> ", end="")
                node = node.forward[level]
            print("None")


class CuckooFilter:
    """
    布谷鸟过滤器
    支持动态删除的概率型数据结构
    """
    
    def __init__(self, capacity, bucket_size=4, fingerprint_size=1):
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = 500
        
        # 初始化桶
        self.buckets = [[None for _ in range(bucket_size)] 
                       for _ in range(capacity)]
        self.size = 0
    
    def _hash(self, item):
        """计算哈希值"""
        h = hashlib.sha256(str(item).encode()).hexdigest()
        return int(h, 16)
    
    def _fingerprint(self, item):
        """计算指纹"""
        return self._hash(item) % (256 ** self.fingerprint_size)
    
    def _get_indices(self, item):
        """获取两个候选位置"""
        fp = self._fingerprint(item)
        h1 = self._hash(item) % self.capacity
        h2 = (h1 ^ self._hash(fp)) % self.capacity
        return h1, h2, fp
    
    def insert(self, item):
        """插入元素"""
        i1, i2, fp = self._get_indices(item)
        
        # 尝试插入第一个位置
        for j in range(self.bucket_size):
            if self.buckets[i1][j] is None:
                self.buckets[i1][j] = fp
                self.size += 1
                return True
        
        # 尝试插入第二个位置
        for j in range(self.bucket_size):
            if self.buckets[i2][j] is None:
                self.buckets[i2][j] = fp
                self.size += 1
                return True
        
        # 需要踢出操作
        i = random.choice([i1, i2])
        for _ in range(self.max_kicks):
            j = random.randrange(self.bucket_size)
            fp, self.buckets[i][j] = self.buckets[i][j], fp
            
            # 计算被踢出元素的另一个位置
            i = (i ^ self._hash(fp)) % self.capacity
            
            # 尝试插入
            for k in range(self.bucket_size):
                if self.buckets[i][k] is None:
                    self.buckets[i][k] = fp
                    self.size += 1
                    return True
        
        # 插入失败
        return False
    
    def contains(self, item):
        """检查元素是否存在"""
        i1, i2, fp = self._get_indices(item)
        
        # 检查两个位置
        for j in range(self.bucket_size):
            if self.buckets[i1][j] == fp or self.buckets[i2][j] == fp:
                return True
        
        return False
    
    def delete(self, item):
        """删除元素"""
        i1, i2, fp = self._get_indices(item)
        
        # 尝试从第一个位置删除
        for j in range(self.bucket_size):
            if self.buckets[i1][j] == fp:
                self.buckets[i1][j] = None
                self.size -= 1
                return True
        
        # 尝试从第二个位置删除
        for j in range(self.bucket_size):
            if self.buckets[i2][j] == fp:
                self.buckets[i2][j] = None
                self.size -= 1
                return True
        
        return False
    
    def load_factor(self):
        """计算负载因子"""
        return self.size / (self.capacity * self.bucket_size)


class TreapNode:
    """Treap节点"""
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.priority = random.random()
        self.left = None
        self.right = None
        self.size = 1


class Treap:
    """
    Treap（树堆）
    结合BST和堆的性质
    期望时间复杂度：O(log n)
    """
    
    def __init__(self):
        self.root = None
    
    def _update_size(self, node):
        """更新节点大小"""
        if node:
            left_size = node.left.size if node.left else 0
            right_size = node.right.size if node.right else 0
            node.size = 1 + left_size + right_size
    
    def _rotate_right(self, node):
        """右旋"""
        new_root = node.left
        node.left = new_root.right
        new_root.right = node
        
        self._update_size(node)
        self._update_size(new_root)
        
        return new_root
    
    def _rotate_left(self, node):
        """左旋"""
        new_root = node.right
        node.right = new_root.left
        new_root.left = node
        
        self._update_size(node)
        self._update_size(new_root)
        
        return new_root
    
    def _insert(self, node, key, value):
        """递归插入"""
        if not node:
            return TreapNode(key, value)
        
        if key < node.key:
            node.left = self._insert(node.left, key, value)
            # 维护堆性质
            if node.left.priority > node.priority:
                node = self._rotate_right(node)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
            # 维护堆性质
            if node.right.priority > node.priority:
                node = self._rotate_left(node)
        else:
            # 更新值
            node.value = value
        
        self._update_size(node)
        return node
    
    def insert(self, key, value=None):
        """插入元素"""
        self.root = self._insert(self.root, key, value)
    
    def _delete(self, node, key):
        """递归删除"""
        if not node:
            return None
        
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # 找到要删除的节点
            if not node.left and not node.right:
                return None
            elif not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # 有两个子节点，旋转使其变为叶节点
                if node.left.priority > node.right.priority:
                    node = self._rotate_right(node)
                    node.right = self._delete(node.right, key)
                else:
                    node = self._rotate_left(node)
                    node.left = self._delete(node.left, key)
        
        self._update_size(node)
        return node
    
    def delete(self, key):
        """删除元素"""
        self.root = self._delete(self.root, key)
    
    def _search(self, node, key):
        """递归搜索"""
        if not node:
            return None
        
        if key < node.key:
            return self._search(node.left, key)
        elif key > node.key:
            return self._search(node.right, key)
        else:
            return node.value
    
    def search(self, key):
        """搜索元素"""
        return self._search(self.root, key)
    
    def _kth_smallest(self, node, k):
        """找第k小的元素"""
        if not node:
            return None
        
        left_size = node.left.size if node.left else 0
        
        if k <= left_size:
            return self._kth_smallest(node.left, k)
        elif k == left_size + 1:
            return node.key
        else:
            return self._kth_smallest(node.right, k - left_size - 1)
    
    def kth_smallest(self, k):
        """返回第k小的元素"""
        return self._kth_smallest(self.root, k)
    
    def _inorder(self, node, result):
        """中序遍历"""
        if node:
            self._inorder(node.left, result)
            result.append((node.key, node.value))
            self._inorder(node.right, result)
    
    def inorder(self):
        """返回中序遍历结果"""
        result = []
        self._inorder(self.root, result)
        return result


class LSMTree:
    """
    LSM树（Log-Structured Merge Tree）简化实现
    用于优化写入性能的数据结构
    """
    
    def __init__(self, memtable_size=100, level_ratio=10):
        self.memtable = {}  # 内存表
        self.memtable_size = memtable_size
        self.level_ratio = level_ratio
        self.levels = []  # 磁盘层级
        self.write_count = 0
    
    def put(self, key, value):
        """写入键值对"""
        self.memtable[key] = value
        self.write_count += 1
        
        # 如果内存表满了，刷写到磁盘
        if len(self.memtable) >= self.memtable_size:
            self._flush()
    
    def get(self, key):
        """读取值"""
        # 先查内存表
        if key in self.memtable:
            return self.memtable[key]
        
        # 再查各层级（从新到旧）
        for level in self.levels:
            for sstable in reversed(level):
                if key in sstable:
                    return sstable[key]
        
        return None
    
    def _flush(self):
        """将内存表刷写到磁盘"""
        if not self.memtable:
            return
        
        # 创建有序字符串表（SSTable）
        sstable = dict(sorted(self.memtable.items()))
        
        # 添加到第0层
        if not self.levels:
            self.levels.append([])
        
        self.levels[0].append(sstable)
        self.memtable.clear()
        
        # 触发合并
        self._compact(0)
    
    def _compact(self, level):
        """合并层级"""
        if level >= len(self.levels):
            return
        
        # 检查是否需要合并
        if len(self.levels[level]) < self.level_ratio:
            return
        
        # 合并当前层的所有SSTable
        merged = {}
        for sstable in self.levels[level]:
            merged.update(sstable)
        
        # 清空当前层
        self.levels[level].clear()
        
        # 将合并结果添加到下一层
        if level + 1 >= len(self.levels):
            self.levels.append([])
        
        self.levels[level + 1].append(dict(sorted(merged.items())))
        
        # 递归合并下一层
        self._compact(level + 1)
    
    def range_query(self, start_key, end_key):
        """范围查询"""
        result = {}
        
        # 查询内存表
        for k, v in self.memtable.items():
            if start_key <= k <= end_key:
                result[k] = v
        
        # 查询各层级
        for level in self.levels:
            for sstable in level:
                for k, v in sstable.items():
                    if start_key <= k <= end_key:
                        if k not in result:  # 新的值优先
                            result[k] = v
        
        return dict(sorted(result.items()))
    
    def delete(self, key):
        """删除键（墓碑标记）"""
        self.put(key, None)  # 使用None作为墓碑标记
    
    def stats(self):
        """统计信息"""
        total_keys = len(self.memtable)
        for level in self.levels:
            for sstable in level:
                total_keys += len(sstable)
        
        return {
            'memtable_size': len(self.memtable),
            'levels': len(self.levels),
            'total_keys': total_keys,
            'write_count': self.write_count
        }


def test_advanced_structures():
    """测试高级数据结构"""
    print("=== 高级数据结构测试 ===\n")
    
    # 测试跳表
    print("1. 跳表测试:")
    skip_list = SkipList()
    
    # 插入数据
    data = [(3, 'three'), (7, 'seven'), (1, 'one'), 
            (4, 'four'), (9, 'nine'), (2, 'two')]
    
    for key, value in data:
        skip_list.insert(key, value)
    
    print(f"   插入数据: {data}")
    print(f"   查找3: {skip_list.search(3)}")
    print(f"   查找5: {skip_list.search(5)}")
    
    skip_list.delete(4)
    print(f"   删除4后查找: {skip_list.search(4)}")
    
    # 测试布谷鸟过滤器
    print("\n2. 布谷鸟过滤器测试:")
    cf = CuckooFilter(capacity=100)
    
    # 插入元素
    items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    for item in items:
        cf.insert(item)
    
    print(f"   插入元素: {items}")
    print(f"   'apple'存在: {cf.contains('apple')}")
    print(f"   'grape'存在: {cf.contains('grape')}")
    
    cf.delete('banana')
    print(f"   删除'banana'后存在: {cf.contains('banana')}")
    print(f"   负载因子: {cf.load_factor():.2f}")
    
    # 测试Treap
    print("\n3. Treap测试:")
    treap = Treap()
    
    # 插入数据
    values = [5, 3, 7, 2, 4, 6, 8, 1, 9]
    for v in values:
        treap.insert(v, f"value_{v}")
    
    print(f"   插入数据: {values}")
    print(f"   中序遍历: {[k for k, _ in treap.inorder()]}")
    print(f"   查找5: {treap.search(5)}")
    print(f"   第3小的元素: {treap.kth_smallest(3)}")
    
    treap.delete(5)
    print(f"   删除5后中序遍历: {[k for k, _ in treap.inorder()]}")
    
    # 测试LSM树
    print("\n4. LSM树测试:")
    lsm = LSMTree(memtable_size=5)
    
    # 批量写入
    for i in range(20):
        lsm.put(f"key_{i:02d}", f"value_{i}")
    
    print(f"   写入20个键值对")
    print(f"   读取key_05: {lsm.get('key_05')}")
    print(f"   读取key_99: {lsm.get('key_99')}")
    
    # 范围查询
    range_result = lsm.range_query('key_05', 'key_10')
    print(f"   范围查询[key_05, key_10]: {list(range_result.keys())}")
    
    # 统计信息
    stats = lsm.stats()
    print(f"   统计信息: {stats}")
    
    # 性能分析
    print("\n=== 数据结构特点 ===")
    print("┌──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ 数据结构      │ 插入         │ 查找         │ 删除         │")
    print("├──────────────┼──────────────┼──────────────┼──────────────┤")
    print("│ 跳表         │ O(log n)     │ O(log n)     │ O(log n)     │")
    print("│ 布谷鸟过滤器  │ O(1)         │ O(1)         │ O(1)         │")
    print("│ Treap        │ O(log n)*    │ O(log n)*    │ O(log n)*    │")
    print("│ LSM树        │ O(1)         │ O(log n)     │ O(1)         │")
    print("└──────────────┴──────────────┴──────────────┴──────────────┘")
    print("*期望时间复杂度")
    
    print("\n=== 应用场景 ===")
    print("• 跳表：Redis有序集合、内存数据库索引")
    print("• 布谷鸟过滤器：去重、黑名单、缓存")
    print("• Treap：需要随机化的平衡树场景")
    print("• LSM树：写密集型存储系统（LevelDB、RocksDB）")


if __name__ == '__main__':
    test_advanced_structures()