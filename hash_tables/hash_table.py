import hashlib
import math


class HashTable:
    """
    哈希表实现（链式法解决冲突）
    """
    
    def __init__(self, initial_size=16, load_factor=0.75):
        self.size = initial_size
        self.load_factor = load_factor
        self.count = 0
        self.buckets = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        """简单哈希函数"""
        return hash(key) % self.size
    
    def _rehash(self):
        """重新哈希，当负载因子过高时扩容"""
        old_buckets = self.buckets
        self.size *= 2
        self.count = 0
        self.buckets = [[] for _ in range(self.size)]
        
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key, value):
        """插入或更新键值对"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
        self.count += 1
        
        # 检查是否需要重新哈希
        if self.count > self.size * self.load_factor:
            self._rehash()
    
    def get(self, key):
        """获取值"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """删除键值对"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                return
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """检查键是否存在"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def keys(self):
        """返回所有键"""
        result = []
        for bucket in self.buckets:
            for key, _ in bucket:
                result.append(key)
        return result
    
    def values(self):
        """返回所有值"""
        result = []
        for bucket in self.buckets:
            for _, value in bucket:
                result.append(value)
        return result
    
    def items(self):
        """返回所有键值对"""
        result = []
        for bucket in self.buckets:
            result.extend(bucket)
        return result


class OpenAddressingHashTable:
    """
    开放寻址法哈希表
    使用线性探测解决冲突
    """
    
    def __init__(self, initial_size=16):
        self.size = initial_size
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.deleted = [False] * self.size
        self.count = 0
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _probe(self, key):
        """线性探测"""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if not self.deleted[index] and self.keys[index] == key:
                return index
            index = (index + 1) % self.size
        
        return index
    
    def _rehash(self):
        """重新哈希"""
        old_keys = self.keys
        old_values = self.values
        old_deleted = self.deleted
        
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.deleted = [False] * self.size
        self.count = 0
        
        for i in range(len(old_keys)):
            if old_keys[i] is not None and not old_deleted[i]:
                self.put(old_keys[i], old_values[i])
    
    def put(self, key, value):
        """插入或更新"""
        if self.count >= self.size // 2:
            self._rehash()
        
        index = self._probe(key)
        
        if self.keys[index] is None or self.deleted[index]:
            self.count += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def get(self, key):
        """获取值"""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if not self.deleted[index] and self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """删除（惰性删除）"""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if not self.deleted[index] and self.keys[index] == key:
                self.deleted[index] = True
                self.count -= 1
                return
            index = (index + 1) % self.size
        
        raise KeyError(f"Key '{key}' not found")


class QuadraticProbingHashTable:
    """
    二次探测哈希表
    """
    
    def __init__(self, initial_size=16):
        self.size = initial_size
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _probe(self, key):
        """二次探测"""
        index = self._hash(key)
        i = 0
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return index
            i += 1
            index = (self._hash(key) + i * i) % self.size
        
        return index
    
    def put(self, key, value):
        """插入或更新"""
        if self.count >= self.size // 2:
            self._rehash()
        
        index = self._probe(key)
        
        if self.keys[index] is None:
            self.count += 1
        
        self.keys[index] = key
        self.values[index] = value
    
    def _rehash(self):
        """重新哈希"""
        old_keys = self.keys
        old_values = self.values
        
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        
        for i in range(len(old_keys)):
            if old_keys[i] is not None:
                self.put(old_keys[i], old_values[i])


class DoubleHashingTable:
    """
    双重哈希表
    """
    
    def __init__(self, initial_size=17):  # 使用质数
        self.size = initial_size
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
    
    def _hash1(self, key):
        return hash(key) % self.size
    
    def _hash2(self, key):
        """第二个哈希函数"""
        return 1 + (hash(key) % (self.size - 1))
    
    def _probe(self, key):
        """双重哈希探测"""
        index = self._hash1(key)
        step = self._hash2(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return index
            index = (index + step) % self.size
        
        return index
    
    def put(self, key, value):
        """插入或更新"""
        index = self._probe(key)
        
        if self.keys[index] is None:
            self.count += 1
        
        self.keys[index] = key
        self.values[index] = value


class BloomFilter:
    """
    布隆过滤器实现
    用于快速判断元素是否可能存在于集合中
    """
    
    def __init__(self, capacity, error_rate=0.01):
        """
        capacity: 预期元素数量
        error_rate: 可接受的误判率
        """
        # 计算最优的位数组大小和哈希函数数量
        self.capacity = capacity
        self.error_rate = error_rate
        
        # m = -n * ln(p) / (ln(2)^2)
        self.bit_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        
        # k = m * ln(2) / n
        self.hash_count = int(self.bit_size * math.log(2) / capacity)
        
        # 初始化位数组（使用列表模拟）
        self.bit_array = [0] * self.bit_size
        
        self.count = 0
    
    def _hash(self, item, seed):
        """使用SHA256生成哈希值"""
        h = hashlib.sha256()
        h.update(str(item).encode('utf-8'))
        h.update(str(seed).encode('utf-8'))
        return int(h.hexdigest(), 16)
    
    def _get_hash_values(self, item):
        """生成k个哈希值"""
        hash_values = []
        
        # 使用不同的种子生成多个哈希值
        for i in range(self.hash_count):
            hash_val = self._hash(item, i) % self.bit_size
            hash_values.append(hash_val)
        
        return hash_values
    
    def add(self, item):
        """添加元素"""
        for hash_val in self._get_hash_values(item):
            self.bit_array[hash_val] = 1
        self.count += 1
    
    def contains(self, item):
        """
        检查元素是否可能存在
        返回False表示一定不存在
        返回True表示可能存在（有误判率）
        """
        for hash_val in self._get_hash_values(item):
            if self.bit_array[hash_val] == 0:
                return False
        return True
    
    def __contains__(self, item):
        return self.contains(item)
    
    def estimate_false_positive_rate(self):
        """估算当前的误判率"""
        # (1 - e^(-kn/m))^k
        ratio = self.count / self.bit_size
        return (1 - math.exp(-self.hash_count * ratio)) ** self.hash_count
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'capacity': self.capacity,
            'bit_size': self.bit_size,
            'hash_count': self.hash_count,
            'items_added': self.count,
            'designed_error_rate': self.error_rate,
            'current_error_rate': self.estimate_false_positive_rate(),
            'memory_usage_bytes': self.bit_size // 8
        }


class CountingBloomFilter:
    """
    计数布隆过滤器
    支持删除操作
    """
    
    def __init__(self, capacity, error_rate=0.01, counter_bits=4):
        self.capacity = capacity
        self.error_rate = error_rate
        self.counter_bits = counter_bits
        self.max_count = (1 << counter_bits) - 1
        
        # 计算参数
        self.bit_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.bit_size * math.log(2) / capacity)
        
        # 使用计数数组而不是位数组
        self.counters = [0] * self.bit_size
        self.count = 0
    
    def _hash(self, item, seed):
        """使用SHA256生成哈希值"""
        h = hashlib.sha256()
        h.update(str(item).encode('utf-8'))
        h.update(str(seed).encode('utf-8'))
        return int(h.hexdigest(), 16)
    
    def _get_hash_values(self, item):
        """生成k个哈希值"""
        return [self._hash(item, i) % self.bit_size 
                for i in range(self.hash_count)]
    
    def add(self, item):
        """添加元素"""
        for hash_val in self._get_hash_values(item):
            if self.counters[hash_val] < self.max_count:
                self.counters[hash_val] += 1
        self.count += 1
    
    def remove(self, item):
        """删除元素"""
        # 先检查是否存在
        if not self.contains(item):
            return False
        
        for hash_val in self._get_hash_values(item):
            if self.counters[hash_val] > 0:
                self.counters[hash_val] -= 1
        
        self.count -= 1
        return True
    
    def contains(self, item):
        """检查元素是否可能存在"""
        for hash_val in self._get_hash_values(item):
            if self.counters[hash_val] == 0:
                return False
        return True


class CuckooHashTable:
    """
    布谷鸟哈希表
    使用两个哈希函数，保证最坏情况O(1)查找
    """
    
    def __init__(self, initial_size=16):
        self.size = initial_size
        self.table1 = [None] * self.size
        self.table2 = [None] * self.size
        self.count = 0
        self.max_iterations = 500
    
    def _hash1(self, key):
        return hash(key) % self.size
    
    def _hash2(self, key):
        return (hash(key) // self.size) % self.size
    
    def put(self, key, value):
        """插入键值对"""
        # 检查是否已存在
        if self.contains(key):
            # 更新值
            h1 = self._hash1(key)
            h2 = self._hash2(key)
            
            if self.table1[h1] and self.table1[h1][0] == key:
                self.table1[h1] = (key, value)
            elif self.table2[h2] and self.table2[h2][0] == key:
                self.table2[h2] = (key, value)
            return
        
        # 尝试插入
        current = (key, value)
        
        for _ in range(self.max_iterations):
            h1 = self._hash1(current[0])
            
            if self.table1[h1] is None:
                self.table1[h1] = current
                self.count += 1
                return
            
            # 踢出现有元素
            current, self.table1[h1] = self.table1[h1], current
            
            h2 = self._hash2(current[0])
            
            if self.table2[h2] is None:
                self.table2[h2] = current
                self.count += 1
                return
            
            # 踢出现有元素
            current, self.table2[h2] = self.table2[h2], current
        
        # 需要重新哈希
        self._rehash()
        self.put(current[0], current[1])
    
    def get(self, key):
        """获取值"""
        h1 = self._hash1(key)
        h2 = self._hash2(key)
        
        if self.table1[h1] and self.table1[h1][0] == key:
            return self.table1[h1][1]
        elif self.table2[h2] and self.table2[h2][0] == key:
            return self.table2[h2][1]
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """检查键是否存在"""
        h1 = self._hash1(key)
        h2 = self._hash2(key)
        
        return ((self.table1[h1] and self.table1[h1][0] == key) or
                (self.table2[h2] and self.table2[h2][0] == key))
    
    def _rehash(self):
        """重新哈希"""
        old_items = []
        
        for item in self.table1:
            if item:
                old_items.append(item)
        
        for item in self.table2:
            if item:
                old_items.append(item)
        
        self.size *= 2
        self.table1 = [None] * self.size
        self.table2 = [None] * self.size
        self.count = 0
        
        for key, value in old_items:
            self.put(key, value)


def test_hash_tables():
    """测试哈希表实现"""
    print("=== 哈希表测试 ===\n")
    
    # 测试链式哈希表
    print("链式哈希表测试:")
    ht = HashTable()
    
    # 插入数据
    for i in range(20):
        ht.put(f"key{i}", f"value{i}")
    
    print(f"插入20个元素后，大小: {ht.size}, 元素数: {ht.count}")
    print(f"获取key5: {ht.get('key5')}")
    
    # 测试开放寻址
    print("\n开放寻址哈希表测试:")
    oht = OpenAddressingHashTable()
    
    for i in range(10):
        oht.put(i, i * i)
    
    print(f"获取5: {oht.get(5)}")
    oht.delete(5)
    print("删除5后，包含5:", 5 in [oht.keys[i] for i in range(oht.size) 
                                   if oht.keys[i] is not None and not oht.deleted[i]])
    
    # 测试布隆过滤器
    print("\n布隆过滤器测试:")
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    
    # 添加元素
    for i in range(100):
        bf.add(f"item{i}")
    
    # 测试存在性
    print(f"item50 存在: {bf.contains('item50')}")
    print(f"item500 存在: {bf.contains('item500')}")
    
    # 统计信息
    stats = bf.get_stats()
    print(f"统计信息:")
    print(f"  位数组大小: {stats['bit_size']}")
    print(f"  哈希函数数: {stats['hash_count']}")
    print(f"  当前误判率: {stats['current_error_rate']:.4f}")
    print(f"  内存使用: {stats['memory_usage_bytes']} bytes")
    
    # 测试布谷鸟哈希
    print("\n布谷鸟哈希表测试:")
    cht = CuckooHashTable()
    
    for i in range(15):
        cht.put(i, i * 2)
    
    print(f"元素数: {cht.count}")
    print(f"获取10: {cht.get(10)}")
    print(f"包含100: {cht.contains(100)}")
    
    # 性能对比
    print("\n=== 性能对比 ===")
    print("┌──────────────┬───────────┬────────────┬──────────┐")
    print("│ 实现方式      │ 插入复杂度 │ 查找复杂度  │ 空间效率  │")
    print("├──────────────┼───────────┼────────────┼──────────┤")
    print("│ 链式法        │ O(1)均摊   │ O(1)平均    │ 中等     │")
    print("│ 开放寻址      │ O(1)均摊   │ O(1)平均    │ 高       │")
    print("│ 布谷鸟哈希    │ O(1)均摊   │ O(1)最坏    │ 中等     │")
    print("│ 布隆过滤器    │ O(k)      │ O(k)       │ 极高     │")
    print("└──────────────┴───────────┴────────────┴──────────┘")


if __name__ == '__main__':
    test_hash_tables()