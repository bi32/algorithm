import random


def heap_sort(nums):
    """
    堆排序算法
    时间复杂度：O(n log n)
    空间复杂度：O(1)
    稳定排序：否
    """
    nums = nums.copy()
    
    def heapify(nums, n, i):
        """
        将以i为根的子树调整为最大堆
        n: 堆的大小
        i: 当前节点索引
        """
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        # 找出三个节点中的最大值
        if left < n and nums[left] > nums[largest]:
            largest = left
        
        if right < n and nums[right] > nums[largest]:
            largest = right
        
        # 如果最大值不是根节点，交换并继续调整
        if largest != i:
            nums[i], nums[largest] = nums[largest], nums[i]
            heapify(nums, n, largest)
    
    n = len(nums)
    
    # 构建最大堆
    # 从最后一个非叶子节点开始，自底向上调整
    for i in range(n // 2 - 1, -1, -1):
        heapify(nums, n, i)
    
    # 逐个取出堆顶元素，放到数组末尾
    for i in range(n - 1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        heapify(nums, i, 0)
    
    return nums


def heap_sort_min_heap(nums):
    """
    使用最小堆的堆排序（结果为降序）
    """
    nums = nums.copy()
    
    def min_heapify(nums, n, i):
        """调整为最小堆"""
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and nums[left] < nums[smallest]:
            smallest = left
        
        if right < n and nums[right] < nums[smallest]:
            smallest = right
        
        if smallest != i:
            nums[i], nums[smallest] = nums[smallest], nums[i]
            min_heapify(nums, n, smallest)
    
    n = len(nums)
    
    # 构建最小堆
    for i in range(n // 2 - 1, -1, -1):
        min_heapify(nums, n, i)
    
    # 取出元素
    for i in range(n - 1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        min_heapify(nums, i, 0)
    
    return nums


class MaxHeap:
    """
    最大堆数据结构实现
    支持动态插入和删除操作
    """
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        """获取父节点索引"""
        return (i - 1) // 2
    
    def left_child(self, i):
        """获取左子节点索引"""
        return 2 * i + 1
    
    def right_child(self, i):
        """获取右子节点索引"""
        return 2 * i + 2
    
    def swap(self, i, j):
        """交换两个元素"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, value):
        """
        插入新元素
        时间复杂度：O(log n)
        """
        self.heap.append(value)
        self._bubble_up(len(self.heap) - 1)
    
    def _bubble_up(self, i):
        """向上调整堆"""
        while i > 0 and self.heap[i] > self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def extract_max(self):
        """
        提取最大值
        时间复杂度：O(log n)
        """
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._bubble_down(0)
        
        return max_val
    
    def _bubble_down(self, i):
        """向下调整堆"""
        n = len(self.heap)
        
        while True:
            largest = i
            left = self.left_child(i)
            right = self.right_child(i)
            
            if left < n and self.heap[left] > self.heap[largest]:
                largest = left
            
            if right < n and self.heap[right] > self.heap[largest]:
                largest = right
            
            if largest == i:
                break
            
            self.swap(i, largest)
            i = largest
    
    def peek(self):
        """查看最大值但不删除"""
        return self.heap[0] if self.heap else None
    
    def size(self):
        """返回堆的大小"""
        return len(self.heap)
    
    def is_empty(self):
        """检查堆是否为空"""
        return len(self.heap) == 0
    
    def build_heap(self, nums):
        """
        从数组构建堆
        时间复杂度：O(n)
        """
        self.heap = nums.copy()
        n = len(self.heap)
        
        # 从最后一个非叶子节点开始调整
        for i in range(n // 2 - 1, -1, -1):
            self._bubble_down(i)


def heap_sort_with_class(nums):
    """使用MaxHeap类实现堆排序"""
    heap = MaxHeap()
    heap.build_heap(nums)
    
    result = []
    while not heap.is_empty():
        result.append(heap.extract_max())
    
    return result[::-1]  # 逆序得到升序结果


def find_kth_largest(nums, k):
    """
    使用堆找到第k大的元素
    时间复杂度：O(n log k)
    """
    import heapq
    
    # 使用最小堆维护k个最大元素
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap[0] if min_heap else None


if __name__ == '__main__':
    nums = list(range(20))
    random.shuffle(nums)
    
    print("原始数组:", nums)
    print("堆排序(升序):", heap_sort(nums))
    print("最小堆排序(降序):", heap_sort_min_heap(nums))
    print("使用堆类排序:", heap_sort_with_class(nums))
    
    # 测试MaxHeap类
    print("\n测试MaxHeap类:")
    heap = MaxHeap()
    test_nums = [3, 1, 4, 1, 5, 9, 2, 6]
    for num in test_nums:
        heap.insert(num)
    
    print("插入元素:", test_nums)
    print("提取最大值序列:", end=" ")
    while not heap.is_empty():
        print(heap.extract_max(), end=" ")
    
    # 测试查找第k大元素
    print("\n\n查找第k大元素:")
    nums = [3, 2, 1, 5, 6, 4]
    for k in range(1, len(nums) + 1):
        print(f"第{k}大的元素: {find_kth_largest(nums, k)}")