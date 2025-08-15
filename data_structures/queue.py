from collections import deque


class Queue:
    """
    队列的实现（FIFO - 先进先出）
    使用Python的deque实现高效的队列操作
    """
    
    def __init__(self):
        """初始化空队列"""
        self.items = deque()
    
    def enqueue(self, item):
        """
        入队操作
        时间复杂度：O(1)
        """
        self.items.append(item)
    
    def dequeue(self):
        """
        出队操作
        时间复杂度：O(1)
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()
    
    def front(self):
        """
        查看队首元素但不删除
        时间复杂度：O(1)
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def rear(self):
        """
        查看队尾元素但不删除
        时间复杂度：O(1)
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[-1]
    
    def is_empty(self):
        """检查队列是否为空"""
        return len(self.items) == 0
    
    def size(self):
        """返回队列的大小"""
        return len(self.items)
    
    def clear(self):
        """清空队列"""
        self.items.clear()
    
    def __str__(self):
        """字符串表示"""
        return f"Queue({list(self.items)})"


class CircularQueue:
    """
    循环队列的实现
    使用固定大小的数组，通过循环利用空间
    """
    
    def __init__(self, capacity):
        """初始化循环队列"""
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        """入队"""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.rear = (self.rear + 1) % self.capacity
        self.items[self.rear] = item
        self.size += 1
    
    def dequeue(self):
        """出队"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.items[self.front]
        self.items[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self):
        """查看队首元素"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def get_size(self):
        return self.size


class Deque:
    """
    双端队列的实现
    支持在两端进行插入和删除操作
    """
    
    def __init__(self):
        self.items = deque()
    
    def add_front(self, item):
        """在前端添加元素"""
        self.items.appendleft(item)
    
    def add_rear(self, item):
        """在后端添加元素"""
        self.items.append(item)
    
    def remove_front(self):
        """从前端删除元素"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.popleft()
    
    def remove_rear(self):
        """从后端删除元素"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.pop()
    
    def peek_front(self):
        """查看前端元素"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items[0]
    
    def peek_rear(self):
        """查看后端元素"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)


class PriorityQueue:
    """
    优先队列的简单实现
    使用列表，每次插入后排序
    """
    
    def __init__(self):
        self.items = []
    
    def enqueue(self, item, priority):
        """
        入队，带优先级
        priority: 数字越小优先级越高
        """
        self.items.append((priority, item))
        self.items.sort(key=lambda x: x[0])
    
    def dequeue(self):
        """出队，返回优先级最高的元素"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self.items.pop(0)[1]
    
    def peek(self):
        """查看优先级最高的元素"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self.items[0][1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)


def hot_potato(names, num):
    """
    热土豆游戏（约瑟夫问题）
    使用队列模拟传递过程
    """
    queue = Queue()
    
    for name in names:
        queue.enqueue(name)
    
    while queue.size() > 1:
        for _ in range(num):
            queue.enqueue(queue.dequeue())
        queue.dequeue()  # 被淘汰的人
    
    return queue.dequeue()


def sliding_window_maximum(nums, k):
    """
    滑动窗口最大值
    使用双端队列维护窗口内的最大值
    """
    if not nums or k <= 0:
        return []
    
    dq = deque()  # 存储索引
    result = []
    
    for i in range(len(nums)):
        # 移除超出窗口的元素
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # 移除小于当前元素的所有元素
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # 当窗口形成后，添加最大值到结果
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


class TaskScheduler:
    """
    任务调度器
    使用队列实现简单的任务调度
    """
    
    def __init__(self):
        self.task_queue = Queue()
        self.completed_tasks = []
    
    def add_task(self, task):
        """添加任务"""
        self.task_queue.enqueue(task)
    
    def execute_next(self):
        """执行下一个任务"""
        if not self.task_queue.is_empty():
            task = self.task_queue.dequeue()
            print(f"执行任务: {task}")
            self.completed_tasks.append(task)
            return task
        return None
    
    def has_pending_tasks(self):
        """检查是否有待处理任务"""
        return not self.task_queue.is_empty()
    
    def get_pending_count(self):
        """获取待处理任务数量"""
        return self.task_queue.size()


def level_order_traversal(root):
    """
    二叉树的层序遍历
    使用队列实现
    """
    if not root:
        return []
    
    result = []
    queue = Queue()
    queue.enqueue(root)
    
    while not queue.is_empty():
        level_size = queue.size()
        level = []
        
        for _ in range(level_size):
            node = queue.dequeue()
            level.append(node.val)
            
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)
        
        result.append(level)
    
    return result


if __name__ == '__main__':
    # 测试基本队列操作
    print("基本队列操作:")
    queue = Queue()
    for i in range(5):
        queue.enqueue(i)
        print(f"入队 {i}: {queue}")
    
    print(f"队首元素: {queue.front()}")
    print(f"队尾元素: {queue.rear()}")
    
    while not queue.is_empty():
        print(f"出队: {queue.dequeue()}")
    
    # 测试循环队列
    print("\n循环队列测试:")
    cq = CircularQueue(5)
    for i in range(5):
        cq.enqueue(i)
        print(f"入队 {i}, 大小: {cq.get_size()}")
    
    print(f"队列已满: {cq.is_full()}")
    
    for _ in range(2):
        print(f"出队: {cq.dequeue()}")
    
    for i in range(5, 7):
        cq.enqueue(i)
        print(f"入队 {i}")
    
    # 测试双端队列
    print("\n双端队列测试:")
    dq = Deque()
    dq.add_front(1)
    dq.add_rear(2)
    dq.add_front(0)
    dq.add_rear(3)
    print(f"双端队列: 前端={dq.peek_front()}, 后端={dq.peek_rear()}")
    
    # 测试优先队列
    print("\n优先队列测试:")
    pq = PriorityQueue()
    tasks = [(3, "低优先级任务"), (1, "高优先级任务"), (2, "中优先级任务")]
    for priority, task in tasks:
        pq.enqueue(task, priority)
    
    while not pq.is_empty():
        print(f"处理: {pq.dequeue()}")
    
    # 测试热土豆游戏
    print("\n热土豆游戏:")
    players = ["Alice", "Bob", "Charlie", "David", "Eve"]
    winner = hot_potato(players, 3)
    print(f"获胜者: {winner}")
    
    # 测试滑动窗口最大值
    print("\n滑动窗口最大值:")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = sliding_window_maximum(nums, k)
    print(f"数组: {nums}")
    print(f"窗口大小: {k}")
    print(f"最大值序列: {result}")
    
    # 测试任务调度器
    print("\n任务调度器:")
    scheduler = TaskScheduler()
    for i in range(1, 4):
        scheduler.add_task(f"任务{i}")
    
    while scheduler.has_pending_tasks():
        scheduler.execute_next()