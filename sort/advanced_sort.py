import random


def shell_sort(arr):
    """
    希尔排序
    时间复杂度：O(n^1.3) 到 O(n²)，取决于间隔序列
    空间复杂度：O(1)
    """
    n = len(arr)
    arr = arr.copy()
    
    # 使用Knuth序列：1, 4, 13, 40, 121...
    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1
    
    while gap > 0:
        # 对每个子序列进行插入排序
        for i in range(gap, n):
            temp = arr[i]
            j = i
            
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            
            arr[j] = temp
        
        gap //= 3
    
    return arr


def comb_sort(arr):
    """
    梳排序（改进的冒泡排序）
    时间复杂度：平均O(n²/2^p)，最好O(n log n)
    空间复杂度：O(1)
    """
    arr = arr.copy()
    n = len(arr)
    
    # 初始间隔
    gap = n
    shrink = 1.3
    sorted = False
    
    while not sorted:
        # 计算下一个间隔
        gap = int(gap / shrink)
        
        if gap <= 1:
            gap = 1
            sorted = True
        
        # 使用当前间隔进行比较和交换
        i = 0
        while i + gap < n:
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted = False
            i += 1
    
    return arr


def cocktail_sort(arr):
    """
    鸡尾酒排序（双向冒泡排序）
    时间复杂度：O(n²)
    空间复杂度：O(1)
    """
    arr = arr.copy()
    n = len(arr)
    left = 0
    right = n - 1
    
    while left < right:
        # 从左到右
        for i in range(left, right):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        right -= 1
        
        # 从右到左
        for i in range(right, left, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
        left += 1
    
    return arr


def gnome_sort(arr):
    """
    侏儒排序（地精排序）
    时间复杂度：O(n²)
    空间复杂度：O(1)
    """
    arr = arr.copy()
    n = len(arr)
    index = 0
    
    while index < n:
        if index == 0:
            index += 1
        elif arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1
    
    return arr


def pancake_sort(arr):
    """
    煎饼排序
    时间复杂度：O(n²)
    空间复杂度：O(1)
    """
    arr = arr.copy()
    
    def flip(arr, k):
        """翻转前k个元素"""
        left = 0
        right = k
        while left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    
    n = len(arr)
    for size in range(n, 1, -1):
        # 找到最大元素的位置
        max_idx = 0
        for i in range(size):
            if arr[i] > arr[max_idx]:
                max_idx = i
        
        # 如果最大元素不在正确位置
        if max_idx != size - 1:
            # 先翻转到顶部
            if max_idx != 0:
                flip(arr, max_idx)
            # 再翻转到正确位置
            flip(arr, size - 1)
    
    return arr


def bogo_sort(arr):
    """
    Bogo排序（猴子排序）- 仅供娱乐
    平均时间复杂度：O(n × n!)
    """
    arr = arr.copy()
    
    def is_sorted(arr):
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return False
        return True
    
    attempts = 0
    max_attempts = 10000  # 防止无限循环
    
    while not is_sorted(arr) and attempts < max_attempts:
        random.shuffle(arr)
        attempts += 1
    
    return arr, attempts


def cycle_sort(arr):
    """
    循环排序
    最小化写入次数，适合写入昂贵的场景
    时间复杂度：O(n²)
    写入次数：O(n)
    """
    arr = arr.copy()
    n = len(arr)
    writes = 0
    
    for cycle_start in range(n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        
        # 找到item的正确位置
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1
        
        # 如果已经在正确位置，继续
        if pos == cycle_start:
            continue
        
        # 跳过重复元素
        while item == arr[pos]:
            pos += 1
        
        # 放置item到正确位置
        arr[pos], item = item, arr[pos]
        writes += 1
        
        # 循环处理剩余元素
        while pos != cycle_start:
            pos = cycle_start
            
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
            
            while item == arr[pos]:
                pos += 1
            
            arr[pos], item = item, arr[pos]
            writes += 1
    
    return arr, writes


def odd_even_sort(arr):
    """
    奇偶排序（砖排序）
    时间复杂度：O(n²)
    适合并行处理
    """
    arr = arr.copy()
    n = len(arr)
    sorted = False
    
    while not sorted:
        sorted = True
        
        # 奇数索引对
        for i in range(1, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sorted = False
        
        # 偶数索引对
        for i in range(0, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sorted = False
    
    return arr


def bitonic_sort(arr):
    """
    双调排序
    时间复杂度：O(log²n × n)
    适合并行处理，要求数组大小为2的幂
    """
    def bitonic_compare(arr, i, j, direction):
        if direction == (arr[i] > arr[j]):
            arr[i], arr[j] = arr[j], arr[i]
    
    def bitonic_merge(arr, low, count, direction):
        if count > 1:
            k = count // 2
            for i in range(low, low + k):
                bitonic_compare(arr, i, i + k, direction)
            bitonic_merge(arr, low, k, direction)
            bitonic_merge(arr, low + k, k, direction)
    
    def bitonic_sort_recursive(arr, low, count, direction):
        if count > 1:
            k = count // 2
            bitonic_sort_recursive(arr, low, k, True)
            bitonic_sort_recursive(arr, low + k, k, False)
            bitonic_merge(arr, low, count, direction)
    
    arr = arr.copy()
    n = len(arr)
    
    # 填充到2的幂
    next_power = 1
    while next_power < n:
        next_power *= 2
    
    while len(arr) < next_power:
        arr.append(float('inf'))
    
    bitonic_sort_recursive(arr, 0, next_power, True)
    
    # 移除填充的元素
    return [x for x in arr if x != float('inf')]


def strand_sort(arr):
    """
    串排序
    时间复杂度：O(n²)
    适合部分有序的数据
    """
    arr = arr.copy()
    result = []
    
    while arr:
        # 提取递增子序列
        sublist = [arr.pop(0)]
        i = 0
        
        while i < len(arr):
            if arr[i] > sublist[-1]:
                sublist.append(arr.pop(i))
            else:
                i += 1
        
        # 合并到结果
        if not result:
            result = sublist
        else:
            # 合并两个有序列表
            merged = []
            i = j = 0
            
            while i < len(result) and j < len(sublist):
                if result[i] < sublist[j]:
                    merged.append(result[i])
                    i += 1
                else:
                    merged.append(sublist[j])
                    j += 1
            
            merged.extend(result[i:])
            merged.extend(sublist[j:])
            result = merged
    
    return result


def tim_sort(arr):
    """
    TimSort（Python内置排序算法的简化版）
    结合归并排序和插入排序
    时间复杂度：O(n log n)
    """
    MIN_MERGE = 32
    
    def insertion_sort_for_tim(arr, left, right):
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def merge_for_tim(arr, left, mid, right):
        left_part = arr[left:mid + 1]
        right_part = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        while i < len(left_part) and j < len(right_part):
            if left_part[i] <= right_part[j]:
                arr[k] = left_part[i]
                i += 1
            else:
                arr[k] = right_part[j]
                j += 1
            k += 1
        
        while i < len(left_part):
            arr[k] = left_part[i]
            i += 1
            k += 1
        
        while j < len(right_part):
            arr[k] = right_part[j]
            j += 1
            k += 1
    
    arr = arr.copy()
    n = len(arr)
    
    # 使用插入排序对小块排序
    for start in range(0, n, MIN_MERGE):
        end = min(start + MIN_MERGE - 1, n - 1)
        insertion_sort_for_tim(arr, start, end)
    
    # 开始归并
    size = MIN_MERGE
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min(start + size * 2 - 1, n - 1)
            
            if mid < end:
                merge_for_tim(arr, start, mid, end)
        
        size *= 2
    
    return arr


def test_advanced_sort():
    """测试高级排序算法"""
    print("=== 高级排序算法测试 ===\n")
    
    # 测试数据
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9, 3, 7, 4, 6],
        list(range(10, 0, -1))  # 逆序数组
    ]
    
    algorithms = [
        ("希尔排序", shell_sort),
        ("梳排序", comb_sort),
        ("鸡尾酒排序", cocktail_sort),
        ("侏儒排序", gnome_sort),
        ("煎饼排序", pancake_sort),
        ("循环排序", cycle_sort),
        ("奇偶排序", odd_even_sort),
        ("双调排序", bitonic_sort),
        ("串排序", strand_sort),
        ("TimSort", tim_sort)
    ]
    
    for i, arr in enumerate(test_arrays):
        print(f"测试数组 {i+1}: {arr}")
        print("-" * 50)
        
        for name, func in algorithms:
            try:
                if name == "循环排序":
                    sorted_arr, writes = func(arr.copy())
                    print(f"{name:12s}: {sorted_arr} (写入次数: {writes})")
                else:
                    sorted_arr = func(arr.copy())
                    print(f"{name:12s}: {sorted_arr}")
            except Exception as e:
                print(f"{name:12s}: 错误 - {e}")
        
        print()
    
    # 性能测试
    print("=== 性能测试（1000个随机数）===")
    import time
    
    test_size = 1000
    test_data = [random.randint(1, 1000) for _ in range(test_size)]
    
    perf_algorithms = [
        ("希尔排序", shell_sort),
        ("TimSort", tim_sort),
        ("梳排序", comb_sort)
    ]
    
    for name, func in perf_algorithms:
        data_copy = test_data.copy()
        start = time.time()
        func(data_copy)
        end = time.time()
        print(f"{name}: {(end - start) * 1000:.2f}ms")
    
    # 复杂度分析
    print("\n=== 算法特点 ===")
    print("┌──────────────┬────────────┬──────────┬──────────────┐")
    print("│ 算法          │ 平均复杂度  │ 稳定性   │ 特点          │")
    print("├──────────────┼────────────┼──────────┼──────────────┤")
    print("│ 希尔排序      │ O(n^1.3)   │ 不稳定   │ 改进插入排序   │")
    print("│ 梳排序        │ O(n log n) │ 不稳定   │ 改进冒泡排序   │")
    print("│ 鸡尾酒排序    │ O(n²)      │ 稳定     │ 双向冒泡      │")
    print("│ TimSort      │ O(n log n) │ 稳定     │ 混合算法      │")
    print("│ 循环排序      │ O(n²)      │ 不稳定   │ 最少写入      │")
    print("│ 双调排序      │ O(log²n×n) │ 不稳定   │ 适合并行      │")
    print("└──────────────┴────────────┴──────────┴──────────────┘")


if __name__ == '__main__':
    test_advanced_sort()