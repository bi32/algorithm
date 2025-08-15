import random


def merge_sort(nums):
    """
    归并排序算法
    时间复杂度：O(n log n)
    空间复杂度：O(n)
    稳定排序：是
    """
    nums = nums.copy()
    
    if len(nums) <= 1:
        return nums
    
    def merge(left, right):
        """合并两个有序数组"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # 添加剩余元素
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    # 分割数组
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    
    # 合并
    return merge(left, right)


def merge_sort_in_place(nums):
    """
    归并排序原地版本（使用辅助数组）
    时间复杂度：O(n log n)
    空间复杂度：O(n)
    """
    nums = nums.copy()
    aux = [0] * len(nums)
    
    def merge(nums, aux, left, mid, right):
        """原地合并"""
        # 复制到辅助数组
        for k in range(left, right + 1):
            aux[k] = nums[k]
        
        i = left
        j = mid + 1
        
        for k in range(left, right + 1):
            if i > mid:
                nums[k] = aux[j]
                j += 1
            elif j > right:
                nums[k] = aux[i]
                i += 1
            elif aux[i] <= aux[j]:
                nums[k] = aux[i]
                i += 1
            else:
                nums[k] = aux[j]
                j += 1
    
    def sort(nums, aux, left, right):
        """递归排序"""
        if left >= right:
            return
        
        mid = left + (right - left) // 2
        sort(nums, aux, left, mid)
        sort(nums, aux, mid + 1, right)
        merge(nums, aux, left, mid, right)
    
    sort(nums, aux, 0, len(nums) - 1)
    return nums


def merge_sort_bottom_up(nums):
    """
    自底向上的归并排序（迭代版本）
    时间复杂度：O(n log n)
    空间复杂度：O(n)
    """
    nums = nums.copy()
    n = len(nums)
    aux = [0] * n
    
    def merge(nums, aux, left, mid, right):
        """合并操作"""
        for k in range(left, right + 1):
            aux[k] = nums[k]
        
        i = left
        j = mid + 1
        
        for k in range(left, right + 1):
            if i > mid:
                nums[k] = aux[j]
                j += 1
            elif j > right:
                nums[k] = aux[i]
                i += 1
            elif aux[i] <= aux[j]:
                nums[k] = aux[i]
                i += 1
            else:
                nums[k] = aux[j]
                j += 1
    
    # 从大小为1的子数组开始，逐步合并
    size = 1
    while size < n:
        for left in range(0, n - size, size * 2):
            mid = left + size - 1
            right = min(left + size * 2 - 1, n - 1)
            merge(nums, aux, left, mid, right)
        size *= 2
    
    return nums


def merge_sort_three_way(nums):
    """
    三路归并排序
    将数组分成三部分进行归并
    """
    nums = nums.copy()
    
    if len(nums) <= 1:
        return nums
    
    def merge_three(first, second, third):
        """合并三个有序数组"""
        result = []
        i = j = k = 0
        
        while i < len(first) or j < len(second) or k < len(third):
            min_val = float('inf')
            min_idx = -1
            
            if i < len(first) and first[i] <= min_val:
                min_val = first[i]
                min_idx = 0
            if j < len(second) and second[j] <= min_val:
                min_val = second[j]
                min_idx = 1
            if k < len(third) and third[k] <= min_val:
                min_val = third[k]
                min_idx = 2
            
            result.append(min_val)
            
            if min_idx == 0:
                i += 1
            elif min_idx == 1:
                j += 1
            else:
                k += 1
        
        return result
    
    # 分割成三部分
    n = len(nums)
    one_third = n // 3
    two_third = 2 * n // 3
    
    if n < 3:
        return merge_sort(nums)
    
    first = merge_sort_three_way(nums[:one_third])
    second = merge_sort_three_way(nums[one_third:two_third])
    third = merge_sort_three_way(nums[two_third:])
    
    return merge_three(first, second, third)


if __name__ == '__main__':
    nums = list(range(20))
    random.shuffle(nums)
    
    print("原始数组:", nums)
    print("基本归并排序:", merge_sort(nums))
    print("原地归并排序:", merge_sort_in_place(nums))
    print("自底向上归并:", merge_sort_bottom_up(nums))
    print("三路归并排序:", merge_sort_three_way(nums))
    
    # 测试稳定性
    data = [(3, 'a'), (1, 'b'), (3, 'c'), (2, 'd'), (3, 'e')]
    print("\n稳定性测试 - 原始:", data)
    sorted_data = sorted(data, key=lambda x: x[0])
    print("稳定性测试 - 排序后:", sorted_data)