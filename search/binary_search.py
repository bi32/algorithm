import random


def binary_search(nums, target):
    """
    二分搜索算法
    前提：数组必须是有序的
    时间复杂度：O(log n)
    空间复杂度：O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def binary_search_recursive(nums, target, left=0, right=None):
    """
    二分搜索递归实现
    时间复杂度：O(log n)
    空间复杂度：O(log n) - 递归调用栈
    """
    if right is None:
        right = len(nums) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        return binary_search_recursive(nums, target, mid + 1, right)
    else:
        return binary_search_recursive(nums, target, left, mid - 1)


def binary_search_first_occurrence(nums, target):
    """
    查找目标值的第一次出现位置
    适用于有重复元素的有序数组
    """
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # 继续在左侧查找
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def binary_search_last_occurrence(nums, target):
    """
    查找目标值的最后一次出现位置
    适用于有重复元素的有序数组
    """
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            left = mid + 1  # 继续在右侧查找
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


if __name__ == '__main__':
    # 测试基本二分搜索
    nums = sorted(list(range(20)))
    target = random.choice(nums)
    
    print("有序数组:", nums)
    print(f"目标值: {target}")
    print(f"迭代版本 - 索引: {binary_search(nums, target)}")
    print(f"递归版本 - 索引: {binary_search_recursive(nums, target)}")
    
    # 测试有重复元素的情况
    nums_with_duplicates = [1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 7]
    target = 5
    
    print("\n有重复元素的数组:", nums_with_duplicates)
    print(f"目标值: {target}")
    print(f"第一次出现位置: {binary_search_first_occurrence(nums_with_duplicates, target)}")
    print(f"最后一次出现位置: {binary_search_last_occurrence(nums_with_duplicates, target)}")