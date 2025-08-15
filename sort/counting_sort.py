import random


def counting_sort(nums):
    """
    计数排序算法
    适用于整数排序，特别是范围不大的情况
    时间复杂度：O(n + k)，k为数值范围
    空间复杂度：O(k)
    稳定排序：是
    """
    if not nums:
        return []
    
    nums = nums.copy()
    
    # 找出最小值和最大值
    min_val = min(nums)
    max_val = max(nums)
    
    # 计数数组的大小
    range_size = max_val - min_val + 1
    count = [0] * range_size
    
    # 统计每个元素出现的次数
    for num in nums:
        count[num - min_val] += 1
    
    # 累加计数，使count[i]表示小于等于i的元素个数
    for i in range(1, range_size):
        count[i] += count[i - 1]
    
    # 构建输出数组
    output = [0] * len(nums)
    
    # 从后向前遍历，保证稳定性
    for i in range(len(nums) - 1, -1, -1):
        index = count[nums[i] - min_val] - 1
        output[index] = nums[i]
        count[nums[i] - min_val] -= 1
    
    return output


def counting_sort_simple(nums):
    """
    简化版计数排序
    直接根据计数重建数组
    """
    if not nums:
        return []
    
    min_val = min(nums)
    max_val = max(nums)
    
    # 统计计数
    count = [0] * (max_val - min_val + 1)
    for num in nums:
        count[num - min_val] += 1
    
    # 重建数组
    result = []
    for i, c in enumerate(count):
        result.extend([i + min_val] * c)
    
    return result


def counting_sort_for_radix(nums, exp):
    """
    为基数排序定制的计数排序
    exp: 当前处理的位数（1, 10, 100...）
    """
    n = len(nums)
    output = [0] * n
    count = [0] * 10  # 0-9的计数
    
    # 统计当前位的数字分布
    for num in nums:
        index = (num // exp) % 10
        count[index] += 1
    
    # 累加计数
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # 构建输出数组
    for i in range(n - 1, -1, -1):
        index = (nums[i] // exp) % 10
        output[count[index] - 1] = nums[i]
        count[index] -= 1
    
    return output


def bucket_sort(nums, bucket_count=10):
    """
    桶排序算法
    将数据分布到多个桶中，每个桶内部排序后合并
    时间复杂度：平均O(n + k)，最坏O(n²)
    空间复杂度：O(n + k)
    """
    if not nums:
        return []
    
    nums = nums.copy()
    
    # 找出最大值和最小值
    min_val = min(nums)
    max_val = max(nums)
    
    # 计算桶的大小
    bucket_size = (max_val - min_val) / bucket_count + 1
    
    # 创建桶
    buckets = [[] for _ in range(bucket_count)]
    
    # 将元素分配到桶中
    for num in nums:
        bucket_index = int((num - min_val) / bucket_size)
        buckets[bucket_index].append(num)
    
    # 对每个桶内部进行排序
    result = []
    for bucket in buckets:
        # 可以使用任何排序算法，这里使用内置排序
        bucket.sort()
        result.extend(bucket)
    
    return result


def counting_sort_negative(nums):
    """
    处理包含负数的计数排序
    """
    if not nums:
        return []
    
    # 分离正数和负数
    positive = [x for x in nums if x >= 0]
    negative = [-x for x in nums if x < 0]
    
    # 分别排序
    sorted_positive = counting_sort_simple(positive) if positive else []
    sorted_negative = counting_sort_simple(negative) if negative else []
    
    # 合并结果（负数需要反转并取负）
    result = [-x for x in reversed(sorted_negative)]
    result.extend(sorted_positive)
    
    return result


def counting_sort_with_key(items, key_func, max_key_value):
    """
    根据键函数进行计数排序
    items: 要排序的元素列表
    key_func: 提取排序键的函数
    max_key_value: 键的最大值
    """
    count = [0] * (max_key_value + 1)
    
    # 统计每个键值的出现次数
    for item in items:
        count[key_func(item)] += 1
    
    # 累加计数
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # 构建输出数组
    output = [None] * len(items)
    for i in range(len(items) - 1, -1, -1):
        key = key_func(items[i])
        output[count[key] - 1] = items[i]
        count[key] -= 1
    
    return output


if __name__ == '__main__':
    # 测试基本计数排序
    nums = [random.randint(0, 20) for _ in range(15)]
    print("原始数组:", nums)
    print("计数排序:", counting_sort(nums))
    print("简化计数排序:", counting_sort_simple(nums))
    
    # 测试包含负数的情况
    nums_with_negative = [random.randint(-10, 10) for _ in range(15)]
    print("\n包含负数的数组:", nums_with_negative)
    print("处理负数的计数排序:", counting_sort_negative(nums_with_negative))
    
    # 测试桶排序
    float_nums = [random.random() * 100 for _ in range(15)]
    print("\n浮点数数组:", [round(x, 2) for x in float_nums])
    print("桶排序结果:", [round(x, 2) for x in bucket_sort(float_nums)])
    
    # 测试基于键的排序
    students = [
        ('Alice', 85),
        ('Bob', 92),
        ('Charlie', 85),
        ('David', 78),
        ('Eve', 92)
    ]
    print("\n学生成绩排序:")
    print("原始:", students)
    sorted_students = counting_sort_with_key(
        students,
        key_func=lambda x: x[1] - 70,  # 假设成绩范围70-100
        max_key_value=30
    )
    print("按成绩排序:", sorted_students)