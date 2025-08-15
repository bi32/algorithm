import random


def radix_sort(nums):
    """
    基数排序算法（LSD - 从最低位开始）
    时间复杂度：O(d * (n + k))，d为位数，k为基数（通常为10）
    空间复杂度：O(n + k)
    稳定排序：是
    """
    if not nums:
        return []
    
    nums = nums.copy()
    
    # 处理负数的情况
    min_num = min(nums)
    if min_num < 0:
        # 将所有数字偏移，使最小值变为0
        offset = -min_num
        nums = [num + offset for num in nums]
    else:
        offset = 0
    
    # 找出最大值，确定位数
    max_num = max(nums)
    
    # 从个位开始，对每一位进行计数排序
    exp = 1
    while max_num // exp > 0:
        nums = counting_sort_for_digit(nums, exp)
        exp *= 10
    
    # 如果有偏移，恢复原始值
    if offset > 0:
        nums = [num - offset for num in nums]
    
    return nums


def counting_sort_for_digit(nums, exp):
    """
    对特定位进行计数排序
    exp: 当前位的基数（1, 10, 100...）
    """
    n = len(nums)
    output = [0] * n
    count = [0] * 10
    
    # 统计当前位的数字分布
    for num in nums:
        index = (num // exp) % 10
        count[index] += 1
    
    # 累加计数
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # 从后向前构建输出数组，保证稳定性
    for i in range(n - 1, -1, -1):
        index = (nums[i] // exp) % 10
        output[count[index] - 1] = nums[i]
        count[index] -= 1
    
    return output


def radix_sort_msd(nums):
    """
    基数排序算法（MSD - 从最高位开始）
    递归实现
    """
    if not nums:
        return []
    
    nums = nums.copy()
    
    # 处理负数
    min_num = min(nums)
    if min_num < 0:
        offset = -min_num
        nums = [num + offset for num in nums]
    else:
        offset = 0
    
    # 找出最大值和位数
    max_num = max(nums)
    max_exp = 1
    while max_num // (max_exp * 10) > 0:
        max_exp *= 10
    
    def msd_sort(nums, exp):
        """递归MSD排序"""
        if len(nums) <= 1 or exp < 1:
            return nums
        
        # 创建10个桶
        buckets = [[] for _ in range(10)]
        
        # 根据当前位分配到桶中
        for num in nums:
            digit = (num // exp) % 10
            buckets[digit].append(num)
        
        # 递归排序每个桶
        result = []
        for bucket in buckets:
            if len(bucket) > 1:
                result.extend(msd_sort(bucket, exp // 10))
            else:
                result.extend(bucket)
        
        return result
    
    nums = msd_sort(nums, max_exp)
    
    # 恢复原始值
    if offset > 0:
        nums = [num - offset for num in nums]
    
    return nums


def radix_sort_binary(nums):
    """
    二进制基数排序
    使用位运算，基数为2
    """
    if not nums:
        return []
    
    nums = nums.copy()
    
    # 处理负数
    min_num = min(nums)
    if min_num < 0:
        offset = -min_num
        nums = [num + offset for num in nums]
    else:
        offset = 0
    
    # 找出最大值的位数
    max_num = max(nums)
    max_bits = max_num.bit_length()
    
    # 对每一位进行排序
    for bit in range(max_bits):
        # 分成两个桶：0和1
        zeros = []
        ones = []
        
        for num in nums:
            if (num >> bit) & 1:
                ones.append(num)
            else:
                zeros.append(num)
        
        # 合并桶
        nums = zeros + ones
    
    # 恢复原始值
    if offset > 0:
        nums = [num - offset for num in nums]
    
    return nums


def radix_sort_strings(strings):
    """
    字符串的基数排序
    从右到左（LSD）处理每个字符位置
    """
    if not strings:
        return []
    
    strings = strings.copy()
    
    # 找出最长字符串的长度
    max_len = max(len(s) for s in strings)
    
    # 从最后一个字符位置开始排序
    for pos in range(max_len - 1, -1, -1):
        # 创建256个桶（ASCII字符）
        buckets = [[] for _ in range(256)]
        
        for string in strings:
            # 如果字符串长度不够，使用0（最小值）
            if pos < len(string):
                index = ord(string[pos])
            else:
                index = 0
            buckets[index].append(string)
        
        # 合并桶
        strings = []
        for bucket in buckets:
            strings.extend(bucket)
    
    return strings


def radix_sort_float(nums, precision=6):
    """
    浮点数的基数排序
    通过转换为整数处理
    precision: 小数位精度
    """
    if not nums:
        return []
    
    # 转换为整数
    multiplier = 10 ** precision
    int_nums = [int(num * multiplier) for num in nums]
    
    # 使用整数基数排序
    sorted_ints = radix_sort(int_nums)
    
    # 转换回浮点数
    return [num / multiplier for num in sorted_ints]


def hybrid_radix_sort(nums):
    """
    混合基数排序
    对于小数组使用插入排序
    """
    if not nums:
        return []
    
    # 小数组使用插入排序
    if len(nums) < 10:
        return insertion_sort(nums)
    
    return radix_sort(nums)


def insertion_sort(nums):
    """辅助函数：插入排序"""
    nums = nums.copy()
    for i in range(1, len(nums)):
        key = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = key
    return nums


if __name__ == '__main__':
    # 测试整数基数排序
    nums = [random.randint(0, 999) for _ in range(15)]
    print("原始数组:", nums)
    print("LSD基数排序:", radix_sort(nums))
    print("MSD基数排序:", radix_sort_msd(nums))
    print("二进制基数排序:", radix_sort_binary(nums))
    
    # 测试负数
    nums_with_negative = [random.randint(-500, 500) for _ in range(10)]
    print("\n包含负数的数组:", nums_with_negative)
    print("基数排序结果:", radix_sort(nums_with_negative))
    
    # 测试字符串排序
    strings = ['apple', 'app', 'application', 'apply', 'banana', 'band', 'can', 'car']
    print("\n字符串数组:", strings)
    print("字符串基数排序:", radix_sort_strings(strings))
    
    # 测试浮点数排序
    float_nums = [random.uniform(0, 100) for _ in range(10)]
    print("\n浮点数数组:", [round(x, 2) for x in float_nums])
    sorted_floats = radix_sort_float(float_nums, precision=2)
    print("浮点数基数排序:", [round(x, 2) for x in sorted_floats])
    
    # 性能测试提示
    print("\n基数排序特点:")
    print("- 适用于整数或可以转换为整数的数据")
    print("- 当数据范围不大时，性能优于比较排序")
    print("- 稳定排序，保持相等元素的相对顺序")
    print("- 不是原地排序，需要额外空间")