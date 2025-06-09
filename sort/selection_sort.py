import random


def selection_sort(nums):
    
    nums = nums.copy()  # 创建副本，避免修改原数组
    n = len(nums)

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if nums[j] < nums[min_idx]:
                min_idx = j

        nums[i], nums[min_idx] = nums[min_idx], nums[i]

    return nums


if __name__ == '__main__':

    nums = list(range(20))
    random.shuffle(nums)

    print(nums)
    print(selection_sort(nums))
