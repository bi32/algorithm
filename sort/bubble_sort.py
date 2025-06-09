import random


def bubble_sort(nums):

    nums = nums.copy()  # 创建副本，避免修改原数组
    n = len(nums)

    for i in range(n):
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]

    return nums


if __name__ == '__main__':

    nums = list(range(20))
    random.shuffle(nums)

    print(nums)
    print(bubble_sort(nums))
