import random


def insertion_sort(nums):

    nums = nums.copy()  # 创建副本，避免修改原数组

    for i in range(1, len(nums)):
        key = nums[i]
        j = i - 1

        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1

        nums[j + 1] = key

    return nums


if __name__ == '__main__':

    nums = list(range(20))
    random.shuffle(nums)

    print(nums)
    print(insertion_sort(nums))
