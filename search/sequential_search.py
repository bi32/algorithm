import random


def sequential_search(nums, target):
    for i in range(len(nums)):
        if nums[i] == target:
            return i
    return -1


if __name__ == '__main__':

    nums = list(range(20))
    random.shuffle(nums)

    target = random.choice(nums)  # 随机选择一个存在的目标

    print(nums)
    print(f"Target: {target}")
    print(f"Index: {sequential_search(nums, target)}")
