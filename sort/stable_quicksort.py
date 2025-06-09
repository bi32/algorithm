import random


def qs(nums):

    if len(nums) < 2: return nums

    pivot = random.choice(nums)
    left, mid, right = [], [], []
    for i in nums:
        if i < pivot: left.append(i)
        elif i > pivot: right.append(i)
        else: mid.append(i)

    return qs(left) + mid + qs(right)


if __name__ == '__main__':

    nums = list(range(20))
    random.shuffle(nums)

    print(nums)
    print(qs(nums))
