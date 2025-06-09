from functools import lru_cache
import math


# 四种计算方法的实现
@lru_cache(maxsize=1000)
def fibonacci_lru(n):
    if n < 2:
        return 1
    else:
        return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)


def fibonacci_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n < 2:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


def fibonacci_iterative(n):
    if n < 2:
        return 1
    a, b = 1, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(n):
    sequence = []
    a, b = 1, 1
    for i in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence


def fibonacci_binet(n):
    phi = (1 + math.sqrt(5)) / 2
    return round((phi ** (n + 1)) / math.sqrt(5))


# 计算前10项并生成表格
n = 10
methods = ["lru_cache递归", "手动字典缓存递归", "循环迭代", "数学公式"]
results = [
    [fibonacci_lru(i) for i in range(n)],
    [fibonacci_memo(i) for i in range(n)],
    fibonacci_sequence(n),
    [fibonacci_binet(i) for i in range(n)]
]

# 打印表格，第一列宽度设为18
method_col_width = 18
header_row = f"{'方法'.ljust(method_col_width)} | " + " | ".join(f"{i:2}" for i in range(n))
print(header_row)
print('-' * len(header_row))

for i, method in enumerate(methods):
    row = method.ljust(method_col_width) + " | " + " | ".join(f"{results[i][j]:2}" for j in range(n))
    print(row)
