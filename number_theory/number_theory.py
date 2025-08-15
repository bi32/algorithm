import random
import math


def gcd(a, b):
    """
    欧几里得算法求最大公约数
    时间复杂度：O(log min(a, b))
    """
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a, b):
    """
    扩展欧几里得算法
    返回 gcd(a, b) 和满足 ax + by = gcd(a, b) 的 x, y
    """
    if b == 0:
        return a, 1, 0
    
    gcd_val, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return gcd_val, x, y


def lcm(a, b):
    """
    最小公倍数
    lcm(a, b) = |a * b| / gcd(a, b)
    """
    return abs(a * b) // gcd(a, b)


def mod_inverse(a, m):
    """
    模逆元
    返回 x 使得 (a * x) % m = 1
    """
    gcd_val, x, _ = extended_gcd(a, m)
    
    if gcd_val != 1:
        return None  # 模逆元不存在
    
    return (x % m + m) % m


def fast_power(base, exp, mod=None):
    """
    快速幂算法
    计算 base^exp % mod
    时间复杂度：O(log exp)
    """
    result = 1
    base = base % mod if mod else base
    
    while exp > 0:
        if exp & 1:  # 如果exp是奇数
            result = (result * base) % mod if mod else result * base
        
        base = (base * base) % mod if mod else base * base
        exp >>= 1
    
    return result


def is_prime_naive(n):
    """
    朴素素性测试
    时间复杂度：O(√n)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True


def sieve_of_eratosthenes(n):
    """
    埃拉托斯特尼筛法
    找出所有小于等于n的素数
    时间复杂度：O(n log log n)
    空间复杂度：O(n)
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            # 标记i的所有倍数为非素数
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]


def segmented_sieve(low, high):
    """
    分段筛法
    找出[low, high]范围内的所有素数
    """
    # 先找出√high以内的所有素数
    limit = int(math.sqrt(high)) + 1
    base_primes = sieve_of_eratosthenes(limit)
    
    # 标记[low, high]范围内的合数
    is_prime = [True] * (high - low + 1)
    
    for prime in base_primes:
        # 找到第一个大于等于low的prime的倍数
        start = max(prime * prime, ((low + prime - 1) // prime) * prime)
        
        for j in range(start, high + 1, prime):
            is_prime[j - low] = False
    
    # 特殊处理1
    if low == 1:
        is_prime[0] = False
    
    return [i + low for i in range(high - low + 1) if is_prime[i]]


def miller_rabin(n, k=5):
    """
    Miller-Rabin素性测试
    概率算法，错误率约为 1/4^k
    时间复杂度：O(k log³n)
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # 将 n-1 写成 2^r * d 的形式
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # 进行k轮测试
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = fast_power(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    
    return True


def pollard_rho(n):
    """
    Pollard's rho算法
    整数分解算法
    """
    if n % 2 == 0:
        return 2
    
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = gcd(abs(x - y), n)
    
    return d if d != n else None


def chinese_remainder_theorem(remainders, moduli):
    """
    中国剩余定理
    求解同余方程组
    x ≡ remainders[i] (mod moduli[i])
    """
    if len(remainders) != len(moduli):
        return None
    
    # 检查模数两两互质
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                return None
    
    # 计算总模数
    M = 1
    for m in moduli:
        M *= m
    
    x = 0
    for i in range(len(moduli)):
        Mi = M // moduli[i]
        # 找Mi的模moduli[i]逆元
        _, ti, _ = extended_gcd(Mi, moduli[i])
        x += remainders[i] * Mi * ti
    
    return x % M


def euler_phi(n):
    """
    欧拉函数 φ(n)
    返回小于等于n且与n互质的正整数个数
    """
    result = n
    p = 2
    
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    
    if n > 1:
        result -= result // n
    
    return result


def discrete_log(base, target, mod):
    """
    离散对数问题（Baby-step Giant-step算法）
    求解 base^x ≡ target (mod mod)
    时间复杂度：O(√mod)
    """
    m = int(math.sqrt(mod)) + 1
    
    # Baby steps: 计算 base^j mod mod for j = 0, 1, ..., m-1
    baby_steps = {}
    value = 1
    
    for j in range(m):
        if value == target:
            return j
        baby_steps[value] = j
        value = (value * base) % mod
    
    # Giant steps: 计算 target * (base^(-m))^i
    factor = mod_inverse(fast_power(base, m, mod), mod)
    if factor is None:
        return None
    
    gamma = target
    for i in range(m):
        if gamma in baby_steps:
            return i * m + baby_steps[gamma]
        gamma = (gamma * factor) % mod
    
    return None


def primitive_root(p):
    """
    找出素数p的原根
    """
    if p == 2:
        return 1
    
    # 找出p-1的所有质因数
    factors = []
    phi = p - 1
    n = phi
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            while n % i == 0:
                n //= i
    
    if n > 1:
        factors.append(n)
    
    # 测试每个可能的原根
    for g in range(2, p):
        is_root = True
        
        for factor in factors:
            if fast_power(g, phi // factor, p) == 1:
                is_root = False
                break
        
        if is_root:
            return g
    
    return None


def jacobi_symbol(a, n):
    """
    雅可比符号
    推广的勒让德符号
    """
    if n <= 0 or n % 2 == 0:
        return None
    
    a = a % n
    result = 1
    
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result
        
        a, n = n, a
        
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        
        a = a % n
    
    if n == 1:
        return result
    else:
        return 0


def modular_sqrt(a, p):
    """
    模平方根
    求解 x² ≡ a (mod p)，其中p是素数
    """
    if not is_prime_naive(p):
        return None
    
    # 检查a是否为二次剩余
    if fast_power(a, (p - 1) // 2, p) != 1:
        return None
    
    # 特殊情况
    if p % 4 == 3:
        return fast_power(a, (p + 1) // 4, p)
    
    # Tonelli-Shanks算法
    # 找到Q和S使得 p - 1 = Q * 2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # 找到二次非剩余
    z = 2
    while fast_power(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    M = S
    c = fast_power(z, Q, p)
    t = fast_power(a, Q, p)
    R = fast_power(a, (Q + 1) // 2, p)
    
    while True:
        if t == 0:
            return 0
        if t == 1:
            return R
        
        # 找到最小的i使得t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        b = fast_power(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


def test_number_theory():
    """测试数论算法"""
    print("=== 数论算法测试 ===\n")
    
    # GCD和LCM
    print("1. 最大公约数和最小公倍数:")
    a, b = 48, 18
    print(f"   gcd({a}, {b}) = {gcd(a, b)}")
    print(f"   lcm({a}, {b}) = {lcm(a, b)}")
    
    g, x, y = extended_gcd(a, b)
    print(f"   扩展欧几里得: {a}×{x} + {b}×{y} = {g}")
    
    # 模运算
    print("\n2. 模运算:")
    print(f"   2^10 mod 1000 = {fast_power(2, 10, 1000)}")
    print(f"   3的模7逆元 = {mod_inverse(3, 7)}")
    
    # 素数相关
    print("\n3. 素数测试:")
    primes = sieve_of_eratosthenes(30)
    print(f"   30以内的素数: {primes}")
    
    n = 97
    print(f"   {n}是素数: {is_prime_naive(n)}")
    print(f"   Miller-Rabin测试{n}: {miller_rabin(n)}")
    
    # 区间素数
    print("\n4. 区间素数:")
    segment_primes = segmented_sieve(100, 120)
    print(f"   [100, 120]内的素数: {segment_primes}")
    
    # 中国剩余定理
    print("\n5. 中国剩余定理:")
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    result = chinese_remainder_theorem(remainders, moduli)
    print(f"   x ≡ 2 (mod 3)")
    print(f"   x ≡ 3 (mod 5)")
    print(f"   x ≡ 2 (mod 7)")
    print(f"   解: x = {result}")
    
    # 欧拉函数
    print("\n6. 欧拉函数:")
    for n in [10, 15, 17]:
        print(f"   φ({n}) = {euler_phi(n)}")
    
    # 原根
    print("\n7. 原根:")
    p = 11
    root = primitive_root(p)
    print(f"   {p}的原根: {root}")
    
    # 离散对数
    print("\n8. 离散对数:")
    base, target, mod = 3, 13, 17
    x = discrete_log(base, target, mod)
    if x is not None:
        print(f"   {base}^{x} ≡ {target} (mod {mod})")
        print(f"   验证: {fast_power(base, x, mod)} = {target}")
    
    # 模平方根
    print("\n9. 模平方根:")
    a, p = 10, 13
    sqrt = modular_sqrt(a, p)
    if sqrt:
        print(f"   {sqrt}² ≡ {a} (mod {p})")
        print(f"   验证: {(sqrt * sqrt) % p} = {a}")
    
    # 复杂度总结
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬───────────────┐")
    print("│ 算法              │ 时间复杂度     │")
    print("├──────────────────┼───────────────┤")
    print("│ 欧几里得算法       │ O(log n)      │")
    print("│ 快速幂            │ O(log n)      │")
    print("│ 埃氏筛            │ O(n log log n)│")
    print("│ Miller-Rabin     │ O(k log³n)    │")
    print("│ Pollard's rho    │ O(n^(1/4))    │")
    print("│ 离散对数(BSGS)    │ O(√n)         │")
    print("└──────────────────┴───────────────┘")


if __name__ == '__main__':
    test_number_theory()