import random
import hashlib
import math
from collections import deque


class RSA:
    """
    RSA加密算法（简化教学版）
    注意：这是教学实现，不应用于生产环境
    """
    
    @staticmethod
    def is_prime(n, k=5):
        """Miller-Rabin素性测试"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # 将n-1写成2^r * d的形式
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # 进行k轮测试
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    @staticmethod
    def generate_prime(bits):
        """生成指定位数的素数"""
        while True:
            num = random.getrandbits(bits)
            num |= (1 << bits - 1) | 1  # 确保最高位和最低位为1
            
            if RSA.is_prime(num):
                return num
    
    @staticmethod
    def gcd(a, b):
        """最大公约数"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def extended_gcd(a, b):
        """扩展欧几里得算法"""
        if b == 0:
            return a, 1, 0
        
        gcd_val, x1, y1 = RSA.extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        
        return gcd_val, x, y
    
    @staticmethod
    def mod_inverse(a, m):
        """模逆元"""
        gcd_val, x, _ = RSA.extended_gcd(a, m)
        
        if gcd_val != 1:
            return None
        
        return (x % m + m) % m
    
    def __init__(self, key_size=512):
        """初始化RSA，生成密钥对"""
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self.generate_keypair()
    
    def generate_keypair(self):
        """生成RSA密钥对"""
        # 生成两个大素数
        p = self.generate_prime(self.key_size // 2)
        q = self.generate_prime(self.key_size // 2)
        
        # 计算n = p * q
        n = p * q
        
        # 计算欧拉函数φ(n) = (p-1)(q-1)
        phi = (p - 1) * (q - 1)
        
        # 选择公钥指数e（通常选择65537）
        e = 65537
        while self.gcd(e, phi) != 1:
            e = random.randrange(3, phi, 2)
        
        # 计算私钥指数d
        d = self.mod_inverse(e, phi)
        
        # 公钥：(n, e)
        self.public_key = (n, e)
        
        # 私钥：(n, d)
        self.private_key = (n, d)
    
    def encrypt(self, message, public_key=None):
        """加密消息"""
        if public_key is None:
            public_key = self.public_key
        
        n, e = public_key
        
        # 将消息转换为整数
        if isinstance(message, str):
            message = int.from_bytes(message.encode(), 'big')
        
        # 确保消息小于n
        if message >= n:
            raise ValueError("消息太长")
        
        # 加密：c = m^e mod n
        ciphertext = pow(message, e, n)
        return ciphertext
    
    def decrypt(self, ciphertext, private_key=None):
        """解密消息"""
        if private_key is None:
            private_key = self.private_key
        
        n, d = private_key
        
        # 解密：m = c^d mod n
        message = pow(ciphertext, d, n)
        
        # 将整数转换回字符串
        try:
            message_bytes = message.to_bytes((message.bit_length() + 7) // 8, 'big')
            return message_bytes.decode()
        except:
            return message
    
    def sign(self, message, private_key=None):
        """数字签名"""
        if private_key is None:
            private_key = self.private_key
        
        # 计算消息的哈希
        if isinstance(message, str):
            message = message.encode()
        
        hash_value = int(hashlib.sha256(message).hexdigest(), 16)
        
        # 使用私钥签名
        n, d = private_key
        signature = pow(hash_value % n, d, n)
        return signature
    
    def verify(self, message, signature, public_key=None):
        """验证签名"""
        if public_key is None:
            public_key = self.public_key
        
        # 计算消息的哈希
        if isinstance(message, str):
            message = message.encode()
        
        hash_value = int(hashlib.sha256(message).hexdigest(), 16)
        
        # 使用公钥验证
        n, e = public_key
        decrypted_hash = pow(signature, e, n)
        
        return decrypted_hash == (hash_value % n)


class DiffieHellman:
    """
    Diffie-Hellman密钥交换算法
    """
    
    @staticmethod
    def generate_prime(bits):
        """生成素数"""
        while True:
            num = random.getrandbits(bits)
            num |= (1 << bits - 1) | 1
            
            if RSA.is_prime(num):
                return num
    
    @staticmethod
    def primitive_root(p):
        """找原根（简化版）"""
        # 对于小素数，直接尝试
        if p == 2:
            return 1
        
        # 简化：随机选择一个可能的原根
        for g in range(2, min(p, 100)):
            if pow(g, p - 1, p) == 1:
                # 检查g的幂次
                is_root = True
                for i in range(2, int(math.sqrt(p - 1)) + 1):
                    if (p - 1) % i == 0:
                        if pow(g, i, p) == 1 or pow(g, (p - 1) // i, p) == 1:
                            is_root = False
                            break
                
                if is_root:
                    return g
        
        return 2  # 默认返回2
    
    def __init__(self, key_size=512):
        """初始化DH参数"""
        # 生成大素数p和生成元g
        self.p = self.generate_prime(key_size)
        self.g = self.primitive_root(self.p)
        
        # 生成私钥
        self.private_key = random.randint(2, self.p - 2)
        
        # 计算公钥
        self.public_key = pow(self.g, self.private_key, self.p)
    
    def generate_shared_secret(self, other_public_key):
        """生成共享密钥"""
        shared_secret = pow(other_public_key, self.private_key, self.p)
        return shared_secret
    
    def get_parameters(self):
        """获取DH参数"""
        return {
            'p': self.p,
            'g': self.g,
            'public_key': self.public_key
        }


class SHA256:
    """
    SHA-256哈希算法（简化教学版）
    """
    
    # SHA-256常量
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    
    # 初始哈希值
    H = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]
    
    @staticmethod
    def rotr(n, b, w=32):
        """右旋转"""
        return ((n >> b) | (n << (w - b))) & ((1 << w) - 1)
    
    @staticmethod
    def shr(n, b):
        """右移"""
        return n >> b
    
    @staticmethod
    def ch(x, y, z):
        """Choice函数"""
        return (x & y) ^ (~x & z)
    
    @staticmethod
    def maj(x, y, z):
        """Majority函数"""
        return (x & y) ^ (x & z) ^ (y & z)
    
    @staticmethod
    def sigma0(x):
        """Σ0函数"""
        return SHA256.rotr(x, 2) ^ SHA256.rotr(x, 13) ^ SHA256.rotr(x, 22)
    
    @staticmethod
    def sigma1(x):
        """Σ1函数"""
        return SHA256.rotr(x, 6) ^ SHA256.rotr(x, 11) ^ SHA256.rotr(x, 25)
    
    @staticmethod
    def gamma0(x):
        """σ0函数"""
        return SHA256.rotr(x, 7) ^ SHA256.rotr(x, 18) ^ SHA256.shr(x, 3)
    
    @staticmethod
    def gamma1(x):
        """σ1函数"""
        return SHA256.rotr(x, 17) ^ SHA256.rotr(x, 19) ^ SHA256.shr(x, 10)
    
    @staticmethod
    def padding(message):
        """消息填充"""
        if isinstance(message, str):
            message = message.encode()
        
        msg_len = len(message) * 8
        message += b'\x80'
        
        while (len(message) * 8) % 512 != 448:
            message += b'\x00'
        
        message += msg_len.to_bytes(8, 'big')
        return message
    
    @staticmethod
    def process_block(block, h):
        """处理单个512位块"""
        # 准备消息调度
        w = []
        for i in range(16):
            w.append(int.from_bytes(block[i*4:(i+1)*4], 'big'))
        
        for i in range(16, 64):
            s0 = SHA256.gamma0(w[i-15])
            s1 = SHA256.gamma1(w[i-2])
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)
        
        # 初始化工作变量
        a, b, c, d, e, f, g, h_var = h
        
        # 主循环
        for i in range(64):
            S1 = SHA256.sigma1(e)
            ch = SHA256.ch(e, f, g)
            temp1 = (h_var + S1 + ch + SHA256.K[i] + w[i]) & 0xffffffff
            S0 = SHA256.sigma0(a)
            maj = SHA256.maj(a, b, c)
            temp2 = (S0 + maj) & 0xffffffff
            
            h_var = g
            g = f
            f = e
            e = (d + temp1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xffffffff
        
        # 更新哈希值
        return [
            (h[0] + a) & 0xffffffff,
            (h[1] + b) & 0xffffffff,
            (h[2] + c) & 0xffffffff,
            (h[3] + d) & 0xffffffff,
            (h[4] + e) & 0xffffffff,
            (h[5] + f) & 0xffffffff,
            (h[6] + g) & 0xffffffff,
            (h[7] + h_var) & 0xffffffff
        ]
    
    @staticmethod
    def hash(message):
        """计算SHA-256哈希"""
        # 消息填充
        padded = SHA256.padding(message)
        
        # 初始化哈希值
        h = SHA256.H.copy()
        
        # 处理每个512位块
        for i in range(0, len(padded), 64):
            block = padded[i:i+64]
            h = SHA256.process_block(block, h)
        
        # 生成最终哈希
        return ''.join(format(x, '08x') for x in h)


class AES:
    """
    AES加密算法（简化教学版 - 仅实现AES-128）
    注意：这是极简化的教学实现
    """
    
    # S盒（简化版）
    SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        # ... 省略完整S盒
    ]
    
    @staticmethod
    def xor_bytes(a, b):
        """字节异或"""
        return bytes(x ^ y for x, y in zip(a, b))
    
    @staticmethod
    def simple_encrypt_block(block, key):
        """简化的块加密（仅用于演示）"""
        # 这是一个极度简化的版本，仅用于教学
        # 实际AES包含SubBytes, ShiftRows, MixColumns, AddRoundKey等步骤
        
        # 简单异或加密
        encrypted = SHA256.xor_bytes(block, key[:len(block)])
        
        # 简单置换
        result = bytearray(encrypted)
        for i in range(len(result)):
            result[i] = (result[i] + key[i % len(key)]) & 0xff
        
        return bytes(result)
    
    @staticmethod
    def simple_decrypt_block(block, key):
        """简化的块解密"""
        # 逆置换
        result = bytearray(block)
        for i in range(len(result)):
            result[i] = (result[i] - key[i % len(key)]) & 0xff
        
        # 异或解密
        decrypted = SHA256.xor_bytes(bytes(result), key[:len(block)])
        
        return decrypted
    
    @staticmethod
    def encrypt(plaintext, key):
        """ECB模式加密（简化版）"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        if isinstance(key, str):
            key = key.encode()
        
        # 确保密钥长度为16字节
        if len(key) < 16:
            key = key + b'\x00' * (16 - len(key))
        else:
            key = key[:16]
        
        # PKCS7填充
        pad_len = 16 - (len(plaintext) % 16)
        plaintext += bytes([pad_len] * pad_len)
        
        # 分块加密
        ciphertext = b''
        for i in range(0, len(plaintext), 16):
            block = plaintext[i:i+16]
            ciphertext += AES.simple_encrypt_block(block, key)
        
        return ciphertext
    
    @staticmethod
    def decrypt(ciphertext, key):
        """ECB模式解密（简化版）"""
        if isinstance(key, str):
            key = key.encode()
        
        # 确保密钥长度为16字节
        if len(key) < 16:
            key = key + b'\x00' * (16 - len(key))
        else:
            key = key[:16]
        
        # 分块解密
        plaintext = b''
        for i in range(0, len(ciphertext), 16):
            block = ciphertext[i:i+16]
            plaintext += AES.simple_decrypt_block(block, key)
        
        # 去除PKCS7填充
        pad_len = plaintext[-1]
        plaintext = plaintext[:-pad_len]
        
        return plaintext.decode()


class Caesar:
    """凯撒密码"""
    
    @staticmethod
    def encrypt(text, shift=3):
        """加密"""
        result = []
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                encrypted_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                result.append(encrypted_char)
            else:
                result.append(char)
        return ''.join(result)
    
    @staticmethod
    def decrypt(text, shift=3):
        """解密"""
        return Caesar.encrypt(text, -shift)


class Vigenere:
    """维吉尼亚密码"""
    
    @staticmethod
    def encrypt(text, key):
        """加密"""
        result = []
        key = key.upper()
        key_index = 0
        
        for char in text:
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - ord('A')
                ascii_offset = 65 if char.isupper() else 97
                encrypted_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                result.append(encrypted_char)
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    @staticmethod
    def decrypt(text, key):
        """解密"""
        result = []
        key = key.upper()
        key_index = 0
        
        for char in text:
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - ord('A')
                ascii_offset = 65 if char.isupper() else 97
                decrypted_char = chr((ord(char) - ascii_offset - shift) % 26 + ascii_offset)
                result.append(decrypted_char)
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)


def test_cryptography():
    """测试密码学算法"""
    print("=== 密码学算法测试 ===\n")
    
    # 测试RSA
    print("1. RSA加密:")
    rsa = RSA(key_size=512)
    message = "Hello RSA!"
    
    # 加密
    encrypted = rsa.encrypt(message)
    print(f"   原文: {message}")
    print(f"   密文: {encrypted}")
    
    # 解密
    decrypted = rsa.decrypt(encrypted)
    print(f"   解密: {decrypted}")
    
    # 数字签名
    signature = rsa.sign(message)
    verified = rsa.verify(message, signature)
    print(f"   签名验证: {verified}")
    
    # 测试Diffie-Hellman
    print("\n2. Diffie-Hellman密钥交换:")
    alice = DiffieHellman(key_size=256)
    bob = DiffieHellman(key_size=256)
    
    # 设置相同的p和g
    bob.p = alice.p
    bob.g = alice.g
    bob.private_key = random.randint(2, alice.p - 2)
    bob.public_key = pow(bob.g, bob.private_key, bob.p)
    
    # 交换公钥并生成共享密钥
    alice_shared = alice.generate_shared_secret(bob.public_key)
    bob_shared = bob.generate_shared_secret(alice.public_key)
    
    print(f"   Alice公钥: {alice.public_key}")
    print(f"   Bob公钥: {bob.public_key}")
    print(f"   共享密钥相同: {alice_shared == bob_shared}")
    
    # 测试SHA-256
    print("\n3. SHA-256哈希:")
    messages = ["Hello", "World", "Hello World"]
    for msg in messages:
        hash_value = SHA256.hash(msg)
        print(f"   SHA256('{msg}'): {hash_value[:16]}...")
    
    # 测试标准库对比
    import hashlib
    test_msg = "Test message"
    custom_hash = SHA256.hash(test_msg)
    standard_hash = hashlib.sha256(test_msg.encode()).hexdigest()
    print(f"   自定义实现: {custom_hash[:16]}...")
    print(f"   标准库实现: {standard_hash[:16]}...")
    
    # 测试简化AES
    print("\n4. AES加密（简化版）:")
    plaintext = "Secret Message"
    key = "MySecretKey12345"
    
    encrypted = AES.encrypt(plaintext, key)
    print(f"   原文: {plaintext}")
    print(f"   密文: {encrypted.hex()[:32]}...")
    
    decrypted = AES.decrypt(encrypted, key)
    print(f"   解密: {decrypted}")
    
    # 测试古典密码
    print("\n5. 古典密码:")
    text = "HELLO WORLD"
    
    # 凯撒密码
    caesar_encrypted = Caesar.encrypt(text, shift=3)
    caesar_decrypted = Caesar.decrypt(caesar_encrypted, shift=3)
    print(f"   凯撒密码:")
    print(f"     原文: {text}")
    print(f"     密文: {caesar_encrypted}")
    print(f"     解密: {caesar_decrypted}")
    
    # 维吉尼亚密码
    vigenere_key = "KEY"
    vigenere_encrypted = Vigenere.encrypt(text, vigenere_key)
    vigenere_decrypted = Vigenere.decrypt(vigenere_encrypted, vigenere_key)
    print(f"   维吉尼亚密码:")
    print(f"     原文: {text}")
    print(f"     密钥: {vigenere_key}")
    print(f"     密文: {vigenere_encrypted}")
    print(f"     解密: {vigenere_decrypted}")
    
    # 算法特点
    print("\n=== 算法特点 ===")
    print("┌──────────────┬──────────────┬──────────────┐")
    print("│ 算法          │ 类型         │ 安全性        │")
    print("├──────────────┼──────────────┼──────────────┤")
    print("│ RSA          │ 非对称加密    │ 高（大密钥）   │")
    print("│ DH           │ 密钥交换      │ 高            │")
    print("│ SHA-256      │ 哈希函数      │ 高            │")
    print("│ AES          │ 对称加密      │ 高            │")
    print("│ 凯撒密码      │ 古典密码      │ 极低          │")
    print("│ 维吉尼亚      │ 古典密码      │ 低            │")
    print("└──────────────┴──────────────┴──────────────┘")
    
    print("\n⚠️  注意：这些实现仅用于教学目的，不应用于生产环境！")


if __name__ == '__main__':
    test_cryptography()