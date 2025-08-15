class Stack:
    """
    栈的实现（LIFO - 后进先出）
    使用Python列表作为底层存储
    """
    
    def __init__(self):
        """初始化空栈"""
        self.items = []
    
    def push(self, item):
        """
        入栈操作
        时间复杂度：O(1)
        """
        self.items.append(item)
    
    def pop(self):
        """
        出栈操作
        时间复杂度：O(1)
        """
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """
        查看栈顶元素但不删除
        时间复杂度：O(1)
        """
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        """检查栈是否为空"""
        return len(self.items) == 0
    
    def size(self):
        """返回栈的大小"""
        return len(self.items)
    
    def clear(self):
        """清空栈"""
        self.items = []
    
    def __str__(self):
        """字符串表示"""
        return f"Stack({self.items})"
    
    def __repr__(self):
        return self.__str__()


class MinStack:
    """
    支持O(1)时间获取最小值的栈
    """
    
    def __init__(self):
        self.stack = []
        self.min_stack = []  # 辅助栈，存储最小值
    
    def push(self, val):
        """入栈"""
        self.stack.append(val)
        
        # 更新最小值栈
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """出栈"""
        if not self.stack:
            raise IndexError("Stack is empty")
        
        val = self.stack.pop()
        
        # 如果出栈的是最小值，同时从最小值栈出栈
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        
        return val
    
    def top(self):
        """获取栈顶元素"""
        if not self.stack:
            raise IndexError("Stack is empty")
        return self.stack[-1]
    
    def get_min(self):
        """
        获取最小值
        时间复杂度：O(1)
        """
        if not self.min_stack:
            raise IndexError("Stack is empty")
        return self.min_stack[-1]
    
    def is_empty(self):
        return len(self.stack) == 0


class ArrayStack:
    """
    使用固定大小数组实现的栈
    """
    
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.items = [None] * capacity
        self.top = -1
    
    def push(self, item):
        """入栈"""
        if self.is_full():
            raise OverflowError("Stack is full")
        self.top += 1
        self.items[self.top] = item
    
    def pop(self):
        """出栈"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        item = self.items[self.top]
        self.items[self.top] = None  # 清理引用
        self.top -= 1
        return item
    
    def peek(self):
        """查看栈顶元素"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[self.top]
    
    def is_empty(self):
        return self.top == -1
    
    def is_full(self):
        return self.top == self.capacity - 1
    
    def size(self):
        return self.top + 1


def is_valid_parentheses(s):
    """
    使用栈验证括号匹配
    示例应用：检查括号是否有效
    """
    stack = Stack()
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # 遇到右括号
            if stack.is_empty() or stack.pop() != mapping[char]:
                return False
        elif char in mapping.values():
            # 遇到左括号
            stack.push(char)
    
    return stack.is_empty()


def evaluate_postfix(expression):
    """
    计算后缀表达式
    示例：["2", "1", "+", "3", "*"] -> 9
    """
    stack = Stack()
    operators = {'+', '-', '*', '/'}
    
    for token in expression:
        if token in operators:
            # 注意操作数的顺序
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.push(a + b)
            elif token == '-':
                stack.push(a - b)
            elif token == '*':
                stack.push(a * b)
            elif token == '/':
                stack.push(int(a / b))  # 整数除法
        else:
            stack.push(int(token))
    
    return stack.pop()


def infix_to_postfix(expression):
    """
    中缀表达式转后缀表达式
    示例："3 + 4 * 2" -> "3 4 2 * +"
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}
    stack = Stack()
    postfix = []
    
    tokens = expression.split()
    
    for token in tokens:
        if token.isdigit():
            postfix.append(token)
        elif token == '(':
            stack.push(token)
        elif token == ')':
            while not stack.is_empty() and stack.peek() != '(':
                postfix.append(stack.pop())
            stack.pop()  # 弹出'('
        elif token in precedence:
            while (not stack.is_empty() and 
                   stack.peek() != '(' and
                   precedence.get(stack.peek(), 0) >= precedence[token]):
                postfix.append(stack.pop())
            stack.push(token)
    
    while not stack.is_empty():
        postfix.append(stack.pop())
    
    return ' '.join(postfix)


def reverse_string(s):
    """使用栈反转字符串"""
    stack = Stack()
    for char in s:
        stack.push(char)
    
    result = []
    while not stack.is_empty():
        result.append(stack.pop())
    
    return ''.join(result)


if __name__ == '__main__':
    # 测试基本栈操作
    print("基本栈操作:")
    stack = Stack()
    for i in range(5):
        stack.push(i)
        print(f"入栈 {i}: {stack}")
    
    print(f"栈顶元素: {stack.peek()}")
    print(f"栈大小: {stack.size()}")
    
    while not stack.is_empty():
        print(f"出栈: {stack.pop()}")
    
    # 测试最小值栈
    print("\n最小值栈测试:")
    min_stack = MinStack()
    values = [3, 5, 2, 6, 1, 4]
    for val in values:
        min_stack.push(val)
        print(f"入栈 {val}, 当前最小值: {min_stack.get_min()}")
    
    # 测试括号匹配
    print("\n括号匹配测试:")
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    for case in test_cases:
        print(f"{case}: {is_valid_parentheses(case)}")
    
    # 测试后缀表达式计算
    print("\n后缀表达式计算:")
    postfix = ["2", "1", "+", "3", "*"]
    print(f"{postfix} = {evaluate_postfix(postfix)}")
    
    # 测试中缀转后缀
    print("\n中缀转后缀:")
    infix = "3 + 4 * 2"
    postfix = infix_to_postfix(infix)
    print(f"中缀: {infix}")
    print(f"后缀: {postfix}")
    
    # 测试字符串反转
    print("\n字符串反转:")
    s = "Hello World"
    print(f"原始: {s}")
    print(f"反转: {reverse_string(s)}")