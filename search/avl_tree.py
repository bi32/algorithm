class AVLNode:
    """AVL树节点"""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """
    AVL树实现（自平衡二叉搜索树）
    性质：任意节点的左右子树高度差不超过1
    所有操作的时间复杂度：O(log n)
    """
    
    def __init__(self):
        self.root = None
    
    def get_height(self, node):
        """获取节点高度"""
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node):
        """获取节点的平衡因子"""
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def update_height(self, node):
        """更新节点高度"""
        if node:
            node.height = 1 + max(self.get_height(node.left), 
                                 self.get_height(node.right))
    
    def rotate_right(self, y):
        """
        右旋转
            y                x
           / \              / \
          x   C    =>      A   y
         / \                  / \
        A   B                B   C
        """
        x = y.left
        B = x.right
        
        # 执行旋转
        x.right = y
        y.left = B
        
        # 更新高度
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def rotate_left(self, x):
        """
        左旋转
          x                  y
         / \                / \
        A   y      =>      x   C
           / \            / \
          B   C          A   B
        """
        y = x.right
        B = y.left
        
        # 执行旋转
        y.left = x
        x.right = B
        
        # 更新高度
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, val):
        """插入节点并保持平衡"""
        def _insert(node, val):
            # 1. 执行标准BST插入
            if not node:
                return AVLNode(val)
            
            if val < node.val:
                node.left = _insert(node.left, val)
            elif val > node.val:
                node.right = _insert(node.right, val)
            else:
                # 不允许重复值
                return node
            
            # 2. 更新节点高度
            self.update_height(node)
            
            # 3. 获取平衡因子
            balance = self.get_balance(node)
            
            # 4. 如果节点不平衡，有四种情况
            
            # 左左情况
            if balance > 1 and val < node.left.val:
                return self.rotate_right(node)
            
            # 右右情况
            if balance < -1 and val > node.right.val:
                return self.rotate_left(node)
            
            # 左右情况
            if balance > 1 and val > node.left.val:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)
            
            # 右左情况
            if balance < -1 and val < node.right.val:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)
            
            return node
        
        self.root = _insert(self.root, val)
    
    def delete(self, val):
        """删除节点并保持平衡"""
        def _get_min_value_node(node):
            current = node
            while current.left:
                current = current.left
            return current
        
        def _delete(node, val):
            # 1. 执行标准BST删除
            if not node:
                return node
            
            if val < node.val:
                node.left = _delete(node.left, val)
            elif val > node.val:
                node.right = _delete(node.right, val)
            else:
                # 找到要删除的节点
                if not node.left:
                    return node.right
                elif not node.right:
                    return node.left
                else:
                    # 有两个子节点
                    temp = _get_min_value_node(node.right)
                    node.val = temp.val
                    node.right = _delete(node.right, temp.val)
            
            if not node:
                return node
            
            # 2. 更新节点高度
            self.update_height(node)
            
            # 3. 获取平衡因子
            balance = self.get_balance(node)
            
            # 4. 如果节点不平衡，有四种情况
            
            # 左左情况
            if balance > 1 and self.get_balance(node.left) >= 0:
                return self.rotate_right(node)
            
            # 左右情况
            if balance > 1 and self.get_balance(node.left) < 0:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)
            
            # 右右情况
            if balance < -1 and self.get_balance(node.right) <= 0:
                return self.rotate_left(node)
            
            # 右左情况
            if balance < -1 and self.get_balance(node.right) > 0:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)
            
            return node
        
        self.root = _delete(self.root, val)
    
    def search(self, val):
        """搜索节点"""
        def _search(node, val):
            if not node:
                return False
            
            if val == node.val:
                return True
            elif val < node.val:
                return _search(node.left, val)
            else:
                return _search(node.right, val)
        
        return _search(self.root, val)
    
    def find_min(self):
        """找到最小值"""
        if not self.root:
            return None
        
        node = self.root
        while node.left:
            node = node.left
        return node.val
    
    def find_max(self):
        """找到最大值"""
        if not self.root:
            return None
        
        node = self.root
        while node.right:
            node = node.right
        return node.val
    
    def inorder_traversal(self):
        """中序遍历"""
        result = []
        
        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.val)
                _inorder(node.right)
        
        _inorder(self.root)
        return result
    
    def preorder_traversal(self):
        """前序遍历"""
        result = []
        
        def _preorder(node):
            if node:
                result.append(node.val)
                _preorder(node.left)
                _preorder(node.right)
        
        _preorder(self.root)
        return result
    
    def level_order_traversal(self):
        """层序遍历"""
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.pop(0)
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    
    def is_balanced(self):
        """检查是否为平衡树"""
        def _check(node):
            if not node:
                return True
            
            balance = self.get_balance(node)
            if abs(balance) > 1:
                return False
            
            return _check(node.left) and _check(node.right)
        
        return _check(self.root)
    
    def get_tree_height(self):
        """获取树的高度"""
        return self.get_height(self.root)
    
    def count_nodes(self):
        """计算节点总数"""
        def _count(node):
            if not node:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        
        return _count(self.root)
    
    def range_query(self, low, high):
        """范围查询"""
        result = []
        
        def _range_query(node, low, high):
            if not node:
                return
            
            if node.val > low:
                _range_query(node.left, low, high)
            
            if low <= node.val <= high:
                result.append(node.val)
            
            if node.val < high:
                _range_query(node.right, low, high)
        
        _range_query(self.root, low, high)
        return result
    
    def print_tree(self):
        """打印树的结构（用于调试）"""
        def _print_tree(node, prefix="", is_tail=True):
            if not node:
                return
            
            print(prefix + ("└── " if is_tail else "├── ") + 
                  f"{node.val} (h={node.height}, b={self.get_balance(node)})")
            
            children = []
            if node.left:
                children.append((node.left, False))
            if node.right:
                children.append((node.right, True))
            
            for i, (child, is_last) in enumerate(children):
                if i == len(children) - 1:
                    _print_tree(child, prefix + ("    " if is_tail else "│   "), True)
                else:
                    _print_tree(child, prefix + ("    " if is_tail else "│   "), False)
        
        if self.root:
            _print_tree(self.root)
        else:
            print("空树")


def test_avl_operations():
    """测试AVL树的各种操作"""
    avl = AVLTree()
    
    # 测试插入和自动平衡
    print("=== AVL树插入测试 ===")
    values = [10, 20, 30, 40, 50, 25]
    for val in values:
        avl.insert(val)
        print(f"插入 {val} 后，树高度: {avl.get_tree_height()}, 是否平衡: {avl.is_balanced()}")
    
    print(f"\n中序遍历: {avl.inorder_traversal()}")
    print(f"前序遍历: {avl.preorder_traversal()}")
    print(f"层序遍历: {avl.level_order_traversal()}")
    
    print("\n树的结构:")
    avl.print_tree()
    
    # 测试搜索
    print("\n=== 搜索测试 ===")
    search_values = [25, 35, 40]
    for val in search_values:
        print(f"搜索 {val}: {avl.search(val)}")
    
    # 测试删除
    print("\n=== 删除测试 ===")
    delete_values = [10, 40]
    for val in delete_values:
        avl.delete(val)
        print(f"删除 {val} 后，中序遍历: {avl.inorder_traversal()}")
        print(f"是否平衡: {avl.is_balanced()}")
    
    print("\n删除后的树结构:")
    avl.print_tree()
    
    # 测试范围查询
    print("\n=== 范围查询测试 ===")
    print(f"范围 [20, 35]: {avl.range_query(20, 35)}")
    
    # 测试最值
    print("\n=== 最值测试 ===")
    print(f"最小值: {avl.find_min()}")
    print(f"最大值: {avl.find_max()}")
    
    # 测试大量插入
    print("\n=== 大量插入测试 ===")
    import random
    avl2 = AVLTree()
    test_values = random.sample(range(1, 101), 20)
    
    for val in test_values:
        avl2.insert(val)
    
    print(f"插入20个随机值后:")
    print(f"树高度: {avl2.get_tree_height()}")
    print(f"节点数: {avl2.count_nodes()}")
    print(f"是否平衡: {avl2.is_balanced()}")
    print(f"理论最小高度: {import_math_log2(20):.2f}")
    
    # 比较AVL树和普通BST的高度
    print("\n=== AVL树 vs 普通BST高度比较 ===")
    sequential_values = list(range(1, 16))
    
    avl3 = AVLTree()
    for val in sequential_values:
        avl3.insert(val)
    
    print(f"顺序插入1-15:")
    print(f"AVL树高度: {avl3.get_tree_height()} (平衡)")
    print(f"普通BST高度: {len(sequential_values) - 1} (退化为链表)")


def import_math_log2(n):
    """辅助函数：计算log2"""
    import math
    return math.log2(n)


if __name__ == '__main__':
    test_avl_operations()