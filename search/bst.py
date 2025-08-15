class TreeNode:
    """二叉搜索树节点"""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class BinarySearchTree:
    """
    二叉搜索树（BST）实现
    性质：左子树的所有节点值 < 根节点值 < 右子树的所有节点值
    """
    
    def __init__(self):
        self.root = None
        self.size = 0
    
    def insert(self, val):
        """
        插入节点
        平均时间复杂度：O(log n)
        最坏时间复杂度：O(n)
        """
        def _insert(node, val):
            if not node:
                return TreeNode(val)
            
            if val < node.val:
                node.left = _insert(node.left, val)
            elif val > node.val:
                node.right = _insert(node.right, val)
            
            return node
        
        self.root = _insert(self.root, val)
        self.size += 1
    
    def search(self, val):
        """
        搜索节点
        平均时间复杂度：O(log n)
        最坏时间复杂度：O(n)
        """
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
    
    def delete(self, val):
        """
        删除节点
        平均时间复杂度：O(log n)
        最坏时间复杂度：O(n)
        """
        def _find_min(node):
            while node.left:
                node = node.left
            return node
        
        def _delete(node, val):
            if not node:
                return None
            
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
                    # 有两个子节点，找到右子树的最小节点
                    successor = _find_min(node.right)
                    node.val = successor.val
                    node.right = _delete(node.right, successor.val)
            
            return node
        
        self.root = _delete(self.root, val)
        self.size -= 1
    
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
        """
        中序遍历（返回有序数组）
        时间复杂度：O(n)
        """
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
    
    def postorder_traversal(self):
        """后序遍历"""
        result = []
        
        def _postorder(node):
            if node:
                _postorder(node.left)
                _postorder(node.right)
                result.append(node.val)
        
        _postorder(self.root)
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
    
    def height(self):
        """
        计算树的高度
        时间复杂度：O(n)
        """
        def _height(node):
            if not node:
                return -1
            return 1 + max(_height(node.left), _height(node.right))
        
        return _height(self.root)
    
    def is_balanced(self):
        """
        检查树是否平衡
        平衡条件：任意节点的左右子树高度差不超过1
        """
        def _check_balance(node):
            if not node:
                return True, -1
            
            left_balanced, left_height = _check_balance(node.left)
            if not left_balanced:
                return False, 0
            
            right_balanced, right_height = _check_balance(node.right)
            if not right_balanced:
                return False, 0
            
            is_balanced = abs(left_height - right_height) <= 1
            height = 1 + max(left_height, right_height)
            
            return is_balanced, height
        
        balanced, _ = _check_balance(self.root)
        return balanced
    
    def is_valid_bst(self):
        """
        验证是否为有效的二叉搜索树
        """
        def _validate(node, min_val, max_val):
            if not node:
                return True
            
            if node.val <= min_val or node.val >= max_val:
                return False
            
            return (_validate(node.left, min_val, node.val) and 
                    _validate(node.right, node.val, max_val))
        
        return _validate(self.root, float('-inf'), float('inf'))
    
    def get_predecessor(self, val):
        """
        找到前驱节点（小于val的最大值）
        """
        predecessor = None
        node = self.root
        
        while node:
            if node.val < val:
                predecessor = node.val
                node = node.right
            else:
                node = node.left
        
        return predecessor
    
    def get_successor(self, val):
        """
        找到后继节点（大于val的最小值）
        """
        successor = None
        node = self.root
        
        while node:
            if node.val > val:
                successor = node.val
                node = node.left
            else:
                node = node.right
        
        return successor
    
    def find_lca(self, val1, val2):
        """
        找到两个节点的最近公共祖先
        """
        def _lca(node, val1, val2):
            if not node:
                return None
            
            if val1 < node.val and val2 < node.val:
                return _lca(node.left, val1, val2)
            elif val1 > node.val and val2 > node.val:
                return _lca(node.right, val1, val2)
            else:
                return node.val
        
        return _lca(self.root, val1, val2)
    
    def count_nodes(self):
        """计算节点总数"""
        def _count(node):
            if not node:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        
        return _count(self.root)
    
    def range_query(self, low, high):
        """
        范围查询：返回[low, high]范围内的所有值
        """
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
    
    def kth_smallest(self, k):
        """
        找到第k小的元素
        """
        def _inorder(node):
            if not node:
                return []
            return _inorder(node.left) + [node.val] + _inorder(node.right)
        
        sorted_vals = _inorder(self.root)
        return sorted_vals[k - 1] if k <= len(sorted_vals) else None
    
    def serialize(self):
        """
        序列化二叉树
        """
        def _serialize(node):
            if not node:
                return "null"
            return f"{node.val},{_serialize(node.left)},{_serialize(node.right)}"
        
        return _serialize(self.root)
    
    def deserialize(self, data):
        """
        反序列化二叉树
        """
        def _deserialize(vals):
            val = vals.pop(0)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = _deserialize(vals)
            node.right = _deserialize(vals)
            return node
        
        vals = data.split(',')
        self.root = _deserialize(vals)


if __name__ == '__main__':
    # 测试BST基本操作
    print("二叉搜索树测试:")
    bst = BinarySearchTree()
    
    # 插入节点
    values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 65]
    for val in values:
        bst.insert(val)
    
    print(f"插入值: {values}")
    print(f"中序遍历（有序）: {bst.inorder_traversal()}")
    print(f"前序遍历: {bst.preorder_traversal()}")
    print(f"后序遍历: {bst.postorder_traversal()}")
    print(f"层序遍历: {bst.level_order_traversal()}")
    
    # 搜索操作
    print(f"\n搜索40: {bst.search(40)}")
    print(f"搜索45: {bst.search(45)}")
    
    # 查找最值
    print(f"\n最小值: {bst.find_min()}")
    print(f"最大值: {bst.find_max()}")
    
    # 树的属性
    print(f"\n树的高度: {bst.height()}")
    print(f"是否平衡: {bst.is_balanced()}")
    print(f"是否为有效BST: {bst.is_valid_bst()}")
    print(f"节点总数: {bst.count_nodes()}")
    
    # 前驱后继
    print(f"\n40的前驱: {bst.get_predecessor(40)}")
    print(f"40的后继: {bst.get_successor(40)}")
    
    # 范围查询
    print(f"\n范围查询[25, 65]: {bst.range_query(25, 65)}")
    
    # 第k小元素
    print(f"\n第5小的元素: {bst.kth_smallest(5)}")
    
    # 最近公共祖先
    print(f"\n20和40的LCA: {bst.find_lca(20, 40)}")
    
    # 删除操作
    print(f"\n删除30")
    bst.delete(30)
    print(f"删除后中序遍历: {bst.inorder_traversal()}")
    
    # 序列化和反序列化
    serialized = bst.serialize()
    print(f"\n序列化: {serialized[:50]}...")  # 只显示前50个字符
    
    new_bst = BinarySearchTree()
    new_bst.deserialize(serialized)
    print(f"反序列化后中序遍历: {new_bst.inorder_traversal()}")