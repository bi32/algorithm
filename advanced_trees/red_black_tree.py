class RBNode:
    """红黑树节点"""
    def __init__(self, val, color='RED'):
        self.val = val
        self.color = color  # RED or BLACK
        self.left = None
        self.right = None
        self.parent = None


class RedBlackTree:
    """
    红黑树实现
    性质：
    1. 每个节点要么是红色，要么是黑色
    2. 根节点是黑色
    3. 所有叶子节点（NIL）是黑色
    4. 如果一个节点是红色，则它的两个子节点都是黑色
    5. 从任一节点到其每个叶子的所有简单路径都包含相同数目的黑色节点
    """
    
    def __init__(self):
        self.NIL = RBNode(None, 'BLACK')  # 哨兵节点
        self.root = self.NIL
    
    def rotate_left(self, x):
        """
        左旋转
          x                  y
         / \                / \
        a   y      =>      x   c
           / \            / \
          b   c          a   b
        """
        y = x.right
        x.right = y.left
        
        if y.left != self.NIL:
            y.left.parent = x
        
        y.parent = x.parent
        
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x
        x.parent = y
    
    def rotate_right(self, y):
        """
        右旋转
            y                x
           / \              / \
          x   c    =>      a   y
         / \                  / \
        a   b                b   c
        """
        x = y.left
        y.left = x.right
        
        if x.right != self.NIL:
            x.right.parent = y
        
        x.parent = y.parent
        
        if y.parent == None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        
        x.right = y
        y.parent = x
    
    def insert(self, val):
        """插入节点"""
        # 创建新节点
        new_node = RBNode(val)
        new_node.left = self.NIL
        new_node.right = self.NIL
        
        # BST插入
        parent = None
        current = self.root
        
        while current != self.NIL:
            parent = current
            if new_node.val < current.val:
                current = current.left
            else:
                current = current.right
        
        new_node.parent = parent
        
        if parent == None:
            self.root = new_node
        elif new_node.val < parent.val:
            parent.left = new_node
        else:
            parent.right = new_node
        
        # 如果新节点是根节点，设为黑色并返回
        if new_node.parent == None:
            new_node.color = 'BLACK'
            return
        
        # 如果父节点是黑色，不需要修复
        if new_node.parent.parent == None:
            return
        
        # 修复红黑树性质
        self._fix_insert(new_node)
    
    def _fix_insert(self, node):
        """修复插入后的红黑树性质"""
        while node.parent and node.parent.color == 'RED':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                
                # 情况1：叔叔节点是红色
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    # 情况2：叔叔是黑色，且当前节点是右子节点
                    if node == node.parent.right:
                        node = node.parent
                        self.rotate_left(node)
                    
                    # 情况3：叔叔是黑色，且当前节点是左子节点
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self.rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                
                # 情况1：叔叔节点是红色
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    # 情况2：叔叔是黑色，且当前节点是左子节点
                    if node == node.parent.left:
                        node = node.parent
                        self.rotate_right(node)
                    
                    # 情况3：叔叔是黑色，且当前节点是右子节点
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self.rotate_left(node.parent.parent)
            
            if node == self.root:
                break
        
        self.root.color = 'BLACK'
    
    def _transplant(self, u, v):
        """用v替换u"""
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent
    
    def _minimum(self, node):
        """找到以node为根的子树的最小节点"""
        while node.left != self.NIL:
            node = node.left
        return node
    
    def delete(self, val):
        """删除节点"""
        # 找到要删除的节点
        node = self._search_node(val)
        if node == self.NIL:
            return
        
        self._delete_node(node)
    
    def _delete_node(self, node):
        """删除指定节点"""
        y = node
        y_original_color = y.color
        
        if node.left == self.NIL:
            x = node.right
            self._transplant(node, node.right)
        elif node.right == self.NIL:
            x = node.left
            self._transplant(node, node.left)
        else:
            y = self._minimum(node.right)
            y_original_color = y.color
            x = y.right
            
            if y.parent == node:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = node.right
                y.right.parent = y
            
            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y.color = node.color
        
        if y_original_color == 'BLACK':
            self._fix_delete(x)
    
    def _fix_delete(self, node):
        """修复删除后的红黑树性质"""
        while node != self.root and node.color == 'BLACK':
            if node == node.parent.left:
                sibling = node.parent.right
                
                # 情况1：兄弟节点是红色
                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    node.parent.color = 'RED'
                    self.rotate_left(node.parent)
                    sibling = node.parent.right
                
                # 情况2：兄弟节点是黑色，且兄弟的两个子节点都是黑色
                if sibling.left.color == 'BLACK' and sibling.right.color == 'BLACK':
                    sibling.color = 'RED'
                    node = node.parent
                else:
                    # 情况3：兄弟节点是黑色，兄弟的右子节点是黑色
                    if sibling.right.color == 'BLACK':
                        sibling.left.color = 'BLACK'
                        sibling.color = 'RED'
                        self.rotate_right(sibling)
                        sibling = node.parent.right
                    
                    # 情况4：兄弟节点是黑色，兄弟的右子节点是红色
                    sibling.color = node.parent.color
                    node.parent.color = 'BLACK'
                    sibling.right.color = 'BLACK'
                    self.rotate_left(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left
                
                # 情况1：兄弟节点是红色
                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    node.parent.color = 'RED'
                    self.rotate_right(node.parent)
                    sibling = node.parent.left
                
                # 情况2：兄弟节点是黑色，且兄弟的两个子节点都是黑色
                if sibling.right.color == 'BLACK' and sibling.left.color == 'BLACK':
                    sibling.color = 'RED'
                    node = node.parent
                else:
                    # 情况3：兄弟节点是黑色，兄弟的左子节点是黑色
                    if sibling.left.color == 'BLACK':
                        sibling.right.color = 'BLACK'
                        sibling.color = 'RED'
                        self.rotate_left(sibling)
                        sibling = node.parent.left
                    
                    # 情况4：兄弟节点是黑色，兄弟的左子节点是红色
                    sibling.color = node.parent.color
                    node.parent.color = 'BLACK'
                    sibling.left.color = 'BLACK'
                    self.rotate_right(node.parent)
                    node = self.root
        
        node.color = 'BLACK'
    
    def search(self, val):
        """搜索值"""
        return self._search_node(val) != self.NIL
    
    def _search_node(self, val):
        """搜索节点"""
        current = self.root
        while current != self.NIL and val != current.val:
            if val < current.val:
                current = current.left
            else:
                current = current.right
        return current
    
    def find_min(self):
        """找到最小值"""
        if self.root == self.NIL:
            return None
        
        current = self.root
        while current.left != self.NIL:
            current = current.left
        return current.val
    
    def find_max(self):
        """找到最大值"""
        if self.root == self.NIL:
            return None
        
        current = self.root
        while current.right != self.NIL:
            current = current.right
        return current.val
    
    def inorder_traversal(self):
        """中序遍历"""
        result = []
        
        def _inorder(node):
            if node != self.NIL:
                _inorder(node.left)
                result.append(node.val)
                _inorder(node.right)
        
        _inorder(self.root)
        return result
    
    def get_height(self):
        """获取树的高度"""
        def _height(node):
            if node == self.NIL:
                return 0
            return 1 + max(_height(node.left), _height(node.right))
        
        return _height(self.root)
    
    def get_black_height(self):
        """获取黑高（从根到叶子的黑色节点数）"""
        def _black_height(node):
            if node == self.NIL:
                return 1
            
            left_height = _black_height(node.left)
            right_height = _black_height(node.right)
            
            if left_height != right_height:
                return -1  # 违反红黑树性质
            
            if node.color == 'BLACK':
                return left_height + 1
            else:
                return left_height
        
        return _black_height(self.root)
    
    def verify_properties(self):
        """验证红黑树的所有性质"""
        def _verify(node, black_count, path_black_count):
            if node == self.NIL:
                if path_black_count == -1:
                    path_black_count = black_count
                return path_black_count == black_count, path_black_count
            
            # 性质4：红色节点的子节点必须是黑色
            if node.color == 'RED':
                if (node.left != self.NIL and node.left.color == 'RED') or \
                   (node.right != self.NIL and node.right.color == 'RED'):
                    return False, -1
            
            if node.color == 'BLACK':
                black_count += 1
            
            left_valid, path_count = _verify(node.left, black_count, path_black_count)
            if not left_valid:
                return False, -1
            
            right_valid, path_count = _verify(node.right, black_count, path_count)
            if not right_valid:
                return False, -1
            
            return True, path_count
        
        # 性质2：根节点必须是黑色
        if self.root != self.NIL and self.root.color != 'BLACK':
            return False, "根节点不是黑色"
        
        # 验证其他性质
        valid, _ = _verify(self.root, 0, -1)
        if not valid:
            return False, "违反红黑树性质"
        
        return True, "满足所有红黑树性质"
    
    def print_tree(self):
        """打印树的结构"""
        def _print_tree(node, prefix="", is_tail=True):
            if node != self.NIL:
                color_symbol = "●" if node.color == 'BLACK' else "○"
                print(prefix + ("└── " if is_tail else "├── ") + 
                      f"{color_symbol} {node.val}")
                
                children = []
                if node.left != self.NIL:
                    children.append((node.left, False))
                if node.right != self.NIL:
                    children.append((node.right, True))
                
                for i, (child, is_last) in enumerate(children):
                    if i == len(children) - 1:
                        _print_tree(child, prefix + ("    " if is_tail else "│   "), True)
                    else:
                        _print_tree(child, prefix + ("    " if is_tail else "│   "), False)
        
        if self.root != self.NIL:
            _print_tree(self.root)
        else:
            print("空树")


def test_red_black_tree():
    """测试红黑树"""
    print("=== 红黑树测试 ===")
    
    rb_tree = RedBlackTree()
    
    # 插入测试
    print("\n插入测试:")
    values = [7, 3, 18, 10, 22, 8, 11, 26, 2, 6, 13]
    for val in values:
        rb_tree.insert(val)
        valid, msg = rb_tree.verify_properties()
        print(f"插入 {val:2d} - 验证: {msg}")
    
    print(f"\n中序遍历: {rb_tree.inorder_traversal()}")
    print(f"树高度: {rb_tree.get_height()}")
    print(f"黑高: {rb_tree.get_black_height()}")
    
    print("\n树结构（● = 黑色, ○ = 红色）:")
    rb_tree.print_tree()
    
    # 搜索测试
    print("\n搜索测试:")
    search_values = [10, 15, 22]
    for val in search_values:
        print(f"搜索 {val}: {rb_tree.search(val)}")
    
    # 删除测试
    print("\n删除测试:")
    delete_values = [18, 11, 3]
    for val in delete_values:
        rb_tree.delete(val)
        valid, msg = rb_tree.verify_properties()
        print(f"删除 {val:2d} - 验证: {msg}")
        print(f"中序遍历: {rb_tree.inorder_traversal()}")
    
    print("\n删除后的树结构:")
    rb_tree.print_tree()
    
    # 性能测试
    print("\n=== 性能对比测试 ===")
    import random
    
    rb_tree2 = RedBlackTree()
    test_size = 1000
    test_values = list(range(test_size))
    random.shuffle(test_values)
    
    for val in test_values:
        rb_tree2.insert(val)
    
    print(f"插入 {test_size} 个随机值:")
    print(f"树高度: {rb_tree2.get_height()}")
    print(f"黑高: {rb_tree2.get_black_height()}")
    print(f"理论最大高度: {2 * import_math_log2(test_size + 1):.0f}")
    
    valid, msg = rb_tree2.verify_properties()
    print(f"验证结果: {msg}")


def import_math_log2(n):
    """辅助函数：计算log2"""
    import math
    return math.log2(n)


if __name__ == '__main__':
    test_red_black_tree()