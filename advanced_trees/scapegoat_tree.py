import math


class SGNode:
    """替罪羊树节点"""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class ScapegoatTree:
    """
    替罪羊树实现
    特点：
    1. 不使用额外的平衡信息（如高度、颜色等）
    2. 通过部分重建来保持平衡
    3. 平衡因子α（通常为0.5-1之间）
    4. 当子树不平衡时，重建整个子树
    """
    
    def __init__(self, alpha=0.75):
        """
        初始化替罪羊树
        alpha: 平衡因子，通常设为0.75
        """
        self.root = None
        self.size = 0
        self.alpha = alpha
        self.max_size = 0  # 用于决定何时重建整棵树
    
    def _get_size(self, node):
        """获取子树大小"""
        if not node:
            return 0
        return 1 + self._get_size(node.left) + self._get_size(node.right)
    
    def _is_balanced(self, node):
        """
        检查节点是否平衡
        平衡条件：子树大小不超过父树大小的α倍
        """
        if not node:
            return True
        
        node_size = self._get_size(node)
        left_size = self._get_size(node.left)
        right_size = self._get_size(node.right)
        
        return (left_size <= self.alpha * node_size and 
                right_size <= self.alpha * node_size)
    
    def _flatten(self, node, nodes):
        """将树展平为有序列表"""
        if not node:
            return
        
        self._flatten(node.left, nodes)
        nodes.append(node)
        self._flatten(node.right, nodes)
    
    def _build_balanced(self, nodes, start, end):
        """从有序列表构建平衡树"""
        if start > end:
            return None
        
        mid = (start + end) // 2
        node = nodes[mid]
        
        node.left = self._build_balanced(nodes, start, mid - 1)
        node.right = self._build_balanced(nodes, mid + 1, end)
        
        return node
    
    def _rebuild(self, node):
        """重建子树使其平衡"""
        nodes = []
        self._flatten(node, nodes)
        return self._build_balanced(nodes, 0, len(nodes) - 1)
    
    def insert(self, val):
        """
        插入节点
        时间复杂度：均摊O(log n)
        """
        def _insert(node, val, depth):
            if not node:
                self.size += 1
                self.max_size = max(self.max_size, self.size)
                return SGNode(val), None
            
            if val < node.val:
                node.left, scapegoat = _insert(node.left, val, depth + 1)
                if scapegoat:
                    return node, scapegoat
            else:
                node.right, scapegoat = _insert(node.right, val, depth + 1)
                if scapegoat:
                    return node, scapegoat
            
            # 检查是否需要重建
            if not self._is_balanced(node):
                return node, node
            
            # 检查深度是否过深
            if depth > math.log(self.size) / math.log(1/self.alpha):
                return node, node
            
            return node, None
        
        if not self.root:
            self.root = SGNode(val)
            self.size = 1
            self.max_size = 1
            return
        
        self.root, scapegoat = _insert(self.root, val, 0)
        
        if scapegoat:
            # 找到替罪羊并重建
            if scapegoat == self.root:
                self.root = self._rebuild(self.root)
            else:
                self._find_and_rebuild_scapegoat(val)
    
    def _find_and_rebuild_scapegoat(self, val):
        """找到替罪羊节点并重建"""
        parent = None
        node = self.root
        path = []
        
        while node:
            path.append((parent, node))
            if not self._is_balanced(node):
                # 找到替罪羊，重建
                if parent:
                    if parent.left == node:
                        parent.left = self._rebuild(node)
                    else:
                        parent.right = self._rebuild(node)
                else:
                    self.root = self._rebuild(node)
                return
            
            parent = node
            if val < node.val:
                node = node.left
            else:
                node = node.right
    
    def delete(self, val):
        """
        删除节点（惰性删除）
        时间复杂度：O(log n)
        """
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
                    # 找到后继节点
                    successor = node.right
                    while successor.left:
                        successor = successor.left
                    node.val = successor.val
                    node.right = _delete(node.right, successor.val)
            
            return node
        
        self.root = _delete(self.root, val)
        self.size -= 1
        
        # 如果树变得太小，考虑重建
        if self.size < self.alpha * self.max_size:
            self.root = self._rebuild(self.root)
            self.max_size = self.size
    
    def search(self, val):
        """
        搜索节点
        时间复杂度：O(log n)
        """
        node = self.root
        while node:
            if val == node.val:
                return True
            elif val < node.val:
                node = node.left
            else:
                node = node.right
        return False
    
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
    
    def get_height(self):
        """获取树的高度"""
        def _height(node):
            if not node:
                return 0
            return 1 + max(_height(node.left), _height(node.right))
        
        return _height(self.root)
    
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
    
    def get_statistics(self):
        """获取树的统计信息"""
        def _calculate_depths(node, depth=0):
            if not node:
                return []
            
            depths = [depth]
            depths.extend(_calculate_depths(node.left, depth + 1))
            depths.extend(_calculate_depths(node.right, depth + 1))
            return depths
        
        depths = _calculate_depths(self.root)
        
        if not depths:
            return {
                'size': 0,
                'height': 0,
                'avg_depth': 0,
                'max_depth': 0,
                'balance_factor': self.alpha
            }
        
        return {
            'size': self.size,
            'height': self.get_height(),
            'avg_depth': sum(depths) / len(depths),
            'max_depth': max(depths),
            'balance_factor': self.alpha,
            'theoretical_height': math.ceil(math.log2(self.size + 1)) if self.size > 0 else 0
        }
    
    def print_tree(self):
        """打印树的结构"""
        def _print_tree(node, prefix="", is_tail=True):
            if node:
                size = self._get_size(node)
                balanced = "✓" if self._is_balanced(node) else "✗"
                print(prefix + ("└── " if is_tail else "├── ") + 
                      f"{node.val} (size={size}, balanced={balanced})")
                
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


def test_scapegoat_tree():
    """测试替罪羊树"""
    print("=== 替罪羊树测试 ===\n")
    
    # 测试不同的平衡因子
    for alpha in [0.6, 0.75, 0.9]:
        print(f"--- 平衡因子 α = {alpha} ---")
        sg_tree = ScapegoatTree(alpha=alpha)
        
        # 插入测试
        values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
        for val in values:
            sg_tree.insert(val)
        
        stats = sg_tree.get_statistics()
        print(f"插入 {len(values)} 个节点后:")
        print(f"  树高度: {stats['height']}")
        print(f"  平均深度: {stats['avg_depth']:.2f}")
        print(f"  理论最优高度: {stats['theoretical_height']}")
        print(f"  中序遍历: {sg_tree.inorder_traversal()}")
        print()
    
    # 详细测试
    print("\n=== 详细测试 (α = 0.75) ===")
    sg_tree = ScapegoatTree(alpha=0.75)
    
    # 顺序插入测试（最坏情况）
    print("\n顺序插入测试:")
    for i in range(1, 16):
        sg_tree.insert(i)
    
    print(f"顺序插入1-15后:")
    print(f"中序遍历: {sg_tree.inorder_traversal()}")
    print(f"层序遍历: {sg_tree.level_order_traversal()}")
    
    stats = sg_tree.get_statistics()
    print(f"树高度: {stats['height']} (理论最优: {stats['theoretical_height']})")
    
    print("\n树结构:")
    sg_tree.print_tree()
    
    # 搜索测试
    print("\n搜索测试:")
    search_vals = [7, 16, 10]
    for val in search_vals:
        print(f"搜索 {val}: {sg_tree.search(val)}")
    
    # 删除测试
    print("\n删除测试:")
    delete_vals = [1, 8, 15]
    for val in delete_vals:
        sg_tree.delete(val)
        print(f"删除 {val} 后: {sg_tree.inorder_traversal()}")
    
    # 随机插入测试
    print("\n=== 随机插入性能测试 ===")
    import random
    
    sg_tree2 = ScapegoatTree(alpha=0.75)
    test_size = 100
    values = list(range(test_size))
    random.shuffle(values)
    
    for val in values:
        sg_tree2.insert(val)
    
    stats = sg_tree2.get_statistics()
    print(f"随机插入 {test_size} 个值后:")
    print(f"  树高度: {stats['height']}")
    print(f"  理论最优高度: {stats['theoretical_height']}")
    print(f"  平均深度: {stats['avg_depth']:.2f}")
    print(f"  最大深度: {stats['max_depth']}")
    
    # 比较不同数据结构
    print("\n=== 数据结构比较 ===")
    print("特性比较:")
    print("┌─────────────┬──────────┬─────────┬──────────┬───────────┐")
    print("│ 数据结构     │ 最坏插入  │ 平均插入 │ 额外空间  │ 实现复杂度 │")
    print("├─────────────┼──────────┼─────────┼──────────┼───────────┤")
    print("│ BST         │ O(n)     │ O(log n)│ O(1)     │ 简单      │")
    print("│ AVL树       │ O(log n) │ O(log n)│ O(n)     │ 中等      │")
    print("│ 红黑树      │ O(log n) │ O(log n)│ O(n)     │ 复杂      │")
    print("│ 替罪羊树    │ O(log n) │ O(log n)│ O(1)     │ 简单      │")
    print("└─────────────┴──────────┴─────────┴──────────┴───────────┘")


if __name__ == '__main__':
    test_scapegoat_tree()