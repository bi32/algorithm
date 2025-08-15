class BTreeNode:
    """B树节点"""
    def __init__(self, t, leaf=False):
        """
        t: 最小度数（节点最少有t-1个键，最多有2t-1个键）
        leaf: 是否为叶子节点
        """
        self.keys = []
        self.children = []
        self.leaf = leaf
        self.t = t
    
    def split_child(self, i, y):
        """分裂第i个子节点y"""
        t = self.t
        z = BTreeNode(t, y.leaf)
        
        # 将y的后半部分键移到z
        z.keys = y.keys[t:]
        y.keys = y.keys[:t-1]
        
        # 如果不是叶子节点，也要分裂子节点
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        
        # 将y的中间键提升到父节点
        self.keys.insert(i, y.keys[t-1])
        self.children.insert(i + 1, z)
    
    def insert_non_full(self, k):
        """在未满的节点中插入键k"""
        i = len(self.keys) - 1
        
        if self.leaf:
            # 叶子节点，直接插入
            self.keys.append(None)
            while i >= 0 and self.keys[i] > k:
                self.keys[i + 1] = self.keys[i]
                i -= 1
            self.keys[i + 1] = k
        else:
            # 内部节点，找到合适的子节点
            while i >= 0 and self.keys[i] > k:
                i -= 1
            i += 1
            
            # 如果子节点已满，先分裂
            if len(self.children[i].keys) == 2 * self.t - 1:
                self.split_child(i, self.children[i])
                if self.keys[i] < k:
                    i += 1
            
            self.children[i].insert_non_full(k)
    
    def merge(self, idx):
        """合并第idx个子节点和它的兄弟"""
        child = self.children[idx]
        sibling = self.children[idx + 1]
        
        # 将键从当前节点下移，并与右兄弟合并
        child.keys.append(self.keys[idx])
        child.keys.extend(sibling.keys)
        
        # 复制子节点指针
        if not child.leaf:
            child.children.extend(sibling.children)
        
        # 移除键和子节点
        self.keys.pop(idx)
        self.children.pop(idx + 1)
    
    def borrow_from_prev(self, idx):
        """从前一个兄弟借一个键"""
        child = self.children[idx]
        sibling = self.children[idx - 1]
        
        # 将一个键从父节点移到子节点
        child.keys.insert(0, self.keys[idx - 1])
        
        # 将兄弟的最后一个键移到父节点
        self.keys[idx - 1] = sibling.keys.pop()
        
        # 移动子节点指针
        if not child.leaf:
            child.children.insert(0, sibling.children.pop())
    
    def borrow_from_next(self, idx):
        """从后一个兄弟借一个键"""
        child = self.children[idx]
        sibling = self.children[idx + 1]
        
        # 将一个键从父节点移到子节点
        child.keys.append(self.keys[idx])
        
        # 将兄弟的第一个键移到父节点
        self.keys[idx] = sibling.keys.pop(0)
        
        # 移动子节点指针
        if not child.leaf:
            child.children.append(sibling.children.pop(0))
    
    def fill(self, idx):
        """填充子节点，使其至少有t个键"""
        t = self.t
        
        # 如果前一个兄弟有至少t个键，借一个
        if idx != 0 and len(self.children[idx - 1].keys) >= t:
            self.borrow_from_prev(idx)
        # 如果后一个兄弟有至少t个键，借一个
        elif idx != len(self.children) - 1 and len(self.children[idx + 1].keys) >= t:
            self.borrow_from_next(idx)
        # 如果都不够，合并
        else:
            if idx != len(self.children) - 1:
                self.merge(idx)
            else:
                self.merge(idx - 1)


class BTree:
    """
    B树实现
    特点：
    1. 多路平衡搜索树
    2. 所有叶子节点在同一层
    3. 节点可以有多个键和子节点
    4. 适合磁盘存储和数据库索引
    """
    
    def __init__(self, t=3):
        """
        初始化B树
        t: 最小度数（通常t>=2）
        """
        self.root = BTreeNode(t, True)
        self.t = t
    
    def search(self, k, node=None):
        """
        搜索键k
        时间复杂度：O(log n)
        """
        if node is None:
            node = self.root
        
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and k == node.keys[i]:
            return True
        
        if node.leaf:
            return False
        
        return self.search(k, node.children[i])
    
    def insert(self, k):
        """
        插入键k
        时间复杂度：O(log n)
        """
        root = self.root
        
        if len(root.keys) == 2 * self.t - 1:
            # 根节点已满，需要分裂
            new_root = BTreeNode(self.t, False)
            new_root.children.append(self.root)
            new_root.split_child(0, self.root)
            self.root = new_root
            new_root.insert_non_full(k)
        else:
            root.insert_non_full(k)
    
    def delete(self, k):
        """
        删除键k
        时间复杂度：O(log n)
        """
        self._delete(self.root, k)
        
        # 如果根节点为空，更新根
        if len(self.root.keys) == 0:
            if not self.root.leaf and self.root.children:
                self.root = self.root.children[0]
    
    def _delete(self, node, k):
        """从节点中删除键k"""
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and k == node.keys[i]:
            if node.leaf:
                node.keys.pop(i)
            else:
                self._delete_internal_node(node, k, i)
        elif not node.leaf:
            is_in_subtree = (i == len(node.keys))
            
            if len(node.children[i].keys) < self.t:
                node.fill(i)
            
            if is_in_subtree and i > len(node.keys):
                self._delete(node.children[i - 1], k)
            else:
                self._delete(node.children[i], k)
    
    def _delete_internal_node(self, node, k, i):
        """从内部节点删除键k"""
        t = self.t
        
        if node.leaf:
            node.keys.pop(i)
        else:
            if len(node.children[i].keys) >= t:
                predecessor = self._get_predecessor(node, i)
                node.keys[i] = predecessor
                self._delete(node.children[i], predecessor)
            elif len(node.children[i + 1].keys) >= t:
                successor = self._get_successor(node, i)
                node.keys[i] = successor
                self._delete(node.children[i + 1], successor)
            else:
                node.merge(i)
                self._delete(node.children[i], k)
    
    def _get_predecessor(self, node, i):
        """获取前驱键"""
        curr = node.children[i]
        while not curr.leaf:
            curr = curr.children[-1]
        return curr.keys[-1]
    
    def _get_successor(self, node, i):
        """获取后继键"""
        curr = node.children[i + 1]
        while not curr.leaf:
            curr = curr.children[0]
        return curr.keys[0]
    
    def find_min(self):
        """找到最小键"""
        node = self.root
        while not node.leaf:
            node = node.children[0]
        return node.keys[0] if node.keys else None
    
    def find_max(self):
        """找到最大键"""
        node = self.root
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1] if node.keys else None
    
    def inorder_traversal(self):
        """中序遍历"""
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node, result):
        """递归中序遍历"""
        i = 0
        for key in node.keys:
            if not node.leaf:
                self._inorder(node.children[i], result)
                i += 1
            result.append(key)
        
        if not node.leaf:
            self._inorder(node.children[i], result)
    
    def level_order_traversal(self):
        """层序遍历"""
        if not self.root:
            return []
        
        result = []
        queue = [(self.root, 0)]
        
        while queue:
            node, level = queue.pop(0)
            
            if level >= len(result):
                result.append([])
            
            result[level].extend(node.keys)
            
            if not node.leaf:
                for child in node.children:
                    queue.append((child, level + 1))
        
        return result
    
    def get_height(self):
        """获取树的高度"""
        def _height(node):
            if node.leaf:
                return 1
            return 1 + _height(node.children[0])
        
        return _height(self.root)
    
    def count_nodes(self):
        """计算节点总数"""
        def _count(node):
            if not node:
                return 0
            
            count = 1
            if not node.leaf:
                for child in node.children:
                    count += _count(child)
            
            return count
        
        return _count(self.root)
    
    def count_keys(self):
        """计算键的总数"""
        def _count_keys(node):
            count = len(node.keys)
            if not node.leaf:
                for child in node.children:
                    count += _count_keys(child)
            return count
        
        return _count_keys(self.root)
    
    def print_tree(self):
        """打印B树结构"""
        def _print_tree(node, prefix="", is_tail=True):
            if node:
                # 打印当前节点的所有键
                keys_str = ", ".join(str(k) for k in node.keys)
                node_type = "Leaf" if node.leaf else "Internal"
                print(prefix + ("└── " if is_tail else "├── ") + 
                      f"[{keys_str}] ({node_type})")
                
                if not node.leaf:
                    # 打印所有子节点
                    for i, child in enumerate(node.children):
                        is_last = (i == len(node.children) - 1)
                        _print_tree(child, 
                                  prefix + ("    " if is_tail else "│   "), 
                                  is_last)
        
        print("B-Tree (t = {}):".format(self.t))
        _print_tree(self.root)
    
    def validate(self):
        """验证B树的性质"""
        def _validate(node, min_key, max_key, depth, expected_depth):
            # 检查键的顺序
            for i in range(len(node.keys) - 1):
                if node.keys[i] >= node.keys[i + 1]:
                    return False, "键顺序错误"
            
            # 检查键的范围
            if node.keys:
                if min_key is not None and node.keys[0] <= min_key:
                    return False, "键超出最小范围"
                if max_key is not None and node.keys[-1] >= max_key:
                    return False, "键超出最大范围"
            
            # 检查节点的键数量
            if node != self.root:
                if len(node.keys) < self.t - 1:
                    return False, f"节点键太少: {len(node.keys)} < {self.t - 1}"
                if len(node.keys) > 2 * self.t - 1:
                    return False, f"节点键太多: {len(node.keys)} > {2 * self.t - 1}"
            
            # 检查叶子节点深度
            if node.leaf:
                if expected_depth == -1:
                    expected_depth = depth
                elif depth != expected_depth:
                    return False, "叶子节点不在同一层"
                return True, expected_depth
            
            # 检查子节点数量
            if len(node.children) != len(node.keys) + 1:
                return False, "子节点数量错误"
            
            # 递归检查子节点
            for i, child in enumerate(node.children):
                child_min = min_key if i == 0 else node.keys[i - 1]
                child_max = max_key if i == len(node.keys) else node.keys[i]
                
                valid, expected_depth = _validate(child, child_min, child_max, 
                                                 depth + 1, expected_depth)
                if not valid:
                    return False, expected_depth
            
            return True, expected_depth
        
        valid, depth_or_error = _validate(self.root, None, None, 0, -1)
        
        if valid:
            return True, "B树性质满足"
        else:
            return False, depth_or_error


def test_btree():
    """测试B树"""
    print("=== B树测试 ===\n")
    
    # 基本操作测试
    print("基本操作测试 (t=3):")
    btree = BTree(t=3)
    
    # 插入测试
    values = [10, 20, 5, 6, 12, 30, 7, 17, 3, 8, 4, 2, 9, 1]
    print(f"插入值: {values}")
    
    for val in values:
        btree.insert(val)
    
    print(f"中序遍历: {btree.inorder_traversal()}")
    print(f"层序遍历: {btree.level_order_traversal()}")
    print(f"树高度: {btree.get_height()}")
    print(f"节点数: {btree.count_nodes()}")
    print(f"键总数: {btree.count_keys()}")
    
    valid, msg = btree.validate()
    print(f"验证: {msg}")
    
    print("\n树结构:")
    btree.print_tree()
    
    # 搜索测试
    print("\n搜索测试:")
    search_vals = [6, 15, 30]
    for val in search_vals:
        print(f"搜索 {val}: {btree.search(val)}")
    
    # 删除测试
    print("\n删除测试:")
    delete_vals = [6, 7, 3]
    for val in delete_vals:
        btree.delete(val)
        print(f"删除 {val} 后: {btree.inorder_traversal()}")
        valid, msg = btree.validate()
        print(f"  验证: {msg}")
    
    # 不同度数的B树测试
    print("\n=== 不同度数的B树测试 ===")
    
    for t in [2, 4, 10]:
        btree = BTree(t=t)
        test_values = list(range(1, 51))
        
        for val in test_values:
            btree.insert(val)
        
        print(f"\nt={t}的B树 (插入1-50):")
        print(f"  树高度: {btree.get_height()}")
        print(f"  节点数: {btree.count_nodes()}")
        print(f"  每个节点最少键数: {t-1}")
        print(f"  每个节点最多键数: {2*t-1}")
    
    # 大量数据测试
    print("\n=== 大量数据测试 ===")
    import random
    
    btree_large = BTree(t=10)
    test_size = 1000
    values = list(range(test_size))
    random.shuffle(values)
    
    for val in values:
        btree_large.insert(val)
    
    print(f"插入 {test_size} 个随机值 (t=10):")
    print(f"  树高度: {btree_large.get_height()}")
    print(f"  节点数: {btree_large.count_nodes()}")
    print(f"  键总数: {btree_large.count_keys()}")
    
    valid, msg = btree_large.validate()
    print(f"  验证: {msg}")
    
    # 性能比较
    print("\n=== B树应用场景 ===")
    print("1. 数据库索引: 减少磁盘I/O次数")
    print("2. 文件系统: 目录结构管理")
    print("3. 大数据存储: 适合外部存储")
    print("\n优点:")
    print("  - 高度较低，减少查找次数")
    print("  - 适合磁盘存储，减少I/O")
    print("  - 保持平衡，操作时间复杂度稳定")
    print("\n缺点:")
    print("  - 实现复杂")
    print("  - 内存利用率可能不如二叉树")


if __name__ == '__main__':
    test_btree()