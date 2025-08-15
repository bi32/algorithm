class SegmentTree:
    """
    线段树实现
    支持区间查询和单点/区间更新
    """
    
    def __init__(self, arr):
        """
        初始化线段树
        时间复杂度：O(n)
        """
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 懒标记数组
        self.arr = arr.copy()
        
        if self.n > 0:
            self._build(0, 0, self.n - 1)
    
    def _build(self, node, start, end):
        """构建线段树"""
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def update_point(self, idx, val):
        """
        单点更新
        时间复杂度：O(log n)
        """
        self._update_point(0, 0, self.n - 1, idx, val)
    
    def _update_point(self, node, start, end, idx, val):
        """递归更新单点"""
        if start == end:
            self.tree[node] = val
            self.arr[idx] = val
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self._update_point(left_child, start, mid, idx, val)
            else:
                self._update_point(right_child, mid + 1, end, idx, val)
            
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def update_range(self, l, r, val):
        """
        区间更新（增加val）
        时间复杂度：O(log n)
        """
        self._update_range(0, 0, self.n - 1, l, r, val)
    
    def _push_down(self, node, start, end):
        """下推懒标记"""
        if self.lazy[node] != 0:
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            mid = (start + end) // 2
            
            # 更新子节点的值
            self.tree[left_child] += self.lazy[node] * (mid - start + 1)
            self.tree[right_child] += self.lazy[node] * (end - mid)
            
            # 传递懒标记
            self.lazy[left_child] += self.lazy[node]
            self.lazy[right_child] += self.lazy[node]
            
            # 清除当前节点的懒标记
            self.lazy[node] = 0
    
    def _update_range(self, node, start, end, l, r, val):
        """递归区间更新"""
        if l > end or r < start:
            return
        
        if l <= start and end <= r:
            # 完全覆盖
            self.tree[node] += val * (end - start + 1)
            if start != end:
                self.lazy[node] += val
            return
        
        # 下推懒标记
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._update_range(left_child, start, mid, l, r, val)
        self._update_range(right_child, mid + 1, end, l, r, val)
        
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query(self, l, r):
        """
        区间查询
        时间复杂度：O(log n)
        """
        return self._query(0, 0, self.n - 1, l, r)
    
    def _query(self, node, start, end, l, r):
        """递归查询"""
        if l > end or r < start:
            return 0
        
        if l <= start and end <= r:
            return self.tree[node]
        
        # 下推懒标记
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self._query(left_child, start, mid, l, r)
        right_sum = self._query(right_child, mid + 1, end, l, r)
        
        return left_sum + right_sum


class MaxSegmentTree:
    """
    最大值线段树
    支持区间最大值查询
    """
    
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [float('-inf')] * (4 * self.n)
        self.arr = arr.copy()
        
        if self.n > 0:
            self._build(0, 0, self.n - 1)
    
    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            
            self.tree[node] = max(self.tree[left_child], self.tree[right_child])
    
    def update(self, idx, val):
        """单点更新"""
        self._update(0, 0, self.n - 1, idx, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            self.arr[idx] = val
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self._update(left_child, start, mid, idx, val)
            else:
                self._update(right_child, mid + 1, end, idx, val)
            
            self.tree[node] = max(self.tree[left_child], self.tree[right_child])
    
    def query(self, l, r):
        """区间最大值查询"""
        return self._query(0, 0, self.n - 1, l, r)
    
    def _query(self, node, start, end, l, r):
        if l > end or r < start:
            return float('-inf')
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_max = self._query(left_child, start, mid, l, r)
        right_max = self._query(right_child, mid + 1, end, l, r)
        
        return max(left_max, right_max)


class BinaryIndexedTree:
    """
    树状数组（Fenwick Tree）
    支持单点更新和前缀和查询
    """
    
    def __init__(self, n):
        """
        初始化树状数组
        时间复杂度：O(n)
        """
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, idx, delta):
        """
        单点更新（增加delta）
        时间复杂度：O(log n)
        """
        idx += 1  # 树状数组从1开始索引
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)  # 获取下一个需要更新的位置
    
    def query(self, idx):
        """
        前缀和查询 [0, idx]
        时间复杂度：O(log n)
        """
        idx += 1
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)  # 获取下一个需要累加的位置
        return result
    
    def range_query(self, left, right):
        """
        区间和查询 [left, right]
        时间复杂度：O(log n)
        """
        if left == 0:
            return self.query(right)
        return self.query(right) - self.query(left - 1)
    
    def build(self, arr):
        """从数组构建树状数组"""
        for i, val in enumerate(arr):
            self.update(i, val)


class BIT2D:
    """
    二维树状数组
    支持矩形区域更新和查询
    """
    
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, row, col, delta):
        """
        单点更新
        时间复杂度：O(log m × log n)
        """
        row += 1
        original_col = col + 1
        
        while row <= self.rows:
            col = original_col
            while col <= self.cols:
                self.tree[row][col] += delta
                col += col & (-col)
            row += row & (-row)
    
    def query(self, row, col):
        """
        查询(0,0)到(row,col)的矩形和
        """
        row += 1
        col += 1
        result = 0
        original_col = col
        
        while row > 0:
            col = original_col
            while col > 0:
                result += self.tree[row][col]
                col -= col & (-col)
            row -= row & (-row)
        
        return result
    
    def range_query(self, r1, c1, r2, c2):
        """
        查询矩形(r1,c1)到(r2,c2)的和
        """
        result = self.query(r2, c2)
        if r1 > 0:
            result -= self.query(r1 - 1, c2)
        if c1 > 0:
            result -= self.query(r2, c1 - 1)
        if r1 > 0 and c1 > 0:
            result += self.query(r1 - 1, c1 - 1)
        return result


class PersistentSegmentTree:
    """
    可持久化线段树
    支持查询历史版本
    """
    
    class Node:
        def __init__(self, val=0):
            self.val = val
            self.left = None
            self.right = None
    
    def __init__(self, arr):
        self.n = len(arr)
        self.roots = []  # 存储各个版本的根节点
        
        if self.n > 0:
            self.roots.append(self._build(arr, 0, self.n - 1))
    
    def _build(self, arr, start, end):
        """构建初始版本"""
        node = self.Node()
        
        if start == end:
            node.val = arr[start]
        else:
            mid = (start + end) // 2
            node.left = self._build(arr, start, mid)
            node.right = self._build(arr, mid + 1, end)
            node.val = node.left.val + node.right.val
        
        return node
    
    def update(self, version, idx, val):
        """
        创建新版本并更新
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1
    
    def _update(self, node, start, end, idx, val):
        """创建新路径"""
        new_node = self.Node()
        
        if start == end:
            new_node.val = val
        else:
            mid = (start + end) // 2
            
            if idx <= mid:
                new_node.left = self._update(node.left, start, mid, idx, val)
                new_node.right = node.right
            else:
                new_node.left = node.left
                new_node.right = self._update(node.right, mid + 1, end, idx, val)
            
            new_node.val = new_node.left.val + new_node.right.val
        
        return new_node
    
    def query(self, version, l, r):
        """查询特定版本的区间和"""
        return self._query(self.roots[version], 0, self.n - 1, l, r)
    
    def _query(self, node, start, end, l, r):
        if l > end or r < start:
            return 0
        
        if l <= start and end <= r:
            return node.val
        
        mid = (start + end) // 2
        left_sum = self._query(node.left, start, mid, l, r)
        right_sum = self._query(node.right, mid + 1, end, l, r)
        
        return left_sum + right_sum


def test_segment_trees():
    """测试线段树和树状数组"""
    print("=== 线段树和树状数组测试 ===\n")
    
    # 测试基本线段树
    print("1. 线段树（区间和）:")
    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr)
    
    print(f"   原始数组: {arr}")
    print(f"   查询[1,3]: {st.query(1, 3)}")
    
    st.update_point(2, 10)
    print(f"   更新arr[2]=10后，查询[1,3]: {st.query(1, 3)}")
    
    st.update_range(0, 2, 2)
    print(f"   区间[0,2]增加2后，查询[0,2]: {st.query(0, 2)}")
    
    # 测试最大值线段树
    print("\n2. 最大值线段树:")
    max_st = MaxSegmentTree(arr)
    print(f"   查询[1,4]最大值: {max_st.query(1, 4)}")
    
    max_st.update(3, 15)
    print(f"   更新arr[3]=15后，查询[1,4]最大值: {max_st.query(1, 4)}")
    
    # 测试树状数组
    print("\n3. 树状数组:")
    bit = BinaryIndexedTree(6)
    bit.build(arr)
    
    print(f"   前缀和[0,3]: {bit.query(3)}")
    print(f"   区间和[2,4]: {bit.range_query(2, 4)}")
    
    bit.update(2, 5)
    print(f"   arr[2]增加5后，区间和[2,4]: {bit.range_query(2, 4)}")
    
    # 测试二维树状数组
    print("\n4. 二维树状数组:")
    bit2d = BIT2D(3, 3)
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    for i in range(3):
        for j in range(3):
            bit2d.update(i, j, matrix[i][j])
    
    print(f"   矩阵:")
    for row in matrix:
        print(f"   {row}")
    
    print(f"   矩形(0,0)到(1,1)的和: {bit2d.range_query(0, 0, 1, 1)}")
    print(f"   矩形(1,1)到(2,2)的和: {bit2d.range_query(1, 1, 2, 2)}")
    
    # 测试可持久化线段树
    print("\n5. 可持久化线段树:")
    pst = PersistentSegmentTree([1, 2, 3, 4, 5])
    
    print(f"   版本0，查询[0,2]: {pst.query(0, 0, 2)}")
    
    v1 = pst.update(0, 2, 10)
    print(f"   版本1（arr[2]=10），查询[0,2]: {pst.query(v1, 0, 2)}")
    print(f"   版本0仍然不变，查询[0,2]: {pst.query(0, 0, 2)}")
    
    # 复杂度分析
    print("\n=== 复杂度分析 ===")
    print("┌────────────────┬────────────┬────────────┬──────────┐")
    print("│ 数据结构        │ 单点更新    │ 区间查询    │ 空间     │")
    print("├────────────────┼────────────┼────────────┼──────────┤")
    print("│ 线段树          │ O(log n)   │ O(log n)   │ O(n)     │")
    print("│ 树状数组        │ O(log n)   │ O(log n)   │ O(n)     │")
    print("│ 2D树状数组      │ O(log²n)   │ O(log²n)   │ O(n²)    │")
    print("│ 可持久化线段树   │ O(log n)   │ O(log n)   │ O(nlogn) │")
    print("└────────────────┴────────────┴────────────┴──────────┘")
    
    # 应用场景
    print("\n=== 应用场景 ===")
    print("• 线段树：区间查询和更新，RMQ问题")
    print("• 树状数组：前缀和查询，逆序对计数")
    print("• 可持久化：历史版本查询，主席树")


if __name__ == '__main__':
    test_segment_trees()