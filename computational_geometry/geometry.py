import math
from functools import cmp_to_key


class Point:
    """二维点"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y
    
    def distance_to(self, other):
        """计算到另一点的欧几里得距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def manhattan_distance(self, other):
        """曼哈顿距离"""
        return abs(self.x - other.x) + abs(self.y - other.y)


def cross_product(o, a, b):
    """
    计算向量OA和OB的叉积
    返回值 > 0: 逆时针
    返回值 = 0: 共线
    返回值 < 0: 顺时针
    """
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def orientation(p, q, r):
    """
    判断三点的方向
    返回值 0: 共线
    返回值 1: 顺时针
    返回值 2: 逆时针
    """
    val = cross_product(p, q, r)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    """判断点q是否在线段pr上"""
    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
        q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return True
    return False


def segments_intersect(p1, q1, p2, q2):
    """
    判断两条线段是否相交
    时间复杂度：O(1)
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # 一般情况
    if o1 != o2 and o3 != o4:
        return True
    
    # 特殊情况：共线且重叠
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False


def convex_hull_graham_scan(points):
    """
    Graham扫描算法求凸包
    时间复杂度：O(n log n)
    """
    n = len(points)
    if n < 3:
        return points
    
    # 找到最下方的点（y最小，如果相同则x最小）
    start = min(points, key=lambda p: (p.y, p.x))
    
    # 按极角排序
    def polar_angle_cmp(a, b):
        cross = cross_product(start, a, b)
        if cross == 0:
            # 共线，按距离排序
            dist_a = start.distance_to(a)
            dist_b = start.distance_to(b)
            return -1 if dist_a < dist_b else 1
        return -1 if cross > 0 else 1
    
    # 排序（除了起始点）
    other_points = [p for p in points if p != start]
    other_points.sort(key=cmp_to_key(polar_angle_cmp))
    
    # Graham扫描
    hull = [start]
    
    for p in other_points:
        # 删除非左转的点
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    return hull


def convex_hull_jarvis_march(points):
    """
    Jarvis步进算法（礼物包装算法）求凸包
    时间复杂度：O(nh)，h为凸包上的点数
    """
    n = len(points)
    if n < 3:
        return points
    
    # 找到最左边的点
    leftmost = min(points, key=lambda p: p.x)
    
    hull = []
    p = leftmost
    
    while True:
        hull.append(p)
        
        # 找到最逆时针的点
        q = points[0]
        for i in range(n):
            if points[i] == p:
                continue
            
            o = orientation(p, q, points[i])
            if q == p or o == 2 or (o == 0 and p.distance_to(points[i]) > p.distance_to(q)):
                q = points[i]
        
        p = q
        
        # 回到起点
        if p == leftmost:
            break
    
    return hull


def convex_hull_andrew(points):
    """
    Andrew算法求凸包（单调链算法）
    时间复杂度：O(n log n)
    """
    points = sorted(points)
    
    if len(points) <= 1:
        return points
    
    # 构建下凸包
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # 构建上凸包
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # 去除重复的端点
    return lower[:-1] + upper[:-1]


def closest_pair_brute_force(points):
    """
    暴力法求最近点对
    时间复杂度：O(n²)
    """
    n = len(points)
    if n < 2:
        return None, float('inf')
    
    min_dist = float('inf')
    closest = None
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = points[i].distance_to(points[j])
            if dist < min_dist:
                min_dist = dist
                closest = (points[i], points[j])
    
    return closest, min_dist


def closest_pair_divide_conquer(points):
    """
    分治法求最近点对
    时间复杂度：O(n log n)
    """
    def closest_pair_recursive(px, py):
        n = len(px)
        
        # 小规模用暴力法
        if n <= 3:
            return closest_pair_brute_force(px)
        
        # 分割
        mid = n // 2
        midpoint = px[mid]
        
        # 按y坐标分割点集
        pyl = [p for p in py if p.x <= midpoint.x]
        pyr = [p for p in py if p.x > midpoint.x]
        
        # 递归求解
        dl_pair, dl = closest_pair_recursive(px[:mid], pyl)
        dr_pair, dr = closest_pair_recursive(px[mid:], pyr)
        
        # 取左右两边的最小值
        if dl < dr:
            d = dl
            min_pair = dl_pair
        else:
            d = dr
            min_pair = dr_pair
        
        # 检查跨越中线的点对
        strip = [p for p in py if abs(p.x - midpoint.x) < d]
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j].y - strip[i].y) < d:
                dist = strip[i].distance_to(strip[j])
                if dist < d:
                    d = dist
                    min_pair = (strip[i], strip[j])
                j += 1
        
        return min_pair, d
    
    # 预排序
    px = sorted(points, key=lambda p: p.x)
    py = sorted(points, key=lambda p: p.y)
    
    return closest_pair_recursive(px, py)


def polygon_area(vertices):
    """
    计算多边形面积（鞋带公式）
    时间复杂度：O(n)
    """
    n = len(vertices)
    if n < 3:
        return 0
    
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].x * vertices[j].y
        area -= vertices[j].x * vertices[i].y
    
    return abs(area) / 2


def polygon_perimeter(vertices):
    """计算多边形周长"""
    n = len(vertices)
    if n < 2:
        return 0
    
    perimeter = 0
    for i in range(n):
        j = (i + 1) % n
        perimeter += vertices[i].distance_to(vertices[j])
    
    return perimeter


def point_in_polygon(point, vertices):
    """
    判断点是否在多边形内（射线法）
    时间复杂度：O(n)
    """
    n = len(vertices)
    if n < 3:
        return False
    
    # 射线从点向右水平延伸
    count = 0
    p1 = vertices[0]
    
    for i in range(1, n + 1):
        p2 = vertices[i % n]
        
        if point.y > min(p1.y, p2.y):
            if point.y <= max(p1.y, p2.y):
                if point.x <= max(p1.x, p2.x):
                    if p1.y != p2.y:
                        xinters = (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if p1.x == p2.x or point.x <= xinters:
                        count += 1
        
        p1 = p2
    
    return count % 2 == 1


def line_intersection(p1, q1, p2, q2):
    """
    计算两条直线的交点
    返回交点坐标，如果平行返回None
    """
    denom = (p1.x - q1.x) * (p2.y - q2.y) - (p1.y - q1.y) * (p2.x - q2.x)
    
    if abs(denom) < 1e-10:
        return None  # 平行或重合
    
    t = ((p1.x - p2.x) * (p2.y - q2.y) - (p1.y - p2.y) * (p2.x - q2.x)) / denom
    
    x = p1.x + t * (q1.x - p1.x)
    y = p1.y + t * (q1.y - p1.y)
    
    return Point(x, y)


def rotating_calipers(hull):
    """
    旋转卡壳算法
    求凸包的直径（最远点对）
    时间复杂度：O(n)
    """
    n = len(hull)
    if n < 2:
        return None, 0
    
    if n == 2:
        return (hull[0], hull[1]), hull[0].distance_to(hull[1])
    
    # 找到最远的对踵点对
    max_dist = 0
    farthest_pair = None
    
    j = 1
    for i in range(n):
        # 找到距离hull[i]最远的点
        while True:
            dist = hull[i].distance_to(hull[j])
            next_j = (j + 1) % n
            next_dist = hull[i].distance_to(hull[next_j])
            
            if next_dist > dist:
                j = next_j
            else:
                break
        
        dist = hull[i].distance_to(hull[j])
        if dist > max_dist:
            max_dist = dist
            farthest_pair = (hull[i], hull[j])
    
    return farthest_pair, max_dist


def minimum_enclosing_circle(points):
    """
    最小包围圆（Welzl算法）
    时间复杂度：期望O(n)
    """
    import random
    
    def circle_from_points(p1, p2, p3=None):
        """从2或3个点构造圆"""
        if p3 is None:
            # 两点的情况
            center_x = (p1.x + p2.x) / 2
            center_y = (p1.y + p2.y) / 2
            radius = p1.distance_to(p2) / 2
            return Point(center_x, center_y), radius
        
        # 三点的情况
        ax = p2.x - p1.x
        ay = p2.y - p1.y
        bx = p3.x - p1.x
        by = p3.y - p1.y
        
        c = 2 * (ax * by - ay * bx)
        if abs(c) < 1e-10:
            return None, 0
        
        d = ax * ax + ay * ay
        e = bx * bx + by * by
        
        center_x = p1.x + (by * d - ay * e) / c
        center_y = p1.y + (ax * e - bx * d) / c
        
        center = Point(center_x, center_y)
        radius = center.distance_to(p1)
        
        return center, radius
    
    def welzl(points, boundary):
        if not points or len(boundary) == 3:
            if not boundary:
                return Point(0, 0), 0
            elif len(boundary) == 1:
                return boundary[0], 0
            elif len(boundary) == 2:
                return circle_from_points(boundary[0], boundary[1])
            else:
                return circle_from_points(boundary[0], boundary[1], boundary[2])
        
        # 随机选择一个点
        p = points[random.randint(0, len(points) - 1)]
        points_copy = [pt for pt in points if pt != p]
        
        center, radius = welzl(points_copy, boundary)
        
        if center.distance_to(p) <= radius + 1e-10:
            return center, radius
        
        boundary_copy = boundary + [p]
        return welzl(points_copy, boundary_copy)
    
    points_copy = points.copy()
    random.shuffle(points_copy)
    return welzl(points_copy, [])


def test_computational_geometry():
    """测试计算几何算法"""
    print("=== 计算几何算法测试 ===\n")
    
    # 测试点集
    points = [
        Point(0, 3), Point(1, 1), Point(2, 2), Point(4, 4),
        Point(0, 0), Point(1, 2), Point(3, 1), Point(3, 3)
    ]
    
    print("测试点集:", points)
    
    # 凸包测试
    print("\n1. 凸包算法:")
    hull1 = convex_hull_graham_scan(points.copy())
    print(f"   Graham扫描: {hull1}")
    
    hull2 = convex_hull_jarvis_march(points.copy())
    print(f"   Jarvis步进: {hull2}")
    
    hull3 = convex_hull_andrew(points.copy())
    print(f"   Andrew算法: {hull3}")
    
    # 最近点对
    print("\n2. 最近点对:")
    pair1, dist1 = closest_pair_brute_force(points)
    print(f"   暴力法: {pair1}, 距离={dist1:.2f}")
    
    pair2, dist2 = closest_pair_divide_conquer(points)
    print(f"   分治法: {pair2}, 距离={dist2:.2f}")
    
    # 线段相交
    print("\n3. 线段相交测试:")
    seg1 = (Point(0, 0), Point(2, 2))
    seg2 = (Point(0, 2), Point(2, 0))
    seg3 = (Point(3, 0), Point(3, 2))
    
    print(f"   线段1: {seg1}")
    print(f"   线段2: {seg2}")
    print(f"   线段3: {seg3}")
    print(f"   线段1与线段2相交: {segments_intersect(*seg1, *seg2)}")
    print(f"   线段1与线段3相交: {segments_intersect(*seg1, *seg3)}")
    
    # 多边形测试
    print("\n4. 多边形操作:")
    polygon = [Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3)]
    print(f"   多边形顶点: {polygon}")
    print(f"   面积: {polygon_area(polygon)}")
    print(f"   周长: {polygon_perimeter(polygon):.2f}")
    
    test_point = Point(2, 2)
    print(f"   点{test_point}在多边形内: {point_in_polygon(test_point, polygon)}")
    
    test_point2 = Point(5, 2)
    print(f"   点{test_point2}在多边形内: {point_in_polygon(test_point2, polygon)}")
    
    # 旋转卡壳
    print("\n5. 旋转卡壳（凸包直径）:")
    pair, diameter = rotating_calipers(hull1)
    if pair:
        print(f"   最远点对: {pair}")
        print(f"   直径: {diameter:.2f}")
    
    # 最小包围圆
    print("\n6. 最小包围圆:")
    center, radius = minimum_enclosing_circle(points)
    print(f"   圆心: {center}")
    print(f"   半径: {radius:.2f}")
    
    # 复杂度分析
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │")
    print("├──────────────────┼──────────────┤")
    print("│ Graham扫描       │ O(n log n)   │")
    print("│ Jarvis步进       │ O(nh)        │")
    print("│ 最近点对(分治)    │ O(n log n)   │")
    print("│ 点在多边形内      │ O(n)         │")
    print("│ 旋转卡壳         │ O(n)         │")
    print("│ 最小包围圆       │ 期望O(n)     │")
    print("└──────────────────┴──────────────┘")


if __name__ == '__main__':
    test_computational_geometry()