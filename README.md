# 算法大全 - Algorithm Collection

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![Algorithms](https://img.shields.io/badge/Algorithms-200%2B-green)](https://github.com/bi32/algorithm)
[![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-17K%2B-orange)](https://github.com/bi32/algorithm)

一个全面的算法实现库，包含200+经典算法的Python实现，涵盖《算法导论》核心内容、LeetCode高频题和ACM竞赛常用算法。

## 📚 目录结构

```
algorithm/
├── sort/                    # 排序算法
│   ├── bubble_sort.py      # 冒泡排序
│   ├── quick_sort.py        # 快速排序
│   ├── merge_sort.py        # 归并排序
│   ├── heap_sort.py         # 堆排序
│   ├── counting_sort.py     # 计数排序
│   ├── radix_sort.py        # 基数排序
│   └── advanced_sort.py     # 高级排序（希尔、TimSort等）
├── search/                  # 搜索算法
│   ├── binary_search.py    # 二分查找
│   ├── bst.py              # 二叉搜索树
│   └── sequential_search.py # 顺序查找
├── graph/                   # 图算法
│   ├── graph_traversal.py  # BFS/DFS
│   ├── dijkstra.py         # Dijkstra最短路径
│   ├── mst.py              # 最小生成树
│   ├── network_flow.py     # 网络流
│   ├── shortest_path.py    # 最短路径算法集
│   └── advanced_graph.py   # 高级图算法
├── dynamic_programming/     # 动态规划
│   ├── knapsack.py         # 背包问题
│   ├── lcs.py              # 最长公共子序列
│   └── advanced_dp.py      # 高级DP（树形、状态压缩、数位）
├── data_structures/         # 数据结构
│   ├── stack.py            # 栈
│   ├── queue.py            # 队列
│   └── advanced_structures.py # 高级结构（跳表、LRU等）
├── advanced_trees/          # 高级树结构
│   ├── avl_tree.py         # AVL树
│   ├── red_black_tree.py   # 红黑树
│   ├── b_tree.py           # B树
│   ├── trie.py             # 字典树
│   ├── segment_tree.py     # 线段树
│   └── scapegoat_tree.py   # 替罪羊树
├── string_algorithms/       # 字符串算法
│   ├── kmp.py              # KMP算法
│   └── advanced_string.py  # 高级字符串算法
├── computational_geometry/  # 计算几何
│   └── geometry.py         # 凸包、最近点对等
├── number_theory/          # 数论算法
│   └── number_theory.py    # 素数、GCD、快速幂等
├── cryptography/           # 密码学算法
│   └── crypto_basics.py   # RSA、SHA、AES等
├── machine_learning/       # 机器学习基础
│   └── ml_basics.py       # K-means、KNN、决策树等
├── parallel/              # 并行算法
│   └── parallel_algorithms.py # 并行排序、MapReduce等
├── competitive/           # 竞赛算法
│   └── competitive_algorithms.py # 单调栈、双指针等
└── misc/                  # 其他经典算法
    └── classic_algorithms.py # 并查集、LRU缓存等
```

## 🚀 特性

- **全面覆盖**：200+ 算法实现，涵盖算法导论主要内容
- **详细注释**：每个算法都有中文注释和复杂度分析
- **即插即用**：所有代码都可以直接运行，包含测试用例
- **面试友好**：包含LeetCode和面试高频算法
- **竞赛实用**：ACM/ICPC常用算法模板

## 📊 算法分类统计

### 基础算法
| 类别 | 数量 | 主要算法 |
|------|------|----------|
| 排序 | 11+ | 快排、归并、堆排、TimSort等 |
| 搜索 | 5+ | 二分、插值、斐波那契搜索等 |
| 递归/迭代 | 3+ | 阶乘、斐波那契、汉诺塔等 |

### 数据结构
| 类别 | 数量 | 主要结构 |
|------|------|----------|
| 基础结构 | 5+ | 栈、队列、链表、堆等 |
| 树结构 | 10+ | AVL、红黑树、B树、Trie等 |
| 高级结构 | 8+ | 跳表、布隆过滤器、LRU缓存等 |

### 图算法
| 类别 | 数量 | 主要算法 |
|------|------|----------|
| 遍历 | 2 | BFS、DFS |
| 最短路径 | 5+ | Dijkstra、Floyd、Bellman-Ford等 |
| 生成树 | 2 | Kruskal、Prim |
| 网络流 | 4+ | Ford-Fulkerson、Dinic等 |
| 其他 | 8+ | 强连通分量、欧拉路径、拓扑排序等 |

### 动态规划
| 类别 | 数量 | 主要问题 |
|------|------|----------|
| 经典DP | 10+ | 背包、LCS、编辑距离等 |
| 高级DP | 8+ | 树形DP、状态压缩、数位DP等 |

### 专题算法
| 类别 | 数量 | 主要算法 |
|------|------|----------|
| 字符串 | 10+ | KMP、Boyer-Moore、AC自动机等 |
| 计算几何 | 8+ | 凸包、最近点对、线段相交等 |
| 数论 | 15+ | 素数筛、快速幂、中国剩余定理等 |
| 密码学 | 6+ | RSA、SHA-256、AES等 |

## 💻 使用方法

### 环境要求
- Python 3.6+
- 无需外部依赖（纯Python实现）

### 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/bi32/algorithm.git
cd algorithm
```

2. 运行示例：
```bash
# 运行排序算法
python sort/quick_sort.py

# 运行图算法
python graph/dijkstra.py

# 运行动态规划
python dynamic_programming/knapsack.py
```

### 代码示例

```python
# 使用快速排序
from sort.quick_sort import quick_sort

arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = quick_sort(arr)
print(sorted_arr)  # [11, 12, 22, 25, 34, 64, 90]

# 使用二分查找
from search.binary_search import binary_search

arr = [1, 3, 5, 7, 9, 11, 13]
index = binary_search(arr, 7)
print(index)  # 3

# 使用Dijkstra算法
from graph.dijkstra import dijkstra

graph = {
    0: [(1, 4), (2, 2)],
    1: [(2, 1), (3, 5)],
    2: [(3, 8), (4, 10)],
    3: [(4, 2)],
    4: []
}
distances = dijkstra(graph, 0)
print(distances)  # 最短路径距离
```

## 📈 复杂度速查

### 排序算法
| 算法 | 平均时间 | 最坏时间 | 空间 | 稳定性 |
|------|----------|----------|------|--------|
| 快速排序 | O(n log n) | O(n²) | O(log n) | 不稳定 |
| 归并排序 | O(n log n) | O(n log n) | O(n) | 稳定 |
| 堆排序 | O(n log n) | O(n log n) | O(1) | 不稳定 |
| TimSort | O(n log n) | O(n log n) | O(n) | 稳定 |

### 图算法
| 算法 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| BFS/DFS | O(V + E) | O(V) |
| Dijkstra | O((V + E) log V) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| Bellman-Ford | O(VE) | O(V) |

### 数据结构操作
| 结构 | 插入 | 删除 | 查找 | 空间 |
|------|------|------|------|------|
| AVL树 | O(log n) | O(log n) | O(log n) | O(n) |
| 红黑树 | O(log n) | O(log n) | O(log n) | O(n) |
| B树 | O(log n) | O(log n) | O(log n) | O(n) |
| 跳表 | O(log n) | O(log n) | O(log n) | O(n) |
| 哈希表 | O(1) | O(1) | O(1) | O(n) |

## 🎯 学习路线

### 初学者
1. 基础排序（冒泡、选择、插入）
2. 基础搜索（线性、二分）
3. 基础数据结构（栈、队列）
4. 简单递归（阶乘、斐波那契）

### 进阶
1. 高级排序（快排、归并、堆排）
2. 树结构（BST、AVL、红黑树）
3. 图基础（BFS、DFS、最短路径）
4. 基础DP（背包、LCS）

### 高级
1. 高级数据结构（跳表、B树、Trie）
2. 网络流算法
3. 高级DP（树形、状态压缩）
4. 字符串匹配算法

### 竞赛/面试
1. 单调栈/队列
2. 双指针技巧
3. 位运算技巧
4. 并查集
5. 线段树/树状数组

## 📖 参考资料

- 《算法导论》(Introduction to Algorithms)
- 《算法》(Algorithms, 4th Edition)
- LeetCode
- GeeksforGeeks
- CP-Algorithms

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🌟 Star History

如果这个项目对你有帮助，请给一个⭐️支持！

---

**Happy Coding! 🚀**
