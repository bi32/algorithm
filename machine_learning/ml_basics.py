import math
import random
from collections import Counter, defaultdict
import numpy as np


class KMeans:
    """
    K-Means聚类算法
    时间复杂度：O(n * k * i * d)
    n: 样本数, k: 簇数, i: 迭代次数, d: 特征维度
    """
    
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
    
    def euclidean_distance(self, a, b):
        """计算欧几里得距离"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def initialize_centroids(self, X):
        """初始化质心（K-Means++）"""
        n_samples = len(X)
        centroids = []
        
        # 随机选择第一个质心
        centroids.append(X[random.randint(0, n_samples - 1)])
        
        # 选择剩余的质心
        for _ in range(1, self.k):
            distances = []
            for point in X:
                # 计算到最近质心的距离
                min_dist = min(self.euclidean_distance(point, c) for c in centroids)
                distances.append(min_dist ** 2)
            
            # 根据距离概率选择下一个质心
            total = sum(distances)
            if total > 0:
                probs = [d / total for d in distances]
                cumsum = 0
                r = random.random()
                for i, p in enumerate(probs):
                    cumsum += p
                    if r < cumsum:
                        centroids.append(X[i])
                        break
            else:
                centroids.append(X[random.randint(0, n_samples - 1)])
        
        return centroids
    
    def assign_clusters(self, X, centroids):
        """分配样本到最近的簇"""
        labels = []
        for point in X:
            distances = [self.euclidean_distance(point, c) for c in centroids]
            labels.append(distances.index(min(distances)))
        return labels
    
    def update_centroids(self, X, labels):
        """更新质心"""
        n_features = len(X[0]) if X else 0
        new_centroids = []
        
        for i in range(self.k):
            cluster_points = [X[j] for j in range(len(X)) if labels[j] == i]
            
            if cluster_points:
                # 计算均值
                centroid = []
                for d in range(n_features):
                    mean = sum(p[d] for p in cluster_points) / len(cluster_points)
                    centroid.append(mean)
                new_centroids.append(centroid)
            else:
                # 如果簇为空，保持原质心
                new_centroids.append(self.centroids[i])
        
        return new_centroids
    
    def fit(self, X):
        """训练模型"""
        # 初始化质心
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # 分配簇
            labels = self.assign_clusters(X, self.centroids)
            
            # 更新质心
            new_centroids = self.update_centroids(X, labels)
            
            # 检查收敛
            converged = True
            for old, new in zip(self.centroids, new_centroids):
                if self.euclidean_distance(old, new) > self.tolerance:
                    converged = False
                    break
            
            self.centroids = new_centroids
            self.labels = labels
            
            if converged:
                break
        
        return self
    
    def predict(self, X):
        """预测新样本的簇"""
        return self.assign_clusters(X, self.centroids)
    
    def inertia(self, X):
        """计算簇内平方和"""
        total = 0
        for i, point in enumerate(X):
            centroid = self.centroids[self.labels[i]]
            total += self.euclidean_distance(point, centroid) ** 2
        return total


class KNN:
    """
    K最近邻算法
    时间复杂度：O(n * d) for each prediction
    n: 训练样本数, d: 特征维度
    """
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, a, b):
        """计算欧几里得距离"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def manhattan_distance(self, a, b):
        """计算曼哈顿距离"""
        return sum(abs(x - y) for x, y in zip(a, b))
    
    def fit(self, X, y):
        """训练（实际上只是存储数据）"""
        self.X_train = X
        self.y_train = y
        return self
    
    def predict_single(self, x):
        """预测单个样本"""
        # 计算到所有训练样本的距离
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self.euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # 排序并选择k个最近邻
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # 投票（分类）或平均（回归）
        k_nearest_labels = [label for _, label in k_nearest]
        
        # 分类：多数投票
        if isinstance(k_nearest_labels[0], (str, int)):
            counter = Counter(k_nearest_labels)
            return counter.most_common(1)[0][0]
        # 回归：平均值
        else:
            return sum(k_nearest_labels) / len(k_nearest_labels)
    
    def predict(self, X):
        """预测多个样本"""
        return [self.predict_single(x) for x in X]
    
    def weighted_predict_single(self, x):
        """加权预测（距离倒数作为权重）"""
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self.euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # 加权投票
        weights = {}
        for dist, label in k_nearest:
            weight = 1 / (dist + 1e-10)  # 避免除零
            if label in weights:
                weights[label] += weight
            else:
                weights[label] = weight
        
        return max(weights.items(), key=lambda x: x[1])[0]


class DecisionTreeNode:
    """决策树节点"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 分割特征索引
        self.threshold = threshold  # 分割阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的值


class DecisionTree:
    """
    决策树（CART算法）
    时间复杂度：O(n * m * log n)
    n: 样本数, m: 特征数
    """
    
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def gini_impurity(self, y):
        """计算基尼不纯度"""
        if not y:
            return 0
        
        counter = Counter(y)
        n = len(y)
        gini = 1.0
        
        for count in counter.values():
            prob = count / n
            gini -= prob ** 2
        
        return gini
    
    def entropy(self, y):
        """计算信息熵"""
        if not y:
            return 0
        
        counter = Counter(y)
        n = len(y)
        entropy = 0
        
        for count in counter.values():
            if count > 0:
                prob = count / n
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def information_gain(self, y, left_y, right_y):
        """计算信息增益"""
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        if n == 0:
            return 0
        
        parent_entropy = self.entropy(y)
        weighted_entropy = (n_left / n) * self.entropy(left_y) + \
                          (n_right / n) * self.entropy(right_y)
        
        return parent_entropy - weighted_entropy
    
    def split_data(self, X, y, feature, threshold):
        """根据特征和阈值分割数据"""
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y
    
    def find_best_split(self, X, y):
        """找到最佳分割点"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0]) if X else 0
        
        for feature in range(n_features):
            # 获取该特征的所有唯一值
            values = sorted(set(x[feature] for x in X))
            
            # 尝试每个可能的分割点
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                # 分割数据
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature, threshold)
                
                # 检查最小样本数约束
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # 计算信息增益
                gain = self.information_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples = len(y)
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(set(y)) == 1):
            
            # 返回叶节点
            counter = Counter(y)
            value = counter.most_common(1)[0][0] if counter else None
            return DecisionTreeNode(value=value)
        
        # 找到最佳分割
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            # 无法分割，返回叶节点
            counter = Counter(y)
            value = counter.most_common(1)[0][0] if counter else None
            return DecisionTreeNode(value=value)
        
        # 分割数据
        left_X, left_y, right_X, right_y = self.split_data(X, y, feature, threshold)
        
        # 递归构建子树
        left_child = self.build_tree(left_X, left_y, depth + 1)
        right_child = self.build_tree(right_X, right_y, depth + 1)
        
        return DecisionTreeNode(feature=feature, threshold=threshold,
                               left=left_child, right=right_child)
    
    def fit(self, X, y):
        """训练决策树"""
        self.root = self.build_tree(X, y)
        return self
    
    def predict_single(self, x, node=None):
        """预测单个样本"""
        if node is None:
            node = self.root
        
        # 叶节点
        if node.value is not None:
            return node.value
        
        # 内部节点
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)
    
    def predict(self, X):
        """预测多个样本"""
        return [self.predict_single(x) for x in X]
    
    def print_tree(self, node=None, depth=0):
        """打印决策树结构"""
        if node is None:
            node = self.root
        
        if node.value is not None:
            print("  " * depth + f"Predict: {node.value}")
        else:
            print("  " * depth + f"Feature {node.feature} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            print("  " * depth + f"Feature {node.feature} > {node.threshold}")
            self.print_tree(node.right, depth + 1)


class Perceptron:
    """
    感知机
    时间复杂度：O(n * m * iterations)
    n: 样本数, m: 特征数
    """
    
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
    
    def sign(self, x):
        """符号函数"""
        return 1 if x >= 0 else -1
    
    def fit(self, X, y):
        """训练感知机"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # 初始化权重和偏置
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # 确保标签是+1/-1
        y = [1 if label > 0 else -1 for label in y]
        
        for _ in range(self.max_iters):
            errors = 0
            
            for i in range(n_samples):
                # 计算预测值
                linear_output = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                prediction = self.sign(linear_output)
                
                # 更新权重（如果预测错误）
                if y[i] * prediction <= 0:
                    for j in range(n_features):
                        self.weights[j] += self.learning_rate * y[i] * X[i][j]
                    self.bias += self.learning_rate * y[i]
                    errors += 1
            
            # 如果没有错误，提前停止
            if errors == 0:
                break
        
        return self
    
    def predict_single(self, x):
        """预测单个样本"""
        linear_output = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sign(linear_output)
    
    def predict(self, X):
        """预测多个样本"""
        return [self.predict_single(x) for x in X]
    
    def decision_function(self, X):
        """计算决策函数值"""
        scores = []
        for x in X:
            score = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            scores.append(score)
        return scores


class NaiveBayes:
    """
    朴素贝叶斯分类器
    """
    
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
    
    def fit(self, X, y):
        """训练朴素贝叶斯"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # 获取所有类别
        self.classes = list(set(y))
        
        # 计算先验概率
        for c in self.classes:
            class_samples = [X[i] for i in range(n_samples) if y[i] == c]
            self.class_priors[c] = len(class_samples) / n_samples
            
            # 计算每个特征的条件概率
            self.feature_probs[c] = []
            
            for feature_idx in range(n_features):
                feature_values = [sample[feature_idx] for sample in class_samples]
                
                # 假设特征是连续的，使用高斯分布
                mean = sum(feature_values) / len(feature_values) if feature_values else 0
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values) if feature_values else 1e-10
                std = math.sqrt(variance)
                
                self.feature_probs[c].append({
                    'mean': mean,
                    'std': std + 1e-10  # 避免除零
                })
        
        return self
    
    def gaussian_probability(self, x, mean, std):
        """计算高斯概率密度"""
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(exponent)
    
    def predict_single(self, x):
        """预测单个样本"""
        posteriors = {}
        
        for c in self.classes:
            # 先验概率
            posterior = math.log(self.class_priors[c])
            
            # 似然概率
            for i, value in enumerate(x):
                mean = self.feature_probs[c][i]['mean']
                std = self.feature_probs[c][i]['std']
                likelihood = self.gaussian_probability(value, mean, std)
                posterior += math.log(likelihood + 1e-10)
            
            posteriors[c] = posterior
        
        # 返回概率最大的类
        return max(posteriors.items(), key=lambda x: x[1])[0]
    
    def predict(self, X):
        """预测多个样本"""
        return [self.predict_single(x) for x in X]


def test_ml_algorithms():
    """测试机器学习算法"""
    print("=== 机器学习基础算法测试 ===\n")
    
    # 生成测试数据
    random.seed(42)
    
    # 聚类数据
    cluster_data = []
    cluster_labels = []
    for i in range(3):
        center = [random.uniform(i*3, i*3+2) for _ in range(2)]
        for _ in range(20):
            point = [c + random.gauss(0, 0.5) for c in center]
            cluster_data.append(point)
            cluster_labels.append(i)
    
    # 分类数据
    classification_data = []
    classification_labels = []
    for _ in range(50):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        label = 1 if x1 + x2 > 0 else 0
        classification_data.append([x1, x2])
        classification_labels.append(label)
    
    # 测试K-Means
    print("1. K-Means聚类:")
    kmeans = KMeans(k=3)
    kmeans.fit(cluster_data)
    print(f"   聚类中心: {[[round(x, 2) for x in c] for c in kmeans.centroids]}")
    print(f"   簇内平方和: {kmeans.inertia(cluster_data):.2f}")
    
    # 测试KNN
    print("\n2. K最近邻:")
    knn = KNN(k=3)
    train_size = 40
    knn.fit(classification_data[:train_size], classification_labels[:train_size])
    predictions = knn.predict(classification_data[train_size:train_size+5])
    actual = classification_labels[train_size:train_size+5]
    print(f"   预测结果: {predictions}")
    print(f"   实际标签: {actual}")
    accuracy = sum(p == a for p, a in zip(predictions, actual)) / len(actual)
    print(f"   准确率: {accuracy:.2%}")
    
    # 测试决策树
    print("\n3. 决策树:")
    dt = DecisionTree(max_depth=3)
    dt.fit(classification_data[:train_size], classification_labels[:train_size])
    dt_predictions = dt.predict(classification_data[train_size:train_size+5])
    print(f"   预测结果: {dt_predictions}")
    print(f"   实际标签: {actual}")
    dt_accuracy = sum(p == a for p, a in zip(dt_predictions, actual)) / len(actual)
    print(f"   准确率: {dt_accuracy:.2%}")
    
    # 测试感知机
    print("\n4. 感知机:")
    perceptron = Perceptron(learning_rate=0.1)
    # 转换标签为+1/-1
    perc_labels = [1 if l == 1 else -1 for l in classification_labels]
    perceptron.fit(classification_data[:train_size], perc_labels[:train_size])
    perc_predictions = perceptron.predict(classification_data[train_size:train_size+5])
    perc_actual = perc_labels[train_size:train_size+5]
    print(f"   权重: {[round(w, 2) for w in perceptron.weights]}")
    print(f"   偏置: {round(perceptron.bias, 2)}")
    print(f"   预测结果: {perc_predictions}")
    print(f"   实际标签: {perc_actual}")
    
    # 测试朴素贝叶斯
    print("\n5. 朴素贝叶斯:")
    nb = NaiveBayes()
    nb.fit(classification_data[:train_size], classification_labels[:train_size])
    nb_predictions = nb.predict(classification_data[train_size:train_size+5])
    print(f"   预测结果: {nb_predictions}")
    print(f"   实际标签: {actual}")
    nb_accuracy = sum(p == a for p, a in zip(nb_predictions, actual)) / len(actual)
    print(f"   准确率: {nb_accuracy:.2%}")
    
    # 算法特点
    print("\n=== 算法特点 ===")
    print("┌──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ 算法          │ 时间复杂度    │ 空间复杂度    │ 适用场景      │")
    print("├──────────────┼──────────────┼──────────────┼──────────────┤")
    print("│ K-Means      │ O(nki)       │ O(nk)        │ 聚类分析      │")
    print("│ KNN          │ O(nd)        │ O(n)         │ 小数据分类    │")
    print("│ 决策树        │ O(nm log n)  │ O(depth)     │ 可解释分类    │")
    print("│ 感知机        │ O(nm)        │ O(m)         │ 线性分类      │")
    print("│ 朴素贝叶斯    │ O(nm)        │ O(cm)        │ 文本分类      │")
    print("└──────────────┴──────────────┴──────────────┴──────────────┘")
    print("n:样本数 m:特征数 k:簇数/邻居数 i:迭代次数 c:类别数")


if __name__ == '__main__':
    test_ml_algorithms()