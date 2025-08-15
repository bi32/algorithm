import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import random
from functools import reduce
import operator


class ParallelSort:
    """并行排序算法"""
    
    @staticmethod
    def parallel_merge_sort(arr, num_threads=4):
        """
        并行归并排序
        时间复杂度：O(n log n / p)，p为处理器数
        """
        if len(arr) <= 1:
            return arr
        
        def merge(left, right):
            """合并两个有序数组"""
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        def merge_sort_chunk(chunk):
            """对块进行排序"""
            if len(chunk) <= 1:
                return chunk
            
            mid = len(chunk) // 2
            left = merge_sort_chunk(chunk[:mid])
            right = merge_sort_chunk(chunk[mid:])
            return merge(left, right)
        
        # 将数组分成chunks
        chunk_size = max(len(arr) // num_threads, 1)
        chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
        
        # 并行排序每个chunk
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            sorted_chunks = list(executor.map(merge_sort_chunk, chunks))
        
        # 合并所有排序后的chunks
        while len(sorted_chunks) > 1:
            merged_chunks = []
            
            # 并行合并相邻的chunks
            for i in range(0, len(sorted_chunks), 2):
                if i + 1 < len(sorted_chunks):
                    merged_chunks.append(merge(sorted_chunks[i], sorted_chunks[i + 1]))
                else:
                    merged_chunks.append(sorted_chunks[i])
            
            sorted_chunks = merged_chunks
        
        return sorted_chunks[0] if sorted_chunks else []
    
    @staticmethod
    def parallel_quick_sort(arr, num_threads=4):
        """
        并行快速排序
        """
        if len(arr) <= 1:
            return arr
        
        def quick_sort_sequential(arr):
            """顺序快速排序"""
            if len(arr) <= 1:
                return arr
            
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            
            return quick_sort_sequential(left) + middle + quick_sort_sequential(right)
        
        def parallel_quick_sort_helper(arr, depth=0):
            """并行快速排序辅助函数"""
            if len(arr) <= 1000 or depth >= 3:  # 阈值控制
                return quick_sort_sequential(arr)
            
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_left = executor.submit(parallel_quick_sort_helper, left, depth + 1)
                future_right = executor.submit(parallel_quick_sort_helper, right, depth + 1)
                
                sorted_left = future_left.result()
                sorted_right = future_right.result()
            
            return sorted_left + middle + sorted_right
        
        return parallel_quick_sort_helper(arr)
    
    @staticmethod
    def parallel_bucket_sort(arr, num_buckets=10, num_threads=4):
        """
        并行桶排序
        适用于均匀分布的数据
        """
        if not arr:
            return []
        
        min_val = min(arr)
        max_val = max(arr)
        
        # 创建桶
        bucket_range = (max_val - min_val) / num_buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # 分配元素到桶
        for num in arr:
            index = min(int((num - min_val) / bucket_range), num_buckets - 1)
            buckets[index].append(num)
        
        # 并行排序每个桶
        def sort_bucket(bucket):
            return sorted(bucket)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            sorted_buckets = list(executor.map(sort_bucket, buckets))
        
        # 合并所有桶
        result = []
        for bucket in sorted_buckets:
            result.extend(bucket)
        
        return result


class MapReduce:
    """MapReduce模拟实现"""
    
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
    
    def map_phase(self, data, map_func):
        """
        Map阶段：并行处理数据
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            mapped_results = list(executor.map(map_func, data))
        
        # 扁平化结果
        flat_results = []
        for result in mapped_results:
            if isinstance(result, list):
                flat_results.extend(result)
            else:
                flat_results.append(result)
        
        return flat_results
    
    def shuffle_phase(self, mapped_data):
        """
        Shuffle阶段：按key分组
        """
        shuffled = {}
        for key, value in mapped_data:
            if key not in shuffled:
                shuffled[key] = []
            shuffled[key].append(value)
        return shuffled
    
    def reduce_phase(self, shuffled_data, reduce_func):
        """
        Reduce阶段：并行聚合
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for key, values in shuffled_data.items():
                future = executor.submit(reduce_func, key, values)
                futures.append(future)
            
            results = []
            for future in futures:
                results.append(future.result())
        
        return results
    
    def execute(self, data, map_func, reduce_func):
        """
        执行完整的MapReduce流程
        """
        # Map阶段
        mapped_data = self.map_phase(data, map_func)
        
        # Shuffle阶段
        shuffled_data = self.shuffle_phase(mapped_data)
        
        # Reduce阶段
        reduced_data = self.reduce_phase(shuffled_data, reduce_func)
        
        return reduced_data


class WordCount:
    """使用MapReduce实现词频统计"""
    
    @staticmethod
    def map_func(document):
        """Map函数：将文档分割成单词"""
        words = document.lower().split()
        return [(word, 1) for word in words]
    
    @staticmethod
    def reduce_func(word, counts):
        """Reduce函数：统计词频"""
        return (word, sum(counts))
    
    @staticmethod
    def count_words(documents, num_workers=4):
        """统计词频"""
        mr = MapReduce(num_workers)
        results = mr.execute(documents, WordCount.map_func, WordCount.reduce_func)
        return dict(results)


class ParallelPrefixSum:
    """并行前缀和算法"""
    
    @staticmethod
    def sequential_prefix_sum(arr):
        """顺序前缀和"""
        result = [0] * len(arr)
        if arr:
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = result[i-1] + arr[i]
        return result
    
    @staticmethod
    def parallel_prefix_sum_work_efficient(arr, num_threads=4):
        """
        工作高效的并行前缀和（Blelloch算法）
        时间复杂度：O(n/p + log n)
        """
        n = len(arr)
        if n <= 1:
            return arr.copy()
        
        result = arr.copy()
        
        # 上扫阶段（Up-sweep）
        d = 0
        while (1 << d) < n:
            step = 1 << (d + 1)
            
            def up_sweep_task(i):
                if i < n:
                    result[i] = result[i - (1 << d)] + result[i]
            
            indices = list(range((1 << d) - 1, n, step))
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                executor.map(up_sweep_task, 
                           [i + (1 << d) for i in indices if i + (1 << d) < n])
            
            d += 1
        
        # 设置最后一个元素为0
        if n > 0:
            last = result[n - 1]
            result[n - 1] = 0
        
        # 下扫阶段（Down-sweep）
        d -= 1
        while d >= 0:
            step = 1 << (d + 1)
            
            def down_sweep_task(i):
                if i < n:
                    temp = result[i - (1 << d)]
                    result[i - (1 << d)] = result[i]
                    result[i] = temp + result[i]
            
            indices = list(range((1 << d) - 1, n, step))
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                executor.map(down_sweep_task,
                           [i + (1 << d) for i in indices if i + (1 << d) < n])
            
            d -= 1
        
        # 修正：包含原始值
        for i in range(n - 1, 0, -1):
            result[i] = result[i - 1] + arr[i]
        if n > 0:
            result[0] = arr[0]
        
        return result
    
    @staticmethod
    def parallel_segmented_scan(arr, segment_size=1000, num_threads=4):
        """
        分段并行扫描
        适用于大规模数据
        """
        n = len(arr)
        if n <= segment_size:
            return ParallelPrefixSum.sequential_prefix_sum(arr)
        
        # 分段
        num_segments = (n + segment_size - 1) // segment_size
        segments = [arr[i*segment_size:(i+1)*segment_size] 
                   for i in range(num_segments)]
        
        # 并行计算每段的前缀和
        def compute_segment_sum(segment):
            return ParallelPrefixSum.sequential_prefix_sum(segment)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            segment_results = list(executor.map(compute_segment_sum, segments))
        
        # 获取每段的总和
        segment_sums = [seg[-1] if seg else 0 for seg in segment_results]
        
        # 计算段间的前缀和
        segment_prefix = ParallelPrefixSum.sequential_prefix_sum(segment_sums)
        
        # 更新每段的值
        def update_segment(args):
            segment_result, offset = args
            return [x + offset for x in segment_result]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            updated_segments = list(executor.map(
                update_segment,
                [(segment_results[i], segment_prefix[i] - segment_sums[i])
                 for i in range(num_segments)]
            ))
        
        # 合并结果
        result = []
        for segment in updated_segments:
            result.extend(segment)
        
        return result[:n]


class ParallelMatrixOperations:
    """并行矩阵运算"""
    
    @staticmethod
    def parallel_matrix_multiply(A, B, num_threads=4):
        """
        并行矩阵乘法
        """
        rows_A = len(A)
        cols_A = len(A[0]) if A else 0
        rows_B = len(B)
        cols_B = len(B[0]) if B else 0
        
        if cols_A != rows_B:
            raise ValueError("矩阵维度不匹配")
        
        C = [[0] * cols_B for _ in range(rows_A)]
        
        def compute_row(i):
            """计算结果矩阵的第i行"""
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i][j] += A[i][k] * B[k][j]
        
        # 并行计算每一行
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(compute_row, range(rows_A))
        
        return C
    
    @staticmethod
    def parallel_matrix_vector_multiply(A, x, num_threads=4):
        """
        并行矩阵向量乘法
        """
        rows = len(A)
        cols = len(A[0]) if A else 0
        
        if cols != len(x):
            raise ValueError("维度不匹配")
        
        result = [0] * rows
        
        def compute_element(i):
            """计算结果向量的第i个元素"""
            result[i] = sum(A[i][j] * x[j] for j in range(cols))
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(compute_element, range(rows))
        
        return result


class ParallelReduction:
    """并行归约"""
    
    @staticmethod
    def parallel_reduce(arr, operation, identity, num_threads=4):
        """
        通用并行归约
        operation: 二元操作（如加法、乘法、最大值等）
        identity: 单位元
        """
        if not arr:
            return identity
        
        n = len(arr)
        if n == 1:
            return arr[0]
        
        # 分块归约
        chunk_size = max(n // num_threads, 1)
        chunks = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]
        
        def reduce_chunk(chunk):
            """归约一个块"""
            return reduce(operation, chunk, identity)
        
        # 并行归约每个块
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunk_results = list(executor.map(reduce_chunk, chunks))
        
        # 归约块结果
        return reduce(operation, chunk_results, identity)
    
    @staticmethod
    def parallel_sum(arr, num_threads=4):
        """并行求和"""
        return ParallelReduction.parallel_reduce(arr, operator.add, 0, num_threads)
    
    @staticmethod
    def parallel_product(arr, num_threads=4):
        """并行求积"""
        return ParallelReduction.parallel_reduce(arr, operator.mul, 1, num_threads)
    
    @staticmethod
    def parallel_max(arr, num_threads=4):
        """并行求最大值"""
        if not arr:
            return None
        return ParallelReduction.parallel_reduce(arr, max, arr[0], num_threads)
    
    @staticmethod
    def parallel_min(arr, num_threads=4):
        """并行求最小值"""
        if not arr:
            return None
        return ParallelReduction.parallel_reduce(arr, min, arr[0], num_threads)


def test_parallel_algorithms():
    """测试并行算法"""
    print("=== 并行算法测试 ===\n")
    
    # 测试并行排序
    print("1. 并行排序测试:")
    data = [random.randint(1, 100) for _ in range(20)]
    print(f"   原始数据: {data}")
    
    sorted_merge = ParallelSort.parallel_merge_sort(data.copy())
    print(f"   并行归并排序: {sorted_merge}")
    
    sorted_quick = ParallelSort.parallel_quick_sort(data.copy())
    print(f"   并行快速排序: {sorted_quick}")
    
    sorted_bucket = ParallelSort.parallel_bucket_sort(data.copy())
    print(f"   并行桶排序: {sorted_bucket}")
    
    # 测试MapReduce
    print("\n2. MapReduce词频统计:")
    documents = [
        "hello world hello",
        "world of parallel computing",
        "hello parallel world"
    ]
    
    word_counts = WordCount.count_words(documents)
    print(f"   文档: {documents}")
    print(f"   词频: {word_counts}")
    
    # 测试并行前缀和
    print("\n3. 并行前缀和:")
    arr = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"   原始数组: {arr}")
    
    prefix_sequential = ParallelPrefixSum.sequential_prefix_sum(arr)
    print(f"   顺序前缀和: {prefix_sequential}")
    
    prefix_parallel = ParallelPrefixSum.parallel_segmented_scan(arr)
    print(f"   并行前缀和: {prefix_parallel}")
    
    # 测试并行矩阵运算
    print("\n4. 并行矩阵运算:")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    x = [1, 2]
    
    C = ParallelMatrixOperations.parallel_matrix_multiply(A, B)
    print(f"   矩阵A: {A}")
    print(f"   矩阵B: {B}")
    print(f"   A × B = {C}")
    
    y = ParallelMatrixOperations.parallel_matrix_vector_multiply(A, x)
    print(f"   向量x: {x}")
    print(f"   A × x = {y}")
    
    # 测试并行归约
    print("\n5. 并行归约:")
    numbers = list(range(1, 11))
    print(f"   数组: {numbers}")
    
    sum_result = ParallelReduction.parallel_sum(numbers)
    print(f"   并行求和: {sum_result}")
    
    product_result = ParallelReduction.parallel_product(numbers[:5])
    print(f"   并行求积(前5个): {product_result}")
    
    max_result = ParallelReduction.parallel_max(numbers)
    print(f"   并行最大值: {max_result}")
    
    min_result = ParallelReduction.parallel_min(numbers)
    print(f"   并行最小值: {min_result}")
    
    # 性能测试
    print("\n6. 性能对比（大数据集）:")
    large_data = [random.random() for _ in range(10000)]
    
    # 顺序排序
    start = time.time()
    sorted(large_data)
    sequential_time = time.time() - start
    
    # 并行排序
    start = time.time()
    ParallelSort.parallel_merge_sort(large_data, num_threads=4)
    parallel_time = time.time() - start
    
    print(f"   数据大小: {len(large_data)}")
    print(f"   顺序排序时间: {sequential_time:.4f}秒")
    print(f"   并行排序时间: {parallel_time:.4f}秒")
    print(f"   加速比: {sequential_time/parallel_time:.2f}x")
    
    # 算法复杂度
    print("\n=== 算法复杂度 ===")
    print("┌──────────────────┬──────────────┬──────────────┐")
    print("│ 算法              │ 时间复杂度    │ 空间复杂度    │")
    print("├──────────────────┼──────────────┼──────────────┤")
    print("│ 并行归并排序      │ O(n log n/p) │ O(n)         │")
    print("│ 并行前缀和        │ O(n/p+log n) │ O(n)         │")
    print("│ MapReduce        │ O(n/p + m)   │ O(n)         │")
    print("│ 并行矩阵乘法      │ O(n³/p)      │ O(n²)        │")
    print("│ 并行归约         │ O(n/p+log p) │ O(p)         │")
    print("└──────────────────┴──────────────┴──────────────┘")
    print("p = 处理器数量, m = Map/Reduce函数复杂度")


if __name__ == '__main__':
    test_parallel_algorithms()