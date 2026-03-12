"""测试优化效果"""

import time
import numpy as np
from yupoo_scraper.ml.splitter import cluster_images

print("开始测试聚类优化效果...")

# 测试不同规模的特征矩阵
sizes = [100, 500, 1000, 2000]

for size in sizes:
    # 生成随机特征矩阵 (size, 608)
    features = np.random.rand(size, 608).astype(np.float32)
    # L2 归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行聚类
    labels = cluster_images(features, 0.5)
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    # 计算分组数量
    group_count = len(set(labels))
    
    print(f"特征数量: {size}")
    print(f"分组数量: {group_count}")
    print(f"聚类耗时: {total_time:.4f} 秒")
    print(f"平均每个样本耗时: {total_time / size:.6f} 秒")
    print()

print("测试完成!")
