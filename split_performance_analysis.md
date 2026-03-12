# 拆分功能性能分析报告

## 1. 功能概述

拆分功能是 Yupoo Scraper 的核心功能之一，用于将产品图片自动聚类分组，主要包括以下步骤：

1. **特征提取**：使用 CLIP + HSV 组合特征
2. **聚类**：使用层次聚类算法
3. **应用拆分**：将图片移动到对应分组文件夹

## 2. 性能瓶颈分析

### 2.1 主要性能瓶颈

1. **特征提取**：
   - CLIP 模型推理（特别是首次加载模型）
   - 图像处理和特征计算

2. **聚类算法**：
   - 层次聚类的时间复杂度为 O(n²)，对于大样本会很慢
   - 距离矩阵计算（特别是对于大样本）

3. **文件操作**：
   - 图片移动操作
   - 文件夹创建和管理

### 2.2 代码分析

#### 特征提取

```python
# 批量组合特征提取
def extract_combined_batch(self, paths: list[Path], on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    # CLIP 批量推理
    clip_feats = self.extract_clip_batch(paths)  # (N, 512)
    
    # 并行提取 HSV 特征
    import concurrent.futures
    hsv_feats = np.empty((n, HSV_DIM), dtype=np.float32)
    
    def process_hsv(i, path):
        feat = self.extract_hsv(path)
        hsv_feats[i] = feat
        if on_progress:
            on_progress(i + 1, n)
        return i, feat
    
    # 使用线程池并行处理
    import os
    max_workers = min(8, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_hsv, i, p) for i, p in enumerate(paths)]
        for future in concurrent.futures.as_completed(futures):
            pass  # 等待所有任务完成
```

#### 聚类算法

```python
def cluster_images(features: np.ndarray, threshold: float) -> np.ndarray:
    n = features.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32)

    try:
        import faiss
        
        # 使用 FAISS 加速距离计算（适用于大型数据集）
        if n > 1000:
            logger.info("使用 FAISS 加速聚类计算")
            
            # 构建 FAISS 索引
            index = faiss.IndexFlatIP(features.shape[1])
            index.add(features.astype(np.float32))
            
            # 计算最近邻距离
            batch_size = 1000
            cosine_dist = np.zeros((n, n), dtype=np.float32)
            
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch = features[i:end].astype(np.float32)
                distances, _ = index.search(batch, n)
                cosine_dist[i:end] = 1.0 - distances  # 转换为余弦距离
            
            np.fill_diagonal(cosine_dist, 0.0)
            cosine_dist = np.clip(cosine_dist, 0.0, 2.0)
        else:
            # 小规模数据使用原始方法
            cosine_sim = features @ features.T
            cosine_dist = 1.0 - cosine_sim
            np.fill_diagonal(cosine_dist, 0.0)
            cosine_dist = np.clip(cosine_dist, 0.0, 2.0)
    except ImportError:
        # 无 FAISS 时使用原始方法
        logger.warning("FAISS 未安装，使用原始距离计算方法")
        cosine_sim = features @ features.T
        cosine_dist = 1.0 - cosine_sim
        np.fill_diagonal(cosine_dist, 0.0)
        cosine_dist = np.clip(cosine_dist, 0.0, 2.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(cosine_dist)
    return labels.astype(np.int32)
```

## 3. 性能优化建议

### 3.1 特征提取优化

1. **缓存优化**：
   - 扩展缓存机制，支持更多特征类型
   - 实现缓存过期策略，避免缓存过大

2. **并行优化**：
   - 增加 HSV 特征提取的并行度
   - 考虑使用多进程而非多线程处理图像处理

3. **模型优化**：
   - 考虑使用更轻量级的 CLIP 模型变体
   - 实现模型量化，减少内存使用

### 3.2 聚类算法优化

1. **算法选择**：
   - 对于大样本，考虑使用 K-Means 或 DBSCAN 等线性时间复杂度的算法
   - 实现分层聚类，先粗聚类再细聚类

2. **距离计算优化**：
   - 进一步优化 FAISS 索引，使用更适合余弦距离的索引类型
   - 实现距离矩阵的稀疏存储，减少内存使用

3. **参数调优**：
   - 针对不同规模的数据集，自动调整聚类参数
   - 实现自适应阈值选择，根据数据分布动态调整

### 3.3 文件操作优化

1. **批量操作**：
   - 批量创建文件夹，减少 I/O 操作
   - 实现文件移动的批处理，减少系统调用

2. **并行操作**：
   - 并行处理文件移动，提高处理速度
   - 使用异步 I/O 操作，避免阻塞主线程

## 4. 性能测试建议

### 4.1 测试场景

1. **小规模测试**：10-50 张图片
2. **中规模测试**：100-200 张图片
3. **大规模测试**：500-1000 张图片
4. **批量测试**：多个文件夹同时处理

### 4.2 测试指标

1. **总处理时间**：从开始到完成的总时间
2. **特征提取时间**：提取特征的时间
3. **聚类时间**：聚类算法的时间
4. **文件操作时间**：文件移动和文件夹创建的时间
5. **内存使用**：处理过程中的内存占用

## 5. 结论

拆分功能的性能主要受限于以下因素：

1. **特征提取**：CLIP 模型推理是主要瓶颈，特别是在首次加载模型时
2. **聚类算法**：层次聚类的 O(n²) 时间复杂度对于大样本会很慢
3. **文件操作**：I/O 操作可能成为瓶颈，特别是在处理大量文件时

通过实施上述优化建议，可以显著提高拆分功能的性能，特别是对于大规模数据集的处理。

## 6. 后续工作

1. **实现性能监控**：添加性能监控机制，实时跟踪处理时间和资源使用
2. **自动化调优**：实现参数自动调优，根据数据规模和硬件环境动态调整
3. **并行化处理**：进一步优化并行处理，充分利用多核 CPU 和 GPU 资源
4. **用户体验优化**：添加处理进度显示，提高用户体验
