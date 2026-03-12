"""简单验证脚本"""

import numpy as np
from yupoo_scraper.ml.splitter import cluster_images
from yupoo_scraper.ml.feature_cache import FeatureCache
import tempfile
from pathlib import Path

print("测试聚类功能...")
# 创建简单的特征矩阵
features = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1],
], dtype=np.float32)

labels = cluster_images(features, 0.1)
print(f"聚类结果: {labels}")
print(f"聚类数量: {len(set(labels))}")

print("\n测试特征缓存...")
with tempfile.TemporaryDirectory() as tmpdir:
    cache = FeatureCache(cache_dir=Path(tmpdir) / "cache")
    
    # 创建测试文件
    test_file = Path(tmpdir) / "test.jpg"
    test_file.write_text("test")
    
    # 测试设置和获取
    test_features = np.random.rand(512).astype(np.float32)
    cache.set(test_file, test_features)
    
    retrieved_features = cache.get(test_file)
    if retrieved_features is not None:
        print("缓存测试成功！")
        print(f"特征形状: {retrieved_features.shape}")
    else:
        print("缓存测试失败！")

print("\n验证完成！")
