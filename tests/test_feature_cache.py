"""测试特征缓存功能"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from yupoo_scraper.ml.feature_cache import FeatureCache, get_cache


class TestFeatureCache:
    """测试特征缓存功能"""

    def test_cache_basic(self):
        """测试缓存的基本功能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=Path(tmpdir) / "cache")
            
            # 创建测试文件
            test_file = Path(tmpdir) / "test.jpg"
            test_file.write_text("test")
            
            # 测试设置和获取
            test_features = np.random.rand(512).astype(np.float32)
            cache.set(test_file, test_features)
            
            retrieved_features = cache.get(test_file)
            assert retrieved_features is not None
            assert np.array_equal(retrieved_features, test_features)
            
            # 测试缓存大小
            assert cache.size() == 1

    def test_cache_invalidation(self):
        """测试缓存失效"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=Path(tmpdir) / "cache")
            
            # 创建测试文件
            test_file = Path(tmpdir) / "test.jpg"
            test_file.write_text("test")
            
            # 设置缓存
            test_features1 = np.random.rand(512).astype(np.float32)
            cache.set(test_file, test_features1)
            
            # 修改文件
            test_file.write_text("modified")
            
            # 测试缓存是否失效
            retrieved_features = cache.get(test_file)
            assert retrieved_features is None
            
            # 重新设置缓存
            test_features2 = np.random.rand(512).astype(np.float32)
            cache.set(test_file, test_features2)
            
            retrieved_features = cache.get(test_file)
            assert retrieved_features is not None
            assert np.array_equal(retrieved_features, test_features2)

    def test_cache_clear(self):
        """测试清除缓存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=Path(tmpdir) / "cache")
            
            # 创建测试文件
            test_file = Path(tmpdir) / "test.jpg"
            test_file.write_text("test")
            
            # 设置缓存
            test_features = np.random.rand(512).astype(np.float32)
            cache.set(test_file, test_features)
            
            assert cache.size() == 1
            
            # 清除缓存
            cache.clear()
            assert cache.size() == 0
            
            # 测试缓存是否被清除
            retrieved_features = cache.get(test_file)
            assert retrieved_features is None

    def test_global_cache(self):
        """测试全局缓存实例"""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2  # 应该是同一个实例
