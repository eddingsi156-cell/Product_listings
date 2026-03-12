"""测试拆分功能"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from yupoo_scraper.ml.splitter import (
    SplitGroup,
    SplitResult,
    cluster_images,
    build_split_result,
    recluster,
    apply_split,
)
from yupoo_scraper.config import CLUSTER_THRESHOLD_DEFAULT


class TestSplitter:
    """测试拆分功能"""

    def test_cluster_images(self):
        """测试聚类功能"""
        # 创建简单的特征矩阵
        features = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ], dtype=np.float32)
        
        # 测试聚类
        labels = cluster_images(features, 0.1)
        assert len(labels) == 4
        assert len(set(labels)) >= 2

    def test_build_split_result(self):
        """测试构建拆分结果"""
        folder = Path("test_folder")
        paths = [Path(f"image_{i}.jpg") for i in range(4)]
        features = np.random.rand(4, 608).astype(np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        
        result = build_split_result(folder, paths, features, labels)
        assert len(result.groups) == 2
        assert len(result.groups[0].image_paths) == 2
        assert len(result.groups[1].image_paths) == 2

    def test_recluster(self):
        """测试重新聚类"""
        folder = Path("test_folder")
        paths = [Path(f"image_{i}.jpg") for i in range(4)]
        features = np.random.rand(4, 608).astype(np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        
        original_result = SplitResult(
            album_folder=folder,
            groups=[
                SplitGroup(id=0, name="group1", image_paths=paths[:2], original_indices=[0, 1]),
                SplitGroup(id=1, name="group2", image_paths=paths[2:], original_indices=[2, 3]),
            ],
            image_paths=paths,
            features=features,
        )
        
        new_result = recluster(original_result, CLUSTER_THRESHOLD_DEFAULT)
        assert len(new_result.groups) > 0

    def test_apply_split(self):
        """测试应用拆分"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件夹和文件
            album_folder = Path(tmpdir) / "test_album"
            album_folder.mkdir()
            
            # 创建测试文件
            for i in range(4):
                (album_folder / f"image_{i}.jpg").write_text(f"test{i}")
            
            # 创建拆分结果
            paths = list(album_folder.iterdir())
            features = np.random.rand(4, 608).astype(np.float32)
            labels = np.array([0, 0, 1, 1], dtype=np.int32)
            result = build_split_result(album_folder, paths, features, labels)
            
            # 应用拆分
            created_folders = apply_split(result)
            assert len(created_folders) == 2
            assert all(folder.exists() for folder in created_folders)
            
            # 检查原文件夹是否被删除
            assert not album_folder.exists()
