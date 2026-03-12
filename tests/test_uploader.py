"""上架模块单元测试"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yupoo_scraper.uploader import UploadResult, WeidianUploader
from yupoo_scraper.title_generator import ProductInfo, get_title_generator, TitleGenerator
from yupoo_scraper import config


class TestUploadResult:
    """UploadResult 数据类测试"""

    def test_upload_result_success(self):
        """测试成功结果"""
        folder = Path("/test/folder")
        result = UploadResult(folder, True)
        assert result.success is True
        assert result.error is None
        assert result.warning is None
        assert result.folder == folder

    def test_upload_result_failure(self):
        """测试失败结果"""
        folder = Path("/test/folder")
        result = UploadResult(folder, False, error="上传失败")
        assert result.success is False
        assert result.error == "上传失败"
        assert result.folder == folder

    def test_upload_result_with_warning(self):
        """测试带警告的成功结果"""
        folder = Path("/test/folder")
        result = UploadResult(folder, True, warning="详情图上传失败，已使用自动生成")
        assert result.success is True
        assert result.warning == "详情图上传失败，已使用自动生成"


class TestWeidianUploader:
    """WeidianUploader 测试（同步测试）"""

    def test_uploader_init(self):
        """测试 uploader 初始化"""
        uploader = WeidianUploader()
        assert uploader._playwright is None
        assert uploader._browser is None
        assert uploader._context is None

    def test_uploader_is_connected_no_browser(self):
        """测试未连接时的状态"""
        uploader = WeidianUploader()
        assert uploader._browser is None


class TestTitleGenerator:
    """TitleGenerator 测试"""

    def test_title_generator_singleton(self):
        """测试单例模式"""
        gen1 = get_title_generator()
        gen2 = get_title_generator()
        assert gen1 is gen2

    def test_title_generator_class_level_cache(self):
        """测试类级别缓存"""
        assert TitleGenerator._text_features is None
        assert TitleGenerator._category_keys == []
        assert TitleGenerator._initialized is False


class TestConfig:
    """配置测试"""

    def test_upload_config_defaults(self):
        """测试上传配置默认值"""
        assert config.UPLOAD_STEP_TIMEOUT_MS == 30000
        assert config.UPLOAD_IMAGE_POLL_INTERVAL_MS == 2000
        assert config.UPLOAD_IMAGE_MIN_WAIT_PER_IMAGE == 5000
        assert config.UPLOAD_IMAGE_MAX_WAIT_BASE == 60000
        assert config.UPLOAD_STEP_DELAY_MIN == 0.5
        assert config.UPLOAD_STEP_DELAY_MAX == 1.5
        assert config.UPLOAD_RETRY_MAX == 3
        assert config.UPLOAD_RETRY_DELAY == 5

    def test_image_limits(self):
        """测试图片限制配置"""
        assert config.MAIN_IMAGE_MAX == 5
        assert config.DETAIL_IMAGE_MAX == 100

    def test_log_config(self):
        """测试日志配置"""
        assert config.LOG_FILE.name == "app.log"
        assert config.LOG_MAX_BYTES == 10 * 1024 * 1024
        assert config.LOG_BACKUP_COUNT == 5
        assert config.LOG_LEVEL == "INFO"

    def test_proxy_config(self):
        """测试代理配置"""
        assert config.WEIDIAN_PROXY_URL == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
