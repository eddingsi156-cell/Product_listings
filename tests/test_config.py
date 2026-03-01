"""config 模块单元测试"""

import pytest

from yupoo_scraper.config import retry_wait, make_headers


class TestRetryWait:
    def test_exponential_backoff(self):
        """第 N 次重试 = 2^N + 随机抖动"""
        wait = retry_wait(1)
        assert 2.0 <= wait < 3.0  # 2^1 + [0, 1)

        wait = retry_wait(2)
        assert 4.0 <= wait < 5.0  # 2^2 + [0, 1)

        wait = retry_wait(3)
        assert 8.0 <= wait < 9.0  # 2^3 + [0, 1)

    def test_uses_retry_after_header(self):
        """有 Retry-After 头时优先使用"""
        wait = retry_wait(1, {"Retry-After": "30"})
        assert 30.0 <= wait < 31.0

    def test_ignores_non_numeric_retry_after(self):
        """非数字的 Retry-After 头应回退到指数退避"""
        wait = retry_wait(1, {"Retry-After": "not-a-number"})
        assert 2.0 <= wait < 3.0

    def test_no_headers_uses_backoff(self):
        """无 headers 时使用指数退避"""
        wait = retry_wait(2, None)
        assert 4.0 <= wait < 5.0


class TestMakeHeaders:
    def test_contains_referer(self):
        headers = make_headers("testuser")
        assert "Referer" in headers
        assert "testuser" in headers["Referer"]
        assert "x.yupoo.com" in headers["Referer"]

    def test_contains_user_agent(self):
        headers = make_headers("testuser")
        assert "User-Agent" in headers
        assert "Mozilla" in headers["User-Agent"]

    def test_different_users(self):
        h1 = make_headers("user1")
        h2 = make_headers("user2")
        assert h1["Referer"] != h2["Referer"]

    def test_does_not_mutate_global(self):
        """每次调用返回新字典，不应修改全局 HEADERS"""
        h1 = make_headers("user1")
        h1["Custom"] = "value"
        h2 = make_headers("user2")
        assert "Custom" not in h2
