"""自动查找并启动 Chrome 浏览器（带远程调试端口）。"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .config import CHROME_USER_DATA_DIR, WEIDIAN_CDP_PORT

_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Google\Chrome\Application\chrome.exe"),
]


def find_chrome() -> str | None:
    """查找系统中的 Chrome 可执行文件路径。"""
    # 优先用 PATH 里的
    which = shutil.which("chrome") or shutil.which("google-chrome")
    if which:
        return which

    for candidate in _CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    return None


def is_cdp_available(port: int = WEIDIAN_CDP_PORT, timeout: float = 2.0) -> bool:
    """检查 CDP 端口是否已有 Chrome 实例在监听。"""
    try:
        resp = urlopen(f"http://localhost:{port}/json/version", timeout=timeout)
        return resp.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def launch_chrome(
    port: int = WEIDIAN_CDP_PORT,
    user_data_dir: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.Popen | None:
    """启动 Chrome 并开启远程调试端口。

    如果 CDP 端口已占用（Chrome 已运行），直接返回 None 表示无需启动。
    """
    if is_cdp_available(port):
        return None  # 已有 Chrome 在运行

    chrome_path = find_chrome()
    if not chrome_path:
        raise FileNotFoundError(
            "未找到 Chrome 浏览器。请安装 Google Chrome 或手动启动 Chrome:\n"
            f"chrome.exe --remote-debugging-port={port}"
        )

    if user_data_dir is None:
        user_data_dir = CHROME_USER_DATA_DIR
    user_data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
        "--disable-popup-blocking",
        "--no-sandbox",
        "--disable-dev-shm-usage",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等 CDP 端口就绪
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # 检查进程是否已退出（启动失败）
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Chrome 进程已退出（退出码 {proc.returncode}）。请手动启动：\n"
                    f'"{chrome_path}" --remote-debugging-port={port}'
                )
            if is_cdp_available(port, timeout=1.0):
                return proc
            time.sleep(0.5)
    except BaseException:
        # 异常时（包括 KeyboardInterrupt）确保 Chrome 进程被清理
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise

    # 超时
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    raise TimeoutError(
        f"Chrome 启动超时（{timeout}秒）。请手动启动：\n"
        f'"{chrome_path}" --remote-debugging-port={port}'
    )
