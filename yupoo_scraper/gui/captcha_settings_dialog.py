"""验证码打码平台设置对话框 — 可视化配置 + JSON 持久化"""

from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import config

logger = logging.getLogger(__name__)

# ── 凭证加密（Windows DPAPI / fallback base64） ──────────────────

# 需要加密存储的字段
_SECRET_FIELDS = ("ttshitu_password", "twocaptcha_key")


def _encrypt_secret(plain: str) -> str:
    """加密敏感字符串。Windows 使用 DPAPI，其他平台 fallback base64。"""
    if not plain:
        return ""
    raw = plain.encode("utf-8")
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes

            class DATA_BLOB(ctypes.Structure):
                _fields_ = [("cbData", ctypes.wintypes.DWORD),
                             ("pbData", ctypes.POINTER(ctypes.c_char))]

            blob_in = DATA_BLOB(len(raw), ctypes.create_string_buffer(raw, len(raw)))
            blob_out = DATA_BLOB()
            if ctypes.windll.crypt32.CryptProtectData(
                ctypes.byref(blob_in), None, None, None, None, 0,
                ctypes.byref(blob_out),
            ):
                enc = ctypes.string_at(blob_out.pbData, blob_out.cbData)
                ctypes.windll.kernel32.LocalFree(blob_out.pbData)
                import base64
                return "dpapi:" + base64.b64encode(enc).decode("ascii")
        except Exception as e:
            logger.debug("DPAPI 加密失败，fallback base64: %s", e)
    # fallback: base64（至少不明文）
    import base64
    return "b64:" + base64.b64encode(raw).decode("ascii")


def _decrypt_secret(stored: str) -> str:
    """解密敏感字符串。"""
    if not stored:
        return ""
    import base64
    if stored.startswith("dpapi:"):
        enc = base64.b64decode(stored[6:])
        if sys.platform == "win32":
            try:
                import ctypes
                import ctypes.wintypes

                class DATA_BLOB(ctypes.Structure):
                    _fields_ = [("cbData", ctypes.wintypes.DWORD),
                                 ("pbData", ctypes.POINTER(ctypes.c_char))]

                blob_in = DATA_BLOB(len(enc), ctypes.create_string_buffer(enc, len(enc)))
                blob_out = DATA_BLOB()
                if ctypes.windll.crypt32.CryptUnprotectData(
                    ctypes.byref(blob_in), None, None, None, None, 0,
                    ctypes.byref(blob_out),
                ):
                    plain = ctypes.string_at(blob_out.pbData, blob_out.cbData)
                    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
                    return plain.decode("utf-8")
            except Exception as e:
                logger.warning("DPAPI 解密失败: %s", e)
        return ""  # 跨机器无法解密 DPAPI
    if stored.startswith("b64:"):
        return base64.b64decode(stored[4:]).decode("utf-8")
    # 兼容旧版明文存储
    return stored

# 平台选项 (显示名, 内部标识)
_PROVIDERS = [
    ("不使用", ""),
    ("图鉴 (ttshitu.com)", "ttshitu"),
    ("2Captcha (2captcha.com)", "twocaptcha"),
]


class CaptchaSettingsDialog(QDialog):
    """验证码打码平台设置对话框。

    功能:
      - 选择打码平台（图鉴 / 2Captcha / 不使用）
      - 填写账号凭证
      - 测试连接（验证账号有效性）
      - 保存到 JSON 文件 + 同步到 config 模块
    """

    settings_changed = Signal()  # 设置保存后发出

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("验证码设置")
        self.setMinimumWidth(460)
        self.setModal(True)
        self._build_ui()
        self._load_settings()

    # ── UI 构建 ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── 说明 ─────────────────────────────────────────────────
        tip = QLabel(
            "上架时如果遇到滑块验证码，将自动调用打码平台识别。\n"
            "不配置则跳过自动识别（需手动处理）。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666; margin-bottom: 4px;")
        layout.addWidget(tip)

        # ── 平台选择 ─────────────────────────────────────────────
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("打码平台:"))
        self._provider_combo = QComboBox()
        for display_name, _ in _PROVIDERS:
            self._provider_combo.addItem(display_name)
        self._provider_combo.currentIndexChanged.connect(
            self._on_provider_changed
        )
        provider_layout.addWidget(self._provider_combo, 1)
        layout.addLayout(provider_layout)

        # ── 图鉴设置组 ──────────────────────────────────────────
        self._ttshitu_group = QGroupBox("图鉴 (ttshitu.com)")
        ttshitu_form = QFormLayout(self._ttshitu_group)

        self._tt_user_input = QLineEdit()
        self._tt_user_input.setPlaceholderText("注册时的用户名")
        ttshitu_form.addRow("用户名:", self._tt_user_input)

        self._tt_pass_input = QLineEdit()
        self._tt_pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._tt_pass_input.setPlaceholderText("注册时的密码")
        ttshitu_form.addRow("密  码:", self._tt_pass_input)

        # 显示/隐藏密码
        self._tt_show_pass = QPushButton("显示")
        self._tt_show_pass.setFixedWidth(50)
        self._tt_show_pass.setCheckable(True)
        self._tt_show_pass.toggled.connect(
            lambda checked: (
                self._tt_pass_input.setEchoMode(
                    QLineEdit.EchoMode.Normal if checked
                    else QLineEdit.EchoMode.Password
                ),
                self._tt_show_pass.setText("隐藏" if checked else "显示"),
            )
        )
        ttshitu_form.addRow("", self._tt_show_pass)

        tt_tip = QLabel(
            '注册地址: ttshitu.com  |  类型: 滑块拼图(27)  |  约2分/次'
        )
        tt_tip.setStyleSheet("color: #888; font-size: 11px;")
        tt_tip.setWordWrap(True)
        ttshitu_form.addRow(tt_tip)

        layout.addWidget(self._ttshitu_group)

        # ── 2Captcha 设置组 ──────────────────────────────────────
        self._twocaptcha_group = QGroupBox("2Captcha (2captcha.com)")
        tc_form = QFormLayout(self._twocaptcha_group)

        self._tc_key_input = QLineEdit()
        self._tc_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._tc_key_input.setPlaceholderText("Dashboard → API Key")
        tc_form.addRow("API Key:", self._tc_key_input)

        # 显示/隐藏 Key
        self._tc_show_key = QPushButton("显示")
        self._tc_show_key.setFixedWidth(50)
        self._tc_show_key.setCheckable(True)
        self._tc_show_key.toggled.connect(
            lambda checked: (
                self._tc_key_input.setEchoMode(
                    QLineEdit.EchoMode.Normal if checked
                    else QLineEdit.EchoMode.Password
                ),
                self._tc_show_key.setText("隐藏" if checked else "显示"),
            )
        )
        tc_form.addRow("", self._tc_show_key)

        tc_tip = QLabel(
            "国际平台，支持多种验证码  |  按次计费"
        )
        tc_tip.setStyleSheet("color: #888; font-size: 11px;")
        tc_form.addRow(tc_tip)

        layout.addWidget(self._twocaptcha_group)

        # ── 高级设置 ─────────────────────────────────────────────
        adv_group = QGroupBox("高级设置")
        adv_form = QFormLayout(adv_group)

        self._retry_spin = QSpinBox()
        self._retry_spin.setRange(1, 10)
        self._retry_spin.setValue(config.CAPTCHA_MAX_RETRIES)
        self._retry_spin.setToolTip("验证码识别失败后的最大重试次数")
        adv_form.addRow("最大重试次数:", self._retry_spin)

        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1000, 30000)
        self._timeout_spin.setSingleStep(500)
        self._timeout_spin.setSuffix(" ms")
        self._timeout_spin.setValue(config.CAPTCHA_DETECT_TIMEOUT_MS)
        self._timeout_spin.setToolTip("点击创建后等待验证码弹出的超时时间")
        adv_form.addRow("检测超时:", self._timeout_spin)

        layout.addWidget(adv_group)

        # ── 底部按钮 ─────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self._btn_test = QPushButton("测试连接")
        self._btn_test.setToolTip("验证账号是否有效（不消耗次数）")
        self._btn_test.clicked.connect(self._on_test)
        btn_layout.addWidget(self._btn_test)

        btn_layout.addStretch()

        self._btn_save = QPushButton("保存")
        self._btn_save.setDefault(True)
        self._btn_save.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px 24px; }"
        )
        self._btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self._btn_save)

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)

        layout.addLayout(btn_layout)

        # ── 状态标签 ─────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._status_label)

        # 初始化可见性
        self._on_provider_changed()

    # ── 平台切换 ─────────────────────────────────────────────────

    @Slot()
    def _on_provider_changed(self) -> None:
        idx = self._provider_combo.currentIndex()
        provider = _PROVIDERS[idx][1] if idx < len(_PROVIDERS) else ""
        self._ttshitu_group.setVisible(provider == "ttshitu")
        self._twocaptcha_group.setVisible(provider == "twocaptcha")
        self._btn_test.setEnabled(provider != "")

    # ── 持久化 ───────────────────────────────────────────────────

    def _load_settings(self) -> None:
        """从 JSON 文件加载设置并填充到 UI。"""
        path = config.CAPTCHA_SETTINGS_FILE
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("加载验证码设置失败: %s", e)
            return

        # 平台
        provider = data.get("provider", "")
        for i, (_, pid) in enumerate(_PROVIDERS):
            if pid == provider:
                self._provider_combo.setCurrentIndex(i)
                break

        # 解密敏感字段
        for field in _SECRET_FIELDS:
            if data.get(field):
                data[field] = _decrypt_secret(data[field])

        # 图鉴
        self._tt_user_input.setText(data.get("ttshitu_username", ""))
        self._tt_pass_input.setText(data.get("ttshitu_password", ""))

        # 2Captcha
        self._tc_key_input.setText(data.get("twocaptcha_key", ""))

        # 高级
        self._retry_spin.setValue(data.get("max_retries", config.CAPTCHA_MAX_RETRIES))
        self._timeout_spin.setValue(data.get("detect_timeout_ms", config.CAPTCHA_DETECT_TIMEOUT_MS))

        # 同步到 config（已解密的明文）
        self._apply_to_config(data)

        self._status_label.setText(f"已加载设置: {path.name}")

    def _save_settings(self) -> dict:
        """将 UI 值写入 JSON 文件并同步到 config 模块。返回设置字典。"""
        idx = self._provider_combo.currentIndex()
        provider = _PROVIDERS[idx][1] if idx < len(_PROVIDERS) else ""

        # 明文数据（用于 apply_to_config）
        plain_data = {
            "provider": provider,
            "ttshitu_username": self._tt_user_input.text().strip(),
            "ttshitu_password": self._tt_pass_input.text().strip(),
            "twocaptcha_key": self._tc_key_input.text().strip(),
            "max_retries": self._retry_spin.value(),
            "detect_timeout_ms": self._timeout_spin.value(),
        }

        # 磁盘存储数据（敏感字段加密）
        disk_data = dict(plain_data)
        for field in _SECRET_FIELDS:
            if disk_data.get(field):
                disk_data[field] = _encrypt_secret(disk_data[field])

        path = config.CAPTCHA_SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(disk_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 限制文件权限：仅当前用户可读写
        try:
            if sys.platform == "win32":
                username = os.environ.get("USERNAME", "")
                if username:
                    subprocess.run(
                        ["icacls", str(path), "/inheritance:r",
                         "/grant:r", f"{username}:(R,W)"],
                        capture_output=True, check=False,
                    )
            else:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            logger.warning("无法设置文件权限: %s", path)

        self._apply_to_config(plain_data)
        return plain_data

    @staticmethod
    def _apply_to_config(data: dict) -> None:
        """将设置字典同步到 config 模块变量（运行时生效）。"""
        config.CAPTCHA_PROVIDER = data.get("provider", "")
        config.CAPTCHA_TTSHITU_USERNAME = data.get("ttshitu_username", "")
        config.CAPTCHA_TTSHITU_PASSWORD = data.get("ttshitu_password", "")
        config.CAPTCHA_TWOCAPTCHA_KEY = data.get("twocaptcha_key", "")
        config.CAPTCHA_MAX_RETRIES = data.get("max_retries", 5)
        config.CAPTCHA_DETECT_TIMEOUT_MS = data.get("detect_timeout_ms", 5000)

    # ── 测试连接 ─────────────────────────────────────────────────

    @Slot()
    def _on_test(self) -> None:
        """验证打码平台账号有效性。"""
        idx = self._provider_combo.currentIndex()
        provider = _PROVIDERS[idx][1] if idx < len(_PROVIDERS) else ""

        if not provider:
            return

        self._btn_test.setEnabled(False)
        self._status_label.setText("正在测试连接...")
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")

        # 在后台线程中测试，避免阻塞 UI
        from PySide6.QtCore import QThread

        class _TestThread(QThread):
            result = Signal(bool, str)

            def __init__(self, provider, username, password, api_key):
                super().__init__()
                self._provider = provider
                self._username = username
                self._password = password
                self._api_key = api_key

            def run(self):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    ok, msg = loop.run_until_complete(
                        self._test_async()
                    )
                    self.result.emit(ok, msg)
                except Exception as e:
                    self.result.emit(False, str(e))
                finally:
                    loop.close()

            async def _test_async(self):
                if self._provider == "ttshitu":
                    return await self._test_ttshitu()
                elif self._provider == "twocaptcha":
                    return await self._test_twocaptcha()
                return False, "未知平台"

            async def _test_ttshitu(self) -> tuple[bool, str]:
                import aiohttp

                if not self._username or not self._password:
                    return False, "请填写用户名和密码"
                # 图鉴没有专门的余额查询 API，用一个空图片测试会返回错误但能验证账号
                # 改为直接验证账号格式
                try:
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            "username": self._username,
                            "password": self._password,
                            "typeid": 27,
                            "image": "",  # 空图片触发参数错误（但能验证账号）
                        }
                        async with session.post(
                            "https://api.ttshitu.com/predict",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            content_type = resp.headers.get("Content-Type", "")
                            if "json" not in content_type:
                                text = await resp.text()
                                return False, f"服务端返回非 JSON（HTTP {resp.status}）: {text[:100]}"
                            data = await resp.json()
                            msg = data.get("message", "")
                            # 账号错误会返回 "用户名或密码错误"
                            # 空图片会返回 "image不能为空" 等
                            if "用户名" in msg or "密码" in msg or "账号" in msg:
                                return False, f"账号验证失败: {msg}"
                            # 其他错误说明账号是对的，只是图片参数不对
                            return True, f"账号有效 ({msg})"
                except aiohttp.ClientError as e:
                    return False, f"网络连接失败: {e}"
                except Exception as e:
                    return False, f"测试失败: {e}"

            async def _test_twocaptcha(self) -> tuple[bool, str]:
                import aiohttp

                if not self._api_key:
                    return False, "请填写 API Key"
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            "https://2captcha.com/res.php",
                            params={
                                "key": self._api_key,
                                "action": "getbalance",
                                "json": "1",
                            },
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            data = await resp.json()
                            if data.get("status") == 1:
                                balance = data.get("request", "?")
                                return True, f"账号有效，余额: ${balance}"
                            else:
                                return False, f"验证失败: {data.get('request', '未知错误')}"
                except Exception as e:
                    return False, f"连接失败: {e}"

        # 防止上一个测试线程仍在运行时重复启动
        if hasattr(self, '_test_thread') and self._test_thread is not None:
            if self._test_thread.isRunning():
                return

        self._test_thread = _TestThread(
            provider,
            self._tt_user_input.text().strip(),
            self._tt_pass_input.text().strip(),
            self._tc_key_input.text().strip(),
        )
        self._test_thread.result.connect(self._on_test_result)
        self._test_thread.start()

    @Slot(bool, str)
    def _on_test_result(self, ok: bool, msg: str) -> None:
        self._btn_test.setEnabled(True)
        self._test_thread = None  # 释放已结束的线程引用
        if ok:
            self._status_label.setText(f"  {msg}")
            self._status_label.setStyleSheet(
                "color: #2E7D32; font-size: 11px; font-weight: bold;"
            )
        else:
            self._status_label.setText(f"  {msg}")
            self._status_label.setStyleSheet(
                "color: #C62828; font-size: 11px; font-weight: bold;"
            )

    # ── 保存 ─────────────────────────────────────────────────────

    @Slot()
    def _on_save(self) -> None:
        idx = self._provider_combo.currentIndex()
        provider = _PROVIDERS[idx][1] if idx < len(_PROVIDERS) else ""

        # 简单校验
        if provider == "ttshitu":
            if not self._tt_user_input.text().strip():
                QMessageBox.warning(self, "提示", "请填写图鉴用户名")
                self._tt_user_input.setFocus()
                return
            if not self._tt_pass_input.text().strip():
                QMessageBox.warning(self, "提示", "请填写图鉴密码")
                self._tt_pass_input.setFocus()
                return
        elif provider == "twocaptcha":
            if not self._tc_key_input.text().strip():
                QMessageBox.warning(self, "提示", "请填写 2Captcha API Key")
                self._tc_key_input.setFocus()
                return

        self._save_settings()
        self._status_label.setText("设置已保存")
        self._status_label.setStyleSheet(
            "color: #2E7D32; font-size: 11px; font-weight: bold;"
        )
        self.settings_changed.emit()
        self.accept()


def load_captcha_settings_on_startup() -> None:
    """应用启动时加载验证码设置到 config 模块。

    在 GUI 初始化之前调用，确保 uploader 能读到持久化的设置。
    """
    path = config.CAPTCHA_SETTINGS_FILE
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text("utf-8"))
        # 解密敏感字段
        for field in _SECRET_FIELDS:
            if data.get(field):
                data[field] = _decrypt_secret(data[field])
        CaptchaSettingsDialog._apply_to_config(data)
        logger.info("已加载验证码设置: provider=%s", data.get("provider", ""))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("加载验证码设置失败: %s", e)
