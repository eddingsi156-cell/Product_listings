"""微店上架核心逻辑 — Playwright CDP 自动化"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import (
    MAIN_IMAGE_MAX,
    WEIDIAN_PUBLISH_URL,
    WEIDIAN_CDP_URL,
    UPLOAD_STEP_TIMEOUT_MS,
    UPLOAD_IMAGE_POLL_INTERVAL_MS,
    UPLOAD_IMAGE_MIN_WAIT_PER_IMAGE,
    UPLOAD_IMAGE_MAX_WAIT_BASE,
    UPLOAD_STEP_DELAY_MIN,
    UPLOAD_STEP_DELAY_MAX,
    UPLOAD_RETRY_MAX,
    UPLOAD_RETRY_DELAY,
    LOGIN_CHECK_URL,
)

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """单个产品的上架结果。"""

    folder: Path
    success: bool
    error: str | None = None
    title: str = ""
    warning: str | None = None  # 警告信息（如详情图上传失败）


# 进度回调类型
StepCallback = Callable[[str], None]          # 当前步骤描述
ProgressCallback = Callable[[int, int], None]  # (当前, 总数)

# 每步超时（秒）
_STEP_TIMEOUT_MS = UPLOAD_STEP_TIMEOUT_MS
# 截图保存路径（调试用）
_BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent
_SCREENSHOT_DIR = _BASE_DIR / "data"

# 步骤间随机延迟范围（秒），防止反自动化检测
_STEP_DELAY = (UPLOAD_STEP_DELAY_MIN, UPLOAD_STEP_DELAY_MAX)


async def _random_delay() -> None:
    """步骤间随机延迟，模拟人工操作节奏。"""
    await asyncio.sleep(random.uniform(*_STEP_DELAY))


class WeidianUploader:
    """微店自动上架器 — 通过 CDP 连接已登录的浏览器。"""

    LOGIN_INDICATORS = [
        "pc-vue-item",      # 已登录后会进入的发布页路径特征
        "item/edit",        # 发布页
        "weidian-pc",       # 微店后台域名
    ]

    NOT_LOGGED_IN_INDICATORS = [
        "login",            # 登录页
        "login.weidian.com", # 登录域名
    ]

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._login_check_page = None

    async def connect(self, cdp_url: str) -> None:
        """连接到已打开的 Chrome 浏览器。"""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
            if not self._browser.contexts:
                raise RuntimeError(
                    "浏览器无可用上下文，请确保 Chrome 已打开并登录微店"
                )
            self._context = self._browser.contexts[0]
        except Exception:
            await self._cleanup()
            raise

    async def disconnect(self) -> None:
        await self._cleanup()

    async def _cleanup(self) -> None:
        if self._browser:
            try:
                # CDP 连接只需断开，不能 close()——close() 会关闭用户的整个 Chrome
                if hasattr(self._browser, 'disconnect'):
                    await self._browser.disconnect()
                # 旧版 Playwright 没有 disconnect，直接置空即可
            except Exception:
                pass
            self._browser = None
            self._context = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    async def is_connected(self) -> bool:
        if not self._browser:
            return False
        try:
            return self._browser.is_connected()
        except Exception:
            return False

    async def check_login_status(
        self,
        on_status_change: Callable[[bool, str], None] | None = None,
    ) -> bool:
        """检查用户是否已登录微店。

        Args:
            on_status_change: 状态变化回调 (is_logged_in: bool, message: str)

        Returns:
            True 表示已登录，False 表示未登录。
        """
        if not await self.is_connected():
            if on_status_change:
                on_status_change(False, "浏览器未连接")
            return False

        page = None
        try:
            page = await self._context.new_page()

            await page.goto(LOGIN_CHECK_URL, wait_until="domcontentloaded")
            await asyncio.sleep(1)

            url = page.url.lower()
            page_text = (await page.evaluate("document.body.innerText")).lower()

            for not_logged in self.NOT_LOGGED_IN_INDICATORS:
                if not_logged in url or not_logged in page_text:
                    if on_status_change:
                        on_status_change(False, "未检测到登录状态")
                    return False

            for logged in self.LOGIN_INDICATORS:
                if logged in url:
                    if on_status_change:
                        on_status_change(True, "已登录")
                    return True

            try:
                await page.goto(WEIDIAN_PUBLISH_URL, wait_until="domcontentloaded", timeout=10000)
                await asyncio.sleep(1)
                publish_url = page.url.lower()
                if "pc-vue-item" in publish_url or "item/edit" in publish_url:
                    if on_status_change:
                        on_status_change(True, "已登录")
                    return True
            except Exception:
                pass

            if on_status_change:
                on_status_change(False, "无法确认登录状态")
            return False

        except Exception as e:
            logger.warning("登录状态检查失败: %s", e)
            if on_status_change:
                on_status_change(False, f"检查失败: {e}")
            return False
        finally:
            if page:
                await page.close()

    async def wait_for_login(
        self,
        timeout: float = 300.0,
        check_interval: float = 3.0,
        on_waiting: Callable[[int, str], None] | None = None,
    ) -> bool:
        """等待用户登录微店。

        Args:
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
            on_waiting: 等待中回调 (elapsed_seconds: int, message: str)

        Returns:
            True 表示登录成功，False 表示超时。
        """
        start_time = time.monotonic()
        check_count = 0

        while time.monotonic() - start_time < timeout:
            is_logged_in = await self.check_login_status()
            if is_logged_in:
                return True

            elapsed = int(time.monotonic() - start_time)
            if on_waiting:
                on_waiting(elapsed, f"等待登录中... ({elapsed}秒)")

            await asyncio.sleep(check_interval)
            check_count += 1

        return False

    async def upload_product(
        self,
        product: "ProductInfo",
        on_step: StepCallback | None = None,
    ) -> UploadResult:
        """上架单个商品（带重试机制）。"""
        if not await self.is_connected():
            return UploadResult(product.folder, False, "浏览器未连接", title=product.title)

        def step(msg: str) -> None:
            if on_step:
                on_step(msg)

        last_error = None
        for attempt in range(UPLOAD_RETRY_MAX + 1):
            try:
                page = await self._context.new_page()
                try:
                    result = await self._do_upload(page, product, step)
                    result.title = product.title
                    if not result.success:
                        await self._save_debug_screenshot(page, product.folder.name)
                        last_error = result.error
                        # 如果是可重试的错误，等待后重试
                        if attempt < UPLOAD_RETRY_MAX and last_error:
                            step(f"上传失败，{UPLOAD_RETRY_DELAY}秒后重试 ({attempt + 1}/{UPLOAD_RETRY_MAX})...")
                            await asyncio.sleep(UPLOAD_RETRY_DELAY)
                            continue
                finally:
                    await page.close()
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < UPLOAD_RETRY_MAX:
                    step(f"上传异常，{UPLOAD_RETRY_DELAY}秒后重试 ({attempt + 1}/{UPLOAD_RETRY_MAX})...")
                    await asyncio.sleep(UPLOAD_RETRY_DELAY)
                else:
                    return UploadResult(product.folder, False, last_error, title=product.title)

        return UploadResult(product.folder, False, last_error or "重试次数耗尽", title=product.title)

    @staticmethod
    async def _save_debug_screenshot(page, name: str) -> None:
        """失败时保存截图用于调试。"""
        try:
            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c for c in name if c.isalnum() or c in " _-")[:100]
            path = _SCREENSHOT_DIR / f"upload_error_{safe_name}.png"
            await page.screenshot(path=str(path), full_page=True)
            logger.info("调试截图已保存: %s", path)
        except Exception as e:
            logger.warning("保存调试截图失败: %s", e)

    async def _upload_images_via_dialog(
        self,
        page,
        image_paths: list[Path],
        step_prefix: str,
        step: Callable[[str], None],
    ) -> str | None:
        """通用的素材管理弹窗上传流程。

        Args:
            page: Playwright page 对象。
            image_paths: 要上传的图片路径列表。
            step_prefix: 步骤前缀（如 "3/9"）用于进度提示。
            step: 步骤回调。

        Returns:
            错误信息字符串，None 表示成功。
        """
        n_upload = len(image_paths)

        # 切换到"上传图片"标签（默认在"选择图片"标签）
        upload_tab = page.locator("text=上传图片").first
        await upload_tab.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        await upload_tab.click()
        await _random_delay()

        # 设置文件
        file_input = page.locator("input.el-upload__input").first
        str_paths = [str(p) for p in image_paths]
        await file_input.set_input_files(str_paths)

        # 轮询等待上传完成（超时随图片数量增长，每张至少 5 秒）
        poll_interval = UPLOAD_IMAGE_POLL_INTERVAL_MS / 1000
        max_wait = max(UPLOAD_IMAGE_MAX_WAIT_BASE // 1000, n_upload * (UPLOAD_IMAGE_MIN_WAIT_PER_IMAGE // 1000))
        max_polls = int(max_wait / poll_interval)
        visible_imgs = 0
        dialog = page.locator(".el-dialog__wrapper:visible .el-dialog").first
        for _ in range(max_polls):
            await asyncio.sleep(poll_interval)
            visible_imgs = await dialog.evaluate(
                "el => [...el.querySelectorAll('img')].filter("
                "  i => i.offsetParent !== null && i.naturalWidth > 0"
                ").length"
            )
            step(f"{step_prefix} 上传中 ({visible_imgs}/{n_upload})...")
            if visible_imgs >= n_upload:
                break
        else:
            if visible_imgs == 0:
                return f"{step_prefix} 上传超时（{max_wait}s），0/{n_upload} 张图片上传成功"
            step(f"{step_prefix} 上传超时，已完成 {visible_imgs}/{n_upload} 张，继续")

        # 点确定关闭弹窗
        confirm_btn = page.locator(
            ".el-dialog__wrapper:visible .el-dialog__footer button:has-text('确定')"
        ).first
        await confirm_btn.click()

        # 等待弹窗关闭
        await page.locator(".v-modal").first.wait_for(
            state="hidden", timeout=_STEP_TIMEOUT_MS,
        )
        await _random_delay()
        return None

    async def _do_upload(
        self,
        page,
        product: "ProductInfo",
        step: Callable[[str], None],
    ) -> UploadResult:
        """上架流程：上传主图 → 填信息 → 下一步 → 自动生成详情 → 创建。"""

        upload_images = product.main_images[:MAIN_IMAGE_MAX]
        n_main = len(upload_images)
        total_steps = 9

        # ── 1. 打开发布页 ─────────────────────────────────────────
        step(f"1/{total_steps} 打开发布页...")
        try:
            await page.goto("about:blank")
            await page.goto(WEIDIAN_PUBLISH_URL, wait_until="networkidle")
            await page.evaluate("""() => {
                localStorage.clear();
                sessionStorage.clear();
            }""")
            await page.locator(".upload-img-card").first.wait_for(
                state="visible", timeout=_STEP_TIMEOUT_MS,
            )
            await _random_delay()
        except Exception as e:
            return UploadResult(product.folder, False, f"1/{total_steps} 打开发布页失败: {e}")

        # ── 2. 点击上传区域 → 打开素材管理弹窗 ────────────────────
        step(f"2/{total_steps} 打开上传弹窗...")
        try:
            await page.locator(".upload-img-card.img-upload-icon").first.click()
            await _random_delay()
        except Exception as e:
            return UploadResult(product.folder, False, f"2/{total_steps} 打开上传弹窗失败: {e}")

        # ── 3. 上传主图 ──────────────────────────────────────────
        step(f"3/{total_steps} 上传主图 ({n_main} 张)...")
        try:
            err = await self._upload_images_via_dialog(
                page, upload_images, f"3/{total_steps}", step,
            )
            if err:
                return UploadResult(product.folder, False, err)
        except Exception as e:
            return UploadResult(product.folder, False, f"3/{total_steps} 上传主图失败: {e}")

        # ── 4. 填写标题/价格/库存 ────────────────────────────────
        step(f"4/{total_steps} 填写信息...")
        try:
            textarea = page.locator("textarea").first
            await textarea.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await textarea.fill(product.title)
            await _random_delay()

            await page.locator(
                'input[placeholder="请填写商品价格"]'
            ).first.fill(str(product.price))

            await page.locator(
                'input[placeholder="请填写商品库存"]'
            ).first.fill(str(product.stock))
            await _random_delay()
        except Exception as e:
            return UploadResult(product.folder, False, f"4/{total_steps} 填写信息失败: {e}")

        # ── 5. 点击下一步 ────────────────────────────────────────
        step(f"5/{total_steps} 下一步...")
        try:
            next_btn = page.locator("button:has-text('下一步')").first
            await next_btn.scroll_into_view_if_needed()
            await next_btn.click()
            await page.locator("text=自动生成商品详情").first.wait_for(
                state="visible", timeout=_STEP_TIMEOUT_MS,
            )
            await _random_delay()
        except Exception as e:
            return UploadResult(product.folder, False, f"5/{total_steps} 下一步失败: {e}")

        # ── 6. 关闭弹窗（如有） ──────────────────────────────────
        step(f"6/{total_steps} 关闭通知弹窗...")
        for dismiss_text in ["我知道了", "知道了"]:
            try:
                btn = page.locator(f"button:has-text('{dismiss_text}')").first
                if await btn.is_visible(timeout=1500):
                    await btn.click()
                    await _random_delay()
            except Exception:
                pass

        # ── 7. 自动生成商品详情 ────────────────────────────────
        step(f"7/{total_steps} 自动生成商品详情...")
        try:
            auto_btn = page.locator("text=自动生成商品详情").first
            await auto_btn.click()
            await _random_delay()
        except Exception as e:
            step(f"7/{total_steps} 自动生成详情跳过: {e}")

        # ── 8. 点击创建 ──────────────────────────────────────────
        step(f"8/{total_steps} 点击创建...")
        try:
            create_btn = page.locator("button:has-text('创建')").first
            await create_btn.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await create_btn.click()

            # 等待成功标志出现 - 增强检测逻辑
            success_indicators = [
                page.locator("text=创建成功"),
                page.locator("text=发布成功"),
                page.locator("text=上架成功"),
                page.locator("text=商品发布成功"),
                page.locator("text=发布商品成功"),
                page.locator("text=上架商品成功"),
                page.locator("text=保存成功"),
                page.locator("text=操作成功"),
                page.locator(".el-message--success"),
                page.locator("[class*=success]"),
            ]
            confirmed_success = False
            success_text = None

            for indicator in success_indicators:
                try:
                    if await indicator.first.is_visible(timeout=3000):
                        confirmed_success = True
                        success_text = await indicator.first.inner_text()
                        break
                except Exception:
                    continue

            if not confirmed_success:
                await asyncio.sleep(3)
                current_url = page.url
                try:
                    page_text = await page.evaluate(
                        "() => document.body.innerText.slice(0, 500)"
                    )
                    dialog_text = await page.evaluate(
                        """() => {
                            const dialogs = document.querySelectorAll('.el-dialog, .modal, [class*=dialog], [class*=popup]');
                            let texts = '';
                            dialogs.forEach(d => { if (d.offsetParent !== null) texts += d.innerText + ' '; });
                            return texts.slice(0, 300);
                        }"""
                    )
                except Exception:
                    page_text = "(无法获取页面文本)"
                    dialog_text = ""

                all_text = (page_text + " " + dialog_text).lower()

                if any(kw in all_text for kw in ["创建成功", "发布成功", "上架成功", "上架商品成功", "保存成功", "操作成功", "success", "完成"]):
                    confirmed_success = True
                    success_text = "检测到成功提示"

                if not confirmed_success:
                    logger.warning(
                        "未检测到成功提示, URL=%s, 页面文本前200字=%s, 弹窗文本=%s",
                        current_url, page_text[:200], dialog_text[:100],
                    )
                    if "success" in current_url.lower():
                        confirmed_success = True
                        success_text = "URL跳转成功"
                    else:
                        return UploadResult(
                            product.folder, False,
                            f"8/{total_steps} 未检测到创建成功提示，可能上架失败"
                            f"（URL={current_url}，页面文本={page_text[:50]}）",
                        )
        except Exception as e:
            return UploadResult(product.folder, False, f"8/{total_steps} 点击创建失败: {e}")

        step("上架成功")

        return UploadResult(product.folder, True)
