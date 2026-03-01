"""微店上架核心逻辑 — Playwright CDP 自动化"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import MAIN_IMAGE_MAX, WEIDIAN_PUBLISH_URL
from .title_generator import ProductInfo

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """单个产品的上架结果。"""

    folder: Path
    success: bool
    error: str | None = None
    title: str = ""


# 进度回调类型
StepCallback = Callable[[str], None]          # 当前步骤描述
ProgressCallback = Callable[[int, int], None]  # (当前, 总数)

# 每步超时（秒）
_STEP_TIMEOUT_MS = 30000
# 截图保存路径（调试用）
_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "data"


class WeidianUploader:
    """微店自动上架器 — 通过 CDP 连接已登录的浏览器。"""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None

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

    async def upload_product(
        self,
        product: ProductInfo,
        on_step: StepCallback | None = None,
    ) -> UploadResult:
        """上架单个商品。"""
        if not await self.is_connected():
            return UploadResult(product.folder, False, "浏览器未连接", title=product.title)

        def step(msg: str) -> None:
            if on_step:
                on_step(msg)

        try:
            page = await self._context.new_page()
            try:
                result = await self._do_upload(page, product, step)
                result.title = product.title
                if not result.success:
                    # 失败时保存截图便于调试
                    await self._save_debug_screenshot(page, product.folder.name)
            finally:
                await page.close()
            return result
        except Exception as e:
            return UploadResult(product.folder, False, str(e), title=product.title)

    @staticmethod
    async def _save_debug_screenshot(page, name: str) -> None:
        """失败时保存截图用于调试。"""
        try:
            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            path = _SCREENSHOT_DIR / f"upload_error_{name}.png"
            await page.screenshot(path=str(path), full_page=True)
        except Exception:
            pass

    async def _do_upload(
        self,
        page,
        product: ProductInfo,
        step: Callable[[str], None],
    ) -> UploadResult:
        """上架流程：打开弹窗上传图片 → 填信息 → 下一步 → 自动生成详情 → 创建。"""

        upload_images = product.main_images[:MAIN_IMAGE_MAX]
        n_upload = len(upload_images)

        # ── 1. 打开发布页 ─────────────────────────────────────────
        step("1/8 打开发布页...")
        try:
            await page.goto(WEIDIAN_PUBLISH_URL, wait_until="domcontentloaded")
            await page.locator(".upload-img-card").first.wait_for(
                state="visible", timeout=_STEP_TIMEOUT_MS,
            )
            await asyncio.sleep(1)
        except Exception as e:
            return UploadResult(product.folder, False, f"1/8 打开发布页失败: {e}")

        # ── 2. 点击上传区域 → 打开素材管理弹窗 ────────────────────
        step("2/8 打开上传弹窗...")
        try:
            await page.locator(".upload-img-card.img-upload-icon").first.click()
            await asyncio.sleep(2)

            # 切换到"上传图片"标签（默认在"选择图片"标签）
            upload_tab = page.locator("text=上传图片").first
            await upload_tab.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await upload_tab.click()
            await asyncio.sleep(1)
        except Exception as e:
            return UploadResult(product.folder, False, f"2/8 打开上传弹窗失败: {e}")

        # ── 3. 上传图片（前6张）───────────────────────────────────
        step(f"3/8 上传图片 ({n_upload} 张)...")
        try:
            file_input = page.locator("input.el-upload__input").first
            image_paths = [str(p) for p in upload_images]
            await file_input.set_input_files(image_paths)

            # 轮询等待上传完成：弹窗内可见 <img> 数量 >= 上传数
            visible_imgs = 0
            dialog = page.locator(".el-dialog__wrapper:visible .el-dialog").first
            for elapsed in range(0, 60, 2):
                await asyncio.sleep(2)
                visible_imgs = await dialog.evaluate(
                    "el => [...el.querySelectorAll('img')].filter("
                    "  i => i.offsetParent !== null && i.naturalWidth > 0"
                    ").length"
                )
                step(f"3/8 上传中 ({visible_imgs}/{n_upload})...")
                if visible_imgs >= n_upload:
                    break
            else:
                if visible_imgs == 0:
                    return UploadResult(
                        product.folder, False,
                        f"3/8 上传超时，0/{n_upload} 张图片上传成功",
                    )
                step(f"3/8 上传超时，已完成 {visible_imgs}/{n_upload} 张，继续上架")
        except Exception as e:
            return UploadResult(product.folder, False, f"3/8 上传图片失败: {e}")

        # ── 4. 点确定关闭弹窗 ─────────────────────────────────────
        step("4/8 确认图片...")
        try:
            confirm_btn = page.locator(
                ".el-dialog__wrapper:visible .el-dialog__footer button:has-text('确定')"
            ).first
            await confirm_btn.click()

            # 等待弹窗关闭（v-modal 遮罩消失）
            await page.locator(".v-modal").first.wait_for(
                state="hidden", timeout=_STEP_TIMEOUT_MS,
            )
            await asyncio.sleep(0.5)
        except Exception as e:
            return UploadResult(product.folder, False, f"4/8 确认图片失败: {e}")

        # ── 5. 填写标题/价格/库存 ────────────────────────────────
        step("5/8 填写信息...")
        try:
            textarea = page.locator("textarea").first
            await textarea.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await textarea.fill(product.title)

            await page.locator(
                'input[placeholder="请填写商品价格"]'
            ).first.fill(str(product.price))

            await page.locator(
                'input[placeholder="请填写商品库存"]'
            ).first.fill(str(product.stock))
        except Exception as e:
            return UploadResult(product.folder, False, f"5/8 填写信息失败: {e}")

        # ── 6. 点击下一步 ────────────────────────────────────────
        step("6/8 下一步...")
        try:
            next_btn = page.locator("button:has-text('下一步')").first
            await next_btn.scroll_into_view_if_needed()
            await next_btn.click()
            # 等页面切换：「自动生成商品详情」出现表示已到第2步
            await page.locator("text=自动生成商品详情").first.wait_for(
                state="visible", timeout=_STEP_TIMEOUT_MS,
            )
            await asyncio.sleep(1)
        except Exception as e:
            return UploadResult(product.folder, False, f"6/8 下一步失败: {e}")

        # 关闭弹窗（如有）
        for dismiss_text in ["我知道了", "知道了"]:
            try:
                btn = page.locator(f"button:has-text('{dismiss_text}')").first
                if await btn.is_visible(timeout=1500):
                    await btn.click()
                    await asyncio.sleep(0.5)
            except Exception:
                pass

        # ── 7. 点击自动生成商品详情 ────────────────────────────────
        step("7/8 自动生成商品详情...")
        try:
            auto_btn = page.locator("text=自动生成商品详情").first
            await auto_btn.click()
            await asyncio.sleep(3)
        except Exception as e:
            # 非必须步骤，失败继续
            step(f"7/8 自动生成详情跳过: {e}")
            await asyncio.sleep(0.5)

        # ── 8. 点击创建 ──────────────────────────────────────────
        step("8/8 点击创建...")
        try:
            create_btn = page.locator("button:has-text('创建')").first
            await create_btn.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await create_btn.click()

            # 等待成功标志出现（弹窗提示或页面跳转）
            success_indicator = page.locator(
                "text=创建成功"
            ).or_(page.locator(
                "text=发布成功"
            )).or_(page.locator(
                "text=上架成功"
            ))
            try:
                await success_indicator.first.wait_for(
                    state="visible", timeout=10000,
                )
            except Exception:
                # 未检测到成功提示，可能页面结构不同，检查 URL 是否变化作为备选
                await asyncio.sleep(3)
                current_url = page.url
                if "edit" in current_url and "success" not in current_url:
                    return UploadResult(
                        product.folder, False,
                        "8/8 未检测到创建成功提示，可能上架失败",
                    )
        except Exception as e:
            return UploadResult(product.folder, False, f"8/8 点击创建失败: {e}")

        step("上架成功")
        return UploadResult(product.folder, True)

    async def batch_upload(
        self,
        products: list[ProductInfo],
        on_progress: ProgressCallback | None = None,
        on_step: StepCallback | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> list[UploadResult]:
        """批量上架产品。"""
        results: list[UploadResult] = []
        total = len(products)

        for i, product in enumerate(products):
            if is_cancelled and is_cancelled():
                for p in products[i:]:
                    results.append(UploadResult(p.folder, False, "已取消", title=p.title))
                break

            if on_step:
                on_step(f"正在上架 ({i + 1}/{total}): {product.title[:20]}...")

            result = await self.upload_product(product, on_step)
            results.append(result)

            if on_progress:
                on_progress(i + 1, total)

            # 产品间随机延迟，避免触发微店反自动化检测
            if i < total - 1 and not (is_cancelled and is_cancelled()):
                delay = random.uniform(3.0, 6.0)
                await asyncio.sleep(delay)

        return results
