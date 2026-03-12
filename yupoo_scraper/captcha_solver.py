"""滑块验证码识别 — 第三方打码平台集成

支持平台:
  - ttshitu (图鉴, ttshitu.com) — 国内推荐，便宜（~2分/次）
  - twocaptcha (2Captcha, 2captcha.com) — 国际平台

用法:
  solver = create_solver("ttshitu", username="xxx", password="xxx")
  gap_x = await solver.recognize_gap(screenshot_bytes)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random

import aiohttp

logger = logging.getLogger(__name__)


# ── 滑块拖拽模拟 ─────────────────────────────────────────────────


def generate_human_track(distance: int) -> list[tuple[int, int, int]]:
    """生成模拟人类的拖拽轨迹。

    Returns:
        [(dx, dy, duration_ms), ...] 相对位移序列。
        dx 之和精确等于 distance（过冲回调除外，其净额为 0）。
    """
    if distance <= 0:
        return []

    track: list[tuple[int, int, int]] = []
    traveled = 0  # 整数累加，保证精确

    # 先加速到 60%，再减速到终点
    mid = distance * 0.6

    while traveled < distance:
        remaining = distance - traveled
        if traveled < mid:
            # 加速阶段：步长 3~8
            step_f = random.uniform(3, 8)
        else:
            # 减速阶段：步长 1~4，越接近越小
            step_f = max(1.0, random.uniform(1, 4) * (remaining / distance))

        dx = max(1, min(round(step_f), remaining))
        traveled += dx

        # 轻微 Y 轴抖动
        dy = random.randint(-2, 2)
        # 每步间隔 8~25ms
        dt = random.randint(8, 25)
        track.append((dx, dy, dt))

    # 可能略微过冲再回调（净额为 0，不影响总距离）
    if random.random() < 0.3:
        overshoot = random.randint(2, 6)
        track.append((overshoot, 0, random.randint(30, 60)))
        track.append((-overshoot, 0, random.randint(50, 100)))

    return track


# ── 打码平台基类 ──────────────────────────────────────────────────


class CaptchaSolver:
    """打码平台抽象基类。

    在 async with 块中使用以复用 HTTP 连接，或直接调用（每次新建连接）。
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None
        return False

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 session，或创建临时 session。"""
        if self._session and not self._session.closed:
            return self._session
        # 未通过 async with 使用时，创建新 session（由调用方负责关闭）
        self._session = aiohttp.ClientSession()
        return self._session

    async def recognize_gap(self, image_bytes: bytes) -> int:
        """识别滑块缺口的 X 坐标。

        Args:
            image_bytes: 验证码截图的 PNG/JPEG 字节。

        Returns:
            缺口的 X 像素坐标（相对于图片左侧）。
        """
        raise NotImplementedError



# ── TTShitu (图鉴) ────────────────────────────────────────────────


class TTShituSolver(CaptchaSolver):
    """图鉴打码平台 (ttshitu.com)。

    注册后获取 username/password。
    滑块验证码 typeId = 27（滑块拼图，返回 X 坐标）。
    """

    API_URL = "https://api.ttshitu.com/predict"

    def __init__(self, username: str, password: str, type_id: int = 27):
        super().__init__()
        self._username = username
        self._password = password
        self._type_id = type_id  # 27=滑块拼图

    async def recognize_gap(self, image_bytes: bytes) -> int:
        b64 = base64.b64encode(image_bytes).decode()

        payload = {
            "username": self._username,
            "password": self._password,
            "typeid": self._type_id,
            "image": b64,
        }

        session = await self._get_session()
        async with session.post(
            self.API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()

        if not data.get("success"):
            raise RuntimeError(
                f"图鉴识别失败: {data.get('message', '未知错误')}"
            )

        result = data["data"]["result"]
        logger.info("图鉴返回结果: %s", result)

        # 返回格式可能是 "x,y" 或纯数字
        if "," in str(result):
            x_str = str(result).split(",")[0]
            return int(float(x_str))
        return int(float(result))


# ── 2Captcha ──────────────────────────────────────────────────────


class TwoCaptchaSolver(CaptchaSolver):
    """2Captcha 打码平台 (2captcha.com)。

    滑块验证码用 coordinatescaptcha 类型。
    """

    SUBMIT_URL = "https://2captcha.com/in.php"
    RESULT_URL = "https://2captcha.com/res.php"

    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key

    async def recognize_gap(self, image_bytes: bytes) -> int:
        b64 = base64.b64encode(image_bytes).decode()

        # 提交任务
        submit_data = {
            "key": self._api_key,
            "method": "base64",
            "coordinatescaptcha": "1",
            "body": b64,
            "json": "1",
        }

        session = await self._get_session()

        # 提交
        async with session.post(
            self.SUBMIT_URL,
            data=submit_data,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()

        if data.get("status") != 1:
            raise RuntimeError(
                f"2Captcha 提交失败: {data.get('request', '未知错误')}"
            )

        task_id = data["request"]
        logger.info("2Captcha 任务已提交: %s", task_id)

        # 官方建议首次等 5 秒再轮询，之后每 2 秒（最多 ~65 秒）
        await asyncio.sleep(5)
        for _ in range(30):
            await asyncio.sleep(2)

            async with session.get(
                self.RESULT_URL,
                params={
                    "key": self._api_key,
                    "action": "get",
                    "id": task_id,
                    "json": "1",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                result = await resp.json()

            if result.get("status") == 1:
                # 返回格式: "coordinates:x=123,y=456"
                answer = result["request"]
                logger.info("2Captcha 返回结果: %s", answer)
                if "x=" in answer:
                    x_part = answer.split("x=")[1].split(",")[0]
                    return int(float(x_part))
                return int(float(answer))

            if result.get("request") != "CAPCHA_NOT_READY":
                raise RuntimeError(
                    f"2Captcha 识别失败: {result.get('request')}"
                )

        raise RuntimeError("2Captcha 识别超时")


# ── 工厂函数 ──────────────────────────────────────────────────────


def create_solver(
    provider: str,
    *,
    username: str = "",
    password: str = "",
    api_key: str = "",
) -> CaptchaSolver:
    """创建打码平台实例。

    Args:
        provider: "ttshitu" 或 "twocaptcha"。
        username: 图鉴用户名。
        password: 图鉴密码。
        api_key: 2Captcha API Key。

    Returns:
        CaptchaSolver 实例。
    """
    provider = provider.lower().strip()
    if provider == "ttshitu":
        if not username or not password:
            raise ValueError("图鉴平台需要 username 和 password")
        return TTShituSolver(username, password)
    elif provider in ("twocaptcha", "2captcha"):
        if not api_key:
            raise ValueError("2Captcha 需要 api_key")
        return TwoCaptchaSolver(api_key)
    else:
        raise ValueError(f"不支持的打码平台: {provider!r}，可选: ttshitu, twocaptcha")
