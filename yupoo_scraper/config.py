"""全局配置"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

# ── 回调类型别名 ─────────────────────────────────────────────
ProgressCallback = Callable[[int, int], None]    # (current, total)
StatusCallback = Callable[[str], None]           # (message,)

# ── Yupoo 请求配置 ──────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# Referer 必须是 *.yupoo.com 子域名，裸域名会被拒绝 (HTTP 567)
REFERER_TEMPLATE = "https://{username}.x.yupoo.com/"


def make_headers(username: str) -> dict[str, str]:
    """构建带 Referer 的请求头。"""
    headers = dict(HEADERS)
    headers["Referer"] = REFERER_TEMPLATE.format(username=username)
    return headers

# ── 并发 / 限速 ────────────────────────────────────────────────
MAX_CONCURRENCY = 3          # 同时下载图片数 (过高易触发 429)
REQUEST_DELAY = (0.8, 1.5)   # 请求间随机延迟 (秒)
MAX_RETRIES = 5              # 失败重试次数 (含 429 重试)
RETRY_BACKOFF = 2            # 指数退避基数
REQUEST_TIMEOUT = 30         # 单次请求超时 (秒)

# ── HTTP 连接池 ────────────────────────────────────────────────
HTTP_CONN_LIMIT = 20         # 全局最大连接数
HTTP_CONN_LIMIT_PER_HOST = 10 # 单主机最大连接数

# ── 翻页并发（500K 相册优化）─────────────────────────────────
PAGE_CONCURRENCY = 5         # 同时获取的页数（不超过单主机连接数）
PAGE_BATCH_DELAY = (1.5, 3.0)  # 页面批次间延迟（秒），补偿并发突发


def retry_wait(attempt: int, resp_headers: dict | None = None) -> float:
    """计算重试等待时间（秒），优先使用 Retry-After 头。"""
    import random as _random

    if resp_headers:
        retry_after = resp_headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return int(retry_after) + _random.random()
    return RETRY_BACKOFF ** attempt + _random.random()

# ── 图片格式 ───────────────────────────────────────────────────
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# ── 图片尺寸 ───────────────────────────────────────────────────
IMAGE_SIZE = "big"           # big ≈ 1080px，是最大的渲染尺寸

# ── 基础目录（打包后用 exe 所在目录，开发时用项目根目录）───────
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# ── 保存路径 ───────────────────────────────────────────────────
DEFAULT_DOWNLOAD_DIR = BASE_DIR / "downloads"

# ── 用户数据 ───────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
URL_HISTORY_FILE = DATA_DIR / "url_history.json"
URL_HISTORY_MAX = 20  # 最多记住多少条 URL
UPLOAD_MARKS_FILE = DATA_DIR / "upload_marks.json"
DOWNLOAD_LOG_FILE = DATA_DIR / "download_log.json"

# ── HTML 选择器 (Yupoo 页面) ───────────────────────────────────
SEL_ALBUM_LINK = "a.album3__main"
SEL_ALBUM_TITLE = "div.album3__title"
SEL_ALBUM_LINK_GALLERY = "a.album__main"      # 画廊/分类视图
SEL_ALBUM_TITLE_GALLERY = "div.album__title"   # 画廊/分类视图
SEL_PAGINATION_MAX = 'form.pagination__jumpwrap input[name="page"]'
SEL_IMAGE_CONTAINER = "div.showalbum__children"
SEL_CATEGORY_LINK = "ul.showheader__category > a"

# ── 分类 URL 模板 ───────────────────────────────────────────────
CATEGORY_ALBUMS_URL = "https://x.yupoo.com/photos/{username}/albums?tab=gallery&referrercate={category_id}"

# ── CDN 域名 ──────────────────────────────────────────────────
CDN_HOSTS = ["photo.yupoo.com", "photo3.yupoo.com"]

# ── ML / 产品拆分 ────────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# HSV 直方图 bins: H=16, S=3, V=2 → 16×3×2 = 96 维
HSV_BINS = (16, 3, 2)
HSV_RANGES = ([0, 180], [0, 256], [0, 256])

# HSV 前景遮罩阈值 — 排除背景像素（纯黑/纯白底）
HSV_FG_V_MIN = 30    # V 通道最小值（过暗视为背景）
HSV_FG_V_MAX = 240   # V 通道最大值（配合 S 判断纯白）
HSV_FG_S_MIN = 15    # S 通道最小值（低饱和度+高亮度 = 白色背景）

# HSV 权重 — 放大颜色特征在组合距离中的贡献
HSV_WEIGHT = 1.5

# 特征维度
CLIP_DIM = 512
HSV_DIM = math.prod(HSV_BINS)        # 16×3×2 = 96
COMBINED_DIM = CLIP_DIM + HSV_DIM    # 512 + 96 = 608

# 聚类参数
CLUSTER_THRESHOLD_DEFAULT = 0.35
CLUSTER_THRESHOLD_MIN = 0.10
CLUSTER_THRESHOLD_MAX = 0.80

# ── 产品查重 ──────────────────────────────────────────────────
DEDUP_DB_PATH = DATA_DIR / "products.db"
DEDUP_FAISS_PATH = DATA_DIR / "product_index.faiss"
DEDUP_EMBEDDING_DIM = COMBINED_DIM  # 608（CLIP + HSV，区分不同颜色）
DEDUP_THRESHOLD_AUTO = 0.92      # >= 自动标记重复
DEDUP_THRESHOLD_REVIEW = 0.85    # >= 待审核
DEDUP_SEARCH_K = 5               # FAISS 返回 Top-K
DEDUP_BATCH_SIZE = 32            # CLIP 推理批次（默认，实际由设备决定）

if DEDUP_THRESHOLD_REVIEW >= DEDUP_THRESHOLD_AUTO:
    raise ValueError(
        f"REVIEW 阈值({DEDUP_THRESHOLD_REVIEW}) 必须小于 AUTO 阈值({DEDUP_THRESHOLD_AUTO})"
    )

# CLIP 推理批大小（按设备区分）
CLIP_BATCH_SIZE_GPU = 64         # GPU 批大小
CLIP_BATCH_SIZE_CPU = 16         # CPU 批大小

# FAISS 索引策略
FAISS_IVF_THRESHOLD = 5000      # 产品数 >= 此值时使用 IVFFlat（否则 FlatIP）
FAISS_IVFPQ_THRESHOLD = 500000  # 产品数 >= 此值时使用 IVFPQ（极致压缩）
FAISS_PQ_M = 32                 # PQ 子量化器数量（dim 必须能被 M 整除，608/32=19）
FAISS_PQ_NBITS = 8              # PQ 每个子向量的编码位数
FAISS_NPROBE = 16               # IVF 搜索时探测的聚类数（越大越精确但越慢）
FAISS_REBUILD_RATIO = 0.2       # 累计新增 > 此比例时触发 IVF 重新训练
FAISS_SAVE_INTERVAL = 1000      # 增量持久化间隔（每新增 N 条向量保存一次）
REGISTER_BATCH_SIZE = 500       # 批量注册时每批产品数（DB + FAISS 批写入）

# ── 相册去重 (采集阶段) ─────────────────────────────────────────
COVER_PHASH_SIZE = 16            # pHash 精度 (16×16 → 256-bit)
COVER_PHASH_THRESHOLD = 8       # 汉明距离 ≤ 此值视为重复封面
COVER_DOWNLOAD_CONCURRENCY = 5  # 封面下载并发数

# ── 图片处理 (1:1 补齐) ─────────────────────────────────────────
SQUARE_FILL_COLOR = (255, 255, 255)  # 填充颜色 RGB (白色)
SQUARE_JPEG_QUALITY = 95             # JPEG 保存质量

# ── 白底检测 / 主图选择 ───────────────────────────────────────
WHITE_BG_THRESHOLD = 240             # 白色像素 RGB 阈值
WHITE_BG_EDGE_RATIO = 0.90           # 边缘白色像素比例阈值
ORIGINALS_SUBFOLDER = "originals"    # 原图备份子目录

# GUI
THUMBNAIL_SIZE = 120  # 缩略图像素尺寸

# ── 微店上架 ──────────────────────────────────────────────────
WEIDIAN_PUBLISH_URL = "https://d.weidian.com/weidian-pc/weidian-loader/#/pc-vue-item/item/edit"
WEIDIAN_CDP_URL = "http://localhost:9222"
WEIDIAN_CDP_PORT = 9222
LOGIN_CHECK_URL = "https://d.weidian.com/weidian-pc/weidian-loader/"

# Chrome 浏览器用户数据目录（保持登录状态）
CHROME_USER_DATA_DIR = DATA_DIR / "chrome_profile"

# 上架超时配置（毫秒）
UPLOAD_STEP_TIMEOUT_MS = 30000           # 每步超时
UPLOAD_IMAGE_POLL_INTERVAL_MS = 2000    # 图片上传轮询间隔
UPLOAD_IMAGE_MIN_WAIT_PER_IMAGE = 5000  # 每张图片最小等待时间(毫秒)
UPLOAD_IMAGE_MAX_WAIT_BASE = 60000       # 基础最大等待时间(毫秒)

# 上架延迟配置（秒）- 防止反自动化检测
UPLOAD_STEP_DELAY_MIN = 0.5              # 最小延迟（可设为0禁用）
UPLOAD_STEP_DELAY_MAX = 1.5              # 最大延迟

# 上架重试配置
UPLOAD_RETRY_MAX = 3                     # 最大重试次数
UPLOAD_RETRY_DELAY = 5                   # 重试间隔(秒)

# 代理配置
WEIDIAN_PROXY_URL = ""                   # 代理地址，如 "http://127.0.0.1:7890"

# ── 验证码打码平台 ────────────────────────────────────────────────
# 支持平台: "ttshitu" (图鉴) 或 "twocaptcha" (2Captcha)，留空则跳过自动识别
# 注意: 以下值为运行时默认值，实际由 GUI 设置对话框管理并持久化到 JSON 文件
CAPTCHA_SETTINGS_FILE = DATA_DIR / "captcha_settings.json"
CAPTCHA_PROVIDER = ""                    # "ttshitu" 或 "twocaptcha"
CAPTCHA_TTSHITU_USERNAME = ""            # 图鉴用户名
CAPTCHA_TTSHITU_PASSWORD = ""            # 图鉴密码
CAPTCHA_TWOCAPTCHA_KEY = ""              # 2Captcha API Key
CAPTCHA_MAX_RETRIES = 3                  # 验证码识别最大重试次数
CAPTCHA_DETECT_TIMEOUT_MS = 5000         # 验证码检测等待超时(毫秒)

# 日志配置
LOG_FILE = DATA_DIR / "app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024        # 单个日志文件最大 10MB
LOG_BACKUP_COUNT = 5                    # 保留的旧日志文件数量
LOG_LEVEL = "INFO"                      # 日志级别: DEBUG, INFO, WARNING, ERROR

# 图片限制
MAIN_IMAGE_MAX = 5                      # 主图最多张数（微店限制15，计划取前5张）
DETAIL_IMAGE_MAX = 100                  # 详情图最多张数

# 标题生成词池
TITLE_PREFIXES = ["新款", "爆款", "热卖", "推荐"]
TITLE_STYLES = ["潮流", "休闲", "时尚", "经典", "街头", "复古"]
TITLE_SUFFIXES = ["百搭", "时尚单品", "穿搭必备", "好物"]

# CLIP zero-shot 产品类别
CATEGORY_PROMPTS = {
    "shoes":   ("a photo of shoes",       "鞋子"),
    "handbag": ("a photo of a handbag",   "包包"),
    "jacket":  ("a photo of a jacket",    "外套"),
    "tshirt":  ("a photo of a t-shirt",   "T恤"),
    "pants":   ("a photo of pants",       "裤子"),
    "hoodie":  ("a photo of a hoodie",    "卫衣"),
    "hat":     ("a photo of a hat",       "帽子"),
    "watch":   ("a photo of a watch",     "手表"),
    "belt":    ("a photo of a belt",      "腰带"),
    "scarf":   ("a photo of a scarf",     "围巾"),
}

# ── 配置完整性校验 ────────────────────────────────────────────
if DEDUP_EMBEDDING_DIM % FAISS_PQ_M != 0:
    raise ValueError(
        f"DEDUP_EMBEDDING_DIM({DEDUP_EMBEDDING_DIM}) 必须能被 FAISS_PQ_M({FAISS_PQ_M}) 整除"
    )
