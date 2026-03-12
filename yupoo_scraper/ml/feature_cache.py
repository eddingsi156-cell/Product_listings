"""特征缓存管理 — SQLite 后端，支持百万级图片"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from ..config import DATA_DIR


class FeatureCache:
    """特征缓存管理，使用 SQLite 存储，避免重复计算。

    相比旧版内存字典 + JSON 索引 + pickle 文件方案：
    - 无 max_size 限制，按 expiration_days 过期淘汰
    - 启动快（不需加载全部索引到内存）
    - 支持并发读（WAL 模式）
    """

    def __init__(self,
                 cache_dir: Path = DATA_DIR / "feature_cache",
                 expiration_days: int = 90,
                 ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiration_seconds = expiration_days * 24 * 3600
        self._db_path = self.cache_dir / "features.db"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._dirty_count = 0
        self._CLEAN_INTERVAL = 500  # 每 500 次 set() 清理一次过期

        self._open_db()
        self._migrate_from_legacy()

    def _open_db(self) -> None:
        """打开/创建 SQLite 数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS features (
                key         TEXT NOT NULL,
                namespace   TEXT NOT NULL DEFAULT '',
                mtime       REAL NOT NULL,
                timestamp   REAL NOT NULL,
                feature     BLOB NOT NULL,
                PRIMARY KEY (key, namespace)
            );
            CREATE INDEX IF NOT EXISTS idx_timestamp ON features(timestamp);
        """)
        self._conn.commit()

    def _migrate_from_legacy(self) -> None:
        """从旧版 index.json + pickle 文件迁移数据（一次性）。"""
        index_file = self.cache_dir / "index.json"
        if not index_file.exists():
            return

        logger.info("检测到旧版缓存索引，开始迁移到 SQLite...")
        migrated = 0
        failed = 0

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, info in data.items():
                mtime = info.get("mtime", 0.0)
                timestamp = info.get("timestamp", time.time())
                features_path = info.get("features_path", "")
                features_file = self.cache_dir / features_path

                if not features_file.exists():
                    continue

                try:
                    import pickle
                    with open(features_file, "rb") as f:
                        features = pickle.load(f)

                    if isinstance(features, np.ndarray):
                        self._conn.execute(
                            "INSERT OR REPLACE INTO features (key, namespace, mtime, timestamp, feature) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (key, "", mtime, timestamp, features.astype(np.float32).tobytes()),
                        )
                        migrated += 1
                except Exception as e:
                    logger.debug("迁移缓存条目失败 %s: %s", key[:50], e)
                    failed += 1

            self._conn.commit()

            # 删除旧文件
            index_file.unlink(missing_ok=True)
            # 清理旧 pickle 文件
            for pkl_file in self.cache_dir.glob("*.pkl"):
                try:
                    pkl_file.unlink()
                except Exception:
                    pass

            logger.info("缓存迁移完成: 成功 %d，失败 %d", migrated, failed)
        except Exception as e:
            logger.warning("缓存迁移失败，将使用空缓存: %s", e)

    def _get_cache_key(self, image_path: Path, namespace: str = "") -> str:
        """获取缓存键，使用绝对路径。"""
        return str(image_path.absolute())

    def get(self, image_path: Path, namespace: str = "") -> Optional[np.ndarray]:
        """获取缓存的特征。

        Args:
            image_path: 图片路径。
            namespace: 命名空间，用于区分不同类型的特征（如 ""=CLIP, "combined"=组合特征）。
        """
        if not image_path.exists():
            return None

        key = self._get_cache_key(image_path)
        current_mtime = image_path.stat().st_mtime

        with self._lock:
            row = self._conn.execute(
                "SELECT mtime, timestamp, feature FROM features WHERE key = ? AND namespace = ?",
                (key, namespace),
            ).fetchone()

            if row is None:
                return None

            cached_mtime, timestamp, feature_blob = row

            # 检查文件是否已修改或已过期
            if current_mtime != cached_mtime or time.time() - timestamp > self.expiration_seconds:
                self._conn.execute(
                    "DELETE FROM features WHERE key = ? AND namespace = ?",
                    (key, namespace),
                )
                self._conn.commit()
                return None

        # 验证 BLOB 完整性：长度必须是 float32 的整数倍
        if len(feature_blob) == 0 or len(feature_blob) % 4 != 0:
            logger.warning("缓存 BLOB 损坏 (大小=%d bytes), key=%s", len(feature_blob), key[:80])
            with self._lock:
                self._conn.execute(
                    "DELETE FROM features WHERE key = ? AND namespace = ?",
                    (key, namespace),
                )
                self._conn.commit()
            return None
        return np.frombuffer(feature_blob, dtype=np.float32).copy()

    def set(self, image_path: Path, features: np.ndarray, namespace: str = "") -> None:
        """设置缓存的特征。

        Args:
            image_path: 图片路径。
            features: 特征向量。
            namespace: 命名空间，用于区分不同类型的特征。
        """
        if not image_path.exists():
            return

        key = self._get_cache_key(image_path)
        current_mtime = image_path.stat().st_mtime
        current_timestamp = time.time()

        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO features (key, namespace, mtime, timestamp, feature) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, namespace, current_mtime, current_timestamp,
                     features.astype(np.float32).tobytes()),
                )
                self._conn.commit()
                self._dirty_count += 1

                # 定期清理过期条目
                if self._dirty_count >= self._CLEAN_INTERVAL:
                    self._clean_expired_locked()
                    self._dirty_count = 0
            except Exception as e:
                logger.warning("保存缓存特征失败: %s", e)

    def _clean_expired_locked(self) -> None:
        """清理过期缓存（调用时必须持有 _lock）。"""
        cutoff = time.time() - self.expiration_seconds
        try:
            self._conn.execute("DELETE FROM features WHERE timestamp < ?", (cutoff,))
            self._conn.commit()
        except Exception as e:
            logger.warning("清理过期缓存失败: %s", e)

    def flush(self) -> None:
        """强制提交（SQLite 模式下主要用于清理过期条目）。"""
        with self._lock:
            self._clean_expired_locked()

    def clear(self) -> None:
        """清除所有缓存。"""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM features")
                self._conn.commit()
            except Exception as e:
                logger.warning("清除缓存失败: %s", e)

    def clear_namespace(self, namespace: str) -> None:
        """清除指定命名空间的缓存。"""
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM features WHERE namespace = ?", (namespace,),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("清除命名空间 %s 缓存失败: %s", namespace, e)

    def size(self) -> int:
        """返回缓存条目数。"""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM features").fetchone()
        return row[0]

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None


# 全局缓存实例
_cache_instance: Optional[FeatureCache] = None
_cache_lock = threading.Lock()


def get_cache() -> FeatureCache:
    """获取全局缓存实例（双检锁，线程安全）。"""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = FeatureCache()
    return _cache_instance
