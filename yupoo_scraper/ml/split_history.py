"""拆分历史管理 — SQLite 数据库记录拆分状态"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ..config import DATA_DIR


@dataclass
class SplitHistoryRecord:
    """拆分历史记录"""
    id: int
    folder: str
    split_time: str
    group_count: int
    image_count: int
    has_features: bool


class SplitHistoryDatabase:
    """SQLite 拆分历史数据库，管理拆分状态和特征"""

    def __init__(self, db_path: Path = DATA_DIR / "split_history.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """创建/打开数据库，初始化表结构"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS split_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                folder          TEXT NOT NULL UNIQUE,
                split_time      TEXT NOT NULL,
                group_count     INTEGER NOT NULL,
                image_count     INTEGER NOT NULL,
                has_features    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS split_groups (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                history_id      INTEGER NOT NULL,
                group_id        INTEGER NOT NULL,
                group_name      TEXT NOT NULL,
                FOREIGN KEY (history_id) REFERENCES split_history(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS split_images (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                history_id      INTEGER NOT NULL,
                group_id        INTEGER NOT NULL,
                file_path       TEXT NOT NULL,
                FOREIGN KEY (history_id) REFERENCES split_history(id) ON DELETE CASCADE
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def add_split_history(
        self,
        folder: str,
        split_time: str,
        group_count: int,
        image_count: int,
        has_features: bool = False
    ) -> int:
        """添加拆分历史记录，返回历史ID。

        如果该文件夹已有记录，先删除旧记录（CASCADE 自动清理关联的
        split_groups 和 split_images），再插入新记录。
        """
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            # 删除旧记录（如果存在），CASCADE 会自动删除关联数据
            self._conn.execute(
                "DELETE FROM split_history WHERE folder = ?", (folder,)
            )
            cursor = self._conn.execute(
                """INSERT INTO split_history
                   (folder, split_time, group_count, image_count, has_features)
                   VALUES (?, ?, ?, ?, ?)""",
                (folder, split_time, group_count, image_count, 1 if has_features else 0)
            )
            self._conn.commit()
            return cursor.lastrowid

    def add_split_group(
        self,
        history_id: int,
        group_id: int,
        group_name: str
    ) -> None:
        """添加拆分分组信息"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            self._conn.execute(
                """INSERT INTO split_groups
                   (history_id, group_id, group_name)
                   VALUES (?, ?, ?)""",
                (history_id, group_id, group_name)
            )
            self._conn.commit()

    def add_split_image(
        self,
        history_id: int,
        group_id: int,
        file_path: str,
    ) -> None:
        """添加拆分图片信息"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            self._conn.execute(
                """INSERT INTO split_images
                   (history_id, group_id, file_path)
                   VALUES (?, ?, ?)""",
                (history_id, group_id, file_path)
            )
            self._conn.commit()

    def add_split_history_batch(
        self,
        folder: str,
        split_time: str,
        group_count: int,
        image_count: int,
        has_features: bool,
        groups: List[Tuple[int, str]],
        images: List[Tuple[int, str]],
        sub_folders: List[Tuple[str, str, int, int, bool]],
    ) -> int:
        """单事务批量写入拆分历史（主记录 + 分组 + 图片 + 子文件夹记录）。

        Args:
            groups: [(group_id, group_name), ...]
            images: [(group_id, file_path), ...]
            sub_folders: [(folder, split_time, group_count, image_count, has_features), ...]

        Returns:
            主记录的 history_id。
        """
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            # 删除旧记录
            self._conn.execute(
                "DELETE FROM split_history WHERE folder = ?", (folder,)
            )
            cursor = self._conn.execute(
                """INSERT INTO split_history
                   (folder, split_time, group_count, image_count, has_features)
                   VALUES (?, ?, ?, ?, ?)""",
                (folder, split_time, group_count, image_count, 1 if has_features else 0)
            )
            history_id = cursor.lastrowid

            # 批量插入分组
            self._conn.executemany(
                """INSERT INTO split_groups (history_id, group_id, group_name)
                   VALUES (?, ?, ?)""",
                [(history_id, gid, gname) for gid, gname in groups]
            )

            # 批量插入图片
            self._conn.executemany(
                """INSERT INTO split_images (history_id, group_id, file_path)
                   VALUES (?, ?, ?)""",
                [(history_id, gid, fpath) for gid, fpath in images]
            )

            # 批量插入子文件夹记录（原子 upsert，避免 DELETE+INSERT 间隙丢数据）
            self._conn.executemany(
                """INSERT OR REPLACE INTO split_history
                   (folder, split_time, group_count, image_count, has_features)
                   VALUES (?, ?, ?, ?, ?)""",
                [(sf, st, gc, ic, 1 if hf else 0) for sf, st, gc, ic, hf in sub_folders]
            )

            self._conn.commit()
            return history_id

    def get_split_history(self, folder: str) -> Optional[SplitHistoryRecord]:
        """按文件夹路径查询拆分历史"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            row = self._conn.execute(
                """SELECT id, folder, split_time, group_count, image_count, has_features
                   FROM split_history WHERE folder = ?""",
                (folder,)
            ).fetchone()
        if row is None:
            return None
        return SplitHistoryRecord(
            id=row[0],
            folder=row[1],
            split_time=row[2],
            group_count=row[3],
            image_count=row[4],
            has_features=bool(row[5])
        )

    def get_split_groups(self, history_id: int) -> List[Tuple[int, str]]:
        """获取指定历史的分组信息"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            rows = self._conn.execute(
                """SELECT group_id, group_name
                   FROM split_groups WHERE history_id = ?
                   ORDER BY group_id""",
                (history_id,)
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_split_images(self, history_id: int, group_id: int) -> List[str]:
        """获取指定历史和分组的图片路径"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            rows = self._conn.execute(
                """SELECT file_path
                   FROM split_images WHERE history_id = ? AND group_id = ?""",
                (history_id, group_id)
            ).fetchall()
        return [row[0] for row in rows]

    def update_has_features(self, history_id: int, has_features: bool) -> None:
        """更新特征状态"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            self._conn.execute(
                """UPDATE split_history
                   SET has_features = ?
                   WHERE id = ?""",
                (1 if has_features else 0, history_id)
            )
            self._conn.commit()

    def delete_split_history(self, folder: str) -> None:
        """删除指定文件夹的拆分历史"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            self._conn.execute(
                "DELETE FROM split_history WHERE folder = ?",
                (folder,)
            )
            self._conn.commit()

    def get_all_split_history(self) -> List[SplitHistoryRecord]:
        """获取所有拆分历史记录"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, folder, split_time, group_count, image_count, has_features
                   FROM split_history ORDER BY split_time DESC"""
            ).fetchall()
        return [
            SplitHistoryRecord(
                id=row[0],
                folder=row[1],
                split_time=row[2],
                group_count=row[3],
                image_count=row[4],
                has_features=bool(row[5])
            )
            for row in rows
        ]


# 全局拆分历史数据库实例
_split_history_instance: Optional[SplitHistoryDatabase] = None
_split_history_lock = threading.Lock()


def get_split_history() -> SplitHistoryDatabase:
    """获取全局拆分历史数据库实例（线程安全）"""
    global _split_history_instance
    if _split_history_instance is None:
        with _split_history_lock:
            if _split_history_instance is None:
                inst = SplitHistoryDatabase()
                inst.open()
                _split_history_instance = inst
    return _split_history_instance
