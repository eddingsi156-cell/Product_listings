"""产品查重引擎 — SQLite 产品库 + FAISS 向量索引 + 查重逻辑"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

logger = logging.getLogger(__name__)

from ..config import (
    DEDUP_BATCH_SIZE,
    DEDUP_DB_PATH,
    DEDUP_EMBEDDING_DIM,
    DEDUP_FAISS_PATH,
    DEDUP_SEARCH_K,
    DEDUP_THRESHOLD_AUTO,
    DEDUP_THRESHOLD_REVIEW,
)
from .feature_extractor import get_extractor
from ..image_processor import list_images


# ── 数据类 ─────────────────────────────────────────────────────

class DedupStatus(Enum):
    """查重状态"""
    NEW = "new"             # 新产品（无相似项）
    REVIEW = "review"       # 待审核（相似度 0.85~0.92）
    DUPLICATE = "duplicate" # 自动标记重复（>= 0.92）


@dataclass
class DedupMatch:
    """单个匹配结果"""
    existing_product: ProductRecord
    similarity: float


@dataclass
class ProductRecord:
    """产品数据库记录"""
    id: int
    name: str
    store: str
    folder: str
    source_url: str
    download_date: str
    image_count: int


@dataclass
class DedupScanItem:
    """批量扫描中单个文件夹的结果"""
    folder: Path
    name: str
    image_count: int
    status: DedupStatus
    best_match: DedupMatch | None
    all_matches: list[DedupMatch]
    embedding: np.ndarray | None  # (DEDUP_EMBEDDING_DIM,) 产品级向量
    error: str | None  # None = 成功


# ── ProductDatabase — SQLite 管理 ──────────────────────────────

class ProductDatabase:
    """SQLite 产品数据库，管理产品记录和图片嵌入。"""

    def __init__(self, db_path: Path = DEDUP_DB_PATH):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()  # 保护所有数据库读写操作

    def open(self) -> None:
        """创建/打开数据库，初始化表结构。"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                store       TEXT NOT NULL DEFAULT '',
                folder      TEXT NOT NULL UNIQUE,
                source_url  TEXT NOT NULL DEFAULT '',
                download_date TEXT NOT NULL DEFAULT '',
                image_count INTEGER NOT NULL DEFAULT 0,
                embedding   BLOB
            );

            CREATE TABLE IF NOT EXISTS images (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id    INTEGER NOT NULL,
                file_path     TEXT NOT NULL,
                clip_embedding BLOB,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def add_product(
        self,
        name: str,
        store: str,
        folder: str,
        source_url: str,
        download_date: str,
        image_count: int,
        embedding: np.ndarray,
    ) -> int:
        """添加产品，返回新产品 ID。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO products
                   (name, store, folder, source_url, download_date, image_count, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                # embedding 存储为 float32 原始字节，维度由 DEDUP_EMBEDDING_DIM 决定
                (name, store, folder, source_url, download_date, image_count,
                 embedding.tobytes()),
            )
            self._conn.commit()
            return cursor.lastrowid

    def get_product(self, product_id: int) -> ProductRecord | None:
        """按 ID 查询产品。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, store, folder, source_url, download_date, image_count "
                "FROM products WHERE id = ?", (product_id,),
            ).fetchone()
        if row is None:
            return None
        return ProductRecord(*row)

    def get_product_by_folder(self, folder: str) -> ProductRecord | None:
        """按文件夹路径查询产品。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, store, folder, source_url, download_date, image_count "
                "FROM products WHERE folder = ?", (folder,),
            ).fetchone()
        if row is None:
            return None
        return ProductRecord(*row)

    def remove_product(self, product_id: int) -> None:
        """删除产品（级联删除关联图片）。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            self._conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
            self._conn.commit()

    def get_all_products(self) -> list[ProductRecord]:
        """获取所有产品记录。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, store, folder, source_url, download_date, image_count "
                "FROM products ORDER BY id",
            ).fetchall()
        return [ProductRecord(*r) for r in rows]

    def get_product_count(self) -> int:
        """获取产品总数。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM products").fetchone()
        return row[0]

    def get_all_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """获取所有产品的 (id, embedding) 列表，用于重建 FAISS。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM products WHERE embedding IS NOT NULL",
            ).fetchall()
        result = []
        for pid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            result.append((pid, emb))
        return result

    def get_folders_set(self) -> set[str]:
        """获取所有已注册的文件夹路径集合。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            rows = self._conn.execute("SELECT folder FROM products").fetchall()
        return {r[0] for r in rows}


# ── FaissIndex — FAISS 封装 ────────────────────────────────────

class FaissIndex:
    """FAISS IndexIDMap(IndexFlatIP) 封装，内积 = 余弦相似度（向量已 L2 归一化）。"""

    def __init__(self, dim: int = DEDUP_EMBEDDING_DIM):
        self._dim = dim
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(
            faiss.IndexFlatIP(dim),
        )
        self._lock = threading.Lock()

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add(self, product_id: int, embedding: np.ndarray) -> None:
        """添加单个向量（线程安全）。"""
        vec = embedding.reshape(1, -1).astype(np.float32)
        ids = np.array([product_id], dtype=np.int64)
        with self._lock:
            self._index.add_with_ids(vec, ids)

    def search(self, query: np.ndarray, k: int = DEDUP_SEARCH_K) -> list[tuple[int, float]]:
        """搜索最相似的 k 个产品，返回 [(product_id, similarity)]（线程安全）。"""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            actual_k = min(k, self._index.ntotal)
            vec = query.reshape(1, -1).astype(np.float32)
            distances, ids = self._index.search(vec, actual_k)
        results = []
        for i in range(actual_k):
            pid = int(ids[0][i])
            sim = float(distances[0][i])
            if pid >= 0:  # -1 表示无结果
                results.append((pid, sim))
        return results

    def remove(self, product_id: int) -> None:
        """移除指定 ID 的向量（线程安全）。"""
        ids = np.array([product_id], dtype=np.int64)
        with self._lock:
            self._index.remove_ids(ids)

    def remove_batch(self, product_ids: list[int]) -> None:
        """批量移除多个 ID 的向量（线程安全）。"""
        if not product_ids:
            return
        ids = np.array(product_ids, dtype=np.int64)
        with self._lock:
            self._index.remove_ids(ids)

    def save(self, path: Path = DEDUP_FAISS_PATH) -> None:
        """保存索引到文件（原子写入：先写临时文件再替换）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        faiss.write_index(self._index, str(tmp_path))
        os.replace(str(tmp_path), str(path))

    def load(self, path: Path = DEDUP_FAISS_PATH) -> bool:
        """从文件加载索引，成功返回 True。"""
        if not path.exists():
            return False
        loaded = faiss.read_index(str(path))
        if not isinstance(loaded, faiss.IndexIDMap):
            logger.warning("FAISS 索引类型不匹配（期望 IndexIDMap，实际 %s），将重建",
                           type(loaded).__name__)
            return False
        if loaded.d != self._dim:
            logger.warning("FAISS 索引维度不匹配（文件 %d 维，期望 %d 维），将重建",
                           loaded.d, self._dim)
            return False
        self._index = loaded
        return True

    def rebuild_from_db(self, embeddings: list[tuple[int, np.ndarray]]) -> None:
        """从数据库记录重建索引。"""
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
        if not embeddings:
            return
        ids = np.array([pid for pid, _ in embeddings], dtype=np.int64)
        vecs = np.vstack([emb.reshape(1, -1) for _, emb in embeddings]).astype(np.float32)
        self._index.add_with_ids(vecs, ids)


# ── Deduplicator — 门面 ───────────────────────────────────────

class Deduplicator:
    """产品查重门面类：协调 DB、FAISS、CLIP 模型。"""

    def __init__(self):
        self._db = ProductDatabase()
        self._index = FaissIndex()
        self._initialized = False

    @property
    def product_count(self) -> int:
        """已收录的产品数量。"""
        if not self._initialized:
            return 0
        return self._db.get_product_count()

    def initialize(self) -> None:
        """打开 DB + 加载/重建 FAISS 索引。

        如果数据库中存在旧维度（如 512 维）的 embedding，
        会跳过不兼容的记录并以空索引启动（需要重新扫描注册）。
        """
        self._db.open()
        embeddings = self._db.get_all_embeddings()

        loaded = self._index.load()
        if loaded:
            # 通过 ID 集合对比判断 FAISS 与 SQLite 是否一致
            # （仅靠数量不可靠：删 A 加 B 后数量相同但内容不同）
            db_ids = {pid for pid, emb in embeddings
                      if emb.shape[0] == DEDUP_EMBEDDING_DIM}
            faiss_ids = set()
            if self._index.ntotal > 0:
                # FaissIndex._index 是 faiss.IndexIDMap，其 id_map 是 int64 向量
                id_map = faiss.vector_to_array(self._index._index.id_map)
                faiss_ids = set(id_map.tolist())
            if faiss_ids == db_ids:
                self._initialized = True
                return

            logger.info(
                "FAISS 索引与数据库 ID 不一致 (FAISS=%d, DB=%d)，重建索引",
                len(faiss_ids), len(db_ids),
            )

        # 不一致或无索引 — 从 SQLite 重建
        compatible = [
            (pid, emb) for pid, emb in embeddings
            if emb.shape[0] == DEDUP_EMBEDDING_DIM
        ]
        if len(compatible) < len(embeddings):
            logger.warning(
                "查重索引维度已从 %d 变更为 %d，跳过 %d 条旧记录。"
                "这些产品需要重新扫描注册。",
                embeddings[0][1].shape[0] if embeddings else 0,
                DEDUP_EMBEDDING_DIM,
                len(embeddings) - len(compatible),
            )
        self._index.rebuild_from_db(compatible)
        self._index.save()

        self._initialized = True

    def compute_product_embedding(
        self,
        image_paths: list[Path],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """计算产品级组合嵌入向量（CLIP + HSV）。

        分批提取所有图片的 CLIP + HSV 组合特征，取均值后 L2 归一化，
        得到 (DEDUP_EMBEDDING_DIM,) 产品级向量。
        HSV 颜色特征使同款不同色被识别为不同产品。
        """
        if not image_paths:
            return np.zeros(DEDUP_EMBEDDING_DIM, dtype=np.float32)

        extractor = get_extractor()
        extractor.load_model()
        n = len(image_paths)
        all_feats = []

        for start in range(0, n, DEDUP_BATCH_SIZE):
            end = min(start + DEDUP_BATCH_SIZE, n)
            batch = image_paths[start:end]
            feats = extractor.extract_combined_batch(batch)  # (batch, 608)
            all_feats.append(feats)
            if on_progress:
                on_progress(end, n)

        all_feats = np.vstack(all_feats)  # (N, 608)

        # 均值 + L2 归一化 → 产品级向量
        mean_feat = all_feats.mean(axis=0)
        norm = np.linalg.norm(mean_feat)
        if norm > 0:
            mean_feat /= norm

        return mean_feat.astype(np.float32)

    def check_duplicate(
        self,
        embedding: np.ndarray,
        temp_products: dict[int, ProductRecord] | None = None,
    ) -> tuple[DedupStatus, list[DedupMatch]]:
        """检查一个嵌入向量与现有库的相似度。

        Args:
            embedding: (DEDUP_EMBEDDING_DIM,) L2 归一化向量。
            temp_products: 临时产品映射（批量扫描时同批次已扫描的 NEW 产品），
                           key 为临时 ID，value 为 ProductRecord。

        Returns:
            (status, matches): status 表示判定结果，matches 为按相似度降序排列的匹配列表。
        """
        results = self._index.search(embedding, DEDUP_SEARCH_K)
        matches: list[DedupMatch] = []

        for pid, similarity in results:
            # 先查临时产品映射（避免临时 ID 触发跨线程 DB 查询），再查 DB
            product = None
            if temp_products:
                product = temp_products.get(pid)
            if product is None:
                product = self._db.get_product(pid)
            if product is None:
                continue
            matches.append(DedupMatch(
                existing_product=product,
                similarity=similarity,
            ))

        if not matches:
            return DedupStatus.NEW, []

        # 显式按相似度降序排列，确保 matches[0] 一定是最佳匹配
        matches.sort(key=lambda m: m.similarity, reverse=True)

        best_sim = matches[0].similarity
        if best_sim >= DEDUP_THRESHOLD_AUTO:
            return DedupStatus.DUPLICATE, matches
        elif best_sim >= DEDUP_THRESHOLD_REVIEW:
            return DedupStatus.REVIEW, matches
        else:
            return DedupStatus.NEW, matches

    def register_product(
        self,
        name: str,
        store: str,
        folder: str,
        source_url: str,
        download_date: str,
        image_count: int,
        embedding: np.ndarray,
        *,
        save: bool = True,
    ) -> int:
        """注册新产品到数据库和 FAISS 索引（原子性：FAISS 失败则回滚 SQLite）。

        Args:
            save: 是否立即持久化 FAISS 索引。批量注册时可设为 False，
                  最后统一调用 save_index() 一次。
        """
        pid = self._db.add_product(
            name=name, store=store, folder=folder,
            source_url=source_url, download_date=download_date,
            image_count=image_count, embedding=embedding,
        )
        try:
            self._index.add(pid, embedding)
        except Exception:
            # FAISS 操作失败，回滚 SQLite
            self._db.remove_product(pid)
            raise
        if save:
            self._index.save()
        return pid

    def remove_product(self, product_id: int) -> None:
        """从 FAISS 和 DB 同时删除产品（先删 FAISS 再删 DB，避免不一致）。"""
        try:
            self._index.remove(product_id)
        except RuntimeError:
            logger.warning("FAISS 中未找到产品 %d 的向量，仅删除数据库记录", product_id)
        self._db.remove_product(product_id)

    def get_all_products(self) -> list[ProductRecord]:
        """获取所有产品记录。"""
        return self._db.get_all_products()

    def save_index(self) -> None:
        """持久化 FAISS 索引到磁盘。"""
        self._index.save()

    def get_registered_folders(self) -> set[str]:
        """获取所有已注册的文件夹路径集合。"""
        return self._db.get_folders_set()

    def get_product_by_folder(self, folder: str) -> ProductRecord | None:
        """按文件夹查找产品。"""
        return self._db.get_product_by_folder(folder)

    def scan_folder(
        self,
        folder: Path,
        on_progress: Callable[[int, int], None] | None = None,
        temp_products: dict[int, ProductRecord] | None = None,
    ) -> DedupScanItem:
        """扫描单个文件夹，计算嵌入并查重。

        Args:
            folder: 待扫描的文件夹。
            on_progress: 图片级进度回调。
            temp_products: 临时产品映射（批量扫描同批次），传递给 check_duplicate。
        """
        images = list_images(folder)
        name = folder.name

        if not images:
            return DedupScanItem(
                folder=folder, name=name, image_count=0,
                status=DedupStatus.NEW, best_match=None,
                all_matches=[], embedding=None, error="无图片",
            )

        try:
            embedding = self.compute_product_embedding(images, on_progress)
            status, matches = self.check_duplicate(embedding, temp_products)
            best = matches[0] if matches else None

            return DedupScanItem(
                folder=folder, name=name, image_count=len(images),
                status=status, best_match=best,
                all_matches=matches, embedding=embedding, error=None,
            )
        except Exception as e:
            return DedupScanItem(
                folder=folder, name=name, image_count=len(images),
                status=DedupStatus.NEW, best_match=None,
                all_matches=[], embedding=None,
                error=str(e) or type(e).__name__,
            )

    # 临时 ID 使用负数，绝不会与 SQLite autoincrement（正整数）冲突
    _TEMP_ID_BASE = -(1 << 30)  # -1073741824

    def batch_scan(
        self,
        folders: list[Path],
        is_cancelled: Callable[[], bool] | None = None,
        on_folder_start: Callable[[int, int, Path], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_image_progress: Callable[[int, int], None] | None = None,
        on_folder_done: Callable[[int, int, DedupScanItem], None] | None = None,
    ) -> list[DedupScanItem]:
        """批量扫描多个文件夹。

        同一批次内，已扫描的 NEW 产品会被临时加入 FAISS 索引，
        使后续文件夹能检测到批次内的重复。扫描结束后清理临时条目。
        """
        results: list[DedupScanItem] = []
        total = len(folders)

        if total == 0:
            return results

        # 确保模型已加载
        extractor = get_extractor()
        if on_status:
            on_status("正在加载模型...")
        extractor.load_model(on_progress=on_status)

        # 批次内临时状态
        temp_ids: list[int] = []  # 需要清理的临时 FAISS 条目
        temp_products: dict[int, ProductRecord] = {}  # 临时 ID → 产品记录
        next_temp_id = self._TEMP_ID_BASE

        try:
            for idx, folder in enumerate(folders):
                if is_cancelled and is_cancelled():
                    break

                if on_folder_start:
                    on_folder_start(idx, total, folder)

                item = self.scan_folder(
                    folder,
                    on_progress=on_image_progress,
                    temp_products=temp_products,
                )
                results.append(item)

                # 若为 NEW 且有向量，临时加入索引供后续文件夹比对
                if (item.status == DedupStatus.NEW
                        and item.error is None
                        and item.embedding is not None):
                    tid = next_temp_id
                    next_temp_id += 1
                    self._index.add(tid, item.embedding)
                    temp_ids.append(tid)
                    temp_products[tid] = ProductRecord(
                        id=tid, name=item.name, store="",
                        folder=str(item.folder), source_url="",
                        download_date="", image_count=item.image_count,
                    )

                if on_folder_done:
                    on_folder_done(idx, total, item)
        finally:
            # 批量清理所有临时条目，恢复 FAISS 索引到扫描前状态
            if temp_ids:
                try:
                    self._index.remove_batch(temp_ids)
                except RuntimeError:
                    pass

        return results

    def close(self) -> None:
        """保存 FAISS 索引并关闭数据库。"""
        if self._initialized:
            self._index.save()
            self._db.close()
            self._initialized = False
