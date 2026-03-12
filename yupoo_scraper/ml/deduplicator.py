"""产品查重引擎 — SQLite 产品库 + FAISS 向量索引 + 查重逻辑"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import deque
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
    FAISS_IVF_THRESHOLD,
    FAISS_IVFPQ_THRESHOLD,
    FAISS_NPROBE,
    FAISS_PQ_M,
    FAISS_PQ_NBITS,
    FAISS_REBUILD_RATIO,
    FAISS_SAVE_INTERVAL,
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
class DedupMatch:
    """单个匹配结果"""
    existing_product: ProductRecord
    similarity: float


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

            CREATE INDEX IF NOT EXISTS idx_products_folder ON products(folder);
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

    def add_products_batch(
        self,
        products: list[tuple[str, str, str, str, str, int, np.ndarray]],
    ) -> list[int]:
        """批量添加产品（单事务提交），返回新产品 ID 列表。

        每个 tuple: (name, store, folder, source_url, download_date, image_count, embedding)
        """
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        ids: list[int] = []
        with self._lock:
            for name, store, folder, source_url, download_date, image_count, embedding in products:
                cursor = self._conn.execute(
                    """INSERT INTO products
                       (name, store, folder, source_url, download_date, image_count, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (name, store, folder, source_url, download_date, image_count,
                     embedding.tobytes()),
                )
                ids.append(cursor.lastrowid)
            self._conn.commit()  # 单次 commit，避免逐条 fsync
        return ids

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

    def remove_products_batch(self, product_ids: list[int]) -> None:
        """批量删除产品（单事务提交）。"""
        if not product_ids:
            return
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")
        with self._lock:
            self._conn.executemany(
                "DELETE FROM products WHERE id = ?",
                [(pid,) for pid in product_ids],
            )
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
        """获取所有产品的 (id, embedding) 列表，用于重建 FAISS。

        使用 fetchmany 分批读取，避免一次性加载全部 BLOB 到内存。
        """
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open()")

        _BATCH = 1000
        result: list[tuple[int, np.ndarray]] = []

        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, embedding FROM products WHERE embedding IS NOT NULL",
            )
            while True:
                rows = cursor.fetchmany(_BATCH)
                if not rows:
                    break
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
    """FAISS 向量索引封装，自动在 FlatIP / IVFFlat / IVFPQ 之间切换。

    - 产品数 < FAISS_IVF_THRESHOLD: 使用 IndexFlatIP（暴力搜索，精确）
    - 产品数 >= FAISS_IVF_THRESHOLD: 使用 IndexIVFFlat（倒排索引，O(√N)）
    - 产品数 >= FAISS_IVFPQ_THRESHOLD: 使用 IVFPQ（量化压缩，内存极低）
    - 所有索引都包裹在 IndexIDMap 中以支持 ID 映射和 remove_ids
    - 向量已 L2 归一化，内积 = 余弦相似度
    """

    def __init__(self, dim: int = DEDUP_EMBEDDING_DIM):
        self._dim = dim
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(
            faiss.IndexFlatIP(dim),
        )
        self._lock = threading.Lock()
        self._is_ivf = False
        self._is_pq = False           # 是否使用 IVFPQ
        self._trained_count = 0       # IVF 训练时的向量数
        self._since_train_count = 0   # 训练后新增的向量数
        self._since_save_count = 0    # 上次保存后新增的向量数
        self._needs_rebuild = False   # 标记是否需要重建

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def get_id_set(self) -> set[int]:
        """获取索引中所有向量的 ID 集合。"""
        with self._lock:
            if self._index.ntotal == 0:
                return set()
            id_map = faiss.vector_to_array(self._index.id_map)
            return set(id_map.tolist())

    def add(self, product_id: int, embedding: np.ndarray,
            *, _auto_save_path: Path | None = DEDUP_FAISS_PATH) -> None:
        """添加单个向量（线程安全）。

        对已训练的 IVF 索引，直接 add_with_ids 即可（不需重新训练）。
        当累计新增超过训练时数量的 FAISS_REBUILD_RATIO，标记需要重建。
        每累计 FAISS_SAVE_INTERVAL 次添加自动持久化索引。
        """
        vec = embedding.reshape(1, -1).astype(np.float32)
        ids = np.array([product_id], dtype=np.int64)
        need_save = False
        with self._lock:
            self._index.add_with_ids(vec, ids)
            self._since_save_count += 1
            if self._since_save_count >= FAISS_SAVE_INTERVAL:
                need_save = True
                self._since_save_count = 0
            if self._is_ivf:
                self._since_train_count += 1
                if (self._trained_count > 0
                        and self._since_train_count > self._trained_count * FAISS_REBUILD_RATIO):
                    self._needs_rebuild = True
        # 增量持久化（锁外执行，避免长时间持锁）
        if need_save and _auto_save_path:
            self.save(_auto_save_path)

    def search(self, query: np.ndarray, k: int = DEDUP_SEARCH_K) -> list[tuple[int, float]]:
        """搜索最相似的 k 个产品，返回 [(product_id, similarity)]（线程安全）。"""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            actual_k = min(k, self._index.ntotal)
            vec = query.reshape(1, -1).astype(np.float32)
            # IVF 索引需要设置 nprobe
            if self._is_ivf:
                self._set_nprobe()
            distances, ids = self._index.search(vec, actual_k)
        results = []
        for i in range(actual_k):
            pid = int(ids[0][i])
            sim = float(distances[0][i])
            if pid >= 0:  # -1 表示无结果
                results.append((pid, sim))
        return results

    def _set_nprobe(self) -> None:
        """设置 IVF 搜索时的 nprobe 参数（调用时必须持有 _lock）。"""
        # IndexIDMap 包裹的子索引
        sub_index = faiss.downcast_index(self._index.index)
        if hasattr(sub_index, 'nprobe'):
            nprobe = min(FAISS_NPROBE, sub_index.nlist)
            sub_index.nprobe = max(1, nprobe)

    def add_batch(
        self,
        product_ids: list[int],
        embeddings: np.ndarray,
        *,
        _auto_save_path: Path | None = DEDUP_FAISS_PATH,
    ) -> None:
        """批量添加向量（线程安全，单次锁获取）。

        Args:
            product_ids: 产品 ID 列表。
            embeddings: (N, dim) float32 矩阵。
        """
        if len(product_ids) == 0:
            return
        vecs = embeddings.astype(np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        ids = np.array(product_ids, dtype=np.int64)
        need_save = False
        with self._lock:
            self._index.add_with_ids(vecs, ids)
            self._since_save_count += len(product_ids)
            if self._since_save_count >= FAISS_SAVE_INTERVAL:
                need_save = True
                self._since_save_count = 0
            if self._is_ivf:
                self._since_train_count += len(product_ids)
                if (self._trained_count > 0
                        and self._since_train_count > self._trained_count * FAISS_REBUILD_RATIO):
                    self._needs_rebuild = True
        if need_save and _auto_save_path:
            self.save(_auto_save_path)

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
        """保存索引到文件（原子写入：先写临时文件再替换）。

        持有 _lock 期间序列化，防止与 add_with_ids 并发导致状态不一致。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with self._lock:
            faiss.write_index(self._index, str(tmp_path))
        os.replace(str(tmp_path), str(path))

    def load(self, path: Path = DEDUP_FAISS_PATH) -> bool:
        """从文件加载索引，成功返回 True。

        自动检测索引类型（FlatIP 或 IVFFlat），恢复内部状态。
        """
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
        # 检测子索引类型
        sub_index = faiss.downcast_index(loaded.index)
        self._is_ivf = isinstance(sub_index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ))
        self._is_pq = isinstance(sub_index, faiss.IndexIVFPQ)
        if self._is_pq:
            self._trained_count = loaded.ntotal
            self._since_train_count = 0
            self._needs_rebuild = False
            logger.info("加载 IVFPQ 索引: %d 向量, nlist=%d, M=%d",
                        loaded.ntotal, sub_index.nlist, sub_index.pq.M)
        elif self._is_ivf:
            self._trained_count = loaded.ntotal
            self._since_train_count = 0
            self._needs_rebuild = False
            logger.info("加载 IVFFlat 索引: %d 向量, nlist=%d",
                        loaded.ntotal, sub_index.nlist)
        else:
            logger.info("加载 FlatIP 索引: %d 向量", loaded.ntotal)
        self._since_save_count = 0
        return True

    @property
    def needs_rebuild(self) -> bool:
        """是否需要重建索引（IVF 新增过多或规模变化需要切换类型）。"""
        return self._needs_rebuild

    def rebuild_from_db(self, embeddings: list[tuple[int, np.ndarray]]) -> None:
        """从数据库记录重建索引。

        根据数据量自动选择索引类型：
        - < FAISS_IVF_THRESHOLD: 使用 FlatIP（精确搜索）
        - >= FAISS_IVF_THRESHOLD: 使用 IVFFlat（nlist=√N，训练后添加）
        - >= FAISS_IVFPQ_THRESHOLD: 使用 IVFPQ（量化压缩，极致省内存）
        """
        n = len(embeddings)
        if n == 0:
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
            self._is_ivf = False
            self._is_pq = False
            self._trained_count = 0
            self._since_train_count = 0
            self._since_save_count = 0
            self._needs_rebuild = False
            return

        ids = np.empty(n, dtype=np.int64)
        vecs = np.empty((n, self._dim), dtype=np.float32)
        for i, (pid, emb) in enumerate(embeddings):
            ids[i] = pid
            vecs[i] = emb

        if n >= FAISS_IVFPQ_THRESHOLD:
            # 超大规模：IVFPQ（量化压缩）
            # PQ 码本大小 = 2^nbits，训练点必须 >= 码本大小
            pq_nbits = FAISS_PQ_NBITS
            while 2 ** pq_nbits > n and pq_nbits > 4:
                pq_nbits -= 1
            nlist = max(1, int(n ** 0.5))
            # FAISS 要求训练点 >= 39*nlist（IVF 聚类质量要求）
            if n < nlist * 39:
                nlist = max(1, n // 39)
            # PQ_M 必须整除 dim；如果不整除，退回到最近的整除值
            pq_m = FAISS_PQ_M
            while self._dim % pq_m != 0 and pq_m > 1:
                pq_m -= 1
            quantizer = faiss.IndexFlatIP(self._dim)
            ivfpq = faiss.IndexIVFPQ(
                quantizer, self._dim, nlist, pq_m, pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            ivfpq.train(vecs)
            self._index = faiss.IndexIDMap(ivfpq)
            self._index.add_with_ids(vecs, ids)
            self._is_ivf = True
            self._is_pq = True
            self._trained_count = n
            self._since_train_count = 0
            logger.info("构建 IVFPQ 索引: %d 向量, nlist=%d, M=%d, nbits=%d",
                        n, nlist, pq_m, pq_nbits)
        elif n >= FAISS_IVF_THRESHOLD:
            # 大规模：IVFFlat
            nlist = max(1, int(n ** 0.5))
            quantizer = faiss.IndexFlatIP(self._dim)
            ivf = faiss.IndexIVFFlat(quantizer, self._dim, nlist,
                                     faiss.METRIC_INNER_PRODUCT)
            ivf.train(vecs)
            self._index = faiss.IndexIDMap(ivf)
            self._index.add_with_ids(vecs, ids)
            self._is_ivf = True
            self._is_pq = False
            self._trained_count = n
            self._since_train_count = 0
            logger.info("构建 IVFFlat 索引: %d 向量, nlist=%d", n, nlist)
        else:
            # 小规模：暴力搜索
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
            self._index.add_with_ids(vecs, ids)
            self._is_ivf = False
            self._is_pq = False
            self._trained_count = 0
            self._since_train_count = 0
            logger.info("构建 FlatIP 索引: %d 向量", n)

        self._since_save_count = 0
        self._needs_rebuild = False


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

        自动处理索引类型升级：当产品数跨越 IVF 阈值时触发重建。
        """
        self._db.open()
        embeddings = self._db.get_all_embeddings()

        compatible = [
            (pid, emb) for pid, emb in embeddings
            if emb.shape[0] == DEDUP_EMBEDDING_DIM
        ]
        if len(compatible) < len(embeddings):
            old_dim = next(
                (emb.shape[0] for _, emb in embeddings
                 if emb.shape[0] != DEDUP_EMBEDDING_DIM),
                0,
            )
            logger.warning(
                "查重索引维度已从 %d 变更为 %d，跳过 %d 条旧记录。"
                "这些产品需要重新扫描注册。",
                old_dim,
                DEDUP_EMBEDDING_DIM,
                len(embeddings) - len(compatible),
            )

        db_ids = {pid for pid, _ in compatible}

        loaded = self._index.load()
        if loaded:
            faiss_ids = self._index.get_id_set()

            if faiss_ids == db_ids:
                # ID 一致，检查是否需要切换索引类型
                n_ids = len(db_ids)
                should_pq = n_ids >= FAISS_IVFPQ_THRESHOLD
                should_ivf = n_ids >= FAISS_IVF_THRESHOLD
                need_switch = (
                    (should_pq and not self._index._is_pq)
                    or (should_ivf and not self._index._is_ivf)
                    or (not should_ivf and self._index._is_ivf)
                )
                if need_switch:
                    logger.info(
                        "产品数 %d，索引类型需切换 (当前 IVF=%s PQ=%s)",
                        n_ids, self._index._is_ivf, self._index._is_pq,
                    )
                else:
                    self._initialized = True
                    return
            else:
                logger.info(
                    "FAISS 索引与数据库 ID 不一致 (FAISS=%d, DB=%d)，重建索引",
                    len(faiss_ids), len(db_ids),
                )

        # 不一致、需要类型切换、或无索引 — 从 SQLite 重建
        # 持锁重建，防止并发 add() 导致索引不一致
        with self._index._lock:
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

        # 过滤掉全零向量（损坏图片），避免污染产品级嵌入
        norms = np.linalg.norm(all_feats, axis=1)
        valid_mask = norms > 1e-6
        if valid_mask.any():
            all_feats = all_feats[valid_mask]
        else:
            logger.warning("产品所有图片特征均为零向量，无法计算有效嵌入")
            return np.zeros(DEDUP_EMBEDDING_DIM, dtype=np.float32)

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
            # 只保留 >= REVIEW 阈值的匹配，过滤无关的低分结果
            matches = [m for m in matches if m.similarity >= DEDUP_THRESHOLD_REVIEW]
            return DedupStatus.DUPLICATE, matches
        elif best_sim >= DEDUP_THRESHOLD_REVIEW:
            matches = [m for m in matches if m.similarity >= DEDUP_THRESHOLD_REVIEW]
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
        if embedding.shape != (DEDUP_EMBEDDING_DIM,):
            raise ValueError(
                f"embedding shape 不匹配: 期望 ({DEDUP_EMBEDDING_DIM},)，"
                f"实际 {embedding.shape}"
            )
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

    def register_products_batch(
        self,
        items: list[tuple[str, str, str, str, str, int, np.ndarray]],
    ) -> list[int]:
        """批量注册产品到 DB + FAISS（单事务 + 单次批量插入）。

        每个 tuple: (name, store, folder, source_url, download_date, image_count, embedding)

        Returns:
            新产品 ID 列表。
        """
        if not items:
            return []

        # 校验 embedding 维度
        for i, (*_, embedding) in enumerate(items):
            if embedding.shape != (DEDUP_EMBEDDING_DIM,):
                raise ValueError(
                    f"items[{i}] embedding shape 不匹配: "
                    f"期望 ({DEDUP_EMBEDDING_DIM},)，实际 {embedding.shape}"
                )

        # 1. 批量写入 SQLite（单 commit）
        pids = self._db.add_products_batch(items)

        # 2. 批量写入 FAISS
        embeddings = np.stack([emb for *_, emb in items])
        try:
            self._index.add_batch(pids, embeddings)
        except Exception:
            # FAISS 失败 — 单事务回滚所有新增的 DB 记录
            try:
                self._db.remove_products_batch(pids)
            except Exception as rollback_err:
                logger.error(
                    "FAISS 写入失败后 DB 回滚也失败: %s — "
                    "可能残留 %d 条孤立 DB 记录 (ids: %s)",
                    rollback_err, len(pids), pids[:5],
                )
            raise

        return pids

    def remove_product(self, product_id: int) -> None:
        """从 FAISS 和 DB 同时删除产品（先删 FAISS 再删 DB，避免不一致）。"""
        try:
            self._index.remove(product_id)
        except RuntimeError as e:
            logger.warning("FAISS 删除产品 %d 失败 (%s)，仅删除数据库记录", product_id, e)
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

        # ETA 估算：滑动窗口记录最近 N 个文件夹的耗时
        _ETA_WINDOW = 20
        folder_times: deque[float] = deque(maxlen=_ETA_WINDOW)

        try:
            for idx, folder in enumerate(folders):
                if is_cancelled and is_cancelled():
                    break

                # ETA 信息
                eta_str = ""
                if folder_times:
                    avg_time = sum(folder_times) / len(folder_times)
                    remaining = total - idx
                    eta_sec = avg_time * remaining
                    if eta_sec >= 3600:
                        eta_str = f" (预计剩余 {eta_sec/3600:.1f}h)"
                    elif eta_sec >= 60:
                        eta_str = f" (预计剩余 {eta_sec/60:.0f}min)"
                    elif eta_sec >= 5:
                        eta_str = f" (预计剩余 {eta_sec:.0f}s)"

                if on_folder_start:
                    on_folder_start(idx, total, folder)
                if on_status and eta_str:
                    on_status(f"正在扫描: {folder.name}{eta_str}")

                t_start = time.monotonic()
                item = self.scan_folder(
                    folder,
                    on_progress=on_image_progress,
                    temp_products=temp_products,
                )
                folder_times.append(time.monotonic() - t_start)
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
                except RuntimeError as e:
                    logger.error(
                        "清理 %d 个临时 FAISS 条目失败: %s — "
                        "索引可能残留负数 ID，建议重启后重建",
                        len(temp_ids), e,
                    )
                    # 逐个尝试清理，尽可能恢复干净状态
                    orphaned: list[int] = []
                    for tid in temp_ids:
                        try:
                            self._index.remove(tid)
                        except RuntimeError:
                            orphaned.append(tid)
                    if orphaned:
                        logger.error(
                            "清理后仍残留 %d 个临时 ID: %s",
                            len(orphaned), orphaned[:10],
                        )

        return results

    def close(self) -> None:
        """保存 FAISS 索引并关闭数据库。"""
        if self._initialized:
            try:
                self._index.save()
            except Exception as e:
                logger.error("保存 FAISS 索引失败: %s", e)
        # 始终关闭 DB 连接，即使初始化未完成或 FAISS 保存失败
        self._db.close()
        self._initialized = False
