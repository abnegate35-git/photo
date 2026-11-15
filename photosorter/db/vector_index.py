"""記憶體內向量檢索（`core.interfaces.VectorIndex` 的預設實作）。

為什麼不直接上向量資料庫：個人照片庫的規模（10 萬張 x 768 維 float32 = 293 MB）用一次
矩陣乘法做「精確」cosine 檢索，A 電腦（14 核心）實測 top-50 約 7 毫秒，沒有近似索引的召回率
問題，也沒有額外相依。資料庫裡的 BLOB 是唯一的真實來源，這個索引隨時可以重建
（從 BLOB 載入 10 萬筆實測約 1.3 秒）。

要換成 sqlite-vec / FAISS：實作同一個 Protocol 即可，呼叫端不必改。BLOB 格式
（float32 LE）與 sqlite-vec 相同，`DbConfig.enable_sqlite_vec = true` 後可直接對
`embeddings.vector` 下 `vec_distance_cosine()`。
"""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np


class InMemoryVectorIndex:
    """精確 cosine 檢索。執行緒安全（讀寫共用一把 RLock；查詢本身很短）。

    向量必須已 L2 正規化，所以相似度 = 內積。刪除採「與最後一列交換」，O(1) 且不留洞。
    """

    def __init__(self, dim: int, capacity: int = 1024) -> None:
        self.dim = dim
        self._mat = np.empty((max(1, capacity), dim), dtype=np.float32)
        self._ids = np.empty(max(1, capacity), dtype=np.int64)
        self._n = 0
        self._pos: dict[int, int] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_arrays(cls, ids: np.ndarray, matrix: np.ndarray) -> "InMemoryVectorIndex":
        """由 `Repository.load_embedding_matrix()` 的結果建立（直接接手陣列，不複製）。"""
        if matrix.ndim != 2 or matrix.shape[0] != ids.shape[0]:
            raise ValueError("ids 與 matrix 的列數不一致")
        idx = cls(matrix.shape[1], capacity=1)
        idx._mat = np.ascontiguousarray(matrix, dtype=np.float32)
        idx._ids = np.ascontiguousarray(ids, dtype=np.int64)
        idx._n = int(ids.shape[0])
        idx._pos = {int(v): i for i, v in enumerate(idx._ids[: idx._n])}
        if len(idx._pos) != idx._n:
            raise ValueError("ids 有重複")
        if idx._mat.shape[0] == 0:
            idx._mat = np.empty((1, idx.dim), dtype=np.float32)
            idx._ids = np.empty(1, dtype=np.int64)
        return idx

    def __len__(self) -> int:
        return self._n

    def _grow(self, need: int) -> None:
        cap = self._mat.shape[0]
        if need <= cap:
            return
        new_cap = max(need, cap * 2)
        mat = np.empty((new_cap, self.dim), dtype=np.float32)
        ids = np.empty(new_cap, dtype=np.int64)
        mat[: self._n] = self._mat[: self._n]
        ids[: self._n] = self._ids[: self._n]
        self._mat, self._ids = mat, ids

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids 與 vectors 的數量不一致")
        with self._lock:
            self._grow(self._n + len(ids))
            for id_, vec in zip(ids, vectors):
                id_ = int(id_)
                pos = self._pos.get(id_)
                if pos is None:
                    pos = self._n
                    self._n += 1
                    self._pos[id_] = pos
                    self._ids[pos] = id_
                self._mat[pos] = vec

    def remove(self, ids: Sequence[int]) -> None:
        with self._lock:
            for id_ in ids:
                pos = self._pos.pop(int(id_), None)
                if pos is None:
                    continue
                last = self._n - 1
                if pos != last:
                    self._mat[pos] = self._mat[last]
                    moved = int(self._ids[last])
                    self._ids[pos] = moved
                    self._pos[moved] = pos
                self._n = last

    def get(self, id_: int) -> np.ndarray | None:
        with self._lock:
            pos = self._pos.get(int(id_))
            return None if pos is None else self._mat[pos].copy()

    def search(self, query: np.ndarray, top_k: int = 50, min_score: float | None = None
               ) -> list[tuple[int, float]]:
        """以圖搜圖 / 以文搜圖：`query` 是 (dim,) 的正規化向量（影像向量或文字向量皆可）。"""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query 維度 {q.shape[0]} 與索引的 {self.dim} 不符")
        with self._lock:
            n = self._n
            if n == 0 or top_k <= 0:
                return []
            scores = self._mat[:n] @ q
            k = min(top_k, n)
            if k < n:
                part = np.argpartition(-scores, k - 1)[:k]
            else:
                part = np.arange(n)
            order = part[np.argsort(-scores[part], kind="stable")]
            out = [(int(self._ids[i]), float(scores[i])) for i in order]
        if min_score is not None:
            out = [(i, s) for i, s in out if s >= min_score]
        return out

    def matrix_view(self) -> tuple[np.ndarray, np.ndarray]:
        """(ids, matrix) 的**唯讀 view**，給批次運算（全庫重標、找近似重複）。

        呼叫端要自己確保期間沒有 add/remove，或先 `.copy()`。
        """
        with self._lock:
            ids = self._ids[: self._n].view()
            mat = self._mat[: self._n].view()
            ids.flags.writeable = False
            mat.flags.writeable = False
            return ids, mat
