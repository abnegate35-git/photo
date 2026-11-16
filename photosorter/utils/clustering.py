"""聚類演算法模組：人臉無監督分群、時間／地理事件分群、連拍偵測與最佳照片挑選。

兩個引擎都是「全域後處理」：整條管線把逐張照片處理完之後才執行，直接讀寫 SQLite
（讀走呼叫執行緒自己的唯讀連線，寫入每個引擎各自包成一個交易），不碰任何照片檔案。
`pipeline/factory.py` 會把它們掛成 `PipelineEngine` 的後處理步驟；也可以由 GUI 單獨呼叫
（例如使用者改完人物名稱後重跑分群）。兩者都可以重複執行，結果是冪等的。

共同原則 —— 使用者的決定永遠優先於自動結果：
  * 已命名（`persons.name` 非 NULL）或已鎖定（`persons.is_locked = 1`）的人物群組，
    以及使用者手動指派的臉（`faces.assign_src = 'manual'`），任何模式下都不會被改動或拆散。
  * 使用者編輯過的事件（`events.is_user_edited = 1`）與其中的照片不會被重新聚類覆蓋。
  * 使用者標為最愛的照片在連拍挑選時優先；標為廢片的不會被選為最佳照片。
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.config import AppConfig
from ..core.interfaces import ReverseGeocoder
from ..core.types import EventCluster
from ..db.db_manager import DBManager
from ..db.repository import Repository, vec_to_blob
from .geo import haversine_km
from .imageops import hamming64

log = logging.getLogger(__name__)

# ---- 預設參數（設定檔沒有對應欄位的，以這裡為準；可由建構子覆寫）----
FACE_EPS = 0.38                    # DBSCAN 的 cosine distance 半徑
FACE_MIN_SAMPLES = 3
FACE_ASSIGN_THRESHOLD = 0.65       # 與既有人物中心向量的 cosine similarity 達此值就直接歸入
TIME_GAP_HOURS = 6.0               # 相鄰照片時間差小於此值 → 同一事件
TRIP_DISTANCE_KM = 30.0            # 連續行程：與本段最近一張有位置的照片距離小於此值
TRIP_GAP_HOURS = 36.0              # 連續行程最多能跨越的時間空檔（過夜）
BURST_SECONDS = 3.0                # 連拍：相鄰照片時間差小於此值
BURST_HAMMING = 4                  # 連拍：dHash 漢明距離不超過此值
MIN_EVENT_PHOTOS = 3

_NOISE = -1
_CHUNK = 500


def person_display_name(person_id: int, name: str | None) -> str:
    """人物的顯示名稱：使用者命名過就用名字，否則是 `Person_{id}`。

    自動產生的群組 `persons.name` 留 NULL（NULL 才代表「使用者還沒命名」，分群保護靠它判斷），
    UI 與匯出一律透過這個函式取得 `Person_#` 形式的名稱。
    """
    return name if name else f"Person_{person_id}"


def _chunks(seq: Sequence, n: int = _CHUNK):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norm, 1e-12)


# =========================================================================== #
# 人臉分群
# =========================================================================== #
@dataclass(slots=True)
class FaceTable:
    """記憶體中的人臉特徵表（`FaceClusterEngine.load_faces()` 的結果）。各陣列同長、同順序。"""

    face_ids: np.ndarray                 # (N,) int64
    photo_ids: np.ndarray                # (N,) int64
    matrix: np.ndarray                   # (N, dim) float32，每列已 L2 正規化
    person_ids: np.ndarray               # (N,) int64；未指派為 -1
    manual: np.ndarray                   # (N,) bool：faces.assign_src == 'manual'
    det_scores: np.ndarray               # (N,) float32
    areas: np.ndarray                    # (N,) float32：bbox_w * bbox_h（佔原圖面積比例）

    def __len__(self) -> int:
        return int(self.face_ids.shape[0])


@dataclass(slots=True)
class FaceClusterResult:
    total_faces: int = 0                 # 參與的臉（有特徵、未被忽略）
    protected_faces: int = 0             # 屬於已命名/已鎖定人物或手動指派，完全沒動
    assigned_to_existing: int = 0        # 依中心向量歸入既有人物
    new_persons: int = 0                 # DBSCAN 新建立的人物群組
    clustered_faces: int = 0             # 歸入新群組的臉
    noise_faces: int = 0                 # 不屬於任何群組（person_id 維持 NULL）
    dissolved_persons: int = 0           # full_recluster 時被解散重分的未保護群組
    new_person_ids: list[int] = field(default_factory=list)


class FaceClusterEngine:
    """把 `faces` 歸成 `persons`。

    流程（預設為增量模式）：
      1. 從 `faces` 讀出所有 512 維特徵 BLOB，還原成 (N, 512) 矩陣。
      2. 以目前的成員算出每個既有人物的中心向量（成員特徵平均後再正規化）。
      3. 尚未指派的臉（`person_id IS NULL`）先與中心向量比 cosine similarity，
         最高分 `>= assign_threshold` 就直接歸入該人物。
      4. 其餘的臉才跑 `DBSCAN(eps, min_samples, metric='cosine')`，每個群建立一個新人物
         （`name` 留 NULL，顯示為 `Person_{id}`）；雜訊維持未指派，下次執行會再參與。
      5. 更新受影響人物的 `face_count`、`centroid` 與代表頭像 `cover_face_id`。

    保護規則：
      * 已有 `person_id` 的臉在增量模式下一律不動。
      * `full_recluster=True` 會把「未保護」的自動群組解散重分；已命名或 `is_locked = 1` 的人物、
        以及 `assign_src = 'manual'` 的臉仍然不動。寫入時 SQL 本身也帶保護條件，就算使用者在
        計算期間剛好鎖定了某個人物，也不會被覆蓋。
      * 受保護人物的代表頭像只在還沒有頭像時才自動填入，不覆蓋既有的選擇。

    DBSCAN 以暴力法算 cosine 距離，時間是 O(N²)；每次只處理「尚未指派」的臉，
    所以平常的增量執行量很小。
    """

    def __init__(
        self,
        db: DBManager,
        *,
        dim: int = 512,
        eps: float = FACE_EPS,
        min_samples: int = FACE_MIN_SAMPLES,
        assign_threshold: float = FACE_ASSIGN_THRESHOLD,
        cover_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> None:
        """
        Args:
            dim: 特徵維度（buffalo_l 為 512）。
            eps / min_samples: DBSCAN 參數（cosine distance）。
            assign_threshold: 歸入既有人物所需的最低 cosine similarity。
            cover_weights: 代表頭像評分的權重 (偵測分數, 臉部面積, 與中心向量的相似度)。
        """
        self._db = db
        self.dim = dim
        self.eps = eps
        self.min_samples = max(1, min_samples)
        self.assign_threshold = assign_threshold
        self.cover_weights = cover_weights

    @classmethod
    def from_config(cls, cfg: AppConfig, db: DBManager) -> "FaceClusterEngine":
        return cls(db, dim=cfg.face.embed_dim, eps=cfg.face.cluster_eps, min_samples=cfg.face.cluster_min_samples)

    # ------------------------------------------------------------------ #
    def load_faces(self) -> FaceTable:
        """讀出所有可分群的臉（有特徵、未被使用者標為忽略）並矩陣化。

        長度不符或含非有限值的 BLOB 會被略過並記錄警告，不會讓整次分群失敗。
        """
        rows = self._db.query(
            "SELECT id, photo_id, embedding, person_id, assign_src, det_score, bbox_w * bbox_h AS area "
            "FROM faces WHERE embedding IS NOT NULL AND is_ignored = 0 ORDER BY id"
        )
        expected = self.dim * 4
        good = [r for r in rows if len(r["embedding"]) == expected]
        if len(good) != len(rows):
            log.warning("%d 張臉的特徵長度不是 %d 維，已略過", len(rows) - len(good), self.dim)
        n = len(good)
        matrix = np.empty((n, self.dim), dtype=np.float32)
        for i, r in enumerate(good):
            matrix[i] = np.frombuffer(r["embedding"], dtype=np.float32)
        finite = np.isfinite(matrix).all(axis=1) if n else np.zeros(0, dtype=bool)
        if n and not finite.all():
            log.warning("%d 張臉的特徵含非有限值，已略過", int((~finite).sum()))
            good = [r for r, ok in zip(good, finite) if ok]
            matrix = matrix[finite]
            n = len(good)
        return FaceTable(
            face_ids=np.fromiter((r["id"] for r in good), dtype=np.int64, count=n),
            photo_ids=np.fromiter((r["photo_id"] for r in good), dtype=np.int64, count=n),
            matrix=_normalize_rows(matrix) if n else matrix,
            person_ids=np.fromiter((-1 if r["person_id"] is None else r["person_id"] for r in good),
                                   dtype=np.int64, count=n),
            manual=np.fromiter((r["assign_src"] == "manual" for r in good), dtype=bool, count=n),
            det_scores=np.fromiter((r["det_score"] or 0.0 for r in good), dtype=np.float32, count=n),
            areas=np.fromiter((r["area"] or 0.0 for r in good), dtype=np.float32, count=n),
        )

    def protected_person_ids(self) -> set[int]:
        """已命名或已鎖定的人物：任何模式下都不可覆蓋或拆散。"""
        return {r["id"] for r in self._db.query(
            "SELECT id FROM persons WHERE is_locked = 1 OR (name IS NOT NULL AND name != '')")}

    # ------------------------------------------------------------------ #
    def run(self, *, full_recluster: bool = False,
            should_stop: Callable[[], bool] = lambda: False) -> FaceClusterResult:
        """執行分群並寫回資料庫（單一交易）。

        Args:
            full_recluster: False（預設）只處理尚未指派的臉；True 另外把未保護的自動群組解散重分。
            should_stop: 回傳 True 時在寫入之前放棄（資料庫不會有半套結果）。
        """
        faces = self.load_faces()
        result = FaceClusterResult(total_faces=len(faces))
        protected = self.protected_person_ids()
        person_of = faces.person_ids.copy()

        is_protected_face = faces.manual | np.isin(person_of, list(protected)) if len(faces) else np.zeros(0, bool)
        result.protected_faces = int(is_protected_face.sum())

        # ---- full_recluster：解散未保護的自動群組（只在記憶體裡，寫入時才生效）----
        dissolved_faces: list[int] = []
        dissolved_persons: set[int] = set()
        if full_recluster and len(faces):
            dissolve = (person_of >= 0) & ~is_protected_face
            dissolved_persons = set(person_of[dissolve].tolist())
            dissolved_faces = faces.face_ids[dissolve].tolist()
            person_of[dissolve] = -1
            result.dissolved_persons = len(dissolved_persons)

        # ---- 既有人物的中心向量 ----
        centroids = self._centroids(faces.matrix, person_of)
        unassigned = np.flatnonzero(person_of < 0)

        # ---- 先比對既有人物 ----
        to_existing: list[tuple[int, int]] = []          # (person_id, face_id)
        if len(unassigned) and centroids:
            pids = np.fromiter(centroids.keys(), dtype=np.int64, count=len(centroids))
            cmat = np.stack([centroids[int(p)] for p in pids])
            best_pid = np.empty(len(unassigned), dtype=np.int64)
            best_sim = np.empty(len(unassigned), dtype=np.float32)
            for start in range(0, len(unassigned), 8192):          # 分塊，避免 N x P 的相似度矩陣過大
                idx = unassigned[start : start + 8192]
                sims = faces.matrix[idx] @ cmat.T
                arg = sims.argmax(axis=1)
                best_pid[start : start + len(idx)] = pids[arg]
                best_sim[start : start + len(idx)] = sims[np.arange(len(idx)), arg]
            hit = best_sim >= self.assign_threshold
            for i, pid in zip(unassigned[hit], best_pid[hit]):
                person_of[i] = pid
                to_existing.append((int(pid), int(faces.face_ids[i])))
            unassigned = unassigned[~hit]
        result.assigned_to_existing = len(to_existing)

        # ---- 剩下的才跑 DBSCAN ----
        labels = self._dbscan(faces.matrix[unassigned]) if len(unassigned) else np.zeros(0, dtype=np.int64)
        clusters: dict[int, list[int]] = {}                        # 群標籤 -> faces 陣列索引
        noise: list[int] = []
        for i, label in zip(unassigned, labels):
            if label == _NOISE:
                noise.append(int(faces.face_ids[i]))
            else:
                clusters.setdefault(int(label), []).append(int(i))
        result.noise_faces = len(noise)
        result.clustered_faces = sum(len(v) for v in clusters.values())

        if should_stop():
            return result

        # ---- 寫回（單一交易）----
        touched = {pid for pid, _ in to_existing} | dissolved_persons
        cover_w = self.cover_weights
        matrix, det, areas, face_ids = faces.matrix, faces.det_scores, faces.areas, faces.face_ids

        def stats_for(members: np.ndarray) -> tuple[bytes, int]:
            centroid = _normalize_rows(matrix[members].mean(axis=0, keepdims=True))[0]
            return vec_to_blob(centroid), int(face_ids[_pick_cover(matrix[members], centroid, det[members],
                                                                    areas[members], cover_w, members)])

        def op(conn: sqlite3.Connection) -> list[int]:
            # (a) 解散：SQL 再確認一次該人物仍未受保護、該臉不是手動指派
            for part in _chunks(dissolved_faces):
                conn.execute(
                    "UPDATE faces SET person_id = NULL, cluster_id = NULL, assign_src = NULL "
                    f"WHERE id IN ({','.join('?' * len(part))}) "
                    "AND (assign_src IS NULL OR assign_src != 'manual') "
                    "AND person_id IN (SELECT id FROM persons WHERE is_locked = 0 AND (name IS NULL OR name = ''))",
                    tuple(part),
                )
            # (b) 歸入既有人物：只動仍未指派的臉
            conn.executemany(
                "UPDATE faces SET person_id = ?, assign_src = 'auto', cluster_id = NULL "
                "WHERE id = ? AND person_id IS NULL",
                to_existing,
            )
            # (c) 新群組
            new_ids: list[int] = []
            final_members: dict[int, np.ndarray] = {}
            for label in sorted(clusters):
                members = np.asarray(clusters[label], dtype=np.int64)
                person_id = int(conn.execute("INSERT INTO persons DEFAULT VALUES").lastrowid)
                new_ids.append(person_id)
                conn.executemany(
                    "UPDATE faces SET person_id = ?, cluster_id = ?, assign_src = 'auto' "
                    "WHERE id = ? AND person_id IS NULL",
                    [(person_id, label, int(face_ids[i])) for i in members],
                )
                final_members[person_id] = members
            # (d) 雜訊
            conn.executemany("UPDATE faces SET cluster_id = -1 WHERE id = ? AND person_id IS NULL",
                             [(f,) for f in noise])
            # (e) 更新受影響人物的統計與代表頭像
            for pid in touched:
                members = np.flatnonzero(person_of == pid)
                if len(members):
                    final_members[pid] = members
            for pid, members in final_members.items():
                centroid_blob, cover = stats_for(members)
                conn.execute(
                    "UPDATE persons SET centroid = ?, updated_at = unixepoch(), "
                    "face_count = (SELECT COUNT(*) FROM faces WHERE person_id = persons.id), "
                    # 受保護的人物：只在還沒有頭像時才自動填入
                    "cover_face_id = CASE WHEN is_locked = 1 OR (name IS NOT NULL AND name != '') "
                    "                     THEN COALESCE(cover_face_id, ?) ELSE ? END "
                    "WHERE id = ?",
                    (centroid_blob, cover, cover, pid),
                )
            # (f) 解散後沒有任何成員的未保護群組：刪除
            for part in _chunks(sorted(dissolved_persons)):
                conn.execute(
                    f"DELETE FROM persons WHERE id IN ({','.join('?' * len(part))}) "
                    "AND is_locked = 0 AND (name IS NULL OR name = '') "
                    "AND NOT EXISTS (SELECT 1 FROM faces WHERE faces.person_id = persons.id)",
                    tuple(part),
                )
            return new_ids

        result.new_person_ids = self._db.write(op)
        result.new_persons = len(result.new_person_ids)
        log.info("人臉分群：%d 張臉，保護 %d，歸入既有 %d，新群組 %d（%d 張），雜訊 %d",
                 result.total_faces, result.protected_faces, result.assigned_to_existing,
                 result.new_persons, result.clustered_faces, result.noise_faces)
        return result

    def refresh_representatives(self, person_ids: Sequence[int] | None = None) -> int:
        """重新計算人物的 `face_count`、`centroid` 與代表頭像。

        給 GUI 在使用者手動搬動人臉之後呼叫。`person_ids=None` 代表全部人物。回傳更新的人物數。
        受保護人物的代表頭像同樣只在為空時才填入。
        """
        faces = self.load_faces()
        wanted = None if person_ids is None else set(int(p) for p in person_ids)
        updates: list[tuple[bytes, int, int, int]] = []
        for pid in np.unique(faces.person_ids[faces.person_ids >= 0]).tolist():
            if wanted is not None and pid not in wanted:
                continue
            members = np.flatnonzero(faces.person_ids == pid)
            centroid = _normalize_rows(faces.matrix[members].mean(axis=0, keepdims=True))[0]
            cover = int(faces.face_ids[_pick_cover(faces.matrix[members], centroid, faces.det_scores[members],
                                                    faces.areas[members], self.cover_weights, members)])
            updates.append((vec_to_blob(centroid), cover, cover, pid))

        def op(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "UPDATE persons SET centroid = ?, updated_at = unixepoch(), "
                "face_count = (SELECT COUNT(*) FROM faces WHERE person_id = persons.id), "
                "cover_face_id = CASE WHEN is_locked = 1 OR (name IS NOT NULL AND name != '') "
                "                     THEN COALESCE(cover_face_id, ?) ELSE ? END "
                "WHERE id = ?",
                updates,
            )
            # 沒有任何特徵可用的人物（例如只剩手動指派且無特徵的臉）仍要有正確的 face_count
            conn.execute("UPDATE persons SET face_count = (SELECT COUNT(*) FROM faces WHERE person_id = persons.id)")

        self._db.write(op)
        return len(updates)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _centroids(matrix: np.ndarray, person_of: np.ndarray) -> dict[int, np.ndarray]:
        """每個人物的中心向量：成員特徵平均後再 L2 正規化。"""
        out: dict[int, np.ndarray] = {}
        assigned = np.flatnonzero(person_of >= 0)
        if not len(assigned):
            return out
        order = assigned[np.argsort(person_of[assigned], kind="stable")]
        pids, starts = np.unique(person_of[order], return_index=True)
        bounds = list(starts) + [len(order)]
        for k, pid in enumerate(pids.tolist()):
            members = order[bounds[k] : bounds[k + 1]]
            out[pid] = _normalize_rows(matrix[members].mean(axis=0, keepdims=True))[0]
        return out

    def _dbscan(self, x: np.ndarray) -> np.ndarray:
        """DBSCAN（cosine distance）。臉數少於 `min_samples` 時不可能成群，直接全部視為雜訊。"""
        if x.shape[0] < self.min_samples:
            return np.full(x.shape[0], _NOISE, dtype=np.int64)
        from sklearn.cluster import DBSCAN

        # 特徵已 L2 正規化，cosine distance = 1 - A·Bᵀ
        return DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit_predict(x).astype(np.int64)


def _pick_cover(vectors: np.ndarray, centroid: np.ndarray, det: np.ndarray, areas: np.ndarray,
                weights: tuple[float, float, float], members: np.ndarray) -> int:
    """挑代表頭像：偵測分數高、臉部面積大、且最接近群組中心向量。

    三項各自正規化到 0..1 後加權（面積以群內最大值為 1）。回傳 `members` 裡被選中的那個索引值。
    """
    w_det, w_area, w_sim = weights
    area_norm = areas / max(float(areas.max()), 1e-12)
    sim = np.clip(vectors @ centroid, 0.0, 1.0)
    score = w_det * np.clip(det, 0.0, 1.0) + w_area * area_norm + w_sim * sim
    return int(members[int(np.argmax(score))])


# =========================================================================== #
# 事件與連拍
# =========================================================================== #
@dataclass(slots=True)
class _Shot:
    id: int
    ts: int
    path: str
    lat: float | None
    lon: float | None
    country: str | None
    city: str | None
    dhash: int | None
    blur: float
    size: int
    favorite: bool
    rejected: bool

    @property
    def located(self) -> bool:
        return (self.lat is not None and self.lon is not None) or bool(self.city)


@dataclass(slots=True)
class EventClusterResult:
    photos_considered: int = 0           # 拍攝時間可信、可參與事件分群的照片數
    events_created: int = 0
    photos_in_events: int = 0
    home_city: str | None = None         # 自動判定的常駐城市（連續行程規則在這裡不適用）
    burst_groups: int = 0
    burst_photos: int = 0                # 屬於任何連拍群組的照片數
    burst_duplicates: int = 0            # 其中被標為次佳的照片數


class EventClusterEngine:
    """時間密度／地理事件分群，以及連拍群組與最佳照片挑選。

    事件（`cluster_events`）
    ------------------------
    照片依拍攝時間排序，相鄰兩張符合下列任一條件就屬於同一個事件：
      1. 時間差小於 `time_gap_hours`（預設 6 小時）。
      2. 連續行程：時間差小於 `trip_gap_hours`（預設 36 小時，容許過夜），而且位置沒有離開 ——
         與本段最近一張有位置的照片在同一個城市（離線逆地理編碼的結果），或 Haversine 距離小於
         `trip_distance_km`（預設 30 公里）。
    所以「京都三天兩夜」會是一個事件，而不是被每晚的空檔切成三段。

    規則 2 有兩道防護，否則「每天在家拍小孩」會被串成一個長達數月的事件：
      * `trip_gap_hours` 上限。
      * 常駐城市不適用規則 2。常駐城市預設自動判定為「有照片的天數最多」的城市
        （`home_city="auto"`）；傳入城市名稱可指定，傳 None 則停用這道防護。

    只採用拍攝時間可信的照片（來自 EXIF 或檔名）；時間來自檔案 mtime 的照片與螢幕截圖不參與。
    每個事件的預設名稱是 `YYYY-MM-DD_{City}_{TopTag}`（沒有城市或標籤就省略那一段）。
    每次都對全庫重算並取代所有自動事件；使用者編輯過的事件不動。

    連拍（`detect_bursts`）
    -----------------------
    相鄰兩張時間差 `< burst_seconds`（預設 3 秒）且 64-bit dHash 漢明距離 `<= burst_hamming`
    （預設 4）就串成同一組，`burst_group_id` 取群內最小的 photo id。
    每組以「清晰度 + 人臉偵測信心度」綜合評分，最高者 `is_best_shot = 1`，其餘
    `is_burst_duplicate = 1`。使用者標為最愛的優先當選；標為廢片的排在最後。
    """

    def __init__(
        self,
        db: DBManager,
        *,
        time_gap_hours: float = TIME_GAP_HOURS,
        trip_distance_km: float = TRIP_DISTANCE_KM,
        trip_gap_hours: float = TRIP_GAP_HOURS,
        min_photos: int = MIN_EVENT_PHOTOS,
        home_city: str | None = "auto",
        burst_seconds: float = BURST_SECONDS,
        burst_hamming: int = BURST_HAMMING,
        burst_weights: tuple[float, float] = (0.7, 0.3),
        geocoder: ReverseGeocoder | None = None,
    ) -> None:
        """
        Args:
            min_photos: 張數少於此值的段不成立事件（那些照片的 `event_id` 留 NULL）。
            burst_weights: 連拍評分的權重 (清晰度, 人臉偵測信心度)。
            geocoder: 給定時，分群前先替「有 GPS 但還沒有城市」的照片補上離線逆地理編碼。
                管線內執行時地理編碼已是前一個後處理步驟，不需要傳。
        """
        self._db = db
        self._repo = Repository(db)
        self.time_gap_s = time_gap_hours * 3600.0
        self.trip_gap_s = max(trip_gap_hours, time_gap_hours) * 3600.0
        self.trip_distance_km = trip_distance_km
        self.min_photos = max(1, min_photos)
        self.home_city = home_city
        self.burst_seconds = burst_seconds
        self.burst_hamming = burst_hamming
        self.burst_weights = burst_weights
        self._geocoder = geocoder

    @classmethod
    def from_config(cls, cfg: AppConfig, db: DBManager, geocoder: ReverseGeocoder | None = None
                    ) -> "EventClusterEngine":
        e = cfg.events
        return cls(db, time_gap_hours=e.gap_minutes / 60.0, trip_distance_km=e.max_distance_km,
                   min_photos=e.min_photos, geocoder=geocoder)

    # ------------------------------------------------------------------ #
    def run(self) -> EventClusterResult:
        """事件分群 + 連拍偵測。"""
        result = EventClusterResult()
        self.cluster_events(result)
        self.detect_bursts(result)
        return result

    def _load_shots(self) -> list[_Shot]:
        rows = self._db.query(
            "SELECT id, taken_ts, path, gps_lat, gps_lon, country_code, city, dhash, blur_score, file_size, "
            "is_favorite, is_rejected FROM photos "
            "WHERE taken_ts IS NOT NULL AND taken_src IN ('exif', 'filename') "
            "AND is_screenshot = 0 AND is_missing = 0"
        )
        shots = [
            _Shot(r["id"], r["taken_ts"], r["path"], r["gps_lat"], r["gps_lon"], r["country_code"] or None,
                  r["city"] or None, r["dhash"], r["blur_score"] or 0.0, r["file_size"] or 0,
                  bool(r["is_favorite"]), bool(r["is_rejected"]))
            for r in rows
        ]
        # 同一秒內依路徑排序：連拍的檔名通常是連號
        shots.sort(key=lambda s: (s.ts, s.path))
        return shots

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #
    def cluster_events(self, result: EventClusterResult | None = None) -> EventClusterResult:
        result = result or EventClusterResult()
        if self._geocoder is not None:
            pending = self._repo.fetch_ungeocoded()
            for part in _chunks(pending, 5000):
                places = self._geocoder.lookup([(lat, lon) for _id, lat, lon in part])
                self._repo.save_places([(pid, place) for (pid, _lat, _lon), place in zip(part, places)])

        shots = self._load_shots()
        result.photos_considered = len(shots)
        home = self._resolve_home(shots)
        result.home_city = home

        segments: list[list[_Shot]] = []
        current: list[_Shot] = []
        anchor: _Shot | None = None                      # 本段最近一張有位置的照片
        for shot in shots:
            if current and not self._same_event(current[-1], shot, anchor, home):
                segments.append(current)
                current, anchor = [], None
            current.append(shot)
            if shot.located:
                anchor = shot
        if current:
            segments.append(current)

        # 已在「使用者編輯過的事件」裡的照片不會被重新指派。分段時仍把它們算進去（時間軸才連續），
        # 分完再排除，剩下的照片數達到 min_photos 才成立新的自動事件 ——
        # 否則使用者的事件旁邊每新增一張照片，就會冒出一個只有 1 張的事件。
        locked = {r["id"] for r in self._db.query(
            "SELECT p.id FROM photos p JOIN events e ON e.id = p.event_id WHERE e.is_user_edited = 1")}
        free = [[s for s in seg if s.id not in locked] for seg in segments]
        clusters = [self._to_cluster(seg) for seg in free if len(seg) >= self.min_photos]
        result.events_created = self._repo.replace_auto_events(clusters)
        result.photos_in_events = int(self._db.query_value(
            "SELECT COUNT(*) FROM photos WHERE event_id IN (SELECT id FROM events WHERE is_user_edited = 0)",
            default=0))
        self._name_auto_events()
        log.info("事件分群：%d 張照片 -> %d 個事件（常駐城市：%s）",
                 result.photos_considered, result.events_created, home or "無")
        return result

    def _resolve_home(self, shots: list[_Shot]) -> str | None:
        if self.home_city != "auto":
            return self.home_city
        days: dict[str, set[int]] = {}
        for s in shots:
            if s.city:
                days.setdefault(s.city, set()).add(s.ts // 86400)
        if not days:
            return None
        # 有照片的「天數」最多的城市；用天數而不是張數，一次旅行拍很多張也不會被當成住處
        return max(days.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]

    def _same_event(self, prev: _Shot, cur: _Shot, anchor: _Shot | None, home: str | None) -> bool:
        gap = cur.ts - prev.ts
        if gap < self.time_gap_s:
            return True
        if gap >= self.trip_gap_s or anchor is None or not cur.located:
            return False
        if home is not None and cur.city == home and anchor.city == home:
            return False                                 # 在住處：不套用連續行程規則
        if cur.city and anchor.city and cur.city == anchor.city and cur.country == anchor.country:
            return True
        if None not in (cur.lat, cur.lon, anchor.lat, anchor.lon):
            return haversine_km(anchor.lat, anchor.lon, cur.lat, cur.lon) < self.trip_distance_km  # type: ignore[arg-type]
        return False

    @staticmethod
    def _to_cluster(segment: list[_Shot]) -> EventCluster:
        gps = [(s.lat, s.lon) for s in segment if s.lat is not None and s.lon is not None]
        lat = sum(g[0] for g in gps) / len(gps) if gps else None       # type: ignore[misc]
        lon = sum(g[1] for g in gps) / len(gps) if gps else None       # type: ignore[misc]
        return EventCluster(photo_ids=[s.id for s in segment], start_ts=segment[0].ts, end_ts=segment[-1].ts,
                            center_lat=lat, center_lon=lon)

    def _name_auto_events(self) -> None:
        """替自動事件產生預設名稱 `YYYY-MM-DD_{City}_{TopTag}`。"""

        def op(conn: sqlite3.Connection) -> None:
            events = conn.execute("SELECT id, start_ts FROM events WHERE is_user_edited = 0").fetchall()
            for ev in events:
                city = conn.execute(
                    "SELECT city FROM photos WHERE event_id = ? AND city IS NOT NULL AND city != '' "
                    "GROUP BY city ORDER BY COUNT(*) DESC, city LIMIT 1",
                    (ev["id"],),
                ).fetchone()
                tag = conn.execute(
                    "SELECT t.name FROM photos p JOIN photo_tags pt ON pt.photo_id = p.id "
                    "JOIN tags t ON t.id = pt.tag_id WHERE p.event_id = ? AND pt.is_rejected = 0 "
                    "GROUP BY t.id ORDER BY COUNT(*) DESC, SUM(COALESCE(pt.score, 1.0)) DESC, t.name LIMIT 1",
                    (ev["id"],),
                ).fetchone()
                parts = [time.strftime("%Y-%m-%d", time.gmtime(ev["start_ts"])),
                         city["city"] if city else None, tag["name"] if tag else None]
                conn.execute("UPDATE events SET name = ? WHERE id = ?",
                             ("_".join(p for p in parts if p), ev["id"]))

        self._db.write(op)

    # ------------------------------------------------------------------ #
    # 連拍
    # ------------------------------------------------------------------ #
    def detect_bursts(self, result: EventClusterResult | None = None) -> EventClusterResult:
        result = result or EventClusterResult()
        shots = [s for s in self._load_shots() if s.dhash is not None]

        groups: list[list[_Shot]] = []
        current: list[_Shot] = []
        for shot in shots:
            if current and (shot.ts - current[-1].ts < self.burst_seconds
                            and hamming64(shot.dhash, current[-1].dhash) <= self.burst_hamming):  # type: ignore[arg-type]
                current.append(shot)
                continue
            if len(current) >= 2:
                groups.append(current)
            current = [shot]
        if len(current) >= 2:
            groups.append(current)

        face_conf = self._face_confidence([s.id for g in groups for s in g])
        w_blur, w_face = self.burst_weights
        updates: list[tuple[int, int, int, int]] = []     # (group_id, is_best, is_duplicate, photo_id)
        for group in groups:
            max_blur = max(s.blur for s in group) or 1.0

            def rank(s: _Shot) -> tuple:
                score = w_blur * (s.blur / max_blur) + w_face * face_conf.get(s.id, 0.0)
                # 由左到右比較：未被標為廢片 > 最愛 > 綜合分數 > 檔案較大 > id 較小
                return (not s.rejected, s.favorite, score, s.size, -s.id)

            best = max(group, key=rank)
            group_id = min(s.id for s in group)
            for s in group:
                updates.append((group_id, int(s is best), int(s is not best), s.id))

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE photos SET burst_group_id = NULL, is_best_shot = 0, is_burst_duplicate = 0 "
                "WHERE burst_group_id IS NOT NULL OR is_best_shot = 1 OR is_burst_duplicate = 1"
            )
            conn.executemany(
                "UPDATE photos SET burst_group_id = ?, is_best_shot = ?, is_burst_duplicate = ? WHERE id = ?",
                updates,
            )

        self._db.write(op)
        result.burst_groups = len(groups)
        result.burst_photos = len(updates)
        result.burst_duplicates = len(updates) - len(groups)
        log.info("連拍偵測：%d 組，共 %d 張，其中 %d 張標為次佳", result.burst_groups, result.burst_photos,
                 result.burst_duplicates)
        return result

    def _face_confidence(self, photo_ids: list[int]) -> dict[int, float]:
        """每張照片的人臉偵測信心度（該張所有臉的平均 det_score）；沒有臉的照片不在結果裡。"""
        out: dict[int, float] = {}
        for part in _chunks(photo_ids):
            for r in self._db.query(
                "SELECT photo_id, AVG(det_score) AS conf FROM faces "
                f"WHERE is_ignored = 0 AND photo_id IN ({','.join('?' * len(part))}) GROUP BY photo_id",
                tuple(part),
            ):
                out[r["photo_id"]] = float(r["conf"] or 0.0)
        return out

