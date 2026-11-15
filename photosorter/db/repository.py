"""業務查詢層：管線與上層用到的所有 SQL 都集中在這裡。

`DBManager` 只管「怎麼安全地讀寫」，這裡管「讀寫什麼」。GUI 需要的新查詢請加在這個類別
（或另開 `db/queries_ui.py`），不要把 SQL 散落在 UI 程式碼裡。

命名慣例：
  * `fetch_* / load_* / find_* / iter_*` —— 讀取，走呼叫執行緒自己的唯讀連線，立即回傳。
  * 其他 —— 寫入，經 writer 執行緒。預設**同步**（等 commit 完成才回傳）；
    回傳值是 commit 後的結果。需要非同步請直接用 `self.db.submit(...)`。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..core.status import ProcessStatus
from ..core.types import (
    EventCluster,
    EventInput,
    FileCandidate,
    GateRole,
    GeoPlace,
    InferenceResult,
    Stage1Record,
    TagSpec,
)
from .db_manager import DBManager

_ERR = int(ProcessStatus.error_bits())
_STAGE1 = int(ProcessStatus.stage1())
_AI = int(ProcessStatus.ai_bits() | ProcessStatus.FACES_SKIPPED)
_IN_CHUNK = 500


def _chunks(seq: Sequence[Any], n: int = _IN_CHUNK) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def vec_to_blob(v: np.ndarray) -> bytes:
    """向量 -> BLOB。固定 float32 little-endian，與 sqlite-vec 相容。"""
    return np.ascontiguousarray(v, dtype="<f4").tobytes()


def blob_to_vec(b: bytes, dim: int | None = None) -> np.ndarray:
    """BLOB -> 向量（唯讀 view，要改請先 `.copy()`）。"""
    v = np.frombuffer(b, dtype="<f4")
    if dim is not None and v.size != dim:
        raise ValueError(f"向量長度 {v.size} 與預期的 {dim} 不符")
    return v


@dataclass(slots=True)
class ScanIndexEntry:
    """Stage 0 用的精簡索引：資料庫裡一張照片的檔案指紋與狀態。"""

    photo_id: int
    size: int
    mtime_ns: int
    status: ProcessStatus
    sha256: bytes | None
    is_missing: bool


@dataclass(slots=True)
class Sha256Match:
    photo_id: int
    path: str
    status: ProcessStatus


class Repository:
    def __init__(self, db: DBManager) -> None:
        self.db = db
        self._model_ids: dict[tuple[str, str], int] = {}
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 模型登錄
    # ------------------------------------------------------------------ #
    def ensure_model(self, name: str, kind: str, dim: int | None) -> int:
        """取得（必要時建立）模型的 id。結果快取在記憶體。"""
        key = (name, kind)
        with self._model_lock:
            if key in self._model_ids:
                return self._model_ids[key]

        def op(conn: sqlite3.Connection) -> int:
            conn.execute(
                "INSERT INTO models(name, kind, dim) VALUES (?, ?, ?) "
                "ON CONFLICT(name, kind) DO UPDATE SET dim = excluded.dim",
                (name, kind, dim),
            )
            return int(conn.execute(
                "SELECT id FROM models WHERE name = ? AND kind = ?", key
            ).fetchone()[0])

        model_id = self.db.write(op)
        with self._model_lock:
            self._model_ids[key] = model_id
        return model_id

    # ------------------------------------------------------------------ #
    # 掃描工作階段
    # ------------------------------------------------------------------ #
    def begin_scan(self, roots: Sequence[str]) -> int:
        return self.db.execute(
            "INSERT INTO scan_sessions(roots) VALUES (?)", (json.dumps(list(roots), ensure_ascii=False),)
        ).result()

    def finish_scan(self, scan_id: int, state: str, stats: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE scan_sessions SET state = ?, finished_at = unixepoch(), stats = ? WHERE id = ?",
            (state, json.dumps(stats, ensure_ascii=False), scan_id),
        ).result()

    # ------------------------------------------------------------------ #
    # Stage 0
    # ------------------------------------------------------------------ #
    def load_scan_index(self, roots: Sequence[str]) -> dict[str, ScanIndexEntry]:
        """把各根目錄底下已入庫照片的指紋一次載入記憶體（十萬張實測約 0.3 秒）。

        以 `path` 的 UNIQUE 索引做前綴範圍查詢，不是全表掃描。
        """
        out: dict[str, ScanIndexEntry] = {}
        for root in roots:
            prefix = root.rstrip("\\/") + os.sep
            upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)
            for r in self.db.iter_query(
                "SELECT id, path, file_size, mtime_ns, process_status, sha256, is_missing "
                "FROM photos WHERE path >= ? AND path < ?",
                (prefix, upper),
            ):
                out[r["path"]] = ScanIndexEntry(
                    r["id"], r["file_size"], r["mtime_ns"], ProcessStatus(r["process_status"]),
                    r["sha256"], bool(r["is_missing"]),
                )
        return out

    def find_by_sha256(self, sha256: bytes) -> list[Sha256Match]:
        rows = self.db.query(
            "SELECT id, path, process_status FROM photos WHERE sha256 = ? ORDER BY id", (sha256,)
        )
        return [Sha256Match(r["id"], r["path"], ProcessStatus(r["process_status"])) for r in rows]

    def relocate(self, photo_id: int, old_path: str, cand: FileCandidate, scan_id: int | None) -> bool:
        """檔案搬移：只改路徑，AI 結果全部保留。

        以 `WHERE path = old_path` 當樂觀鎖：兩條解碼執行緒同時想把同一列搬到不同新路徑時，
        只有一個會成功，另一個回傳 False（呼叫端把它當新檔處理）。
        """
        d, name = os.path.split(cand.path)

        def op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE photos SET path = ?, dir_path = ?, file_name = ?, ext = ?, file_size = ?, "
                "mtime_ns = ?, is_missing = 0, last_scan_id = ? WHERE id = ? AND path = ?",
                (cand.path, d, name, _ext(name), cand.size, cand.mtime_ns, scan_id, photo_id, old_path),
            )
            return cur.rowcount == 1

        try:
            return self.db.write(op)
        except sqlite3.IntegrityError:
            return False                       # 新路徑已被別列佔用

    def touch_file(self, photo_id: int, size: int, mtime_ns: int, scan_id: int | None) -> None:
        """內容沒變（sha256 相同）只是 mtime 被動過：更新指紋，其餘不動。"""
        self.db.execute(
            "UPDATE photos SET file_size = ?, mtime_ns = ?, is_missing = 0, last_scan_id = ? WHERE id = ?",
            (size, mtime_ns, scan_id, photo_id),
        ).result()

    def mark_missing(self, entries: Sequence[tuple[int, str]]) -> int:
        """把掃描時沒看到的照片標為遺失（保留紀錄，不刪除）。回傳實際標記的列數。

        Args:
            entries: `(photo_id, 掃描當下資料庫裡的路徑)`。

        以 `id + path` 當條件：掃描之後、標記之前，這一列若已被判定為「搬移」而改了路徑
        （`relocate()`），條件就不成立，不會被誤標成遺失。
        """
        if not entries:
            return 0
        rows = list(entries)

        def op(conn: sqlite3.Connection) -> int:
            return conn.executemany(
                "UPDATE photos SET is_missing = 1 WHERE id = ? AND path = ? AND is_missing = 0", rows
            ).rowcount

        return self.db.write(op)

    def clear_missing(self, photo_ids: Sequence[int]) -> None:
        """之前標為遺失的檔案又出現了。"""
        if not photo_ids:
            return
        ids = list(photo_ids)

        def op(conn: sqlite3.Connection) -> None:
            for part in _chunks(ids):
                conn.execute(
                    f"UPDATE photos SET is_missing = 0 WHERE id IN ({','.join('?' * len(part))})",
                    tuple(part),
                )

        self.db.write(op)

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    def upsert_stage1(self, rec: Stage1Record, scan_id: int | None) -> int:
        """寫入 Stage 1 結果並設上 Stage 1 位元。回傳 photo_id。

        路徑已存在時：
          * sha256 相同 -> 保留既有的 AI 結果與狀態（只是補做或重做 Stage 1）。
          * sha256 不同 -> 內容換了：清掉向量、自動標籤、人臉、地名，狀態歸零後只留 Stage 1。
            使用者手動加的標籤、廢片/最愛標記屬於「這個路徑」，予以保留。
        """
        m = rec.meta
        d, name = os.path.split(rec.path)
        cols = dict(
            path=rec.path, dir_path=d, file_name=name, ext=_ext(name),
            file_size=rec.file_size, mtime_ns=rec.mtime_ns, sha256=rec.sha256,
            width=rec.width, height=rec.height, orientation=m.orientation,
            taken_at=m.taken_at, taken_ts=m.taken_ts, taken_src=m.taken_src,
            camera_make=m.camera_make, camera_model=m.camera_model, lens_model=m.lens_model,
            iso=m.iso, f_number=m.f_number, exposure_time=m.exposure_time, focal_length=m.focal_length,
            gps_lat=m.gps_lat, gps_lon=m.gps_lon, gps_alt=m.gps_alt,
            blur_score=rec.blur_score, dhash=rec.dhash, is_screenshot=int(rec.is_screenshot),
            last_scan_id=scan_id,
        )

        def op(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                "SELECT id, sha256, process_status FROM photos WHERE path = ?", (rec.path,)
            ).fetchone()
            if row is None:
                names = list(cols) + ["process_status"]
                cur = conn.execute(
                    f"INSERT INTO photos({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                    (*cols.values(), _STAGE1),
                )
                return int(cur.lastrowid)

            photo_id = int(row["id"])
            if row["sha256"] == rec.sha256:
                status = (int(row["process_status"]) & ~_ERR) | _STAGE1
                extra = ""
            else:
                _delete_derived(conn, photo_id)
                status = _STAGE1
                extra = ", country_code = NULL, admin1 = NULL, city = NULL, phash = NULL, dup_group_id = NULL"
            sets = ", ".join(f"{k} = ?" for k in cols)
            conn.execute(
                f"UPDATE photos SET {sets}, process_status = ?, is_missing = 0, last_error = NULL{extra} "
                "WHERE id = ?",
                (*cols.values(), status, photo_id),
            )
            return photo_id

        return self.db.write(op)

    def record_decode_error(self, cand: FileCandidate, sha256: bytes | None, message: str,
                            scan_id: int | None) -> int:
        """讀檔/解碼失敗也要留一列，否則每次重掃都會再試一次同一個壞檔。"""
        d, name = os.path.split(cand.path)
        err = int(ProcessStatus.ERR_DECODE)

        def op(conn: sqlite3.Connection) -> int:
            conn.execute(
                "INSERT INTO photos(path, dir_path, file_name, ext, file_size, mtime_ns, sha256, "
                "process_status, error_count, last_error, last_scan_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET file_size = excluded.file_size, "
                "mtime_ns = excluded.mtime_ns, sha256 = COALESCE(excluded.sha256, photos.sha256), "
                "process_status = photos.process_status | excluded.process_status, "
                "error_count = photos.error_count + 1, last_error = excluded.last_error, "
                "is_missing = 0, last_scan_id = excluded.last_scan_id",
                (cand.path, d, name, _ext(name), cand.size, cand.mtime_ns, sha256, err,
                 message[:2000], scan_id),
            )
            return int(conn.execute("SELECT id FROM photos WHERE path = ?", (cand.path,)).fetchone()[0])

        return self.db.write(op)

    def clone_derived(self, src_id: int, dst_id: int) -> None:
        """內容完全相同的另一個檔案：直接複製 AI 結果，不重跑推論。"""

        def op(conn: sqlite3.Connection) -> None:
            _delete_derived(conn, dst_id)
            conn.execute(
                "INSERT INTO embeddings(photo_id, model_id, dim, vector) "
                "SELECT ?, model_id, dim, vector FROM embeddings WHERE photo_id = ?",
                (dst_id, src_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO photo_tags(photo_id, tag_id, score, source) "
                "SELECT ?, tag_id, score, 'auto' FROM photo_tags "
                "WHERE photo_id = ? AND source = 'auto' AND is_rejected = 0",
                (dst_id, src_id),
            )
            conn.execute(
                "INSERT INTO faces(photo_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, landmarks, quality, "
                "model_id, embedding, person_id, cluster_id, assign_src) "
                "SELECT ?, bbox_x, bbox_y, bbox_w, bbox_h, det_score, landmarks, quality, "
                "model_id, embedding, person_id, cluster_id, assign_src FROM faces WHERE photo_id = ?",
                (dst_id, src_id),
            )
            conn.execute(
                "UPDATE photos SET process_status = (process_status & ?) | "
                "(SELECT process_status & ? FROM photos WHERE id = ?) WHERE id = ?",
                (~_AI, _AI, src_id, dst_id),
            )

        self.db.write(op)

    # ------------------------------------------------------------------ #
    # Stage 2 / 3
    # ------------------------------------------------------------------ #
    def persist_inference(
        self,
        results: Sequence[InferenceResult],
        image_model_id: int | None,
        face_model_id: int | None,
    ) -> None:
        """把一個 batch 的推論結果連同狀態位元寫進**同一個交易**。

        這是續掃正確性的關鍵：位元與資料要嘛一起在、要嘛一起不在。
        某個階段失敗的照片（`error` 有值）仍會寫入先前階段已完成的結果與對應位元，
        另外設上 ERR_INFER；下次重試只需要補失敗的那個階段。
        """
        err_bit = int(ProcessStatus.ERR_INFER)

        def op(conn: sqlite3.Connection) -> None:
            for r in results:
                if r.embedding is not None:
                    conn.execute(
                        "INSERT INTO embeddings(photo_id, model_id, dim, vector) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(photo_id, model_id) DO UPDATE SET dim = excluded.dim, "
                        "vector = excluded.vector, created_at = unixepoch()",
                        (r.photo_id, image_model_id, int(r.embedding.size), vec_to_blob(r.embedding)),
                    )
                if r.tags is not None:
                    _replace_auto_tags(conn, r.photo_id, [(h.tag_id, h.score) for h in r.tags])
                if r.faces is not None:
                    conn.execute("DELETE FROM faces WHERE photo_id = ?", (r.photo_id,))
                    conn.executemany(
                        "INSERT INTO faces(photo_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, landmarks, "
                        "quality, model_id, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                r.photo_id, *map(float, f.bbox), float(f.det_score),
                                vec_to_blob(f.landmarks) if f.landmarks is not None else None,
                                f.quality, face_model_id,
                                vec_to_blob(f.embedding) if f.embedding is not None else None,
                            )
                            for f in r.faces
                        ],
                    )
                if r.error is not None:
                    conn.execute(
                        "UPDATE photos SET process_status = process_status | ?, "
                        "error_count = error_count + 1, last_error = ? WHERE id = ?",
                        (int(r.status_set) | err_bit, r.error[:2000], r.photo_id),
                    )
                else:
                    conn.execute(
                        "UPDATE photos SET process_status = (process_status & ?) | ?, last_error = NULL "
                        "WHERE id = ?",
                        (~err_bit, int(r.status_set), r.photo_id),
                    )

        self.db.write(op)

    def fetch_gate_roles(self, photo_ids: Sequence[int]) -> dict[int, set[GateRole]]:
        """續掃用：Stage 2 上一輪已做完、這一輪只補 Stage 3 的照片，從資料庫取回它命中的 gate 角色。"""
        out: dict[int, set[GateRole]] = {}
        for part in _chunks(list(photo_ids)):
            for r in self.db.query(
                "SELECT pt.photo_id, t.gate_role FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
                f"WHERE pt.is_rejected = 0 AND t.gate_role != 'none' "
                f"AND pt.photo_id IN ({','.join('?' * len(part))})",
                tuple(part),
            ):
                out.setdefault(r["photo_id"], set()).add(GateRole(r["gate_role"]))
        return out

    # ------------------------------------------------------------------ #
    # 標籤
    # ------------------------------------------------------------------ #
    def upsert_tag(self, name: str, prompt: str | None, *, display_name: str | None = None,
                   threshold: float | None = None, gate_role: GateRole = GateRole.NONE) -> int:
        """新增或更新標籤。prompt 改變時資料庫 trigger 會自動作廢快取的文字向量。"""

        def op(conn: sqlite3.Connection) -> int:
            conn.execute(
                "INSERT INTO tags(name, display_name, prompt, threshold, gate_role) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET display_name = excluded.display_name, "
                "prompt = excluded.prompt, threshold = excluded.threshold, gate_role = excluded.gate_role",
                (name, display_name, prompt, threshold, gate_role.value),
            )
            return int(conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()[0])

        return self.db.write(op)

    def count_tags(self) -> int:
        return int(self.db.query_value("SELECT COUNT(*) FROM tags", default=0))

    def load_auto_tags(self, text_model_id: int | None, default_threshold: float, dim: int) -> list[TagSpec]:
        """所有啟用中、有 prompt 的標籤。文字向量若不是目前的模型算的，回傳 None 讓呼叫端重算。"""
        out: list[TagSpec] = []
        for r in self.db.query(
            "SELECT id, name, prompt, threshold, gate_role, text_vector, text_model_id FROM tags "
            "WHERE is_enabled = 1 AND prompt IS NOT NULL AND prompt != '' ORDER BY id"
        ):
            vec = None
            if r["text_vector"] is not None and r["text_model_id"] == text_model_id:
                v = blob_to_vec(r["text_vector"])
                if v.size == dim:
                    vec = v
            out.append(TagSpec(
                id=r["id"], name=r["name"], prompt=r["prompt"],
                threshold=default_threshold if r["threshold"] is None else float(r["threshold"]),
                gate_role=GateRole(r["gate_role"]), text_vector=vec,
            ))
        return out

    def save_text_vectors(self, text_model_id: int, vectors: dict[int, np.ndarray]) -> None:
        rows = [(vec_to_blob(v), text_model_id, tag_id) for tag_id, v in vectors.items()]
        self.db.executemany(
            "UPDATE tags SET text_vector = ?, text_model_id = ? WHERE id = ?", rows
        ).result()

    def replace_auto_tags_bulk(self, hits: dict[int, list[tuple[int, float]]]) -> None:
        """標籤定義改變後的全庫重標：`{photo_id: [(tag_id, score), …]}`，不需重跑 vision encoder。"""

        def op(conn: sqlite3.Connection) -> None:
            for photo_id, pairs in hits.items():
                _replace_auto_tags(conn, photo_id, pairs)

        self.db.write(op)

    # ------------------------------------------------------------------ #
    # 向量載入（給 VectorIndex、重標、分群）
    # ------------------------------------------------------------------ #
    def load_embedding_matrix(self, model_id: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """一次載入某模型的全部影像向量。

        Returns:
            (ids, matrix)：ids 為 (N,) int64，matrix 為 (N, dim) float32（可寫）。
            10 萬張 x 768 維 ≈ 293 MB。
        """
        n = int(self.db.query_value(
            "SELECT COUNT(*) FROM embeddings WHERE model_id = ? AND dim = ?", (model_id, dim), 0))
        ids = np.empty(n, dtype=np.int64)
        mat = np.empty((n, dim), dtype=np.float32)
        i = 0
        for r in self.db.iter_query(
            "SELECT photo_id, vector FROM embeddings WHERE model_id = ? AND dim = ? ORDER BY photo_id",
            (model_id, dim),
        ):
            if i >= n:                         # 載入途中有新資料寫入：這次先不取，之後增量 add
                break
            ids[i] = r["photo_id"]
            mat[i] = blob_to_vec(r["vector"], dim)
            i += 1
        return ids[:i], mat[:i]

    def load_face_matrix(self, dim: int, *, only_unassigned: bool = False
                         ) -> tuple[np.ndarray, np.ndarray]:
        """載入人臉特徵供分群。排除被使用者標為忽略的臉。Returns: (face_ids, (N, dim) float32)。"""
        where = "embedding IS NOT NULL AND is_ignored = 0"
        if only_unassigned:
            where += " AND person_id IS NULL"
        rows = self.db.query(f"SELECT id, embedding FROM faces WHERE {where} ORDER BY id")
        ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = blob_to_vec(r["embedding"], dim)
        return ids, mat

    # ------------------------------------------------------------------ #
    # 地理
    # ------------------------------------------------------------------ #
    def fetch_ungeocoded(self) -> list[tuple[int, float, float]]:
        rows = self.db.query(
            "SELECT id, gps_lat, gps_lon FROM photos WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL "
            "AND country_code IS NULL"
        )
        return [(r["id"], r["gps_lat"], r["gps_lon"]) for r in rows]

    def save_places(self, places: Sequence[tuple[int, GeoPlace]]) -> None:
        bit = int(ProcessStatus.GEOCODED)
        rows = [(p.country_code or "", p.admin1, p.city, bit, pid) for pid, p in places]
        self.db.executemany(
            "UPDATE photos SET country_code = ?, admin1 = ?, city = ?, "
            "process_status = process_status | ? WHERE id = ?",
            rows,
        ).result()

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #
    def fetch_event_inputs(self) -> list[EventInput]:
        """可參與事件聚類的照片：拍攝時間可信（來自 EXIF 或檔名）、不是截圖、檔案還在。

        時間來源是 mtime 的照片不參與 —— 複製/下載會改 mtime，拿來聚類只會產生假事件。
        """
        rows = self.db.query(
            "SELECT id, taken_ts, gps_lat, gps_lon FROM photos "
            "WHERE taken_ts IS NOT NULL AND taken_src IN ('exif','filename') "
            "AND is_screenshot = 0 AND is_missing = 0"
        )
        return [EventInput(r["id"], r["taken_ts"], r["gps_lat"], r["gps_lon"]) for r in rows]

    def replace_auto_events(self, clusters: Sequence[EventCluster]) -> int:
        """以新的聚類結果取代所有「自動」事件。使用者編輯過的事件（`is_user_edited = 1`）
        與其中的照片完全不動。回傳建立的事件數。"""

        def op(conn: sqlite3.Connection) -> int:
            conn.execute("DELETE FROM events WHERE is_user_edited = 0")   # FK: photos.event_id -> NULL
            locked = {r[0] for r in conn.execute("SELECT id FROM photos WHERE event_id IS NOT NULL")}
            created = 0
            for c in clusters:
                ids = [p for p in c.photo_ids if p not in locked]
                if not ids:
                    continue
                event_id = conn.execute(
                    "INSERT INTO events(start_ts, end_ts, photo_count, center_lat, center_lon) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (c.start_ts, c.end_ts, len(ids), c.center_lat, c.center_lon),
                ).lastrowid
                for part in _chunks(ids):
                    conn.execute(
                        f"UPDATE photos SET event_id = ? WHERE id IN ({','.join('?' * len(part))})",
                        (event_id, *part),
                    )
                place = conn.execute(
                    "SELECT city, country_code FROM photos WHERE event_id = ? AND country_code != '' "
                    "GROUP BY city, country_code ORDER BY COUNT(*) DESC LIMIT 1",
                    (event_id,),
                ).fetchone()
                cover = conn.execute(
                    "SELECT id FROM photos WHERE event_id = ? AND is_rejected = 0 "
                    "ORDER BY blur_score DESC LIMIT 1",
                    (event_id,),
                ).fetchone()
                label = None
                if place is not None:
                    label = ", ".join(x for x in (place["city"], place["country_code"]) if x)
                conn.execute(
                    "UPDATE events SET location_label = ?, cover_photo_id = ? WHERE id = ?",
                    (label, cover["id"] if cover else None, event_id),
                )
                created += 1
            return created

        return self.db.write(op)

    # ------------------------------------------------------------------ #
    # 匯出
    # ------------------------------------------------------------------ #
    def fetch_export_rows(self, photo_ids: Iterable[int] | None = None) -> list[sqlite3.Row]:
        """匯出計畫需要的欄位。`photo_ids=None` 代表全部（未遺失、未標廢片）。

        `primary_tag` 取分數最高的標籤；`primary_person` 取畫面中最大的已命名人臉。
        """
        base = (
            "SELECT p.id, p.path, p.file_name, p.ext, p.file_size, p.sha256, p.taken_at, p.taken_ts, "
            "p.mtime_ns, p.camera_make, p.camera_model, p.country_code, p.admin1, p.city, "
            "p.is_screenshot, e.name AS event_name, e.start_ts AS event_start_ts, e.location_label, "
            "(SELECT COALESCE(t.display_name, t.name) FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            "  WHERE pt.photo_id = p.id AND pt.is_rejected = 0 "
            "  ORDER BY (pt.source = 'manual') DESC, pt.score DESC LIMIT 1) AS primary_tag, "
            "(SELECT pe.name FROM faces f JOIN persons pe ON pe.id = f.person_id "
            "  WHERE f.photo_id = p.id AND pe.name IS NOT NULL "
            "  ORDER BY f.bbox_w * f.bbox_h DESC LIMIT 1) AS primary_person "
            "FROM photos p LEFT JOIN events e ON e.id = p.event_id "
        )
        if photo_ids is None:
            return self.db.query(base + "WHERE p.is_missing = 0 AND p.is_rejected = 0 ORDER BY p.id")
        out: list[sqlite3.Row] = []
        for part in _chunks(list(photo_ids)):
            out.extend(self.db.query(
                base + f"WHERE p.id IN ({','.join('?' * len(part))}) ORDER BY p.id", tuple(part)))
        return out

    # ------------------------------------------------------------------ #
    # 雜項
    # ------------------------------------------------------------------ #
    def kv_get(self, key: str, default: str | None = None) -> str | None:
        return self.db.query_value("SELECT value FROM kv WHERE key = ?", (key,), default)

    def kv_set(self, key: str, value: str | None) -> None:
        self.db.execute(
            "INSERT INTO kv(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = unixepoch()",
            (key, value),
        ).result()

    def library_stats(self) -> dict[str, int]:
        """給 CLI / 狀態列的總覽。"""
        q = self.db.query_value
        return {
            "photos": int(q("SELECT COUNT(*) FROM photos WHERE is_missing = 0", default=0)),
            "missing": int(q("SELECT COUNT(*) FROM photos WHERE is_missing = 1", default=0)),
            "embedded": int(q("SELECT COUNT(*) FROM embeddings", default=0)),
            "tagged_pairs": int(q("SELECT COUNT(*) FROM photo_tags WHERE is_rejected = 0", default=0)),
            "faces": int(q("SELECT COUNT(*) FROM faces", default=0)),
            "persons": int(q("SELECT COUNT(*) FROM persons", default=0)),
            "events": int(q("SELECT COUNT(*) FROM events", default=0)),
            "screenshots": int(q("SELECT COUNT(*) FROM photos WHERE is_screenshot = 1", default=0)),
            "errors": int(q("SELECT COUNT(*) FROM photos WHERE process_status & ? != 0", (_ERR,), 0)),
        }


# --------------------------------------------------------------------------- #
# writer 執行緒內使用的小工具
# --------------------------------------------------------------------------- #
def _ext(file_name: str) -> str:
    return os.path.splitext(file_name)[1].lstrip(".").lower()


def _delete_derived(conn: sqlite3.Connection, photo_id: int) -> None:
    """清掉由內容推導出的資料（向量、自動標籤、人臉）。手動標籤與使用者標記保留。"""
    conn.execute("DELETE FROM embeddings WHERE photo_id = ?", (photo_id,))
    conn.execute("DELETE FROM photo_tags WHERE photo_id = ? AND source = 'auto'", (photo_id,))
    conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))


def _replace_auto_tags(conn: sqlite3.Connection, photo_id: int, pairs: Sequence[tuple[int, float]]) -> None:
    """以新的自動標籤取代舊的。

    使用者的決定永遠優先：手動加的標籤、被使用者否決的標籤（`is_rejected = 1`）都不會被刪除或
    翻案，只更新它們的 score 供參考。
    """
    conn.execute(
        "DELETE FROM photo_tags WHERE photo_id = ? AND source = 'auto' AND is_rejected = 0", (photo_id,)
    )
    conn.executemany(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, ?, 'auto') "
        "ON CONFLICT(photo_id, tag_id) DO UPDATE SET score = excluded.score",
        [(photo_id, tag_id, float(score)) for tag_id, score in pairs],
    )
