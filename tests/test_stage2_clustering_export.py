"""第二階段：聚類（人臉、事件、連拍）與四層多視圖匯出。

全部在 `tempfile.TemporaryDirectory` 裡用模擬的照片檔與 Sidecar（.mov / .xmp / .arw）進行，
走的是真實的資料庫、真實的檔案操作；人臉特徵是合成的 512 維向量（不需要模型權重）。
"""
from __future__ import annotations

import calendar
import hashlib
import os
import re
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from photosorter.core import export_manager as em
from photosorter.core.events import EventBus, EventKind
from photosorter.core.export_manager import (
    GROUP_DIR,
    MASTER_DIR,
    MAX_PATH_LEN,
    NO_DATE_DIR,
    PERSON_DIR,
    SOLO_DIR,
    STAGING_BLURRY,
    STAGING_BURST,
    STAGING_DIR,
    STAGING_SCREENSHOT,
    TAG_DIR,
    ExportManager,
    clean_name,
    device_token,
    verify_archive,
)
from photosorter.db.db_manager import SCHEMA_VERSION, DBManager
from photosorter.db.repository import Repository, blob_to_vec, vec_to_blob
from photosorter.pipeline.decoder import DefaultImageDecoder
from photosorter.pipeline.export_manager import ExportError
from photosorter.pipeline.pipeline_engine import PipelineEngine
from photosorter.utils.clustering import EventClusterEngine, FaceClusterEngine, person_display_name

from .fakes import make_config, make_image

try:
    import sklearn  # noqa: F401

    HAVE_SKLEARN = True
except Exception:  # noqa: BLE001
    HAVE_SKLEARN = False

DIM = 512
NAME_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{3})_([0-9A-Za-z_\-]+)_([0-9a-f]{6})(_v\d+)?\.([a-z0-9]+)$")


def ts(text: str) -> int:
    """'2024-05-01 10:00:00' -> 資料庫用的 taken_ts（把當地時間當 UTC）。"""
    return calendar.timegm(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timetuple())


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def insert_photo(conn: sqlite3.Connection, pid: int, name: str, **cols: Any) -> None:
    """直接寫一列照片（聚類測試不需要真的檔案）。"""
    taken_ts = cols.pop("taken_ts", None)
    base = dict(
        id=pid, path=os.path.join("virtual", name), dir_path="virtual", file_name=name, ext="jpg",
        file_size=1000, mtime_ns=1, width=100, height=100, sha256=hashlib.sha256(name.encode()).digest(),
        taken_ts=taken_ts, taken_src="exif" if taken_ts is not None else None,
        taken_at=None if taken_ts is None
        else datetime.fromtimestamp(taken_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        blur_score=100.0, process_status=7,
    )
    base.update(cols)
    conn.execute(f"INSERT INTO photos({','.join(base)}) VALUES ({','.join('?' * len(base))})", tuple(base.values()))


def insert_face(conn: sqlite3.Connection, face_id: int, photo_id: int, vec: np.ndarray | None, *,
                person_id: int | None = None, assign_src: str | None = None, det: float = 0.9,
                size: float = 0.2) -> None:
    conn.execute(
        "INSERT INTO faces(id, photo_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding, person_id, assign_src) "
        "VALUES (?, ?, 0.1, 0.1, ?, ?, ?, ?, ?, ?)",
        (face_id, photo_id, size, size, det, None if vec is None else vec_to_blob(vec), person_id, assign_src),
    )


class DbCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # realpath：暫存資料夾可能是 8.3 短路徑，而引擎會把路徑正規化成長路徑
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.cfg = make_config(self.tmp, pipeline={"stages": ["meta"], "geocode": False, "cluster_events": False})
        self.db = DBManager(self.cfg.db_path, self.cfg.db).open()
        self.repo = Repository(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self._tmp.cleanup()

    def one(self, sql: str, params: tuple = ()) -> Any:
        return self.db.query_value(sql, params)


# =========================================================================== #
# 資料庫遷移
# =========================================================================== #
class MigrationTest(unittest.TestCase):
    def test_v1_database_is_upgraded_in_place_without_losing_data(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.db"
            with DBManager(path) as db:
                db.write(lambda c: insert_photo(c, 1, "keep.jpg"))
                db.execute("INSERT INTO persons(id, name) VALUES (1, 'Alice')").result()
            # 把它變回第一階段（v1）的樣子
            raw = sqlite3.connect(path)
            raw.executescript(
                "DROP INDEX idx_photos_burst;"
                "ALTER TABLE photos DROP COLUMN burst_group_id;"
                "ALTER TABLE photos DROP COLUMN is_best_shot;"
                "ALTER TABLE photos DROP COLUMN is_burst_duplicate;"
                "ALTER TABLE persons DROP COLUMN is_locked;"
                "PRAGMA user_version = 1;"
            )
            raw.close()

            with DBManager(path) as db:
                self.assertEqual(db.query_value("PRAGMA user_version"), SCHEMA_VERSION)
                photo_cols = {r["name"] for r in db.query("PRAGMA table_info(photos)")}
                self.assertTrue({"burst_group_id", "is_best_shot", "is_burst_duplicate"} <= photo_cols)
                row = db.query_one("SELECT name, is_locked FROM persons WHERE id = 1")
                self.assertEqual((row["name"], row["is_locked"]), ("Alice", 0))
                self.assertEqual(db.query_value("SELECT file_name FROM photos WHERE id = 1"), "keep.jpg")
                self.assertEqual(db.query_value("SELECT is_best_shot FROM photos WHERE id = 1"), 0)


# =========================================================================== #
# FaceClusterEngine
# =========================================================================== #
@unittest.skipUnless(HAVE_SKLEARN, "需要 scikit-learn")
class FaceClusterEngineTest(DbCase):
    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(7)
        # 512 維的隨機方向彼此幾乎正交：每個方向代表一個「人」
        self.base = {k: unit(self.rng.normal(size=DIM)) for k in "ABCDEFGH"}
        self.engine = FaceClusterEngine(self.db)

    def near(self, key: str) -> np.ndarray:
        """某個人的一張臉：基準方向加上少量雜訊（同一人的臉彼此 cosine distance 約 0.1）。"""
        return unit(self.base[key] + 0.015 * self.rng.normal(size=DIM))

    def faces_of(self, person_id: int) -> set[int]:
        return {r["id"] for r in self.db.query("SELECT id FROM faces WHERE person_id = ?", (person_id,))}

    def person_of(self, face_id: int) -> int | None:
        return self.one("SELECT person_id FROM faces WHERE id = ?", (face_id,))

    def seed(self, fn) -> None:
        def op(conn: sqlite3.Connection) -> None:
            insert_photo(conn, 1, "p.jpg")
            fn(conn)

        self.db.write(op)

    # ------------------------------------------------------------------ #
    def test_blob_matrix_roundtrip_and_dbscan_creates_persons(self) -> None:
        def fill(conn: sqlite3.Connection) -> None:
            fid = 1
            for key in "ABC":
                for _ in range(5):
                    insert_face(conn, fid, 1, self.near(key))
                    fid += 1
            insert_face(conn, 100, 1, unit(self.rng.normal(size=DIM)))       # 路人
            insert_face(conn, 101, 1, unit(self.rng.normal(size=DIM)))
            insert_face(conn, 102, 1, None)                                  # 臉太小、沒有特徵
            conn.execute("UPDATE faces SET is_ignored = 1 WHERE id = 101")

        self.seed(fill)
        table = self.engine.load_faces()
        self.assertEqual(table.matrix.shape, (16, DIM))                      # 15 + 1 個路人；忽略與無特徵的不算
        self.assertEqual(table.matrix.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(table.matrix, axis=1), 1.0, atol=1e-5)

        result = self.engine.run()
        self.assertEqual((result.total_faces, result.new_persons, result.clustered_faces, result.noise_faces),
                         (16, 3, 15, 1))
        persons = self.db.query("SELECT * FROM persons ORDER BY id")
        self.assertEqual(len(persons), 3)
        for p in persons:
            self.assertIsNone(p["name"])                                     # 自動群組不命名
            self.assertEqual(person_display_name(p["id"], p["name"]), f"Person_{p['id']}")
            self.assertEqual(p["face_count"], 5)
            self.assertIn(p["cover_face_id"], self.faces_of(p["id"]))
            members = [blob_to_vec(r["embedding"]) for r in
                       self.db.query("SELECT embedding FROM faces WHERE person_id = ?", (p["id"],))]
            np.testing.assert_allclose(blob_to_vec(p["centroid"], DIM), unit(np.mean(members, axis=0)), atol=1e-5)
        # 同一個人的 5 張臉在同一群
        self.assertEqual(len({self.person_of(f) for f in range(1, 6)}), 1)
        self.assertEqual(len({self.person_of(f) for f in (1, 6, 11)}), 3)
        self.assertIsNone(self.person_of(100))
        self.assertEqual(self.one("SELECT cluster_id FROM faces WHERE id = 100"), -1)
        self.assertEqual(self.one("SELECT assign_src FROM faces WHERE id = 1"), "auto")

    def test_locked_and_named_groups_are_never_overwritten_or_split(self) -> None:
        def fill(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO persons(id, name, is_locked) VALUES (1, NULL, 1)")       # 鎖定、未命名
            conn.execute("INSERT INTO persons(id, name) VALUES (2, 'Alice')")                  # 已命名
            conn.execute("INSERT INTO persons(id) VALUES (3)")                                 # 自動群組（未保護）
            conn.execute("INSERT INTO persons(id) VALUES (4)")                                 # 自動群組（未保護）
            # 人物 1：使用者鎖定的群組，刻意混了 A 與 B 兩種臉 —— 演算法「會想拆開」的情況
            for fid, key in zip(range(1, 6), "AAABB"):
                insert_face(conn, fid, 1, self.near(key), person_id=1, assign_src="auto")
            for fid in range(6, 10):
                insert_face(conn, fid, 1, self.near("C"), person_id=2, assign_src="auto")
            # 人物 3：4 張自動指派 + 1 張使用者手動指派
            for fid in range(10, 14):
                insert_face(conn, fid, 1, self.near("E"), person_id=3, assign_src="auto")
            insert_face(conn, 14, 1, self.near("E"), person_id=3, assign_src="manual")
            # 人物 4：把 G（3 張）與 H（2 張）錯誤地併在一起的自動群組
            for fid, key in zip(range(60, 65), "GGGHH"):
                insert_face(conn, fid, 1, self.near(key), person_id=4, assign_src="auto")
            # 新掃描進來、尚未分類的臉
            for fid in range(20, 24):
                insert_face(conn, fid, 1, self.near("A"))
            for fid in range(30, 33):
                insert_face(conn, fid, 1, self.near("C"))
            for fid in range(40, 45):
                insert_face(conn, fid, 1, self.near("D"))
            insert_face(conn, 50, 1, unit(self.rng.normal(size=DIM)))
            conn.execute("UPDATE persons SET cover_face_id = 5 WHERE id = 1")                  # 使用者選的頭像

        self.seed(fill)
        locked_before, alice_before = self.faces_of(1), self.faces_of(2)

        result = self.engine.run(full_recluster=True)

        # ---- 受保護的群組：原本的成員一個都沒少、沒被拆開 ----
        self.assertTrue(locked_before <= self.faces_of(1))
        self.assertEqual({self.person_of(f) for f in range(1, 6)}, {1}, "鎖定的群組被拆散了")
        self.assertTrue(alice_before <= self.faces_of(2))
        locked = self.db.query_one("SELECT * FROM persons WHERE id = 1")
        self.assertEqual((locked["is_locked"], locked["name"], locked["cover_face_id"]), (1, None, 5))
        self.assertEqual(self.one("SELECT name FROM persons WHERE id = 2"), "Alice")
        self.assertIn(self.one("SELECT cover_face_id FROM persons WHERE id = 2"), self.faces_of(2))   # 原本沒有頭像才自動填
        # ---- 手動指派的臉不動 ----
        manual = self.db.query_one("SELECT person_id, assign_src FROM faces WHERE id = 14")
        self.assertEqual((manual["person_id"], manual["assign_src"]), (3, "manual"))

        # ---- 新的臉先比對既有人物的中心向量（>= 0.65 直接歸入）----
        self.assertEqual({self.person_of(f) for f in range(20, 24)}, {1})
        self.assertEqual({self.person_of(f) for f in range(30, 33)}, {2})
        self.assertEqual(self.one("SELECT face_count FROM persons WHERE id = 1"), 9)
        self.assertEqual(self.one("SELECT face_count FROM persons WHERE id = 2"), 7)

        # ---- 未保護的自動群組被解散重分：G 自成一群，H 只有 2 張不成群，人物 4 消失 ----
        self.assertIsNone(self.one("SELECT id FROM persons WHERE id = 4"))
        g_person = self.person_of(60)
        self.assertIsNotNone(g_person)
        self.assertEqual({self.person_of(f) for f in (60, 61, 62)}, {g_person})
        self.assertEqual({self.person_of(f) for f in (63, 64)}, {None})
        # 沒命中任何既有人物的臉才進 DBSCAN，產生新的 Person_# 群組
        d_person = self.person_of(40)
        self.assertEqual({self.person_of(f) for f in range(40, 45)}, {d_person})
        self.assertNotIn(d_person, (1, 2, 3, g_person, None))
        self.assertIsNone(self.person_of(50))
        self.assertEqual(result.new_persons, 2)
        self.assertEqual(sorted(result.new_person_ids), sorted([g_person, d_person]))
        self.assertEqual(result.protected_faces, 5 + 4 + 1)

        # ---- 重跑（增量）是冪等的 ----
        snapshot = {r["id"]: r["person_id"] for r in self.db.query("SELECT id, person_id FROM faces")}
        again = self.engine.run()
        self.assertEqual((again.new_persons, again.assigned_to_existing), (0, 0))
        self.assertEqual({r["id"]: r["person_id"] for r in self.db.query("SELECT id, person_id FROM faces")}, snapshot)

    def test_incremental_mode_leaves_existing_auto_groups_alone(self) -> None:
        def fill(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO persons(id) VALUES (4)")
            for fid, key in zip(range(60, 65), "GGGHH"):
                insert_face(conn, fid, 1, self.near(key), person_id=4, assign_src="auto")

        self.seed(fill)
        result = self.engine.run()
        self.assertEqual({self.person_of(f) for f in range(60, 65)}, {4})
        self.assertEqual((result.dissolved_persons, result.new_persons), (0, 0))

    def test_assign_threshold_is_respected(self) -> None:
        p = self.base["A"]
        o1, o2 = (unit(v - (v @ p) * p) for v in (self.base["B"], self.base["C"]))     # 與 p 正交的方向

        def fill(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO persons(id, name) VALUES (1, 'P')")
            insert_face(conn, 1, 1, p, person_id=1, assign_src="auto")
            insert_face(conn, 2, 1, unit(0.70 * p + np.sqrt(1 - 0.49) * o1))           # 相似度 0.70
            insert_face(conn, 3, 1, unit(0.60 * p + np.sqrt(1 - 0.36) * o2))           # 相似度 0.60

        self.seed(fill)
        result = self.engine.run()
        self.assertEqual(self.person_of(2), 1)
        self.assertIsNone(self.person_of(3))
        self.assertEqual((result.assigned_to_existing, result.noise_faces), (1, 1))

    def test_representative_face_prefers_confident_large_central_face(self) -> None:
        def fill(conn: sqlite3.Connection) -> None:
            insert_face(conn, 1, 1, self.near("A"), det=0.60, size=0.05)
            insert_face(conn, 2, 1, self.near("A"), det=0.99, size=0.40)               # 分數最高、面積最大
            insert_face(conn, 3, 1, self.near("A"), det=0.70, size=0.10)

        self.seed(fill)
        self.engine.run()
        self.assertEqual(self.one("SELECT cover_face_id FROM persons"), 2)
        # 使用者把臉 2 標為忽略後重新整理：頭像換成剩下的裡面最好的
        self.db.execute("UPDATE faces SET is_ignored = 1 WHERE id = 2").result()
        self.assertEqual(self.engine.refresh_representatives(), 1)
        self.assertEqual(self.one("SELECT cover_face_id FROM persons"), 3)


# =========================================================================== #
# EventClusterEngine
# =========================================================================== #
class EventClusterEngineTest(DbCase):
    def events(self) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM events ORDER BY start_ts")

    def event_of(self, pid: int) -> int | None:
        return self.one("SELECT event_id FROM photos WHERE id = ?", (pid,))

    def test_time_gap_groups_and_splits(self) -> None:
        def fill(conn: sqlite3.Connection) -> None:
            times = ["2024-05-01 08:00:00", "2024-05-01 09:00:00", "2024-05-01 14:59:00",     # 相鄰差都 < 6 小時
                     "2024-05-01 21:00:00", "2024-05-01 22:00:00", "2024-05-01 23:00:00",     # 與前一張差 6 小時 1 分
                     "2024-05-09 12:00:00"]                                                   # 落單
            for i, t in enumerate(times, 1):
                insert_photo(conn, i, f"{i}.jpg", taken_ts=ts(t))
            insert_photo(conn, 90, "mtime.jpg", taken_ts=ts("2024-05-01 08:30:00"), taken_src="mtime")
            insert_photo(conn, 91, "shot.jpg", taken_ts=ts("2024-05-01 08:31:00"), is_screenshot=1)

        self.db.write(fill)
        result = EventClusterEngine(self.db).cluster_events()
        self.assertEqual((result.photos_considered, result.events_created, result.photos_in_events), (7, 2, 6))
        self.assertEqual([e["photo_count"] for e in self.events()], [3, 3])
        self.assertEqual(len({self.event_of(i) for i in (1, 2, 3)}), 1)
        self.assertNotEqual(self.event_of(3), self.event_of(4))
        self.assertIsNone(self.event_of(7))                       # 少於 min_photos 不成事件
        self.assertIsNone(self.event_of(90))                      # 時間來自 mtime 不可信
        self.assertIsNone(self.event_of(91))                      # 螢幕截圖不參與

    def test_continuous_trip_spans_nights_but_home_does_not_chain(self) -> None:
        kyoto, taipei = dict(gps_lat=35.01, gps_lon=135.77, city="Kyoto", country_code="JP"), \
            dict(gps_lat=25.03, gps_lon=121.56, city="Taipei", country_code="TW")

        def fill(conn: sqlite3.Connection) -> None:
            # 京都兩天一夜：過夜的空檔 14 小時，但人還在同一個城市
            for i, t in enumerate(["2024-05-10 18:00:00", "2024-05-10 19:00:00",
                                   "2024-05-11 09:00:00", "2024-05-11 10:00:00"], 1):
                insert_photo(conn, i, f"k{i}.jpg", taken_ts=ts(t), **kyoto)
            # 在家：連續 5 天每天中午各一張
            for d in range(5):
                insert_photo(conn, 10 + d, f"h{d}.jpg", taken_ts=ts(f"2024-06-0{d + 1} 12:00:00"), **taipei)
            # 只有 GPS、沒有城市名稱：距離 < 30 公里視為同一趟
            for i, (t, lat) in enumerate([("2024-07-01 08:00:00", 48.85), ("2024-07-01 20:00:00", 48.90),
                                          ("2024-07-02 09:00:00", 48.95)]):
                insert_photo(conn, 20 + i, f"g{i}.jpg", taken_ts=ts(t), gps_lat=lat, gps_lon=2.35)
            # 同樣的時間間隔，但位置跳了幾百公里 → 不是同一趟
            for i, (t, lat) in enumerate([("2024-08-01 08:00:00", 40.0), ("2024-08-01 20:00:00", 45.0),
                                          ("2024-08-02 09:00:00", 50.0)]):
                insert_photo(conn, 30 + i, f"f{i}.jpg", taken_ts=ts(t), gps_lat=lat, gps_lon=2.35)
            tag = conn.execute("INSERT INTO tags(name) VALUES ('temple')").lastrowid
            food = conn.execute("INSERT INTO tags(name) VALUES ('food')").lastrowid
            conn.executemany("INSERT INTO photo_tags(photo_id, tag_id, score) VALUES (?, ?, 0.9)",
                             [(1, tag), (2, tag), (3, tag), (4, food)])

        self.db.write(fill)
        result = EventClusterEngine(self.db).cluster_events()
        self.assertEqual(result.home_city, "Taipei")              # 有照片的天數最多
        events = self.events()
        self.assertEqual([e["photo_count"] for e in events], [4, 3])
        self.assertEqual(len({self.event_of(i) for i in (1, 2, 3, 4)}), 1)
        self.assertEqual({self.event_of(10 + d) for d in range(5)}, {None})
        self.assertEqual(len({self.event_of(20 + i) for i in range(3)}), 1)
        self.assertEqual({self.event_of(30 + i) for i in range(3)}, {None})
        # 預設名稱：YYYY-MM-DD_{City}_{TopTag}；沒有城市或標籤就省略
        self.assertEqual([e["name"] for e in events], ["2024-05-10_Kyoto_temple", "2024-07-01"])
        self.assertEqual(events[0]["location_label"], "Kyoto, JP")

        # 停用常駐城市防護：在家的 5 天會被串成一個事件
        EventClusterEngine(self.db, home_city=None).cluster_events()
        self.assertEqual([e["photo_count"] for e in self.events()], [4, 5, 3])

    def test_user_edited_events_survive_reclustering(self) -> None:
        def fill(conn: sqlite3.Connection) -> None:
            for i in range(1, 5):
                insert_photo(conn, i, f"{i}.jpg", taken_ts=ts(f"2024-05-01 1{i}:00:00"))

        self.db.write(fill)
        engine = EventClusterEngine(self.db)
        engine.cluster_events()
        event_id = self.event_of(1)
        self.db.execute("UPDATE events SET name = '畢業典禮', is_user_edited = 1 WHERE id = ?", (event_id,)).result()
        self.db.write(lambda c: insert_photo(c, 5, "5.jpg", taken_ts=ts("2024-05-01 15:00:00")))
        engine.cluster_events()
        events = self.events()
        self.assertEqual([(e["id"], e["name"], e["photo_count"]) for e in events], [(event_id, "畢業典禮", 4)])
        self.assertEqual({self.event_of(i) for i in range(1, 5)}, {event_id})

    def test_burst_groups_and_best_shot(self) -> None:
        t0 = ts("2024-05-01 10:00:00")

        def fill(conn: sqlite3.Connection) -> None:
            # 第 1 組：三連拍，中間那張最清楚
            insert_photo(conn, 1, "b1.jpg", taken_ts=t0, dhash=0b0000, blur_score=50.0)
            insert_photo(conn, 2, "b2.jpg", taken_ts=t0 + 1, dhash=0b0001, blur_score=120.0)
            insert_photo(conn, 3, "b3.jpg", taken_ts=t0 + 2, dhash=0b0011, blur_score=80.0)
            # 時間夠近但畫面不同（漢明距離 > 4）→ 不是連拍
            insert_photo(conn, 4, "b4.jpg", taken_ts=t0 + 3, dhash=0b1111_1111_0011, blur_score=90.0)
            # 第 2 組：清晰度相近，但第二張的人臉偵測信心度高
            insert_photo(conn, 5, "c1.jpg", taken_ts=t0 + 100, dhash=-1, blur_score=100.0)
            insert_photo(conn, 6, "c2.jpg", taken_ts=t0 + 101, dhash=-1, blur_score=90.0)
            insert_face(conn, 1, 6, None, det=0.99)
            # 第 3 組：使用者標為最愛的優先，標為廢片的不會當選
            insert_photo(conn, 7, "d1.jpg", taken_ts=t0 + 200, dhash=7, blur_score=200.0, is_rejected=1)
            insert_photo(conn, 8, "d2.jpg", taken_ts=t0 + 201, dhash=7, blur_score=60.0, is_favorite=1)
            insert_photo(conn, 9, "d3.jpg", taken_ts=t0 + 202, dhash=7, blur_score=150.0)
            # 畫面一樣但相隔 3 秒以上 → 不是連拍
            insert_photo(conn, 10, "e1.jpg", taken_ts=t0 + 300, dhash=9)
            insert_photo(conn, 11, "e2.jpg", taken_ts=t0 + 303, dhash=9)

        self.db.write(fill)
        engine = EventClusterEngine(self.db)
        result = engine.detect_bursts()
        self.assertEqual((result.burst_groups, result.burst_photos, result.burst_duplicates), (3, 8, 5))
        rows = {r["id"]: (r["burst_group_id"], r["is_best_shot"], r["is_burst_duplicate"])
                for r in self.db.query("SELECT * FROM photos")}
        self.assertEqual([rows[i] for i in (1, 2, 3)], [(1, 0, 1), (1, 1, 0), (1, 0, 1)])
        self.assertEqual(rows[4], (None, 0, 0))
        self.assertEqual([rows[i] for i in (5, 6)], [(5, 0, 1), (5, 1, 0)])
        self.assertEqual([rows[i] for i in (7, 8, 9)], [(7, 0, 1), (7, 1, 0), (7, 0, 1)])
        self.assertEqual([rows[i] for i in (10, 11)], [(None, 0, 0), (None, 0, 0)])

        # 重跑會先清掉舊的標記：把 b2 的畫面改掉之後，第 1 組只剩不成組的照片
        self.db.execute("UPDATE photos SET dhash = ? WHERE id = 2", (0x7FFF_FFFF_FFFF,)).result()
        engine.detect_bursts()
        self.assertEqual(self.one("SELECT COUNT(*) FROM photos WHERE burst_group_id = 1"), 0)
        self.assertEqual(self.one("SELECT is_burst_duplicate FROM photos WHERE id = 1"), 0)


# =========================================================================== #
# ExportManager（四層多視圖）
# =========================================================================== #
def snapshot(root: Path) -> dict[str, str]:
    """資料夾底下所有檔案：相對路徑 -> 內容的 SHA-256。用來證明「100% 還原」。"""
    if not root.exists():
        return {}
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def tree(root: Path) -> set[str]:
    return set(snapshot(root))


class ExportCase(DbCase):
    def setUp(self) -> None:
        super().setUp()
        self.lib = self.tmp / "library"
        self.out = self.tmp / "export"
        self.bus = EventBus()
        self.exporter = ExportManager(self.cfg, self.db, self.repo, self.bus)

    def ingest(self) -> None:
        """用真的管線（Stage 1）把照片庫掃進資料庫。"""
        engine = PipelineEngine(self.cfg, self.db, self.repo, self.bus, decoder=DefaultImageDecoder(self.cfg))
        engine.start([self.lib])
        assert engine.join(60) and engine.state.value == "finished"
        # 原始檔名 -> ShortHash。MOVE 之後資料庫裡的檔名會變成新檔名，所以先記下來
        self.short = {r["file_name"]: bytes(r["sha256"]).hex()[:6]
                      for r in self.db.query("SELECT file_name, sha256 FROM photos")}

    def pid(self, name: str) -> int:
        return self.one("SELECT id FROM photos WHERE file_name = ?", (name,))

    def exported_name(self, name: str, layer: str = MASTER_DIR) -> Path:
        """原始檔名為 `name` 的照片在某一層裡的新路徑（以 ShortHash 找）。"""
        short = self.short[name]
        ext = os.path.splitext(name)[1].lower()
        hits = [p for p in (self.out / layer).rglob(f"*_{short}*{ext}")]
        self.assertEqual(len(hits), 1, f"{name} 在 {layer} 裡應該正好一份，實際：{hits}")
        return hits[0]


class ExportManagerTest(ExportCase):
    def setUp(self) -> None:
        super().setUp()
        trip, misc = self.lib / "trip", self.lib / "misc"
        make_image(trip / "a1.jpg", color=(200, 30, 30), seed=1, taken="2024:05:01 10:00:00", camera="Zenfone 9")
        make_image(trip / "a2.jpg", color=(30, 200, 30), seed=2, taken="2024:05:01 10:05:00", camera="GO Ultra")
        # Live Photo：主照片 + 同主檔名的影片與 XMP
        make_image(trip / "IMG_1234.jpg", seed=3, taken="2024:05:01 10:10:00", camera="iPhone 15 Pro")
        (trip / "IMG_1234.MOV").write_bytes(b"live photo video " * 500)
        (trip / "IMG_1234.xmp").write_bytes(b"<x:xmpmeta>edits</x:xmpmeta>")
        # RAW + JPG
        make_image(trip / "DSC0001.JPG", seed=4, taken="2024:05:01 10:15:00", camera="ILCE-7M4")
        (trip / "DSC0001.ARW").write_bytes(b"raw sensor data " * 2000)
        (trip / "notes.txt").write_bytes(b"not a sidecar")
        # 會進隔離區的三種
        make_image(misc / "Screenshot_20240601-120000.jpg", seed=5, taken=None, camera=None)
        make_image(misc / "blurry.jpg", seed=6, taken="2024:05:02 08:00:00")
        make_image(misc / "burst2.jpg", seed=7, taken="2024:05:02 08:00:01")
        # 沒有 EXIF 日期
        make_image(misc / "nodate.jpg", seed=8, taken=None, camera=None)
        self.ingest()

        start = ts("2024-05-01 10:00:00")

        def annotate(conn: sqlite3.Connection) -> None:
            ids = {r["file_name"]: r["id"] for r in conn.execute("SELECT id, file_name FROM photos")}
            conn.execute("INSERT INTO events(id, name, start_ts, end_ts, photo_count, location_label) "
                         "VALUES (1, '2024-05-01_Taipei_food', ?, ?, 4, 'Taipei, TW')", (start, start + 900))
            for name in ("a1.jpg", "a2.jpg", "IMG_1234.jpg", "DSC0001.JPG"):
                conn.execute("UPDATE photos SET event_id = 1, city = 'Taipei', country_code = 'TW' WHERE id = ?",
                             (ids[name],))
            conn.execute("UPDATE photos SET is_rejected = 1, reject_reason = 'blur' WHERE id = ?", (ids["blurry.jpg"],))
            conn.execute("UPDATE photos SET is_burst_duplicate = 1, burst_group_id = 1 WHERE id = ?",
                         (ids["burst2.jpg"],))
            food = conn.execute("INSERT INTO tags(name) VALUES ('food')").lastrowid
            beach = conn.execute("INSERT INTO tags(name) VALUES ('beach')").lastrowid
            conn.executemany("INSERT INTO photo_tags(photo_id, tag_id, score) VALUES (?, ?, ?)", [
                (ids["a1.jpg"], food, 0.9), (ids["a2.jpg"], food, 0.8), (ids["IMG_1234.jpg"], food, 0.7),
                (ids["a2.jpg"], beach, 0.6), (ids["nodate.jpg"], beach, 0.9),
                (ids["blurry.jpg"], food, 0.9),                       # 隔離區的照片不會有視圖
            ])
            conn.execute("INSERT INTO persons(id, name) VALUES (1, 'Alice')")
            conn.execute("INSERT INTO persons(id) VALUES (2)")                 # 未命名 -> Person_2
            insert_face(conn, 1, ids["a1.jpg"], None, person_id=1)             # 獨照
            insert_face(conn, 2, ids["a2.jpg"], None, person_id=1)             # 合照：Alice + Person_2
            insert_face(conn, 3, ids["a2.jpg"], None, person_id=2)
            insert_face(conn, 4, ids["IMG_1234.jpg"], None, person_id=1)       # Alice + 一張不認識的臉
            insert_face(conn, 5, ids["IMG_1234.jpg"], None)

        self.db.write(annotate)
        self.library_before = snapshot(self.lib)
        self.paths_before = {r["id"]: r["path"] for r in self.db.query("SELECT id, path FROM photos")}
        self.nodate_year = str(datetime.fromtimestamp((misc / "nodate.jpg").stat().st_mtime).year)
        self.event_dir = self.out / MASTER_DIR / "2024" / "2024-05-01_Taipei_food"

    def assert_fully_restored(self, batch_id: int) -> None:
        self.assertEqual(snapshot(self.lib), self.library_before, "照片庫沒有 100% 還原")
        self.assertEqual({r["id"]: r["path"] for r in self.db.query("SELECT id, path FROM photos")}, self.paths_before)
        self.assertEqual(tree(self.out), set(), "匯出根目錄還有殘留檔案")
        for layer in (MASTER_DIR, PERSON_DIR, TAG_DIR, STAGING_DIR):
            self.assertFalse((self.out / layer).exists(), f"{layer} 沒有清掉")
        self.assertEqual(self.one("SELECT state FROM export_batches WHERE id = ?", (batch_id,)), "rolled_back")
        self.assertEqual(self.one("SELECT COUNT(*) FROM export_logs WHERE batch_id = ? AND state = 'done'",
                                  (batch_id,)), 0)

    # ------------------------------------------------------------------ #
    def test_preview_is_a_pure_dry_run_with_space_estimate(self) -> None:
        preview = self.exporter.preview_export(self.out, mode="copy")
        self.assertFalse(self.out.exists(), "預覽不可以建立任何資料夾")
        self.assertEqual(self.one("SELECT COUNT(*) FROM export_batches"), 0)
        self.assertEqual(snapshot(self.lib), self.library_before)

        summary = preview.summary()
        self.assertEqual(summary["master:copy"], 5)                # 事件裡 4 張 + 沒有 EXIF 日期 1 張
        self.assertEqual(summary["staging:copy"], 3)
        self.assertEqual(summary["sidecar:copy"], 3)               # .mov .xmp .arw
        self.assertEqual(summary["person_view:hardlink"], 5)
        self.assertEqual(summary["tag_view:hardlink"], 5)
        # COPY：主檔與 Sidecar 會佔空間，硬連結視圖不佔
        copied = [p for p in self.lib.rglob("*") if p.is_file() and p.name != "notes.txt"]
        self.assertEqual(preview.estimated_bytes, sum(p.stat().st_size for p in copied))
        self.assertIsNotNone(preview.free_bytes)
        self.assertEqual(preview.warnings, [])
        # HARDLINK（同磁碟機）：完全不佔額外空間
        self.assertEqual(self.exporter.preview_export(self.out, mode="hardlink").estimated_bytes, 0)
        with self.assertRaises(ExportError):
            self.exporter.preview_export(self.out, mode="symlink")

    def test_copy_export_builds_four_layers_then_rolls_back_completely(self) -> None:
        events = []
        self.bus.subscribe(events.append)
        result = self.exporter.execute_export(self.exporter.preview_export(self.out, mode="copy"))
        self.assertEqual((result.done, result.failed), (21, 0), result.errors)
        self.assertEqual(snapshot(self.lib), self.library_before, "COPY 不可以動到原始檔")

        # ---- 01 主庫 ----
        master = sorted(p.name for p in self.event_dir.iterdir())
        self.assertEqual(len(master), 7, master)                   # 4 張照片 + .mov + .xmp + .arw
        a1 = self.exported_name("a1.jpg")
        self.assertEqual(a1.parent, self.event_dir)
        m = NAME_RE.match(a1.name)
        self.assertIsNotNone(m, a1.name)
        sha = hashlib.sha256((self.lib / "trip" / "a1.jpg").read_bytes()).hexdigest()
        self.assertEqual(m.groups(), ("20240501", "100000", "000", "Zenfone9", sha[:6], None, "jpg"))
        self.assertIn("_GOUltra_", self.exported_name("a2.jpg").name)
        self.assertEqual(a1.read_bytes(), (self.lib / "trip" / "a1.jpg").read_bytes())
        nodate = self.exported_name("nodate.jpg")
        self.assertEqual(nodate.parent, self.out / MASTER_DIR / NO_DATE_DIR / self.nodate_year)
        self.assertIn("_beach_", nodate.name)                      # 沒有相機型號 -> 用分數最高的標籤

        # ---- Sidecar：與主照片同一個資料夾、共用新檔名主幹、副檔名轉小寫 ----
        live = self.exported_name("IMG_1234.jpg")
        self.assertEqual((live.with_suffix(".mov")).read_bytes(), (self.lib / "trip" / "IMG_1234.MOV").read_bytes())
        self.assertEqual((live.with_suffix(".xmp")).read_bytes(), (self.lib / "trip" / "IMG_1234.xmp").read_bytes())
        raw_main = self.exported_name("DSC0001.JPG")
        self.assertTrue(raw_main.name.endswith(".jpg"))
        self.assertEqual(raw_main.with_suffix(".arw").read_bytes(), (self.lib / "trip" / "DSC0001.ARW").read_bytes())
        self.assertFalse(any("notes" in n for n in tree(self.out)))

        # ---- 02 人物視圖：硬連結、獨照 / 年份 / 合照 ----
        a2 = self.exported_name("a2.jpg")
        person = self.out / PERSON_DIR
        self.assertEqual(tree(person), {
            f"Alice/{SOLO_DIR}/{a1.name}", f"Alice/2024/{a2.name}", f"Alice/2024/{live.name}",
            f"Person_2/2024/{a2.name}", f"{GROUP_DIR}/{a2.name}",
        })
        for rel in tree(person):
            source = a1 if a1.name in rel else a2 if a2.name in rel else live
            self.assertTrue(os.path.samefile(person / rel, source), f"{rel} 不是主庫檔案的硬連結")
        # ---- 03 標籤視圖：一張照片可出現在多個標籤下 ----
        tags = self.out / TAG_DIR
        self.assertEqual(tree(tags), {f"food/{a1.name}", f"food/{a2.name}", f"food/{live.name}",
                                      f"beach/{a2.name}", f"beach/{nodate.name}"})
        self.assertTrue(os.path.samefile(tags / "beach" / a2.name, a2))
        self.assertGreaterEqual(os.stat(a2).st_nlink, 5)           # 主庫 + 3 個人物視圖 + 2 個標籤視圖 - 自己

        # ---- 99 隔離區：不進主庫、沒有視圖 ----
        staging = self.out / STAGING_DIR
        self.assertEqual({p.parent.name for p in staging.rglob("*.jpg")},
                         {STAGING_SCREENSHOT, STAGING_BLURRY, STAGING_BURST})
        self.assertEqual(self.exported_name("blurry.jpg", STAGING_DIR).parent.name, STAGING_BLURRY)
        self.assertEqual(self.exported_name("burst2.jpg", STAGING_DIR).parent.name, STAGING_BURST)
        self.assertEqual(self.exported_name("Screenshot_20240601-120000.jpg", STAGING_DIR).parent.name,
                         STAGING_SCREENSHOT)
        self.assertEqual(len(tree(self.out)), 21)
        for rel in tree(self.out):
            self.assertLess(len(str(self.out / rel)), MAX_PATH_LEN)

        # ---- verify_archive：視圖全是主庫檔案的硬連結，實際佔用只有主庫 + 隔離區那一份 ----
        report = verify_archive(self.out)
        self.assertTrue(report.ok, report.broken_views)
        self.assertEqual((report.master_files, report.staging_files, report.view_links), (8, 3, 10))
        copied = sum(p.stat().st_size for p in self.lib.rglob("*") if p.is_file() and p.name != "notes.txt")
        self.assertEqual(report.physical_bytes, copied)
        self.assertGreater(report.apparent_bytes, report.physical_bytes)   # 檔案總管看到的數字會重複計算硬連結
        self.assertEqual(report.saved_bytes, report.apparent_bytes - copied)
        impostor = self.out / TAG_DIR / "food" / "not_a_link.jpg"          # 視圖裡混進一個實體檔案
        impostor.write_bytes(b"real bytes, not a hardlink")
        self.assertEqual(verify_archive(self.out).broken_views, [str(impostor)])
        impostor.unlink()

        # ---- 交易日誌 ----
        logs = self.db.query("SELECT * FROM export_logs WHERE batch_id = ? ORDER BY seq", (result.batch_id,))
        self.assertEqual(len(logs), 21)
        self.assertEqual({r["state"] for r in logs}, {"done"})
        self.assertEqual({r["action"] for r in logs}, {"copy", "hardlink"})
        self.assertEqual(sum(1 for r in logs if r["photo_id"] is not None), 8)       # 只有主檔記 photo_id
        self.assertEqual([e.kind for e in events][-1], EventKind.EXPORT_DONE)

        # ---- 冪等：再匯出一次什麼都不做，即使事件被改名（照片留在第一次放的位置）----
        self.db.execute("UPDATE events SET name = '2024-05-01_Taipei_renamed' WHERE id = 1").result()
        again = self.exporter.preview_export(self.out, mode="copy")
        self.assertEqual(again.pending(), [])
        self.assertEqual(again.estimated_bytes, 0)
        self.assertEqual(self.exporter.execute_export(again).done, 0)
        self.assertEqual(len(tree(self.out)), 21)

        # ---- 回復 ----
        back = self.exporter.rollback_batch(result.batch_id)
        self.assertEqual((back.done, back.failed), (21, 0), back.errors)
        self.assert_fully_restored(result.batch_id)

    def test_move_export_and_rollback_restores_library_100_percent(self) -> None:
        result = self.exporter.execute_export(self.exporter.preview_export(self.out, mode="move"))
        self.assertEqual(result.failed, 0, result.errors)
        self.assertEqual(tree(self.lib), {"trip/notes.txt"}, "MOVE 之後原始位置只該剩下不相干的檔案")
        self.assertEqual(len(tree(self.out)), 21)
        # 資料庫跟著主檔走
        a1 = self.exported_name("a1.jpg")
        self.assertEqual(self.one("SELECT path FROM photos WHERE file_name = ?", (a1.name,)), os.path.realpath(a1))
        for r in self.db.query("SELECT path FROM photos"):
            self.assertTrue(os.path.isfile(r["path"]), r["path"])
        # MOVE 之後再預覽：都已在定位
        self.assertEqual(self.exporter.preview_export(self.out, mode="move").pending(), [])

        back = self.exporter.rollback_batch(result.batch_id)
        self.assertEqual(back.failed, 0, back.errors)
        self.assert_fully_restored(result.batch_id)

    def test_hardlink_export_shares_data_with_originals(self) -> None:
        result = self.exporter.execute_export(self.exporter.preview_export(self.out, mode="hardlink"))
        self.assertEqual(result.failed, 0, result.errors)
        self.assertEqual(snapshot(self.lib), self.library_before)
        a1 = self.exported_name("a1.jpg")
        self.assertTrue(os.path.samefile(a1, self.lib / "trip" / "a1.jpg"))
        live = self.exported_name("IMG_1234.jpg")
        self.assertTrue(os.path.samefile(live.with_suffix(".mov"), self.lib / "trip" / "IMG_1234.MOV"))
        self.exporter.rollback_batch(result.batch_id)
        self.assert_fully_restored(result.batch_id)

    def test_cross_drive_hardlink_falls_back_to_copy_but_views_stay_hardlinks(self) -> None:
        out_key = os.path.normcase(os.path.realpath(self.out))

        def fake_drive(path: str) -> str:
            return "X:" if os.path.normcase(os.path.realpath(path)).startswith(out_key) else "C:"

        with mock.patch.object(em, "_drive", fake_drive):
            preview = self.exporter.preview_export(self.out, mode="hardlink")
            primary = [a for a in preview.pending() if a.role in ("master", "staging", "sidecar")]
            self.assertEqual({a.op_type for a in primary}, {"copy"})
            self.assertTrue(all(a.note == "hardlink_fallback_copy" for a in primary if a.role != "sidecar"))
            self.assertEqual({a.op_type for a in preview.pending() if a.role.endswith("_view")}, {"hardlink"})
            self.assertGreater(preview.estimated_bytes, 0)
            self.assertTrue(any("不在同一個磁碟機" in w for w in preview.warnings))

            strict = self.exporter.preview_export(self.out, mode="hardlink", cross_volume_fallback=False)
            self.assertEqual(strict.pending(), [])
            self.assertEqual(strict.summary()["master:skip:cross_volume"], 5)

            result = self.exporter.execute_export(preview)
        self.assertEqual(result.failed, 0, result.errors)
        a2 = self.exported_name("a2.jpg")
        self.assertFalse(os.path.samefile(a2, self.lib / "trip" / "a2.jpg"))          # 主庫是複本
        self.assertTrue(os.path.samefile(a2, self.out / TAG_DIR / "beach" / a2.name))  # 視圖仍是零空間的硬連結

    def test_sidecar_failure_reverts_the_whole_atomic_group(self) -> None:
        preview = self.exporter.preview_export(self.out, mode="move", include_views=False)
        mov = next(a for a in preview.pending() if a.role == "sidecar" and a.dst.endswith(".mov"))
        os.makedirs(os.path.dirname(mov.dst))
        Path(mov.dst).write_bytes(b"appeared after the preview")               # 預覽之後才冒出來的同名檔案

        result = self.exporter.execute_export(preview)
        self.assertGreater(result.failed, 0)
        # 主照片已經搬過去了，但因為 Sidecar 失敗，整組被還原
        self.assertTrue((self.lib / "trip" / "IMG_1234.jpg").is_file())
        self.assertTrue((self.lib / "trip" / "IMG_1234.MOV").is_file())
        self.assertTrue((self.lib / "trip" / "IMG_1234.xmp").is_file())
        short = self.short["IMG_1234.jpg"]
        self.assertEqual([p.name for p in self.out.rglob(f"*_{short}*") if p.suffix != ".mov"], [])
        self.assertEqual(Path(mov.dst).read_bytes(), b"appeared after the preview")   # 絕不覆寫
        self.assertEqual(self.one("SELECT path FROM photos WHERE file_name = 'IMG_1234.jpg'"),
                         self.paths_before[self.pid("IMG_1234.jpg")])
        group_states = {r["state"] for r in self.db.query(
            "SELECT state FROM export_logs WHERE src_path LIKE '%IMG_1234%'")}
        self.assertEqual(group_states, {"failed"})
        # 其他群組不受影響
        self.assertFalse((self.lib / "trip" / "a1.jpg").exists())
        self.exporter.rollback_batch(result.batch_id)
        self.assertEqual({k: v for k, v in snapshot(self.lib).items()}, self.library_before)

    def test_recategorized_photo_moves_between_layers_on_next_export(self) -> None:
        first = self.exporter.execute_export(self.exporter.preview_export(self.out, mode="copy"))
        self.assertEqual(first.failed, 0)
        # 使用者確認 blurry.jpg 其實沒問題
        self.db.execute("UPDATE photos SET is_rejected = 0, reject_reason = NULL WHERE file_name = 'blurry.jpg'").result()
        preview = self.exporter.preview_export(self.out, mode="copy")
        moved = [a for a in preview.pending() if a.role == "master"]
        self.assertEqual(len(moved), 1)
        self.assertIn(MASTER_DIR, moved[0].dst)
        self.assertEqual(self.exporter.execute_export(preview).failed, 0)
        self.assertEqual(self.exported_name("blurry.jpg").parent.name, "2024-05-02")
        self.assertTrue((self.out / TAG_DIR / "food" / self.exported_name("blurry.jpg").name).is_file())

    def test_blur_threshold_and_views_switch(self) -> None:
        scores = {r["file_name"]: r["blur_score"] for r in self.db.query("SELECT file_name, blur_score FROM photos")}
        threshold = scores["a1.jpg"] + 0.001
        preview = self.exporter.preview_export(self.out, blur_threshold=threshold, include_views=False)
        a1 = next(a for a in preview.actions if a.photo_id == self.pid("a1.jpg") and a.role == "staging")
        self.assertIn(STAGING_BLURRY, a1.dst)
        self.assertFalse(any(a.role.endswith("_view") for a in preview.actions))


class ExportNamingTest(ExportCase):
    def test_subsec_from_exif_and_collision_gets_v2_suffix(self) -> None:
        folder = self.lib / "burst"
        make_image(folder / "x1.jpg", seed=11, taken="2024:07:04 12:00:00", camera="Cam")
        make_image(folder / "x2.jpg", seed=12, taken="2024:07:04 12:00:00", camera="Cam")
        # 帶 SubSecTimeOriginal 的照片
        img = Image.fromarray(np.full((60, 80, 3), 90, dtype=np.uint8))
        exif = Image.Exif()
        exif[0x0110] = "Cam"
        sub = exif.get_ifd(0x8769)
        sub[0x9003] = "2024:07:04 12:00:00"
        sub[0x9291] = "52"
        img.save(folder / "x3.jpg", exif=exif)
        # 內容完全相同的另一份複本
        (self.lib / "copy").mkdir()
        (self.lib / "copy" / "x1_copy.jpg").write_bytes((folder / "x1.jpg").read_bytes())
        self.ingest()

        # 人為製造「6 碼 ShortHash 相同、完整 SHA-256 不同」的極端情況
        sha1 = bytes(self.one("SELECT sha256 FROM photos WHERE file_name = 'x1.jpg'"))
        fake = sha1[:3] + bytes(29)
        self.db.execute("UPDATE photos SET sha256 = ? WHERE file_name = 'x2.jpg'", (fake,)).result()

        preview = self.exporter.preview_export(self.out, mode="copy")
        by_src = {os.path.basename(a.src): a for a in preview.actions if a.role == "master"}
        short = sha1.hex()[:6]
        self.assertEqual(os.path.basename(by_src["x1.jpg"].dst), f"20240704_120000_000_Cam_{short}.jpg")
        self.assertEqual(os.path.basename(by_src["x2.jpg"].dst), f"20240704_120000_000_Cam_{short}_v2.jpg")
        self.assertEqual(by_src["x2.jpg"].note, "hash_collision_v2")
        self.assertRegex(os.path.basename(by_src["x3.jpg"].dst), r"^20240704_120000_520_Cam_[0-9a-f]{6}\.jpg$")
        self.assertEqual((by_src["x1_copy.jpg"].op_type, by_src["x1_copy.jpg"].note), ("skip", "duplicate_content"))

        result = self.exporter.execute_export(preview)
        self.assertEqual(result.failed, 0, result.errors)
        self.assertEqual(len(tree(self.out)), 3)
        # 再跑一次：_v2 那個檔案也認得出是自己先前匯出的，不會變成 _v3
        self.assertEqual(self.exporter.preview_export(self.out, mode="copy").pending(), [])

    def test_existing_file_not_from_this_tool_is_never_overwritten(self) -> None:
        make_image(self.lib / "one.jpg", seed=21, taken="2024:07:04 12:00:00", camera="Cam")
        self.ingest()
        preview = self.exporter.preview_export(self.out, mode="copy")
        target = Path(next(a.dst for a in preview.pending() if a.role == "master"))
        target.parent.mkdir(parents=True)
        target.write_bytes(b"someone else's file with the very same name")

        again = self.exporter.preview_export(self.out, mode="copy")
        dst = next(a.dst for a in again.pending() if a.role == "master")
        self.assertTrue(dst.endswith("_v2.jpg"), dst)                       # 內容不同 -> 另取檔名
        self.exporter.execute_export(again)
        self.assertEqual(target.read_bytes(), b"someone else's file with the very same name")

        # 反過來：同名檔案內容就是同一張（例如資料庫重建過、日誌沒了）-> 視為已匯出
        target.write_bytes((self.lib / "one.jpg").read_bytes())
        os.remove(dst)
        self.db.execute("DELETE FROM export_batches").result()
        self.assertEqual(self.exporter.preview_export(self.out, mode="copy").pending(), [])

    def test_long_names_are_trimmed_to_fit_windows_path_limit(self) -> None:
        make_image(self.lib / "p.jpg", seed=31, taken="2024:07:04 12:00:00", camera="Cam")
        self.ingest()
        pid = self.pid("p.jpg")

        def annotate(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO events(id, name, start_ts, end_ts, is_user_edited) VALUES (1, ?, ?, ?, 1)",
                         ("很長的事件名稱" * 60, ts("2024-07-04 12:00:00"), ts("2024-07-04 12:00:00")))
            conn.execute("UPDATE photos SET event_id = 1 WHERE id = ?", (pid,))
            conn.execute("INSERT INTO persons(id, name) VALUES (1, ?)", ("N" * 300,))
            insert_face(conn, 1, pid, None, person_id=1)
            tag = conn.execute("INSERT INTO tags(name) VALUES (?)", ("t" * 300,)).lastrowid
            conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score) VALUES (?, ?, 0.9)", (pid, tag))

        self.db.write(annotate)
        preview = self.exporter.preview_export(self.out, mode="copy")
        self.assertEqual(len(preview.pending()), 3)
        for a in preview.pending():
            self.assertLess(len(a.dst), MAX_PATH_LEN, a.dst)
            self.assertTrue(os.path.basename(a.dst).endswith(".jpg"))      # 檔名本身不被截斷
        result = self.exporter.execute_export(preview)
        self.assertEqual((result.done, result.failed), (3, 0), result.errors)
        # 使用者自訂的事件名稱：資料夾是 日期_事件名稱
        self.assertTrue(self.exported_name("p.jpg").parent.name.startswith("2024-07-04_很長的事件名稱"))

    def test_raw_file_that_is_also_in_the_database_travels_as_sidecar_only(self) -> None:
        make_image(self.lib / "DSC0002.JPG", seed=41, taken="2024:07:04 12:00:00", camera="Cam")
        raw = self.lib / "DSC0002.DNG"
        raw.write_bytes(b"dng " * 1000)
        self.ingest()
        # 模擬「RAW 也被掃進資料庫」（scan.enable_raw）的情況
        self.db.write(lambda c: insert_photo(c, 99, "DSC0002.DNG", path=os.path.realpath(raw),
                                             dir_path=os.path.dirname(os.path.realpath(raw)), ext="dng",
                                             taken_ts=ts("2024-07-04 12:00:00")))
        preview = self.exporter.preview_export(self.out, mode="copy")
        # RAW 只以 Sidecar 的身分匯出一次；它自己那一列被略過
        raw_actions = [a for a in preview.pending() if a.src == os.path.realpath(raw)]
        self.assertEqual([(a.role, a.op_type) for a in raw_actions], [("sidecar", "copy")])
        self.assertEqual(preview.summary()["master:skip:exported_as_sidecar"], 1)
        self.exporter.execute_export(preview)
        main = self.exported_name("DSC0002.JPG")
        self.assertTrue(main.with_suffix(".dng").is_file())
        self.assertEqual(len(tree(self.out)), 2)

    def test_name_helpers(self) -> None:
        self.assertEqual(clean_name('a<b>c:d"e/f\\g|h?i*j'), "a_b_c_d_e_f_g_h_i_j")
        self.assertEqual(clean_name("  .hidden name. "), "hidden name")
        self.assertEqual(clean_name("..."), "_")
        self.assertEqual(clean_name("NUL"), "_NUL")
        self.assertEqual(clean_name("東京 2024"), "東京 2024")
        self.assertEqual(device_token("Zenfone 9"), "Zenfone9")
        self.assertEqual(device_token("GO_Ultra"), "GO_Ultra")
        self.assertEqual(device_token("iPhone 15 Pro (Max)"), "iPhone15ProMax")
        self.assertEqual(device_token(None), "")
        self.assertEqual(person_display_name(7, None), "Person_7")
        self.assertEqual(person_display_name(7, "Alice"), "Alice")


if __name__ == "__main__":
    unittest.main()
