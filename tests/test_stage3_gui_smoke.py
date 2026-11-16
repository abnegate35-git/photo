"""第三階段：Tkinter GUI 的單元與冒煙測試。

用暫存的 SQLite v2 資料庫，塞入模擬的 photos（含 `burst_group_id` / `is_best_shot` /
`is_burst_duplicate`）、persons（含 `is_locked = 1`）與 export_logs（含 `action` / `state`），
然後真的建立 `MainWindow` 與 `ThumbnailGrid`，在真實的 Tk 事件迴圈裡驅動它們。

沒有模型權重、也沒有 sv-ttk 的環境就能跑：GUI 的設計本來就要在這種環境下正常運作
（徽章顯示 `Mock / Weights Missing`、主題退回 clam 深色）。沒有顯示器而無法建立 Tk 視窗時整組略過。
"""
from __future__ import annotations

import hashlib
import itertools
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from photosorter.core.events import EventBus
from photosorter.core.export_manager import MASTER_DIR, ExportManager
from photosorter.core.status import ProcessStatus
from photosorter.db import queries_ui
from photosorter.db.db_manager import SCHEMA_VERSION, DBManager
from photosorter.db.repository import Repository
from photosorter.pipeline.decoder import DefaultImageDecoder
from photosorter.pipeline.factory import AppContext, resolve_stages
from photosorter.ui.thumbnail_grid import DEFAULT_CACHE_SIZE, LRUCache, load_thumbnail, thumb_cache_path

from .fakes import make_config, make_image
from .test_stage2_clustering_export import insert_face, insert_photo, ts

try:
    import tkinter as tk

    from photosorter.ui.theme import create_root, releasing_tk

    with releasing_tk():                                    # 試建一個視窗；用完立刻在主執行緒回收
        _probe = create_root()
        _probe.destroy()
        del _probe
    HAVE_TK = True
except Exception:  # noqa: BLE001
    HAVE_TK = False


def pump(root: Any, condition: Callable[[], bool], timeout: float = 20.0, what: str = "") -> None:
    """轉動 Tk 事件迴圈直到條件成立。逾時就讓測試失敗。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"等待逾時：{what}")


class GuiDataCase(unittest.TestCase):
    """暫存資料庫 + 一組涵蓋各種分類的模擬資料。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.lib = self.tmp / "library"
        self.cfg = make_config(self.tmp, pipeline={"geocode": False})     # 預設階段含 siglip / faces，但沒有權重
        self.db = DBManager(self.cfg.db_path, self.cfg.db).open()
        self.repo = Repository(self.db)
        self.bus = EventBus()
        self.ctx = AppContext(self.cfg, self.db, self.repo, self.bus,
                              ExportManager(self.cfg, self.db, self.repo, self.bus))

    def tearDown(self) -> None:
        self.db.close()
        self._tmp.cleanup()

    def one(self, sql: str, params: tuple = ()) -> Any:
        return self.db.query_value(sql, params)

    def seed(self) -> None:
        """8 張照片：一組三連拍、獨照、沒有事件的、螢幕截圖、廢片、遺失的檔案。"""
        start = ts("2024-05-01 10:00:00")
        spec = [
            # (id, 檔名, 秒數位移, 額外欄位)
            (1, "burst_a.jpg", 0, dict(event_id=1, burst_group_id=1, is_burst_duplicate=1, city="Taipei", country_code="TW")),
            (2, "burst_b.jpg", 1, dict(event_id=1, burst_group_id=1, is_best_shot=1, city="Taipei", country_code="TW")),
            (3, "burst_c.jpg", 2, dict(event_id=1, burst_group_id=1, is_burst_duplicate=1, city="Taipei", country_code="TW")),
            (4, "alice.jpg", 600, dict(event_id=1, city="Taipei", country_code="TW", camera_make="ASUS",
                                       camera_model="Zenfone 9", iso=100, f_number=1.9, exposure_time=0.01)),
            (5, "kyoto.jpg", 86400 * 9, dict(city="Kyoto", country_code="JP", gps_lat=35.01, gps_lon=135.77)),
            (6, "Screenshot_1.jpg", 86400 * 20, dict(is_screenshot=1)),
            (7, "blurry.jpg", 86400 * 21, dict(is_rejected=1, reject_reason="blur")),
        ]

        def fill(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO events(id, name, start_ts, end_ts, photo_count, location_label) "
                         "VALUES (1, '2024-05-01_Taipei_food', ?, ?, 4, 'Taipei, TW')", (start, start + 600))
            for pid, name, offset, extra in spec:
                path = make_image(self.lib / name, seed=pid, taken="2024:05:01 10:00:00")
                data = path.read_bytes()
                insert_photo(conn, pid, name, path=str(path), dir_path=str(path.parent), file_size=len(data),
                             sha256=hashlib.sha256(data).digest(), mtime_ns=path.stat().st_mtime_ns, width=320,
                             height=240, taken_ts=start + offset, **extra)
            insert_photo(conn, 8, "gone.jpg", path=str(self.lib / "gone.jpg"), taken_ts=start + 5, is_missing=1)

            conn.execute("INSERT INTO persons(id, name, is_locked) VALUES (1, 'Alice', 1)")
            conn.execute("INSERT INTO persons(id) VALUES (2)")                       # 未命名、未鎖定 -> Person_2
            insert_face(conn, 1, 4, None, person_id=1)
            insert_face(conn, 2, 5, None, person_id=2)
            insert_face(conn, 3, 5, None)                                            # 還沒有歸屬的臉
            food = conn.execute("INSERT INTO tags(name, display_name) VALUES ('food', '美食')").lastrowid
            beach = conn.execute("INSERT INTO tags(name) VALUES ('beach')").lastrowid
            conn.executemany("INSERT INTO photo_tags(photo_id, tag_id, score, is_rejected) VALUES (?, ?, ?, ?)", [
                (1, food, 0.91, 0), (4, food, 0.80, 0), (5, food, 0.70, 0), (5, beach, 0.66, 0),
                (4, beach, 0.90, 1),                                                 # 使用者否決的標籤不算
            ])
            # 一筆既有的匯出歷史：export_logs 以 action / state 記錄
            conn.execute("INSERT INTO export_batches(id, mode, dest_root, template, conflict_policy, state, total) "
                         "VALUES (1, 'copy', ?, 'four_layer_v1', 'shorthash', 'done', 4)", (str(self.tmp / "old"),))
            for seq, (action, state) in enumerate([("copy", "done"), ("copy", "done"), ("hardlink", "done"),
                                                   ("hardlink", "failed")]):
                conn.execute("INSERT INTO export_logs(batch_id, seq, action, src_path, dst_path, state) "
                             "VALUES (1, ?, ?, 'src', ?, ?)", (seq, action, f"dst{seq}", state))

        self.db.write(fill)


# =========================================================================== #
# 不需要視窗的部分
# =========================================================================== #
class SidebarQueryTest(GuiDataCase):
    def setUp(self) -> None:
        super().setUp()
        self.seed()

    def test_schema_is_v2(self) -> None:
        self.assertEqual(self.one("PRAGMA user_version"), SCHEMA_VERSION)
        self.assertGreaterEqual(SCHEMA_VERSION, 2)

    def test_sidebar_counts(self) -> None:
        data = queries_ui.load_sidebar(self.db)
        self.assertEqual(data.total, 7)                                 # 遺失的檔案不算
        self.assertEqual(data.in_events, 4)
        event, orphan = data.events
        self.assertEqual((event.id, event.name, event.count), (1, "2024-05-01_Taipei_food", 4))
        self.assertEqual([(b.group_id, b.count, b.best_file) for b in event.bursts], [(1, 3, "burst_b.jpg")])
        self.assertEqual((orphan.id, orphan.count, orphan.bursts), (None, 3, []))
        self.assertEqual(data.burst_groups, 1)
        self.assertEqual([(p.id, p.display_name, p.is_locked, p.count) for p in data.persons],
                         [(1, "Alice", True, 1), (2, "Person_2", False, 1)])
        self.assertEqual([(t.name, t.display_name, t.count) for t in data.tags],
                         [("food", "美食", 3), ("beach", "beach", 1)])
        self.assertEqual([(c.code, c.count, c.cities) for c in data.locations],
                         [("TW", 4, [("Taipei", 4)]), ("JP", 1, [("Kyoto", 1)])])
        self.assertEqual(data.staging, {"screenshot": 1, "blurry": 1, "burst": 2})

    def test_list_photos_for_every_node_kind(self) -> None:
        expected = {
            "all": 7, "events": 4, "event:1": 4, "event:none": 3, "burst:1": 3, "persons": 2, "person:1": 1,
            "person:2": 1, "tags": 3, "tag:1": 3, "tag:2": 1, "locations": 5, "loc:TW": 4, "city:TW|Taipei": 4,
            "city:JP|Kyoto": 1, "staging": 4, "staging:blurry": 1, "staging:burst": 2, "staging:screenshot": 1,
        }
        for node, count in expected.items():
            self.assertEqual(len(queries_ui.list_photos(self.db, node)), count, node)
        burst = queries_ui.list_photos(self.db, "burst:1")
        self.assertEqual((burst[0].file_name, burst[0].is_best_shot), ("burst_b.jpg", True))   # Best Shot 排最前
        self.assertEqual([p.is_burst_duplicate for p in burst], [False, True, True])
        with self.assertRaises(ValueError):
            queries_ui.list_photos(self.db, "nonsense:1")

    def test_photo_detail(self) -> None:
        detail = queries_ui.photo_detail(self.db, 5)
        self.assertEqual(detail.short_hash, bytes(detail.row["sha256"]).hex()[:6])
        self.assertEqual(len(detail.short_hash), 6)
        self.assertEqual([(name, round(score, 2)) for name, score, _ in detail.tags], [("美食", 0.7), ("beach", 0.66)])
        self.assertEqual([(f.person_name, f.is_locked) for f in detail.faces], [("Person_2", False), ("未知", False)])
        self.assertEqual(queries_ui.photo_detail(self.db, 4).event_name, "2024-05-01_Taipei_food")
        self.assertEqual([(f.person_name, f.is_locked) for f in queries_ui.photo_detail(self.db, 4).faces],
                         [("Alice", True)])
        self.assertIsNone(queries_ui.photo_detail(self.db, 999))

    def test_export_history_reads_action_and_state(self) -> None:
        (batch,) = queries_ui.export_history(self.db)
        self.assertEqual(batch.counts, {"copy": {"done": 2}, "hardlink": {"done": 1, "failed": 1}})
        self.assertEqual(batch.describe_counts(), "copy: done 2｜hardlink: done 1, failed 1")
        self.assertTrue(batch.can_rollback)
        self.assertEqual((batch.id, batch.mode, batch.layout, batch.state), (1, "copy", "four_layer_v1", "done"))

    def test_naming_locks_the_person(self) -> None:
        queries_ui.rename_person(self.db, 2, "  Carol ")
        row = self.db.query_one("SELECT name, is_locked FROM persons WHERE id = 2")
        self.assertEqual((row["name"], row["is_locked"]), ("Carol", 1))
        with self.assertRaises(ValueError):
            queries_ui.rename_person(self.db, 2, "   ")
        with self.assertRaises(ValueError):
            queries_ui.rename_person(self.db, 999, "x")

        # 沒有歸屬的臉：建立新人物、鎖定、這張臉標成手動指派
        new_id = queries_ui.name_face(self.db, 3, "Dave")
        face = self.db.query_one("SELECT person_id, assign_src FROM faces WHERE id = 3")
        self.assertEqual((face["person_id"], face["assign_src"]), (new_id, "manual"))
        person = self.db.query_one("SELECT name, is_locked, face_count FROM persons WHERE id = ?", (new_id,))
        self.assertEqual((person["name"], person["is_locked"], person["face_count"]), ("Dave", 1, 1))
        # 已有歸屬的臉：等同替那個人物改名
        self.assertEqual(queries_ui.name_face(self.db, 1, "Alicia"), 1)
        self.assertEqual(self.one("SELECT name FROM persons WHERE id = 1"), "Alicia")
        queries_ui.set_person_locked(self.db, 1, False)
        self.assertEqual(self.one("SELECT is_locked FROM persons WHERE id = 1"), 0)

    def test_resolve_stages_degrades_without_weights(self) -> None:
        stages, skipped = resolve_stages(self.cfg)
        self.assertEqual(stages, ["meta"])
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all("缺少權重" in s for s in skipped))


class LRUCacheTest(unittest.TestCase):
    def test_default_capacity_is_300_and_oldest_is_evicted(self) -> None:
        cache: LRUCache[int, str] = LRUCache()
        self.assertEqual((cache.maxsize, DEFAULT_CACHE_SIZE), (300, 300))
        for i in range(301):
            cache.put(i, f"img{i}")
        self.assertEqual((len(cache), cache.evictions), (300, 1))
        self.assertNotIn(0, cache)
        self.assertIn(1, cache)

    def test_get_refreshes_recency(self) -> None:
        cache: LRUCache[str, int] = LRUCache(maxsize=3)
        for key in "abc":
            cache.put(key, ord(key))
        self.assertEqual(cache.get("a"), ord("a"))          # a 變成最近用過
        cache.put("d", 1)                                   # 淘汰最久沒用到的 b
        self.assertEqual(cache.keys(), ["c", "a", "d"])
        self.assertIsNone(cache.get("b"))
        cache.put("c", 99)                                  # 更新既有的鍵也算用過
        cache.put("e", 2)
        self.assertEqual(cache.keys(), ["d", "c", "e"])
        self.assertEqual(cache.get("c"), 99)
        cache.clear()
        self.assertEqual(len(cache), 0)
        with self.assertRaises(ValueError):
            LRUCache(maxsize=0)

    def test_thumbnail_disk_cache_is_keyed_by_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = make_image(Path(d) / "a.jpg", size=(800, 600), orientation=6)
            sha = hashlib.sha256(src.read_bytes()).digest()
            cache_dir = Path(d) / "cache"
            thumb = load_thumbnail(str(src), sha, 128, cache_dir)
            self.assertEqual(thumb.size, (96, 128))                     # 已套用 EXIF 方向（直幅），長邊 128
            cached = thumb_cache_path(cache_dir, sha, 128)
            self.assertTrue(cached.is_file())
            self.assertEqual(cached.parent.name, sha.hex()[:2])
            src.unlink()                                                # 原檔不在了，快取仍可用（搬移後也一樣）
            self.assertEqual(load_thumbnail(str(src), sha, 128, cache_dir).size, (96, 128))
            self.assertIsNone(load_thumbnail(str(src), None, 128, cache_dir))
            self.assertIsNone(thumb_cache_path(None, sha, 128))


class CliEntryTest(unittest.TestCase):
    def test_no_subcommand_and_gui_subcommand_launch_the_gui(self) -> None:
        from photosorter.__main__ import main

        with mock.patch("photosorter.ui.app.run_gui", return_value=0) as run_gui:
            self.assertEqual(main([]), 0)
            self.assertEqual(main(["gui"]), 0)
            self.assertEqual(main(["--config", "x.json", "gui"]), 0)
        self.assertEqual([c.args for c in run_gui.call_args_list], [(None,), (None,), ("x.json",)])

    def test_plan_execute_archive_subcommands_still_work(self) -> None:
        from photosorter.__main__ import main

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d, \
                mock.patch.dict(os.environ, {"PHOTOSORTER_HOME": d}), \
                mock.patch("photosorter.__main__._print"):
            dest = os.path.join(d, "out")
            self.assertEqual(main(["plan", dest]), 0)
            self.assertEqual(main(["export", dest]), 0)
            self.assertEqual(main(["archive", dest]), 0)
            self.assertFalse(os.path.exists(dest), "預覽不可以建立任何東西")
            self.assertEqual(main(["execute", dest]), 0)                # 空的照片庫：沒有東西可匯出
            self.assertEqual(main(["archive", dest, "--execute"]), 0)
            self.assertTrue(os.path.isfile(os.path.join(d, "data", "photos.db")))


# =========================================================================== #
# 真的建立視窗
# =========================================================================== #
@unittest.skipUnless(HAVE_TK, "這個環境無法建立 Tk 視窗")
class GuiCase(GuiDataCase):
    def setUp(self) -> None:
        super().setUp()
        from photosorter.ui.theme import apply_theme, create_root

        self.root = create_root()                           # 帶重試：防毒佔住 tk.tcl 時偶爾會建立失敗
        self.root.geometry("1280x820+0+0")
        self.theme = apply_theme(self.root)
        self.tk_errors: list[str] = []
        # Tk 會吞掉 callback 裡的例外只印在 stderr；收集起來，任何一個都算測試失敗
        self.root.report_callback_exception = lambda exc, val, tb: self.tk_errors.append(f"{exc.__name__}: {val}")

    def tearDown(self) -> None:
        from photosorter.ui.theme import releasing_tk

        # Tk 主視窗必須在主執行緒被回收（否則 Tcl 會中止整個行程）：
        # 銷毀、放掉最後的參考、立刻在這條執行緒回收，全部包在 releasing_tk() 裡。
        with releasing_tk():
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self.root = None
        super().tearDown()

    def window(self, **kwargs: Any) -> Any:
        from photosorter.ui.main_window import MainWindow

        win = MainWindow(self.root, self.ctx, confirm=lambda _title, _message: True, **kwargs)
        win.pack(fill="both", expand=True)
        return win

    def alive(self) -> bool:
        try:
            self.root.update()
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def assert_clean(self, win: Any) -> None:
        self.assertEqual(self.tk_errors, [])
        self.assertEqual(win.errors, [])


class MainWindowSmokeTest(GuiCase):
    def setUp(self) -> None:
        super().setUp()
        self.seed()

    def test_full_smoke(self) -> None:
        win = self.window()
        root = self.root
        tree = win.tree
        pump(root, lambda: win.sidebar is not None and len(win.photo_grid.items) == 7, what="側邊欄與照片牆載入")

        # ---- 左側導航樹：五大分類與數量 ----
        text = {iid: tree.item(iid, "text") for iid in win._all_nodes()}
        self.assertEqual(text["all"], "🖼️ 全部照片 (7)")
        self.assertEqual(text["events"], "📅 時間與事件 (Events) (4)")
        self.assertEqual(text["event:1"], "2024-05-01_Taipei_food (4)")
        self.assertEqual(text["burst:1"], "連拍 burst_group_id=1 (3 張)　⭐ Best Shot: burst_b.jpg")
        self.assertEqual(text["event:none"], "未歸入事件 (3)")
        self.assertEqual(text["persons"], "👤 人物群組 (Persons) (2)")
        self.assertEqual(text["person:1"], "🔒 Alice (1)")                 # is_locked = 1
        self.assertEqual(text["person:2"], "Person_2 (1)")
        self.assertEqual(text["tags"], "🏷️ AI 語意標籤 (SigLIP Tags) (2)")
        self.assertEqual(text["tag:1"], "美食 (3)")
        self.assertEqual(text["locations"], "📍 拍攝地點 (Locations) (2)")
        self.assertEqual((text["loc:TW"], text["city:TW|Taipei"]), ("TW (4)", "Taipei (4)"))
        self.assertEqual(text["staging"], "🧹 待清理隔離區 (Staging) (4)")
        self.assertEqual((text["staging:blurry"], text["staging:burst"], text["staging:screenshot"]),
                         ("疑似模糊廢片 (1)", "連拍次佳照片 (2)", "螢幕截圖 (1)"))

        # ---- 硬體徽章：沒有權重不崩潰 ----
        pump(root, lambda: win.badge_var.get() != "偵測硬體中…", what="硬體徽章")
        self.assertEqual(win.badge_var.get(), "Mock / Weights Missing")

        # ---- 點每一個節點都不出錯 ----
        for iid in text:
            tree.selection_set(iid)
            pump(root, lambda iid=iid: win.current_node == iid and win.header_var.get().startswith(text[iid]),
                 what=f"節點 {iid}")
        self.assertEqual(len(win.photo_grid.items), 1)                     # 最後一個是 staging:screenshot

        # ---- 連拍群組：Best Shot 排最前，縮圖真的載入 ----
        win.select_node("burst:1")
        pump(root, lambda: [p.file_name for p in win.photo_grid.items][:1] == ["burst_b.jpg"], what="連拍清單")
        pump(root, lambda: len(win.photo_grid._cell_images) == 3 and win.photo_grid.pending_count == 0, what="縮圖")
        win.photo_grid.select_index(0)
        pump(root, lambda: win.detail is not None and win.detail.row["id"] == 2, what="照片明細")
        info = {win.info.item(i, "text"): win.info.item(i, "values")[0] for i in win.info.get_children()}
        self.assertEqual(info["檔名"], "burst_b.jpg")
        self.assertEqual(info["ShortHash"], bytes(self.one("SELECT sha256 FROM photos WHERE id = 2")).hex()[:6])
        self.assertEqual(info["連拍"], "burst_group_id=1　⭐ Best Shot")
        self.assertIn("⭐ Best Shot", info["標記"])
        self.assertEqual(info["Sidecar"], "無")
        self.assertIn("blur_score", info["清晰度"])
        self.assertIsNotNone(win._preview_photo)

        # ---- Sidecar 提示、EXIF、標籤信心度、人臉 ----
        (self.lib / "alice.MOV").write_bytes(b"live")
        win.select_node("event:1")
        pump(root, lambda: len(win.photo_grid.items) == 4, what="事件清單")
        self.assertTrue(win.photo_grid.select_photo(4))
        pump(root, lambda: win.detail is not None and win.detail.row["id"] == 4, what="alice 明細")
        info = {win.info.item(i, "text"): win.info.item(i, "values")[0] for i in win.info.get_children()}
        self.assertEqual(info["Sidecar"], "alice.MOV")
        self.assertEqual(info["相機"], "ASUS Zenfone 9")
        self.assertEqual(info["曝光"], "f/1.9　1/100 秒　ISO 100")
        self.assertEqual(info["標籤"], "美食 0.80")                        # 被否決的 beach 不顯示
        self.assertEqual(info["事件"], "2024-05-01_Taipei_food")
        self.assertEqual([win.face_list.item(i, "text") for i in win.face_list.get_children()], ["🔒 Alice"])
        self.assertEqual(len(win.preview_canvas.find_all()), 2)            # 預覽圖 + 一個人臉框

        # ---- 人物命名與鎖定（經過對話框）----
        dialog = win.prompt_rename_person(2)
        dialog.name_var.set("Carol")
        dialog.submit()
        pump(root, lambda: self.one("SELECT name FROM persons WHERE id = 2") == "Carol"
             and tree.exists("person:2") and tree.item("person:2", "text") == "🔒 Carol (1)", what="人物改名")
        self.assertEqual(self.one("SELECT is_locked FROM persons WHERE id = 2"), 1)
        dialog = win.prompt_rename_face()                                  # 目前照片只有一張臉 -> 就是它
        dialog.name_var.set("Alicia")
        dialog.submit()
        pump(root, lambda: tree.exists("person:1") and tree.item("person:1", "text") == "🔒 Alicia (1)", what="人臉改名")
        win.rename_face(3, "Dave")                                         # 沒有歸屬的臉 -> 新人物
        pump(root, lambda: self.one("SELECT COUNT(*) FROM persons WHERE name = 'Dave' AND is_locked = 1") == 1
             and any(tree.item(i, "text") == "🔒 Dave (1)" for i in tree.get_children("persons")), what="新人物")

        # ---- 匯出：Dry-Run → 執行 → 歷史 → 一鍵還原 ----
        out = self.tmp / "export"
        win.dest_var.set(str(out))
        win.mode_var.set("HARDLINK")
        self.assertTrue(win.preview_export())
        self.assertFalse(win.preview_export(), "同時間只能有一個獨佔工作")
        pump(root, lambda: win.preview_dialog is not None and not win.busy, what="匯出預覽")
        preview = win.last_preview
        self.assertFalse(out.exists(), "預覽不可以建立任何東西")
        self.assertGreater(len(preview.pending()), 0)
        self.assertEqual(preview.estimated_bytes, 0)                       # 同磁碟機的硬連結不佔空間
        self.assertIn("master:hardlink", win.preview_dialog.summary_text)
        win.preview_dialog.execute()
        pump(root, lambda: win.last_export is not None and not win.busy, what="執行匯出")
        self.assertEqual(win.last_export.failed, 0, win.last_export.errors)
        master_files = [p for p in (out / MASTER_DIR).rglob("*") if p.is_file()]
        self.assertEqual(len(master_files), 4)                             # 3 張正常照片（含 Best Shot）+ alice.mov
        self.assertEqual(self.one("SELECT mode FROM export_batches WHERE id = 2"), "hardlink")

        win.open_rollback()
        pump(root, lambda: win.rollback_dialog is not None, what="復原歷史")
        dlg = win.rollback_dialog
        rows = {iid: dlg.tree.item(iid, "values") for iid in dlg.tree.get_children()}
        self.assertEqual(set(rows), {"1", "2"})
        self.assertEqual(rows["1"][4], "copy: done 2｜hardlink: done 1, failed 1")     # action: state 筆數
        self.assertIn("hardlink: done", rows["2"][4])
        dlg.tree.selection_set("2")
        root.update()
        dlg.rollback()
        pump(root, lambda: self.one("SELECT state FROM export_batches WHERE id = 2") == "rolled_back"
             and not win.busy, what="還原")
        self.assertEqual([p for p in out.rglob("*") if p.is_file()], [])
        self.assertTrue((self.lib / "alice.jpg").is_file())

        # ---- 重新分群（背景執行），完成後導航樹刷新 ----
        self.assertTrue(win.recluster())
        pump(root, lambda: not win.busy and any("重新分群完成" in t for _l, t in win.messages), what="重新分群")
        self.assertEqual(self.one("SELECT is_locked FROM persons WHERE id = 1"), 1)     # 鎖定的人物沒被動到

        # ---- 規格指定的收尾方式：排程 destroy，事件迴圈自然結束 ----
        root.after(300, root.destroy)
        root.mainloop()
        self.assert_clean(win)

    def test_window_survives_being_destroyed_right_after_creation(self) -> None:
        """`root.after(300, root.destroy)`：查詢與縮圖還在背景跑的時候視窗就被關掉，也不可以有任何錯誤。"""
        win = self.window()
        self.root.after(300, self.root.destroy)
        self.root.mainloop()
        self.assertTrue(win._closed)
        self.assert_clean(win)
        self.assertTrue(self.db.is_open)                    # 測試自己擁有 ctx：視窗不可以把資料庫關掉

    def test_theme_falls_back_to_clam_dark_without_sv_ttk(self) -> None:
        from photosorter.ui.theme import apply_theme, enable_high_dpi, ui_scale

        with mock.patch.dict(sys.modules, {"sv_ttk": None}):                # 模擬沒安裝
            self.assertEqual(apply_theme(self.root), "clam-dark")
        self.assertEqual(tk.ttk.Style(self.root).theme_use(), "clam")
        self.assertGreaterEqual(ui_scale(self.root), 1.0)
        enable_high_dpi()                                                   # 重複呼叫或不支援時都不可以丟例外


class ScanFromGuiTest(GuiCase):
    def make_library(self, n: int) -> None:
        for i in range(n):
            make_image(self.lib / f"IMG_{i:03d}.jpg", seed=i, taken=f"2024:06:01 10:{i % 60:02d}:00")

    def test_scan_without_weights_runs_stage1_and_refreshes_tree(self) -> None:
        self.make_library(6)
        win = self.window()
        pump(self.root, lambda: win.sidebar is not None, what="初始載入")
        self.assertFalse(win.start_scan(), "還沒選資料夾不可以開始")
        win.set_library(str(self.lib))
        self.assertTrue(win.start_scan())
        self.assertFalse(win.start_scan(), "掃描中不可以再開始一次")
        self.assertFalse(win.recluster(), "掃描中不可以分群")
        pump(self.root, lambda: win.engine_state == "finished" and not win.busy
             and win.sidebar is not None and win.sidebar.total == 6 and len(win.photo_grid.items) == 6, what="掃描完成")
        self.assertEqual(len(win.skipped_stages), 2)                        # siglip、faces 因缺權重被略過
        full = int(ProcessStatus.stage1())
        self.assertEqual(self.one(f"SELECT COUNT(*) FROM photos WHERE process_status & {full} = {full}"), 6)
        self.assertEqual(self.repo.kv_get("ui.library"), os.path.normpath(str(self.lib)))
        self.assertEqual(str(win.btn_scan.state()), "()")                   # 結束後按鈕恢復可用
        self.assertIn("disabled", win.btn_stop.state())
        self.assertEqual(win.badge_var.get(), "Mock / Weights Missing")
        self.assert_clean(win)

    def test_closing_window_during_scan_stops_safely_and_scan_resumes_later(self) -> None:
        self.make_library(40)
        original = DefaultImageDecoder.decode
        release = threading.Event()
        calls = itertools.count()

        def gated_decode(decoder: Any, *args: Any, **kwargs: Any) -> Any:
            # 前 6 張照常處理，之後的解碼卡住直到測試放行 —— 保證「暫停」與「關閉」都發生在掃描進行中，
            # 不依賴機器快慢。
            if next(calls) >= 6:
                release.wait(timeout=60)
            return original(decoder, *args, **kwargs)

        with mock.patch.object(DefaultImageDecoder, "decode", gated_decode):
            win = self.window()
            win.set_library(str(self.lib))
            self.assertTrue(win.start_scan())
            pump(self.root, lambda: (self.one("SELECT COUNT(*) FROM photos") or 0) >= 6, what="掃描進行中")
            self.assertTrue(win.engine_running)
            pump(self.root, lambda: win.engine_state == "running", what="視窗得知引擎已啟動")
            win.toggle_pause()
            pump(self.root, lambda: win.engine_state == "paused", what="暫停")
            self.assertEqual(win.btn_pause.cget("text"), "繼續")
            win.toggle_pause()
            pump(self.root, lambda: win.engine_state == "running", what="繼續")

            win.close()                                                     # WM_DELETE_WINDOW
            self.assertTrue(win._closing)
            self.assertTrue(self.alive(), "引擎還沒停下來之前視窗不可以關")
            release.set()                                                   # 讓卡住的解碼做完，引擎才能安全停止
            deadline = time.monotonic() + 30
            while self.alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(self.alive(), "視窗應該在引擎安全停止後關閉")

        engine = win.engine
        self.assertFalse(engine.is_running)
        self.assertEqual(engine.state.value, "stopped")
        done = self.one("SELECT COUNT(*) FROM photos")
        self.assertTrue(0 < done < 40, done)
        # 已 commit 的每一列都是完整的 Stage 1 結果（沒有半套交易）
        full = int(ProcessStatus.stage1())
        self.assertEqual(self.one(f"SELECT COUNT(*) FROM photos WHERE process_status & {full} = {full}"), done)
        self.assertEqual(self.one("SELECT state FROM scan_sessions ORDER BY id DESC LIMIT 1"), "stopped")
        self.assertEqual(self.tk_errors, [])

        # 之後再掃一次：從中斷處接著做完
        engine2 = self.ctx.build_engine(["meta"])
        engine2.start([self.lib])
        self.assertTrue(engine2.join(60))
        self.assertEqual(self.one("SELECT COUNT(*) FROM photos"), 40)
        self.assertEqual(engine2.stats().unchanged, done)


class ThumbnailGridTest(GuiCase):
    def test_virtual_scrolling_renders_only_the_viewport_and_lru_is_bounded(self) -> None:
        from photosorter.ui.thumbnail_grid import BUFFER_ROWS, ThumbnailGrid

        files = []
        for i in range(8):
            path = make_image(self.lib / f"t{i}.jpg", seed=i, color=(30 * i, 90, 200 - 20 * i))
            files.append((str(path), hashlib.sha256(path.read_bytes()).digest()))
        items = [queries_ui.PhotoItem(i + 1, *files[i % 8][:1], f"photo_{i:04d}.jpg", files[i % 8][1], None,
                                      i % 50 == 0, False, None, False, False, False) for i in range(1000)]
        items.append(queries_ui.PhotoItem(5000, str(self.lib / "missing.jpg"), "missing.jpg", None, None,
                                          False, False, None, False, False, False))
        selected: list[Any] = []
        grid = ThumbnailGrid(self.root, cache_dir=self.cfg.cache_dir, cache_size=40, on_select=selected.append)
        grid.pack(fill="both", expand=True)
        self.root.update()
        grid.set_items(items)
        pump(self.root, lambda: grid.rendered_count > 0 and grid.pending_count == 0, what="第一屏縮圖")

        cols = grid._cols
        visible_rows = -(-grid.canvas.winfo_height() // grid.cell_h)
        self.assertGreater(cols, 1)
        # 1001 張照片，只建立「可見的列 ± 緩衝列」的格子
        self.assertLessEqual(grid.rendered_count, (visible_rows + 2 * BUFFER_ROWS + 1) * cols)
        self.assertLess(grid.rendered_count, 150)
        first, last = grid.visible_range()
        self.assertEqual((first, grid.rendered_count), (0, last))
        self.assertEqual(len(grid._cell_images), grid.rendered_count)       # 每一格都拿到圖了
        self.assertTrue((self.cfg.cache_dir / "thumbs").is_dir())           # 磁碟快取已寫入

        # 捲到中間：舊的格子刪除、新的建立，總數仍然只有一屏
        grid.canvas.yview_moveto(0.5)
        pump(self.root, lambda: grid.visible_range()[0] > last and grid.pending_count == 0
             and len(grid._cell_images) == grid.rendered_count, what="捲動後的縮圖")
        self.assertLess(grid.rendered_count, 150)
        self.assertNotIn(0, grid._cells)

        # 一路捲到底：LRU 不超過上限，而且發生過淘汰；正在顯示的圖不受淘汰影響
        for fraction in (0.6, 0.7, 0.8, 0.9, 1.0):
            grid.canvas.yview_moveto(fraction)
            pump(self.root, lambda: grid.pending_count == 0, what="捲動")
        self.assertLessEqual(len(grid.cache), 40)
        self.assertGreater(grid.cache.evictions, 0)
        self.assertGreaterEqual(grid.visible_range()[1], 1001)
        missing_index = grid._index_of[5000]
        self.assertIn(missing_index, grid._cells)                           # 讀不到的檔案：留佔位方塊，不出錯
        self.assertNotIn(missing_index, grid._cell_images)
        self.assertEqual(len(grid._cell_images), grid.rendered_count - 1)

        # 選取：通知 callback、必要時捲動到看得見
        grid.select_index(3)
        self.root.update()
        self.assertEqual(selected[-1].id, 4)
        self.assertEqual(grid.visible_range()[0], 0)
        self.assertEqual(grid.selected_item.file_name, "photo_0003.jpg")
        grid.set_items(items[:10], keep_selection=True)
        self.assertEqual(grid.selected_item.id, 4)
        grid.set_items([])
        self.assertEqual((grid.rendered_count, grid.selected_item), (0, None))
        self.assertEqual(self.tk_errors, [])


if __name__ == "__main__":
    unittest.main()
