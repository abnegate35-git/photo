"""PipelineEngine 端到端：用假模型在真實的執行緒、佇列、資料庫上跑完整條管線。"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from photosorter.core.events import Event, EventBus, EventKind
from photosorter.core.status import ProcessStatus
from photosorter.core.types import GateRole
from photosorter.db.db_manager import DBManager
from photosorter.db.repository import Repository, blob_to_vec
from photosorter.models.siglip import SiglipTagger
from photosorter.pipeline.decoder import DefaultImageDecoder
from photosorter.pipeline.event_clustering import TimeGapEventClusterer
from photosorter.pipeline.pipeline_engine import EngineState, PipelineEngine, should_run_faces

from .fakes import (
    EMBED_DIM,
    FACE_DIM,
    FakeFaces,
    FakeGeocoder,
    FakeTextEncoder,
    FakeVision,
    make_config,
    make_image,
    rename,
)

RED, GREEN, BLUE = (230, 15, 15), (15, 230, 15), (15, 15, 230)
FULL = ProcessStatus.stage1() | ProcessStatus.stage2() | ProcessStatus.stage3()


class EngineHarness:
    """一個暫存照片庫 + 資料庫 + 引擎。每次 `run()` 都建立新的引擎，模擬「程式重新啟動」。"""

    def __init__(self, tmp: Path, **cfg_overrides: Any) -> None:
        self.tmp = tmp
        self.library = tmp / "library"
        self.library.mkdir()
        self.cfg = make_config(tmp, **cfg_overrides)
        self.db = DBManager(self.cfg.db_path, self.cfg.db).open()
        self.repo = Repository(self.db)
        # red = 一般標籤；green = 文件類（略過人臉）；blue = 人物語意
        self.repo.upsert_tag("red", "a red photo")
        self.repo.upsert_tag("green", "a green document", gate_role=GateRole.SKIP_FACE)
        self.repo.upsert_tag("blue", "a blue person", gate_role=GateRole.HAS_PERSON)
        self.events: list[Event] = []
        self.vision = FakeVision()
        self.faces = FakeFaces()
        self.text = FakeTextEncoder()
        self.geocoder = FakeGeocoder()

    def build(self) -> tuple[PipelineEngine, EventBus]:
        bus = EventBus()
        bus.subscribe(self.events.append)
        stages = self.cfg.pipeline.stages
        engine = PipelineEngine(
            self.cfg, self.db, self.repo, bus,
            decoder=DefaultImageDecoder(self.cfg),
            vision=self.vision if "siglip" in stages else None,
            tagger=SiglipTagger(self.cfg, self.repo, self.text) if "siglip" in stages else None,
            faces=self.faces if "faces" in stages else None,
            geocoder=self.geocoder,
            event_clusterer=TimeGapEventClusterer(self.cfg.events),
        )
        return engine, bus

    def run(self, timeout: float = 60.0) -> PipelineEngine:
        self.events.clear()
        engine, _bus = self.build()
        engine.start([self.library])
        assert engine.join(timeout), "管線沒有在時限內結束"
        return engine

    def photo(self, name: str) -> Any:
        return self.db.query_one("SELECT * FROM photos WHERE file_name = ?", (name,))

    def tags_of(self, photo_id: int) -> set[str]:
        return {r[0] for r in self.db.query(
            "SELECT t.name FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id WHERE pt.photo_id = ?", (photo_id,))}

    def count(self, table: str, where: str = "1") -> int:
        return int(self.db.query_value(f"SELECT COUNT(*) FROM {table} WHERE {where}"))

    def close(self) -> None:
        self.db.close()


class PipelineEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.h = EngineHarness(Path(self._tmp.name))
        self.lib = self.h.library

    def tearDown(self) -> None:
        self.h.close()
        self._tmp.cleanup()

    # ------------------------------------------------------------------ #
    def _seed_basic(self, n: int = 9) -> None:
        colors = [RED, GREEN, BLUE]
        for i in range(n):
            make_image(self.lib / f"sub{i % 2}" / f"img{i:02d}.jpg", color=colors[i % 3], seed=i,
                       taken=f"2024:05:01 10:{i:02d}:00")

    def test_full_run_writes_every_stage(self) -> None:
        self._seed_basic()
        engine = self.h.run()
        self.assertIs(engine.state, EngineState.FINISHED)
        stats = engine.stats()
        self.assertEqual((stats.to_process, stats.processed, stats.errors), (9, 9, 0))

        for r in self.h.db.query("SELECT * FROM photos"):
            self.assertEqual(ProcessStatus(r["process_status"]) & FULL, FULL, r["file_name"])
            self.assertEqual(len(r["sha256"]), 32)
            self.assertEqual((r["width"], r["height"]), (320, 240))
            self.assertEqual(r["camera_model"], "TestCam X1")
            self.assertEqual(r["taken_src"], "exif")
            self.assertIsNotNone(r["dhash"])
            self.assertGreater(r["blur_score"], 0)
        self.assertEqual(self.h.count("embeddings"), 9)
        vec = blob_to_vec(self.h.db.query_value("SELECT vector FROM embeddings LIMIT 1"), EMBED_DIM)
        self.assertAlmostEqual(float((vec * vec).sum()), 1.0, places=5)

        # 多標籤判定：顏色對應的標籤
        self.assertEqual(self.h.tags_of(self.h.photo("img00.jpg")["id"]), {"red"})
        self.assertEqual(self.h.tags_of(self.h.photo("img01.jpg")["id"]), {"green"})
        self.assertEqual(self.h.tags_of(self.h.photo("img02.jpg")["id"]), {"blue"})

        # 條件式人臉：green（文件類）被略過，其餘有跑
        green_ids = {r["id"] for r in self.h.db.query("SELECT id FROM photos") if self.h.tags_of(r["id"]) == {"green"}}
        self.assertEqual(len(green_ids), 3)
        self.assertTrue(green_ids.isdisjoint(self.h.faces.analyzed))
        self.assertEqual(self.h.count("faces"), 6)
        self.assertEqual(stats.faces_skipped, 3)
        for pid in green_ids:
            st = ProcessStatus(self.h.db.query_value("SELECT process_status FROM photos WHERE id = ?", (pid,)))
            self.assertTrue(st & ProcessStatus.FACES_SKIPPED)
        face = self.h.db.query_one("SELECT * FROM faces LIMIT 1")
        self.assertEqual(blob_to_vec(face["embedding"]).size, FACE_DIM)
        self.assertEqual(blob_to_vec(face["landmarks"]).size, 10)

        # 文字向量只算一次並快取在資料庫
        self.assertEqual(self.h.text.calls, 1)
        self.assertEqual(self.h.count("tags", "text_vector IS NOT NULL"), 3)

        # 推論在單一執行緒上、有做 batching、模型用完有釋放
        self.assertEqual(self.h.vision.loaded_on, "inference")
        self.assertTrue(max(self.h.vision.batch_sizes) > 1)
        self.assertTrue(all(b <= engine.plan.batch_size for b in self.h.vision.batch_sizes))
        self.assertTrue(self.h.vision.closed)

        kinds = [e.kind for e in self.h.events]
        self.assertEqual(kinds[0], EventKind.STATE)
        self.assertEqual(kinds[-1], EventKind.FINISHED)
        self.assertIn(EventKind.SCAN_DONE, kinds)
        self.assertEqual(sum(1 for k in kinds if k is EventKind.PHOTO_DONE), 9)
        scan = self.h.db.query_one("SELECT state, stats FROM scan_sessions")
        self.assertEqual(scan["state"], "finished")

    def test_rescan_skips_everything_and_does_not_reencode_text(self) -> None:
        self._seed_basic()
        self.h.run()
        batches_before = len(self.h.vision.batch_sizes)
        engine = self.h.run()
        stats = engine.stats()
        self.assertEqual((stats.files_seen, stats.unchanged, stats.to_process), (9, 9, 0))
        self.assertEqual(len(self.h.vision.batch_sizes), batches_before)
        self.assertEqual(self.h.text.calls, 1)

    def test_stop_midway_then_resume_completes_without_redoing(self) -> None:
        self._seed_basic(24)
        self.h.vision.delay_s = 0.05
        engine, bus = self.h.build()
        done = threading.Event()
        bus.subscribe(lambda e: done.set() if e.kind is EventKind.PHOTO_DONE else None)
        engine.start([self.lib])
        self.assertTrue(done.wait(30))
        engine.stop()
        engine.stop()                                        # 可重複呼叫
        self.assertTrue(engine.join(30))
        self.assertIs(engine.state, EngineState.STOPPED)
        partial = self.h.count("embeddings")
        self.assertTrue(0 < partial < 24, partial)
        # 一致性：有向量 <=> 狀態位元已設
        flagged = self.h.count("photos", f"process_status & {int(ProcessStatus.EMBEDDED)} != 0")
        self.assertEqual(flagged, partial)
        self.assertEqual(self.h.db.query_value("SELECT state FROM scan_sessions ORDER BY id DESC LIMIT 1"), "stopped")

        self.h.vision.delay_s = 0.0
        inferred_before = sum(self.h.vision.batch_sizes)
        engine2 = self.h.run()
        self.assertIs(engine2.state, EngineState.FINISHED)
        self.assertEqual(self.h.count("embeddings"), 24)
        self.assertEqual(self.h.count("photos", f"process_status & {int(FULL)} = {int(FULL)}"), 24)
        # 續掃只補沒做完的，不重做已完成的
        self.assertEqual(sum(self.h.vision.batch_sizes) - inferred_before, 24 - partial)
        self.assertEqual(engine2.stats().unchanged, partial)

    def test_pause_and_resume(self) -> None:
        self._seed_basic(16)
        self.h.vision.delay_s = 0.03
        engine, bus = self.h.build()
        first = threading.Event()
        bus.subscribe(lambda e: first.set() if e.kind is EventKind.PHOTO_DONE else None)
        engine.start([self.lib])
        self.assertTrue(first.wait(30))
        engine.pause()
        self.assertIs(engine.state, EngineState.PAUSED)
        time.sleep(0.4)                                      # 讓進行中的 batch 做完
        frozen = engine.stats().processed
        time.sleep(0.4)
        self.assertEqual(engine.stats().processed, frozen, "暫停期間不應該有進度")
        self.assertTrue(engine.is_running)
        engine.resume()
        self.assertTrue(engine.join(30))
        self.assertIs(engine.state, EngineState.FINISHED)
        self.assertEqual(engine.stats().processed, 16)

    def test_bounded_queue_backpressure(self) -> None:
        self._seed_basic(30)
        self.h.vision.delay_s = 0.02
        engine, bus = self.h.build()
        depths: list[int] = []
        bus.subscribe(lambda e: depths.append(e.data["queue_depth"]) if e.kind is EventKind.PROGRESS else None)
        engine.start([self.lib])
        self.assertTrue(engine.join(60))
        self.assertEqual(engine.stats().processed, 30)
        self.assertLessEqual(max(depths), engine.plan.queue_maxsize)

    # ------------------------------------------------------------------ #
    # Stage 0
    # ------------------------------------------------------------------ #
    def test_moved_file_keeps_id_and_skips_ai(self) -> None:
        self._seed_basic(3)
        self.h.run()
        before = self.h.photo("img00.jpg")
        inferred = sum(self.h.vision.batch_sizes)
        rename(self.lib / "sub0" / "img00.jpg", self.lib / "moved" / "renamed.jpg")

        engine = self.h.run()
        self.assertEqual(engine.stats().moved, 1)
        after = self.h.photo("renamed.jpg")
        self.assertEqual(after["id"], before["id"])
        self.assertTrue(after["path"].endswith(str(Path("moved") / "renamed.jpg")))
        self.assertEqual(after["is_missing"], 0)
        self.assertIsNone(self.h.photo("img00.jpg"))
        self.assertEqual(sum(self.h.vision.batch_sizes), inferred, "搬移不應該重跑推論")
        self.assertEqual(self.h.count("photos"), 3)

    def test_duplicate_content_clones_results(self) -> None:
        self._seed_basic(3)
        self.h.run()
        inferred = sum(self.h.vision.batch_sizes)
        shutil.copy2(self.lib / "sub0" / "img00.jpg", self.lib / "copy.jpg")

        engine = self.h.run()
        self.assertEqual(engine.stats().duplicates, 1)
        self.assertEqual(sum(self.h.vision.batch_sizes), inferred)
        orig, copy = self.h.photo("img00.jpg"), self.h.photo("copy.jpg")
        self.assertNotEqual(orig["id"], copy["id"])
        self.assertEqual(orig["sha256"], copy["sha256"])
        self.assertEqual(ProcessStatus(copy["process_status"]) & FULL, FULL)
        self.assertEqual(self.h.tags_of(copy["id"]), self.h.tags_of(orig["id"]))
        self.assertEqual(self.h.count("embeddings", f"photo_id = {copy['id']}"), 1)
        self.assertEqual(self.h.count("faces", f"photo_id = {copy['id']}"), 1)

    def test_changed_content_is_reprocessed_but_touch_is_not(self) -> None:
        self._seed_basic(3)
        self.h.run()
        target = self.lib / "sub0" / "img00.jpg"
        pid = self.h.photo("img00.jpg")["id"]
        self.assertEqual(self.h.tags_of(pid), {"red"})
        # 使用者手動標記：內容換了也要保留
        self.h.db.execute("UPDATE photos SET is_favorite = 1 WHERE id = ?", (pid,)).result()

        # 只動 mtime：不重跑
        inferred = sum(self.h.vision.batch_sizes)
        st = target.stat()
        import os
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        self.h.run()
        self.assertEqual(sum(self.h.vision.batch_sizes), inferred)
        self.assertEqual(self.h.photo("img00.jpg")["mtime_ns"], target.stat().st_mtime_ns)

        # 內容換成藍色：重跑，舊的自動標籤被取代
        make_image(target, color=BLUE, seed=99)
        self.h.run()
        row = self.h.photo("img00.jpg")
        self.assertEqual(row["id"], pid)
        self.assertEqual(self.h.tags_of(pid), {"blue"})
        self.assertEqual(row["is_favorite"], 1)
        self.assertEqual(self.h.count("embeddings", f"photo_id = {pid}"), 1)
        self.assertEqual(self.h.count("faces", f"photo_id = {pid}"), 1)

    def test_missing_files_are_flagged_not_deleted(self) -> None:
        self._seed_basic(3)
        self.h.run()
        (self.lib / "sub0" / "img00.jpg").unlink()
        self.h.run()
        self.assertEqual(self.h.photo("img00.jpg")["is_missing"], 1)
        self.assertEqual(self.h.count("photos"), 3)
        # 檔案回來了 -> 取消遺失標記，不重跑
        make_image(self.lib / "sub0" / "img00.jpg", color=RED, seed=0, taken="2024:05:01 10:00:00")
        self.h.run()
        self.assertEqual(self.h.photo("img00.jpg")["is_missing"], 0)

    def test_unreachable_root_does_not_mark_library_missing(self) -> None:
        self._seed_basic(3)
        self.h.run()
        hidden = self.h.tmp / "unplugged"
        self.lib.rename(hidden)                              # 模擬外接硬碟被拔掉
        try:
            engine = self.h.run()
            self.assertIs(engine.state, EngineState.FINISHED)
            self.assertEqual(self.h.count("photos", "is_missing = 1"), 0)
            self.assertTrue(any(e.kind is EventKind.ERROR for e in self.h.events))
        finally:
            hidden.rename(self.lib)

    # ------------------------------------------------------------------ #
    # 錯誤處理
    # ------------------------------------------------------------------ #
    def test_corrupt_file_is_recorded_and_pipeline_continues(self) -> None:
        self._seed_basic(3)
        (self.lib / "broken.jpg").write_bytes(b"this is not a jpeg" * 10)
        engine = self.h.run()
        self.assertIs(engine.state, EngineState.FINISHED)
        self.assertEqual(engine.stats().errors, 1)
        row = self.h.photo("broken.jpg")
        self.assertTrue(ProcessStatus(row["process_status"]) & ProcessStatus.ERR_DECODE)
        self.assertIn("decode", row["last_error"])
        self.assertEqual(self.h.count("embeddings"), 3)
        self.assertTrue(any(e.kind is EventKind.PHOTO_ERROR for e in self.h.events))
        # 重掃不會一直重試壞檔
        engine2 = self.h.run()
        self.assertEqual(engine2.stats().to_process, 0)

    def test_inference_failure_is_isolated_to_one_photo(self) -> None:
        self._seed_basic(6)
        make_image(self.lib / "poison.jpg", color=(77, 10, 10), seed=500)
        self.h.vision.fail_on_red = 77
        engine = self.h.run()
        self.assertIs(engine.state, EngineState.FINISHED)
        self.assertEqual(engine.stats().errors, 1)
        poison = self.h.photo("poison.jpg")
        st = ProcessStatus(poison["process_status"])
        self.assertTrue(st & ProcessStatus.ERR_INFER)
        self.assertFalse(st & ProcessStatus.EMBEDDED)
        self.assertEqual(st & ProcessStatus.stage1(), ProcessStatus.stage1())    # Stage 1 的結果保留
        self.assertEqual(self.h.count("embeddings"), 6)
        self.assertEqual(self.h.count("photos", f"process_status & {int(FULL)} = {int(FULL)}"), 6)

        # 修好之後以 retry_errors 重跑：只補那一張
        self.h.vision.fail_on_red = None
        self.h.cfg.scan.retry_errors = True
        engine2 = self.h.run()
        self.assertEqual(engine2.stats().to_process, 1)
        st = ProcessStatus(self.h.photo("poison.jpg")["process_status"])
        self.assertEqual(st & FULL, FULL)
        self.assertFalse(st & ProcessStatus.error_bits())

    def test_unimplemented_component_fails_fast_without_polluting_photos(self) -> None:
        self._seed_basic(4)
        self.h.faces.not_implemented = True
        engine = self.h.run()
        self.assertIs(engine.state, EngineState.FAILED)
        finished = self.h.events[-1]
        self.assertIs(finished.kind, EventKind.FINISHED)
        self.assertIn("NotImplementedError", finished.data["error"])
        self.assertEqual(self.h.count("photos", f"process_status & {int(ProcessStatus.ERR_INFER)} != 0"), 0)

    def test_model_load_failure_reports_failed(self) -> None:
        self._seed_basic(2)

        def boom() -> list:
            raise RuntimeError("no weights")

        self.h.vision.load = boom                            # type: ignore[method-assign]
        engine = self.h.run()
        self.assertIs(engine.state, EngineState.FAILED)
        self.assertTrue(any(e.kind is EventKind.MODEL and e.data["phase"] == "failed" for e in self.h.events))
        # Stage 1 還沒開始派工，資料庫不應該有半成品
        self.assertEqual(self.h.count("embeddings"), 0)

    def test_start_twice_is_rejected(self) -> None:
        self._seed_basic(8)
        self.h.vision.delay_s = 0.05
        engine, _ = self.h.build()
        engine.start([self.lib])
        with self.assertRaises(RuntimeError):
            engine.start([self.lib])
        engine.stop()
        self.assertTrue(engine.join(30))

    # ------------------------------------------------------------------ #
    # Stage 1 細節與後處理
    # ------------------------------------------------------------------ #
    def test_screenshot_rule_skips_faces(self) -> None:
        make_image(self.lib / "Screenshots" / "Screenshot_20240501-101500.jpg", color=RED, camera=None, taken=None)
        make_image(self.lib / "normal.jpg", color=RED, seed=1)
        self.h.run()
        shot = self.h.photo("Screenshot_20240501-101500.jpg")
        self.assertEqual(shot["is_screenshot"], 1)
        self.assertEqual(shot["taken_src"], "filename")
        self.assertEqual(shot["taken_at"], "2024-05-01T10:15:00")
        self.assertNotIn(shot["id"], self.h.faces.analyzed)
        self.assertIn(self.h.photo("normal.jpg")["id"], self.h.faces.analyzed)

    def test_orientation_is_applied_before_ai(self) -> None:
        make_image(self.lib / "rot.jpg", color=RED, size=(320, 240), orientation=6)
        self.h.run()
        row = self.h.photo("rot.jpg")
        self.assertEqual((row["width"], row["height"], row["orientation"]), (240, 320, 6))

    def test_geocode_and_event_clustering_post_steps(self) -> None:
        # 兩個事件：台北（上午連拍 4 張）與隔天的另一地（4 張）；另有 1 張孤立照片不成事件
        for i in range(4):
            make_image(self.lib / f"a{i}.jpg", color=RED, seed=i, taken=f"2024:05:01 09:{i * 5:02d}:00",
                       gps=(25.03, 121.56))
        for i in range(4):
            make_image(self.lib / f"b{i}.jpg", color=RED, seed=10 + i, taken=f"2024:05:02 15:{i * 5:02d}:00",
                       gps=(22.63, 120.30))
        make_image(self.lib / "lonely.jpg", color=RED, seed=50, taken="2024:06:15 12:00:00")
        engine = self.h.run()
        self.assertIs(engine.state, EngineState.FINISHED)

        a0 = self.h.photo("a0.jpg")
        self.assertAlmostEqual(a0["gps_lat"], 25.03, places=3)
        self.assertEqual((a0["country_code"], a0["city"]), ("TW", "City25"))
        self.assertTrue(ProcessStatus(a0["process_status"]) & ProcessStatus.GEOCODED)
        self.assertEqual(self.h.geocoder.calls, 1, "地理編碼應該整批查一次")

        events = self.h.db.query("SELECT * FROM events ORDER BY start_ts")
        self.assertEqual([e["photo_count"] for e in events], [4, 4])
        self.assertEqual(events[0]["location_label"], "City25, TW")
        self.assertIsNotNone(events[0]["cover_photo_id"])
        self.assertIsNone(self.h.photo("lonely.jpg")["event_id"])
        self.assertEqual(self.h.photo("b3.jpg")["event_id"], events[1]["id"])

        # 使用者改過的事件，重新聚類不可覆蓋
        self.h.db.execute("UPDATE events SET name = '台北行', is_user_edited = 1 WHERE id = ?", (events[0]["id"],)).result()
        make_image(self.lib / "b9.jpg", color=RED, seed=77, taken="2024:05:02 15:40:00", gps=(22.63, 120.30))
        self.h.run()
        events2 = self.h.db.query("SELECT * FROM events ORDER BY start_ts")
        self.assertEqual([(e["name"], e["photo_count"]) for e in events2], [("台北行", 4), (None, 5)])
        self.assertEqual(self.h.photo("a0.jpg")["event_id"], events[0]["id"])
        steps = [(e.data["name"], e.data["phase"]) for e in self.h.events if e.kind is EventKind.POST_STEP]
        self.assertEqual(steps, [("geocode", "start"), ("geocode", "done"), ("events", "start"), ("events", "done")])

    def test_meta_only_run_needs_no_models_then_ai_stages_resume(self) -> None:
        self._seed_basic(5)
        h = self.h
        h.cfg.pipeline.stages = ["meta"]
        engine = h.run()
        self.assertIs(engine.state, EngineState.FINISHED)
        self.assertIsNone(h.vision.loaded_on)
        self.assertEqual(h.count("photos", f"process_status = {int(ProcessStatus.stage1())}"), 5)
        self.assertEqual(h.count("embeddings"), 0)

        # 之後開啟 AI 階段：只補 Stage 2/3，不重做 Stage 1
        h.cfg.pipeline.stages = ["meta", "siglip", "faces"]
        engine2 = h.run()
        self.assertEqual(engine2.stats().to_process, 5)
        self.assertEqual(h.count("embeddings"), 5)
        self.assertEqual(h.count("photos", f"process_status & {int(FULL)} = {int(FULL)}"), 5)

    def test_resume_of_faces_only_uses_stored_tags_for_gating(self) -> None:
        self._seed_basic(6)
        h = self.h
        h.cfg.pipeline.stages = ["meta", "siglip"]
        h.run()
        self.assertEqual(h.count("faces"), 0)
        h.cfg.pipeline.stages = ["meta", "siglip", "faces"]
        batches = len(h.vision.batch_sizes)
        h.run()
        self.assertEqual(len(h.vision.batch_sizes), batches, "Stage 2 已完成，不應重跑 vision")
        self.assertEqual(h.count("faces"), 4)                # 6 張裡 2 張 green 被略過
        self.assertEqual(h.count("photos", f"process_status & {int(ProcessStatus.FACES_SKIPPED)} != 0"), 2)


class GateRuleTest(unittest.TestCase):
    def test_should_run_faces(self) -> None:
        self.assertTrue(should_run_faces(False, []))
        self.assertFalse(should_run_faces(True, []))
        self.assertFalse(should_run_faces(False, [GateRole.SKIP_FACE]))
        self.assertTrue(should_run_faces(True, [GateRole.SKIP_FACE, GateRole.HAS_PERSON]))
        self.assertTrue(should_run_faces(False, [GateRole.NONE]))


if __name__ == "__main__":
    unittest.main()
