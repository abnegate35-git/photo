"""不需要執行緒的單元測試：設定、影像運算、前處理、向量索引、中繼資料、事件、硬體規劃。"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from photosorter.core.config import AppConfig, ConfigError, dump_default_config, load_config
from photosorter.core.events import Event, EventBus, EventKind, EventQueue
from photosorter.core.hardware import CPU, CUDA, DML, plan_runtime, provider_candidates
from photosorter.core.paths import project_root
from photosorter.core.status import ProcessStatus, required_mask
from photosorter.core.types import EventInput, GateRole, HardwareInfo, PhotoMeta, RawFile
from photosorter.db.db_manager import DBManager
from photosorter.db.repository import Repository, blob_to_vec, vec_to_blob
from photosorter.db.vector_index import InMemoryVectorIndex
from photosorter.models.preprocess import (
    ARCFACE_SPEC,
    SCRFD_SPEC,
    SIGLIP_SPEC,
    l2_normalize,
    sigmoid,
    to_model_input,
)
from photosorter.models.siglip import DEFAULT_TAGS, SiglipTagger, rescore_library, seed_default_tags, siglip_probabilities
from photosorter.pipeline.decoder import apply_orientation
from photosorter.pipeline.event_clustering import TimeGapEventClusterer
from photosorter.pipeline.metadata import (
    BasicMetadataExtractor,
    ScreenshotDetector,
    parse_exif_datetime,
    parse_filename_datetime,
)
from photosorter.ui.bridge import TkEventPump
from photosorter.utils import imageops
from photosorter.utils.geo import haversine_km

from .fakes import EMBED_DIM, FakeTextEncoder, make_config, make_image

ROOT = project_root()


class ConfigTest(unittest.TestCase):
    def test_example_file_matches_code_defaults(self) -> None:
        example = (ROOT / "config.example.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(example), json.loads(dump_default_config()),
                         "config.example.json 過期了：請執行 python -m photosorter dump-config -o config.example.json")

    def test_paths_resolve_against_root_not_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(root=Path(d))
            root = Path(d).resolve()
            self.assertEqual(cfg.db_path, root / "data" / "photos.db")
            self.assertEqual(cfg.weight("siglip/vision_model.onnx"), root / "weights" / "siglip" / "vision_model.onnx")
            self.assertTrue(cfg.db_path.is_absolute())

    def test_local_config_overrides_only_given_keys(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(
                json.dumps({"runtime": {"batch_size": 32}, "paths": {"db_path": "elsewhere/x.db"}}), encoding="utf-8")
            cfg = load_config(root=Path(d))
            self.assertEqual(cfg.runtime.batch_size, 32)
            self.assertEqual(cfg.runtime.queue_maxsize, 64)
            self.assertEqual(cfg.db_path, Path(d).resolve() / "elsewhere" / "x.db")

    def test_unknown_or_mistyped_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for bad in ({"runtme": {}}, {"runtime": {"batch_sizes": 1}}, {"runtime": {"batch_size": "8"}},
                        {"runtime": {"warmup": 1}}, {"scan": {"extensions": "jpg"}}):
                with self.assertRaises(ConfigError, msg=str(bad)):
                    load_config(root=Path(d), overrides=bad)

    def test_defaults_carry_no_machine_specific_values(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.runtime.providers, "auto")
        self.assertEqual((cfg.runtime.decode_workers, cfg.runtime.intra_op_threads, cfg.runtime.batch_size), (0, 0, 0))
        for value in vars(cfg.paths).values():
            self.assertFalse(Path(value).is_absolute(), value)

    def test_source_has_no_hardcoded_absolute_paths(self) -> None:
        pattern = re.compile(r"""["'][A-Za-z]:[\\/]|/home/|/Users/|\\\\Users\\\\""")
        offenders = []
        for py in (ROOT / "photosorter").rglob("*.py"):
            for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{py.relative_to(ROOT)}:{n}: {line.strip()}")
        self.assertEqual(offenders, [])


class StatusTest(unittest.TestCase):
    def test_bit_values_are_frozen(self) -> None:
        """資料庫裡存的是整數：這些值一旦發布就不可更動。要新增階段請往高位加。"""
        self.assertEqual(
            {s.name: int(s) for s in ProcessStatus if s.name and s.value and (s.value & (s.value - 1)) == 0},
            {"HASHED": 1, "META": 2, "QUALITY": 4, "EMBEDDED": 8, "TAGGED": 16, "FACES": 32,
             "FACES_SKIPPED": 64, "GEOCODED": 128, "ERR_DECODE": 65536, "ERR_INFER": 131072},
        )

    def test_required_mask(self) -> None:
        self.assertEqual(required_mask([]), ProcessStatus.stage1())
        self.assertEqual(int(required_mask(["meta", "siglip", "faces"])), 63)
        with self.assertRaises(ValueError):
            required_mask(["nope"])


class ImageOpsTest(unittest.TestCase):
    def _photo(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        base = rng.integers(0, 255, size=(30, 40, 3), dtype=np.uint8)
        return np.asarray(Image.fromarray(base).resize((400, 300), Image.Resampling.BICUBIC))

    def test_orientation_matches_pillow(self) -> None:
        img = self._photo()
        for o in range(1, 9):
            pil = Image.fromarray(img)
            exif = Image.Exif()
            exif[0x0112] = o
            pil.info["exif"] = exif.tobytes()
            pil.getexif()[0x0112] = o
            expected = np.asarray(ImageOps.exif_transpose(pil))
            np.testing.assert_array_equal(apply_orientation(img, o), expected, err_msg=f"orientation {o}")

    def test_dhash_is_stable_under_resize_and_distinguishes_images(self) -> None:
        img = self._photo()
        gray = imageops.to_gray(img)
        h1 = imageops.dhash64(gray)
        h2 = imageops.dhash64(imageops.to_gray(imageops.resize(img, 200, 150)))
        other = imageops.dhash64(imageops.to_gray(np.ascontiguousarray(img[::-1, ::-1])))
        self.assertTrue(-(1 << 63) <= h1 < (1 << 63))
        self.assertLessEqual(imageops.hamming64(h1, h2), 6)
        self.assertGreater(imageops.hamming64(h1, other), 12)
        self.assertEqual(imageops.hamming64(-1, 0), 64)

    def test_blur_score_drops_when_blurred(self) -> None:
        rng = np.random.default_rng(2)
        sharp = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
        blurred = imageops.resize(imageops.resize(sharp, 25, 25), 200, 200)
        self.assertGreater(imageops.laplacian_variance(sharp), 10 * imageops.laplacian_variance(blurred))

    def test_letterbox_geometry(self) -> None:
        img = np.full((300, 600, 3), 200, dtype=np.uint8)
        canvas, lb = imageops.letterbox_square(img, 640)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertEqual((lb.content_w, lb.content_h), (640, 320))
        self.assertAlmostEqual(lb.scale, 640 / 600)
        self.assertTrue((canvas[:320] == 200).all() and (canvas[320:] == 0).all())

    def test_resize_long_edge_never_upscales(self) -> None:
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        self.assertIs(imageops.resize_long_edge(img, 1280), img)
        self.assertEqual(imageops.resize_long_edge(img, 50).shape[:2], (50, 25))


class PreprocessTest(unittest.TestCase):
    def test_to_model_input_layout_and_values(self) -> None:
        batch = np.zeros((2, 4, 5, 3), dtype=np.uint8)
        batch[0, :, :, 0] = 255                                  # 第一張全紅
        batch[1] = 127
        x = to_model_input(batch, SIGLIP_SPEC)
        self.assertEqual((x.shape, x.dtype), ((2, 3, 4, 5), np.float32))
        self.assertTrue(x.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(x[0, 0], 1.0, atol=1e-6)      # (255/255 - 0.5) / 0.5
        np.testing.assert_allclose(x[0, 1], -1.0, atol=1e-6)
        np.testing.assert_allclose(to_model_input(batch, SCRFD_SPEC)[1], (127 - 127.5) / 128, atol=1e-6)
        np.testing.assert_allclose(to_model_input(batch, ARCFACE_SPEC)[0, 0], 1.0, atol=1e-6)
        with self.assertRaises(ValueError):
            to_model_input(batch.astype(np.float32), SIGLIP_SPEC)

    def test_l2_normalize_and_sigmoid(self) -> None:
        v = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(v, [[0.6, 0.8], [0.0, 0.0]], atol=1e-6)
        s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(s, [0.0, 0.5, 1.0], atol=1e-6)
        self.assertTrue(np.isfinite(s).all())

    def test_siglip_probabilities_are_independent_per_label(self) -> None:
        img = l2_normalize(np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
        txt = l2_normalize(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        p = siglip_probabilities(img, txt, logit_scale=100.0, logit_bias=-10.0)
        self.assertEqual(p.shape, (2, 3))
        self.assertTrue(p[0, 0] > 0.99 and p[0, 1] < 0.01)
        self.assertTrue(p[1, 0] > 0.99 and p[1, 1] > 0.99 and p[1, 2] < 0.01)   # 不是 softmax：兩個標籤可同時成立


class VectorIndexTest(unittest.TestCase):
    def test_add_search_update_remove(self) -> None:
        idx = InMemoryVectorIndex(dim=4, capacity=2)
        vecs = l2_normalize(np.eye(4, dtype=np.float32))
        idx.add([10, 20, 30, 40], vecs)                          # 觸發擴容
        self.assertEqual(len(idx), 4)
        self.assertEqual(idx.search(vecs[2], top_k=1), [(30, 1.0)])
        q = l2_normalize(np.array([1.0, 0.9, 0.0, 0.0]))
        self.assertEqual([i for i, _ in idx.search(q, top_k=2)], [10, 20])
        self.assertEqual(idx.search(q, top_k=4, min_score=0.5)[-1][0], 20)

        idx.add([20], vecs[3:4])                                 # 覆寫
        self.assertEqual(len(idx), 4)
        np.testing.assert_array_equal(idx.get(20), vecs[3])
        idx.remove([10, 999])
        self.assertEqual(len(idx), 3)
        self.assertIsNone(idx.get(10))
        self.assertEqual({i for i, _ in idx.search(vecs[3], top_k=10)}, {20, 30, 40})
        with self.assertRaises(ValueError):
            idx.search(np.zeros(3))

    def test_from_arrays_and_blob_roundtrip(self) -> None:
        mat = l2_normalize(np.random.default_rng(0).normal(size=(5, 8)))
        idx = InMemoryVectorIndex.from_arrays(np.arange(5) + 100, mat)
        self.assertEqual(idx.search(mat[3], top_k=1)[0][0], 103)
        blob = vec_to_blob(mat[0])
        self.assertEqual(len(blob), 8 * 4)
        np.testing.assert_array_equal(blob_to_vec(blob, 8), mat[0])
        with self.assertRaises(ValueError):
            blob_to_vec(blob, 7)


class MetadataTest(unittest.TestCase):
    def test_exif_fields(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = make_image(Path(d) / "x.jpg", taken="2023:12:31 23:59:58", gps=(-33.8688, 151.2093), orientation=3)
            data = p.read_bytes()
            import io
            meta = BasicMetadataExtractor().extract(RawFile(str(p), data, b""), Image.open(io.BytesIO(data)))
        self.assertEqual((meta.taken_at, meta.taken_src), ("2023-12-31T23:59:58", "exif"))
        self.assertEqual(meta.taken_ts, 1704067198)
        self.assertEqual((meta.camera_make, meta.camera_model, meta.orientation), ("TestMake", "TestCam X1", 3))
        self.assertAlmostEqual(meta.gps_lat, -33.8688, places=3)
        self.assertAlmostEqual(meta.gps_lon, 151.2093, places=3)

    def test_datetime_parsers(self) -> None:
        self.assertIsNone(parse_exif_datetime("0000:00:00 00:00:00"))
        self.assertIsNone(parse_exif_datetime(None))
        self.assertEqual(parse_exif_datetime("2024:02:29 08:00:00\x00").day, 29)
        f = parse_filename_datetime
        self.assertEqual(f("IMG_20230815_142355.jpg").isoformat(), "2023-08-15T14:23:55")
        self.assertEqual(f("2023-08-15 14.23.55.png").isoformat(), "2023-08-15T14:23:55")
        self.assertEqual(f("IMG-20230815-WA0001.jpg").isoformat(), "2023-08-15T00:00:00")
        self.assertIsNone(f("DSC01234.JPG"))
        self.assertIsNone(f("20231345_000000.jpg"))

    def test_screenshot_detector_is_conservative(self) -> None:
        det = ScreenshotDetector(AppConfig().screenshot)
        no_cam, cam = PhotoMeta(), PhotoMeta(camera_model="Pixel 8")
        self.assertTrue(det.is_screenshot("D:/x/Screenshot_1.png", 800, 600, cam))
        self.assertTrue(det.is_screenshot("D:/x/螢幕擷取畫面 2024.png", 800, 600, no_cam))
        self.assertTrue(det.is_screenshot("D:/x/a.png", 1080, 2400, no_cam))
        self.assertTrue(det.is_screenshot("D:/x/a.png", 1920, 1080, no_cam))
        self.assertFalse(det.is_screenshot("D:/x/a.jpg", 1920, 1080, cam))       # 相機拍的 16:9
        self.assertFalse(det.is_screenshot("D:/x/a.jpg", 4000, 2250, no_cam))    # 只是比例像，不算


class EventClusteringTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = AppConfig().events
        cfg.gap_minutes, cfg.max_distance_km, cfg.min_photos = 60, 20.0, 2
        self.c = TimeGapEventClusterer(cfg)

    def test_splits_on_time_gap_and_filters_small_groups(self) -> None:
        items = [EventInput(i, t) for i, t in enumerate([0, 600, 1200, 9000, 9100, 50000])]
        clusters = self.c.cluster(list(reversed(items)))         # 輸入不需排序
        self.assertEqual([c.photo_ids for c in clusters], [[0, 1, 2], [3, 4]])
        self.assertEqual((clusters[0].start_ts, clusters[0].end_ts), (0, 1200))

    def test_splits_on_distance_even_without_time_gap(self) -> None:
        taipei, kaohsiung = (25.03, 121.56), (22.63, 120.30)
        items = [EventInput(1, 0, *taipei), EventInput(2, 60, *taipei), EventInput(3, 120),
                 EventInput(4, 180, *kaohsiung), EventInput(5, 240, *kaohsiung)]
        clusters = self.c.cluster(items)
        self.assertEqual([c.photo_ids for c in clusters], [[1, 2, 3], [4, 5]])   # 沒 GPS 的 3 只看時間
        self.assertAlmostEqual(clusters[0].center_lat, 25.03, places=4)
        self.assertEqual(self.c.cluster([]), [])

    def test_haversine(self) -> None:
        self.assertAlmostEqual(haversine_km(25.03, 121.56, 22.63, 120.30), 296, delta=5)
        self.assertEqual(haversine_km(10, 10, 10, 10), 0.0)


class HardwareTest(unittest.TestCase):
    def test_provider_chain(self) -> None:
        self.assertEqual(provider_candidates("auto", [DML, CPU]), [DML, CPU])
        self.assertEqual(provider_candidates("auto", [CPU, DML, CUDA, "TensorrtExecutionProvider"]), [CUDA, DML, CPU])
        self.assertEqual(provider_candidates([CUDA], [DML, CPU]), [CPU])           # 指定的不可用 -> 仍有 CPU 墊底
        self.assertEqual(provider_candidates([CPU], [CUDA, CPU]), [CPU])           # 明確指定 CPU 就不碰 GPU
        self.assertEqual(provider_candidates("auto", []), [])

    def test_plan_runtime_scales_with_machine(self) -> None:
        rt = AppConfig().runtime
        small = HardwareInfo(4, 2, 8.0, "x", [CPU])
        big = HardwareInfo(32, 16, 64.0, "x", [CUDA, CPU])
        self.assertEqual(plan_runtime(rt, big, CUDA).batch_size, 16)
        self.assertEqual(plan_runtime(rt, big, DML).batch_size, 8)
        cpu_plan = plan_runtime(rt, small, CPU)
        self.assertEqual((cpu_plan.batch_size, cpu_plan.decode_workers, cpu_plan.intra_op_threads), (4, 1, 1))
        # CPU 模式：解碼與推論的執行緒加起來不超過實體核心數
        p = plan_runtime(rt, big, CPU)
        self.assertLessEqual(p.decode_workers + p.intra_op_threads, 16)
        rt.decode_workers, rt.batch_size = 3, 5                                    # 使用者明確指定則照用
        p = plan_runtime(rt, big, CUDA)
        self.assertEqual((p.decode_workers, p.batch_size), (3, 5))


class EventBusTest(unittest.TestCase):
    def test_queue_drops_progress_but_never_critical_events(self) -> None:
        q = EventQueue(maxsize=3)
        for i in range(10):
            q.put(Event(EventKind.PROGRESS, {"i": i}))
        q.put(Event(EventKind.FINISHED, {}))
        q.put(Event(EventKind.ERROR, {}))
        self.assertEqual(q.dropped, 7)
        kinds = [e.kind for e in q.drain()]
        self.assertEqual(kinds, [EventKind.PROGRESS] * 3 + [EventKind.FINISHED, EventKind.ERROR])
        self.assertEqual(len(q), 0)

    def test_subscriber_errors_are_contained(self) -> None:
        bus, seen = EventBus(), []
        bus.subscribe(lambda e: 1 / 0)
        unsubscribe = bus.subscribe(seen.append, kinds=[EventKind.STATE])
        with self.assertLogs("photosorter.core.events", level="ERROR"):
            bus.publish(EventKind.STATE, state="running")
            bus.publish(EventKind.LOG, message="filtered out")
        self.assertEqual([e.kind for e in seen], [EventKind.STATE])
        unsubscribe()
        with self.assertLogs("photosorter.core.events", level="ERROR"):
            bus.publish(EventKind.STATE, state="x")
        self.assertEqual(len(seen), 1)

    def test_tk_event_pump_delivers_on_the_ui_tick(self) -> None:
        class FakeWidget:
            def __init__(self) -> None:
                self.scheduled: list = []

            def after(self, ms: int, fn: object) -> int:
                self.scheduled.append(fn)
                return len(self.scheduled)

            def after_cancel(self, _id: int) -> None:
                self.scheduled.clear()

        bus, widget, got = EventBus(), FakeWidget(), []
        pump = TkEventPump(widget, bus, got.append)
        pump.start()
        bus.publish(EventKind.PROGRESS, processed=1)
        self.assertEqual(got, [], "事件不可以在發布的執行緒上直接送進 UI")
        widget.scheduled.pop()()                                 # 模擬 Tk 主迴圈觸發計時器
        self.assertEqual([e.kind for e in got], [EventKind.PROGRESS])
        pump.stop()
        bus.publish(EventKind.PROGRESS, processed=2)
        self.assertEqual(len(got), 1)


class RetagTest(unittest.TestCase):
    """改標籤定義後不必重跑 vision encoder：用已存的影像向量重新判定。"""

    def test_rescore_library_respects_manual_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(Path(d))
            with DBManager(cfg.db_path, cfg.db) as db:
                repo = Repository(db)
                red = repo.upsert_tag("red", "a red photo")
                green = repo.upsert_tag("green", "a green photo")
                model_id = repo.ensure_model(cfg.siglip.model_name, "image_embed", EMBED_DIM)

                def seed(conn) -> None:
                    for pid, axis in ((1, 0), (2, 1)):
                        conn.execute("INSERT INTO photos(id, path, dir_path, file_name, ext, file_size, mtime_ns) "
                                     "VALUES (?, ?, 'd', 'f', 'jpg', 1, 1)", (pid, f"p{pid}"))
                        v = np.zeros(EMBED_DIM, dtype=np.float32)
                        v[axis] = 1.0
                        conn.execute("INSERT INTO embeddings(photo_id, model_id, dim, vector) VALUES (?, ?, ?, ?)",
                                     (pid, model_id, EMBED_DIM, vec_to_blob(v)))
                    # 使用者否決了照片 1 的 red，並手動幫照片 2 加上 red
                    conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score, source, is_rejected) "
                                 "VALUES (1, ?, 0.9, 'auto', 1)", (red,))
                    conn.execute("INSERT INTO photo_tags(photo_id, tag_id, source) VALUES (2, ?, 'manual')", (red,))

                db.write(seed)
                text = FakeTextEncoder()
                tagger = SiglipTagger(cfg, repo, text)
                tagger.load()
                self.assertEqual(rescore_library(tagger, repo), 2)
                rows = {(r["photo_id"], r["tag_id"]): (r["source"], r["is_rejected"])
                        for r in db.query("SELECT * FROM photo_tags")}
                self.assertEqual(rows, {(1, red): ("auto", 1), (2, red): ("manual", 0), (2, green): ("auto", 0)})

                # 新增一個標籤：只編碼新的那一個，再重標一次
                repo.upsert_tag("blue", "a blue photo")
                tagger2 = SiglipTagger(cfg, repo, text)
                tagger2.load()
                self.assertEqual(text.calls, 2)
                self.assertEqual(len(tagger2.tags), 3)
                self.assertEqual(tagger2.gate_role(red), GateRole.NONE)

    def test_default_tags_are_seeded_once(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(Path(d))
            with DBManager(cfg.db_path, cfg.db) as db:
                repo = Repository(db)
                self.assertEqual(seed_default_tags(repo, cfg.siglip.prompt_template), len(DEFAULT_TAGS))
                self.assertEqual(seed_default_tags(repo, cfg.siglip.prompt_template), 0)
                self.assertEqual(db.query_value("SELECT prompt FROM tags WHERE name = 'food'"), "a photo of food.")
                roles = {r["gate_role"] for r in db.query("SELECT gate_role FROM tags")}
                self.assertEqual(roles, {"none", "skip_face", "has_person"})


if __name__ == "__main__":
    unittest.main()
