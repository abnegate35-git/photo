"""ExportManager：dry-run 不動檔案、四種模式、衝突解決、絕不覆寫、rollback、當機收尾。"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from photosorter.core.events import EventBus, EventKind
from photosorter.core.types import ConflictPolicy, ExportMode
from photosorter.db.db_manager import DBManager
from photosorter.db.repository import Repository
from photosorter.pipeline.decoder import DefaultImageDecoder
from photosorter.pipeline.export_manager import (
    _PART_SUFFIX,
    ExportError,
    ExportManager,
    can_symlink,
    sanitize_component,
)
from photosorter.pipeline.pipeline_engine import PipelineEngine

from .fakes import make_config, make_image


def tree(root: Path) -> set[str]:
    """資料夾底下所有檔案的相對路徑。"""
    if not root.exists():
        return set()
    return {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file() or p.is_symlink()}


class ExportManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # realpath：暫存資料夾可能是 8.3 短路徑，而引擎會把路徑正規化成長路徑
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.lib = self.tmp / "library"
        self.dest = self.tmp / "sorted"
        self.cfg = make_config(self.tmp, pipeline={"stages": ["meta"], "geocode": False, "cluster_events": False})
        self.db = DBManager(self.cfg.db_path, self.cfg.db).open()
        self.repo = Repository(self.db)
        self.bus = EventBus()
        self.exporter = ExportManager(self.cfg, self.db, self.repo, self.bus)

        make_image(self.lib / "a" / "one.jpg", seed=1, taken="2023:03:05 08:00:00")
        make_image(self.lib / "b" / "one.jpg", seed=2, taken="2023:03:09 09:00:00")      # 同名、不同內容
        make_image(self.lib / "c" / "two.jpg", seed=3, taken="2024:11:20 10:00:00")
        engine = PipelineEngine(self.cfg, self.db, self.repo, self.bus, decoder=DefaultImageDecoder(self.cfg))
        engine.start([self.lib])
        assert engine.join(30)
        self.before = tree(self.lib)

    def tearDown(self) -> None:
        self.db.close()
        self._tmp.cleanup()

    def _logs(self, batch_id: int) -> list[tuple[str, str]]:
        return [(r["action"], r["state"]) for r in self.db.query(
            "SELECT action, state FROM export_logs WHERE batch_id = ? ORDER BY seq", (batch_id,))]

    # ------------------------------------------------------------------ #
    def test_plan_is_a_pure_dry_run(self) -> None:
        plan = self.exporter.plan(self.dest, mode="copy", template="{year}/{year}-{month}/{filename}")
        self.assertFalse(self.dest.exists(), "dry-run 不可以建立任何資料夾")
        self.assertEqual(self.db.query_value("SELECT COUNT(*) FROM export_batches"), 0)
        self.assertEqual(tree(self.lib), self.before)
        rel = sorted((os.path.relpath(op.dst, self.dest).replace("\\", "/"), op.note) for op in plan.ops)
        self.assertEqual(rel, [("2023/2023-03/one.jpg", ""), ("2023/2023-03/one_1.jpg", "renamed"),
                               ("2024/2024-11/two.jpg", "")])
        self.assertEqual(plan.summary(), {"copy": 2, "copy:renamed": 1})

    def test_unknown_template_field_is_reported(self) -> None:
        with self.assertRaises(ExportError) as ctx:
            self.exporter.plan(self.dest, template="{yaer}/{filename}")
        self.assertIn("yaer", str(ctx.exception))

    def test_copy_then_rollback_restores_everything(self) -> None:
        events = []
        self.bus.subscribe(events.append)
        plan = self.exporter.plan(self.dest, mode=ExportMode.COPY, template="{year}/{filename}")
        result = self.exporter.execute(plan)
        self.assertEqual((result.done, result.failed, result.skipped), (3, 0, 0))
        self.assertEqual(tree(self.dest), {"2023/one.jpg", "2023/one_1.jpg", "2024/two.jpg"})
        self.assertEqual(tree(self.lib), self.before, "copy 不可以動到原始檔")
        self.assertEqual(self._logs(result.batch_id), [("copy", "done")] * 3)
        self.assertEqual(sum(1 for e in events if e.kind is EventKind.EXPORT_PROGRESS), 3)
        self.assertIs(events[-1].kind, EventKind.EXPORT_DONE)
        self.assertFalse(any(p.endswith(_PART_SUFFIX) for p in tree(self.dest)))

        back = self.exporter.rollback(result.batch_id)
        self.assertEqual((back.done, back.failed), (3, 0))
        self.assertEqual(tree(self.dest), set())
        self.assertFalse((self.dest / "2023").exists(), "回復後空資料夾要清掉")
        self.assertEqual(self.db.query_value("SELECT state FROM export_batches"), "rolled_back")

    def test_move_updates_db_path_and_rollback_puts_files_back(self) -> None:
        plan = self.exporter.plan(self.dest, mode="move", template="{date}/{filename}")
        result = self.exporter.execute(plan)
        self.assertEqual(result.done, 3)
        self.assertEqual(tree(self.lib), set())
        moved = {r["path"] for r in self.db.query("SELECT path FROM photos")}
        self.assertEqual(moved, {str(self.dest / "2023-03-05" / "one.jpg"), str(self.dest / "2023-03-09" / "one.jpg"),
                                 str(self.dest / "2024-11-20" / "two.jpg")})
        self.assertTrue(all(os.path.isfile(p) for p in moved))

        back = self.exporter.rollback(result.batch_id)
        self.assertEqual((back.done, back.failed), (3, 0))
        self.assertEqual(tree(self.lib), self.before)
        # 資料庫裡的路徑是掃描時正規化過的（8.3 短檔名會被還原成長檔名），所以用 realpath 比
        self.assertEqual(
            {r["path"] for r in self.db.query("SELECT path FROM photos")},
            {os.path.realpath(self.lib / d / n) for d, n in (("a", "one.jpg"), ("b", "one.jpg"), ("c", "two.jpg"))},
        )

    def test_hardlink_shares_data_and_rollback_keeps_original(self) -> None:
        plan = self.exporter.plan(self.dest, mode="hardlink", template="{filename}")
        result = self.exporter.execute(plan)
        self.assertEqual(result.done, 3)
        self.assertTrue(os.path.samefile(self.lib / "c" / "two.jpg", self.dest / "two.jpg"))
        self.exporter.rollback(result.batch_id)
        self.assertEqual(tree(self.dest), set())
        self.assertEqual(tree(self.lib), self.before)

    @unittest.skipUnless(can_symlink(), "這台電腦沒有建立符號連結的權限（需開發人員模式或系統管理員）")
    def test_symlink_mode(self) -> None:
        plan = self.exporter.plan(self.dest, mode="symlink", template="{filename}")
        self.assertEqual(plan.warnings, [])
        result = self.exporter.execute(plan)
        self.assertEqual(result.done, 3)
        self.assertTrue(os.path.islink(self.dest / "two.jpg"))
        self.exporter.rollback(result.batch_id)
        self.assertEqual(tree(self.dest), set())

    def test_symlink_without_privilege_warns_in_the_plan(self) -> None:
        if can_symlink():
            self.skipTest("這台電腦有權限")
        plan = self.exporter.plan(self.dest, mode="symlink", template="{filename}")
        self.assertTrue(any("符號連結" in w for w in plan.warnings))

    # ------------------------------------------------------------------ #
    def test_conflict_policies_against_existing_files(self) -> None:
        template = "{filename}"
        first = self.exporter.execute(self.exporter.plan(self.dest, template=template))
        self.assertEqual(first.done, 3)
        snapshot = {p: (self.dest / p).read_bytes() for p in tree(self.dest)}

        # skip_identical（預設）：內容相同的略過，重跑是冪等的
        plan = self.exporter.plan(self.dest, template=template)
        notes = sorted(op.note for op in plan.ops)
        self.assertEqual(notes, ["identical", "identical", "renamed"])       # b/one.jpg 與已存在的 one.jpg 內容不同
        # skip：存在就略過
        plan = self.exporter.plan(self.dest, template=template, conflict_policy=ConflictPolicy.SKIP)
        self.assertEqual({op.action for op in plan.ops}, {"skip"})
        # rename：一律改名，不覆寫
        plan = self.exporter.plan(self.dest, template=template, conflict_policy="rename")
        self.assertEqual({op.note for op in plan.ops}, {"renamed"})
        result = self.exporter.execute(plan)
        self.assertEqual(result.done, 3)
        for p, data in snapshot.items():
            self.assertEqual((self.dest / p).read_bytes(), data, f"{p} 被覆寫了")
        self.assertEqual(len(tree(self.dest)), 6)

    def test_never_overwrites_a_file_that_appeared_after_planning(self) -> None:
        plan = self.exporter.plan(self.dest, template="{filename}")
        victim = self.dest / "two.jpg"
        victim.parent.mkdir(parents=True)
        victim.write_bytes(b"user data that must survive")
        result = self.exporter.execute(plan)
        self.assertEqual((result.done, result.failed), (2, 1))
        self.assertEqual(victim.read_bytes(), b"user data that must survive")
        self.assertIn(("copy", "failed"), self._logs(result.batch_id))
        # 回復只處理成功的那幾筆，不會去刪使用者的檔案
        self.exporter.rollback(result.batch_id)
        self.assertEqual(tree(self.dest), {"two.jpg"})

    def test_rollback_refuses_to_delete_modified_target(self) -> None:
        result = self.exporter.execute(self.exporter.plan(self.dest, template="{filename}"))
        edited = self.dest / "two.jpg"
        edited.write_bytes(b"edited by the user after export")
        back = self.exporter.rollback(result.batch_id)
        self.assertEqual((back.done, back.failed), (2, 1))
        self.assertTrue(edited.exists())
        self.assertEqual(self.db.query_value("SELECT state FROM export_batches"), "partial_rollback")
        self.assertEqual(self.db.query_value("SELECT COUNT(*) FROM export_logs WHERE state = 'rollback_failed'"), 1)

    def test_cancel_midway_and_missing_source(self) -> None:
        (self.lib / "c" / "two.jpg").unlink()
        plan = self.exporter.plan(self.dest, template="{filename}")
        self.assertEqual(sum(1 for op in plan.ops if op.note == "missing_source"), 1)
        calls = iter([False, True])
        result = self.exporter.execute(plan, should_stop=lambda: next(calls))
        self.assertTrue(result.cancelled)
        self.assertEqual((result.done, result.skipped), (1, 1))
        self.assertEqual(self.db.query_value("SELECT state FROM export_batches"), "cancelled")

    def test_recover_resolves_interrupted_operations(self) -> None:
        """模擬當機：日誌寫了 pending、檔案操作做了一半。"""
        src = str(self.lib / "c" / "two.jpg")
        done_dst, lost_dst = str(self.dest / "done.jpg"), str(self.dest / "lost.jpg")
        self.dest.mkdir()
        Path(done_dst).write_bytes(Path(src).read_bytes())                    # 其實已經複製完
        Path(lost_dst + _PART_SUFFIX).write_bytes(b"half written")            # 複製到一半

        def seed(conn) -> None:
            b = conn.execute("INSERT INTO export_batches(mode, dest_root, template, conflict_policy, total) "
                             "VALUES ('copy', ?, 't', 'rename', 2)", (str(self.dest),)).lastrowid
            for seq, dst in enumerate((done_dst, lost_dst)):
                conn.execute("INSERT INTO export_logs(batch_id, seq, action, src_path, dst_path) "
                             "VALUES (?, ?, 'copy', ?, ?)", (b, seq, src, dst))

        self.db.write(seed)
        self.assertEqual(self.exporter.recover(), 2)
        states = [r["state"] for r in self.db.query("SELECT state FROM export_logs ORDER BY seq")]
        self.assertEqual(states, ["done", "failed"])
        self.assertFalse(os.path.exists(lost_dst + _PART_SUFFIX))
        self.assertEqual(self.db.query_value("SELECT state FROM export_batches"), "cancelled")
        self.assertEqual(self.exporter.recover(), 0)

    def test_sanitize_component(self) -> None:
        self.assertEqual(sanitize_component('a/b\\c:d*e?"f<g>h|i'), "a_b_c_d_e__f_g_h_i")
        self.assertEqual(sanitize_component("  trailing dots... "), "trailing dots")
        self.assertEqual(sanitize_component(".."), "_")
        self.assertEqual(sanitize_component(""), "_")
        self.assertEqual(sanitize_component("CON"), "_CON")
        self.assertEqual(sanitize_component("nul.txt"), "_nul.txt")
        self.assertEqual(sanitize_component("東京 2024"), "東京 2024")

    def test_template_values_cannot_escape_dest_root(self) -> None:
        self.db.execute("UPDATE photos SET camera_model = '..\\..\\evil'").result()
        plan = self.exporter.plan(self.dest, template="{camera}/{filename}")
        root = os.path.normcase(str(self.dest)) + os.sep
        for op in plan.ops:
            self.assertTrue(os.path.normcase(os.path.normpath(op.dst)).startswith(root), op.dst)


if __name__ == "__main__":
    unittest.main()
