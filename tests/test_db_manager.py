"""DBManager：單一寫入者、group commit、savepoint 隔離、唯讀連線、生命週期。"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from photosorter.core.config import DbConfig
from photosorter.db.db_manager import (
    APPLICATION_ID,
    SCHEMA_VERSION,
    DatabaseClosedError,
    DatabaseError,
    DBManager,
)

EXPECTED_TABLES = {
    "photos", "embeddings", "tags", "photo_tags", "faces", "persons", "events",
    "export_batches", "export_logs", "models", "scan_sessions", "kv",
}


class DBManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sub" / "test.db"
        self.db = DBManager(self.path, DbConfig()).open()

    def tearDown(self) -> None:
        self.db.close()
        self._tmp.cleanup()

    def test_schema_and_pragmas(self) -> None:
        tables = {r[0] for r in self.db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertTrue(EXPECTED_TABLES <= tables, EXPECTED_TABLES - tables)
        self.assertEqual(self.db.query_value("PRAGMA user_version"), SCHEMA_VERSION)
        self.assertEqual(self.db.query_value("PRAGMA application_id"), APPLICATION_ID)
        self.assertEqual(self.db.query_value("PRAGMA journal_mode"), "wal")
        self.assertEqual(self.db.query_value("PRAGMA foreign_keys"), 1)
        # writer 連線的設定要從 writer 執行緒看
        self.assertEqual(self.db.write(lambda c: c.execute("PRAGMA synchronous").fetchone()[0]), 1)   # NORMAL
        self.assertEqual(self.db.write(lambda c: c.execute("PRAGMA foreign_keys").fetchone()[0]), 1)

    def test_write_returns_after_commit_and_is_visible_to_readers(self) -> None:
        rowid = self.db.execute("INSERT INTO kv(key, value) VALUES ('a', '1')").result()
        self.assertGreaterEqual(rowid, 0)
        self.assertEqual(self.db.query_value("SELECT value FROM kv WHERE key = 'a'"), "1")
        self.assertEqual(self.db.execute("UPDATE kv SET value = '2' WHERE key = 'a'").result(), 1)

    def test_concurrent_writers_are_serialized_and_group_committed(self) -> None:
        threads, per_thread = 8, 200
        errors: list[BaseException] = []

        def worker(t: int) -> None:
            try:
                futures = [
                    self.db.execute("INSERT INTO kv(key, value) VALUES (?, ?)", (f"{t}-{i}", str(i)))
                    for i in range(per_thread)
                ]
                for f in futures:
                    f.result(timeout=30)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.db.query_value("SELECT COUNT(*) FROM kv"), threads * per_thread)
        # group commit：交易數遠少於操作數
        self.assertLess(self.db.commits, threads * per_thread / 2)

    def test_failing_op_only_rolls_back_itself(self) -> None:
        gate = threading.Event()
        # 先塞一個會卡住 writer 的操作，確保後面三個落在同一個交易
        blocker = self.db.submit(lambda c: gate.wait(5))
        ok1 = self.db.execute("INSERT INTO kv(key, value) VALUES ('ok1', 'x')")
        bad = self.db.execute("INSERT INTO kv(key, value) VALUES ('ok1', 'dup')")       # 違反 PRIMARY KEY

        def partial(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO kv(key, value) VALUES ('half', 'x')")
            raise ValueError("boom")                                                    # 前一句也要被回滾

        half = self.db.submit(partial)
        ok2 = self.db.execute("INSERT INTO kv(key, value) VALUES ('ok2', 'x')")
        gate.set()
        blocker.result(5)
        ok1.result(5)
        ok2.result(5)
        with self.assertRaises(sqlite3.IntegrityError):
            bad.result(5)
        with self.assertRaises(ValueError):
            half.result(5)
        keys = {r[0] for r in self.db.query("SELECT key FROM kv")}
        self.assertEqual(keys, {"ok1", "ok2"})

    def test_reader_connection_is_read_only_and_per_thread(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            self.db.reader().execute("INSERT INTO kv(key, value) VALUES ('x', 'y')")
        main_conn = self.db.reader()
        self.assertIs(main_conn, self.db.reader())
        other: list[sqlite3.Connection] = []
        t = threading.Thread(target=lambda: other.append(self.db.reader()))
        t.start()
        t.join()
        self.assertIsNot(other[0], main_conn)

    def test_readers_do_not_block_on_open_write_transaction(self) -> None:
        self.db.execute("INSERT INTO kv(key, value) VALUES ('seen', '1')").result()
        entered, release = threading.Event(), threading.Event()

        def slow(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO kv(key, value) VALUES ('uncommitted', '1')")
            entered.set()
            release.wait(5)

        fut = self.db.submit(slow)
        self.assertTrue(entered.wait(5))
        # 寫入交易還開著：讀取立即返回，而且看不到未 commit 的資料
        self.assertEqual({r[0] for r in self.db.query("SELECT key FROM kv")}, {"seen"})
        release.set()
        fut.result(5)
        self.assertEqual(self.db.query_value("SELECT COUNT(*) FROM kv"), 2)

    def test_nested_write_is_rejected_instead_of_deadlocking(self) -> None:
        def nested(conn: sqlite3.Connection) -> None:
            self.db.write(lambda c: None)

        with self.assertRaises(DatabaseError):
            self.db.write(nested, timeout=5)

    def test_flush_waits_for_queued_writes(self) -> None:
        for i in range(500):
            self.db.execute("INSERT INTO kv(key, value) VALUES (?, 'v')", (str(i),))
        self.db.flush(timeout=30)
        self.assertEqual(self.db.query_value("SELECT COUNT(*) FROM kv"), 500)

    def test_close_drains_queue_then_rejects_new_writes(self) -> None:
        futures = [self.db.execute("INSERT INTO kv(key, value) VALUES (?, 'v')", (str(i),)) for i in range(300)]
        self.db.close()
        self.assertTrue(all(f.done() and f.exception() is None for f in futures))
        with self.assertRaises(DatabaseClosedError):
            self.db.execute("INSERT INTO kv(key, value) VALUES ('late', 'v')")
        self.db.close()                                     # 可重複呼叫
        # 重新開啟：資料還在，WAL 已收乾淨
        with DBManager(self.path) as db2:
            self.assertEqual(db2.query_value("SELECT COUNT(*) FROM kv"), 300)

    def test_rejects_foreign_database(self) -> None:
        other = Path(self._tmp.name) / "other.db"
        conn = sqlite3.connect(other)
        conn.execute("CREATE TABLE something(x)")
        conn.commit()
        conn.close()
        with self.assertRaises(DatabaseError):
            DBManager(other).open()

    def test_foreign_keys_cascade_and_triggers(self) -> None:
        def seed(conn: sqlite3.Connection) -> int:
            pid = conn.execute(
                "INSERT INTO photos(path, dir_path, file_name, ext, file_size, mtime_ns) "
                "VALUES ('C:/a.jpg', 'C:/', 'a.jpg', 'jpg', 1, 1)").lastrowid
            mid = conn.execute("INSERT INTO models(name, kind, dim) VALUES ('m', 'image_embed', 4)").lastrowid
            conn.execute("INSERT INTO embeddings(photo_id, model_id, dim, vector) VALUES (?, ?, 4, x'00')", (pid, mid))
            tid = conn.execute("INSERT INTO tags(name, prompt, text_vector, text_model_id) "
                               "VALUES ('t', 'old', x'01', ?)", (mid,)).lastrowid
            conn.execute("INSERT INTO photo_tags(photo_id, tag_id, score) VALUES (?, ?, 0.9)", (pid, tid))
            conn.execute("INSERT INTO faces(photo_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score) "
                         "VALUES (?, 0, 0, 1, 1, 0.9)", (pid,))
            return pid

        pid = self.db.write(seed)
        # prompt 一改，快取的文字向量就作廢
        self.db.execute("UPDATE tags SET prompt = 'new' WHERE name = 't'").result()
        self.assertIsNone(self.db.query_value("SELECT text_vector FROM tags WHERE name = 't'"))
        # 刪照片：向量、標籤關聯、人臉一起消失
        self.db.execute("DELETE FROM photos WHERE id = ?", (pid,)).result()
        for table in ("embeddings", "photo_tags", "faces"):
            self.assertEqual(self.db.query_value(f"SELECT COUNT(*) FROM {table}"), 0, table)


if __name__ == "__main__":
    unittest.main()
