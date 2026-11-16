"""執行緒安全的 SQLite 存取層（基礎設施；不含任何業務 SQL，業務查詢在 `db/repository.py`）。

併發模型 —— 「單一寫入者 + 多個讀取者」
-----------------------------------------
SQLite 在 WAL 模式下允許「一個寫入者與任意多個讀取者同時進行」，但**同一時間只能有一個
寫入者**。與其讓多條執行緒各自開連線搶寫入鎖（會遇到 SQLITE_BUSY、重試、飢餓），這裡把
所有寫入序列化到一條專用的 writer 執行緒：

    任何執行緒 ──submit(fn)──▶ [有界寫入佇列] ──▶ writer 執行緒（唯一的寫入連線）
                    ▲                                    │ 把佇列裡現有的操作合併成一個交易
                    └────────── Future（commit 後才完成）◀┘

  * **寫入**：`submit(fn)` 把 `fn(conn)` 排進佇列，回傳 `Future`。writer 一次把佇列裡
    現有的操作全部撈出來，包在同一個 `BEGIN IMMEDIATE … COMMIT` 裡（group commit）。
    閒置時延遲接近零，忙碌時自動變成大批次 —— 不需要調任何時間參數。
  * **每個操作自己一個 SAVEPOINT**：某個操作丟例外只會回滾它自己，同批次的其他操作不受影響。
  * **Future 在 COMMIT 成功後才完成**：呼叫端拿到結果就代表資料已落盤（WAL），
    所以「狀態位元說做完了」與「資料真的在」永遠一致。
  * **背壓**：寫入佇列有上限，滿了 `submit` 會阻塞，上游自然慢下來。
  * **讀取**：每條執行緒各自一條唯讀連線（thread-local），WAL 下讀取不會被寫入擋住，
    看到的是該查詢開始時的一致快照。

使用規則（交接時必讀）
----------------------
  1. 不要自己 `sqlite3.connect()`。讀用 `query*()`，寫用 `submit()` / `write()`。
  2. 傳給 `submit` 的 `fn` 在 writer 執行緒執行：只做 SQL，不做 I/O、不做運算、不碰 GUI，
     也不可以在裡面 `BEGIN` / `COMMIT`（交易由 writer 管）。
  3. 不可以在 `fn` 裡呼叫 `db.write()` / `future.result()`（會自己等自己而死結；有防呆會丟例外）。
  4. 需要「讀出來、判斷、再寫」保持原子性時，把讀與寫都放進同一個 `fn`（writer 連線可以讀）。
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from ..core.config import DbConfig

log = logging.getLogger(__name__)

T = TypeVar("T")
WriteFn = Callable[[sqlite3.Connection], T]

SCHEMA_VERSION = 2                   # schema.sql 目前的版本（全新資料庫直接建成這一版）
APPLICATION_ID = 0x50485354          # 'PHST'：用來辨識這是不是本工具的資料庫
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")

#: 之後的結構變更：(目標版本, SQL)。只能往後加，不可修改已發布的項目。
#: 範例：(2, "ALTER TABLE photos ADD COLUMN caption TEXT;")
MIGRATIONS: list[tuple[int, str]] = [
    # v2（第二階段：分群與多視圖匯出）—— 純新增，不更動任何既有欄位。
    #   persons.is_locked          使用者鎖定的人物群組，重新分群不可覆蓋或拆散
    #   photos.burst_group_id      連拍群組（同群的照片共用群內最小的 photo id）
    #   photos.is_best_shot        連拍群組中綜合評分最高的那一張
    #   photos.is_burst_duplicate  連拍群組中其餘的次佳照片
    (
        2,
        """
        ALTER TABLE persons ADD COLUMN is_locked INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE photos ADD COLUMN burst_group_id INTEGER;
        ALTER TABLE photos ADD COLUMN is_best_shot INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE photos ADD COLUMN is_burst_duplicate INTEGER NOT NULL DEFAULT 0;
        CREATE INDEX idx_photos_burst ON photos(burst_group_id) WHERE burst_group_id IS NOT NULL;
        """,
    ),
]


class DatabaseError(RuntimeError):
    pass


class DatabaseClosedError(DatabaseError):
    pass


class _Op:
    __slots__ = ("fn", "future")

    def __init__(self, fn: WriteFn[Any], future: "Future[Any]") -> None:
        self.fn = fn
        self.future = future


_STOP = object()


class DBManager:
    """單一寫入執行緒 + thread-local 唯讀連線池。

    典型用法::

        db = DBManager(cfg.db_path, cfg.db).open()
        try:
            rows = db.query("SELECT id, path FROM photos WHERE event_id = ?", (7,))
            db.write(lambda c: c.execute("UPDATE photos SET is_favorite = 1 WHERE id = ?", (3,)))
        finally:
            db.close()
    """

    def __init__(self, db_path: Path, config: DbConfig | None = None) -> None:
        self.path = Path(db_path)
        self.cfg = config or DbConfig()
        self._wconn: sqlite3.Connection | None = None
        self._wq: "queue.Queue[Any]" = queue.Queue(maxsize=self.cfg.write_queue_size)
        self._writer: threading.Thread | None = None
        self._local = threading.local()
        self._readers: list[sqlite3.Connection] = []
        self._readers_lock = threading.Lock()
        self._closed = threading.Event()
        self._opened = False
        self._lifecycle = threading.Lock()
        self.sqlite_vec_loaded = False
        # 統計（除錯與效能調校用）
        self.commits = 0
        self.ops_written = 0

    # ------------------------------------------------------------------ #
    # 生命週期
    # ------------------------------------------------------------------ #
    def open(self) -> "DBManager":
        """建立資料夾、開寫入連線、套 PRAGMA、建表/遷移、啟動 writer 執行緒。可重複呼叫。"""
        with self._lifecycle:
            if self._opened:
                return self
            if self._closed.is_set():
                raise DatabaseClosedError("DBManager 已關閉，請建立新的實例")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.path,
                isolation_level=None,          # 自己明確管交易，不用 sqlite3 模組的隱式 BEGIN
                check_same_thread=False,       # 在這裡建立、在 writer 執行緒使用
                timeout=self.cfg.busy_timeout_ms / 1000.0,
            )
            conn.row_factory = sqlite3.Row
            try:
                self._apply_pragmas(conn, writer=True)
                self._migrate(conn)
                if self.cfg.enable_sqlite_vec:
                    self.sqlite_vec_loaded = _try_load_sqlite_vec(conn)
            except BaseException:
                conn.close()
                raise
            self._wconn = conn
            self._writer = threading.Thread(target=self._writer_loop, name="db-writer", daemon=True)
            self._writer.start()
            self._opened = True
            return self

    def close(self, timeout: float | None = 30.0) -> None:
        """把佇列裡的寫入做完、checkpoint、關閉所有連線。可重複呼叫。

        之後再 `submit()` 會丟 `DatabaseClosedError`。
        """
        with self._lifecycle:
            if not self._opened or self._closed.is_set():
                self._closed.set()
                return
            self._closed.set()
            self._wq.put(_STOP)                # 排在所有已提交的操作之後
            assert self._writer is not None
            self._writer.join(timeout)
            if self._writer.is_alive():
                log.error("db-writer 未在 %.0f 秒內結束，仍強制關閉連線", timeout or 0)
            self._fail_leftovers()             # writer 結束後才擠進來的操作
            with self._readers_lock:
                for c in self._readers:
                    try:
                        c.close()
                    except sqlite3.Error:
                        pass
                self._readers.clear()
            if self._wconn is not None:
                try:
                    self._wconn.execute("PRAGMA optimize")
                    self._wconn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error as e:
                    log.warning("關閉前的 checkpoint 失敗：%s", e)
                self._wconn.close()
                self._wconn = None

    def __enter__(self) -> "DBManager":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed.is_set()

    # ------------------------------------------------------------------ #
    # PRAGMA 與遷移
    # ------------------------------------------------------------------ #
    def _apply_pragmas(self, conn: sqlite3.Connection, *, writer: bool) -> None:
        c = self.cfg
        if writer:
            # WAL：讀寫互不阻塞。這是資料庫檔案的持久屬性，設一次即可。
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                # 網路磁碟或不支援共享記憶體的檔案系統會退回其他模式
                log.warning("無法啟用 WAL（目前 journal_mode=%s）；資料庫請放在本機磁碟", mode)
            # WAL + NORMAL：斷電可能丟最後幾個交易，但資料庫不會損毀；寫入快數倍。
            conn.execute(f"PRAGMA synchronous = {c.synchronous}")
            conn.execute(f"PRAGMA wal_autocheckpoint = {int(c.wal_autocheckpoint_pages)}")
            conn.execute(f"PRAGMA journal_size_limit = {int(c.journal_size_limit_mb) * 1024 * 1024}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {int(c.busy_timeout_ms)}")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(f"PRAGMA cache_size = {-int(c.cache_size_kb)}")        # 負值 = KiB
        conn.execute(f"PRAGMA mmap_size = {int(c.mmap_size_mb) * 1024 * 1024}")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            has_tables = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
            ).fetchone()
            if has_tables:
                raise DatabaseError(f"{self.path} 不是 photosorter 的資料庫（已有其他資料表）")
            log.info("建立新資料庫 %s", self.path)
            script = _SCHEMA_FILE.read_text(encoding="utf-8")
            conn.executescript(
                "BEGIN;\n"
                f"{script}\n"
                f"PRAGMA application_id = {APPLICATION_ID};\n"
                f"PRAGMA user_version = {SCHEMA_VERSION};\n"
                "COMMIT;"
            )
            version = SCHEMA_VERSION
        else:
            app_id = conn.execute("PRAGMA application_id").fetchone()[0]
            if app_id != APPLICATION_ID:
                raise DatabaseError(f"{self.path} 不是 photosorter 的資料庫（application_id={app_id:#x}）")
        if version > SCHEMA_VERSION:
            raise DatabaseError(
                f"資料庫版本 {version} 比程式支援的新，請更新程式（不會自動降版以免毀損資料）"
            )
        for target, sql in MIGRATIONS:
            if target > version:
                log.info("資料庫遷移 %d -> %d", version, target)
                conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {target};\nCOMMIT;")
                version = target

    # ------------------------------------------------------------------ #
    # 讀取
    # ------------------------------------------------------------------ #
    def reader(self) -> sqlite3.Connection:
        """目前執行緒專屬的唯讀連線（第一次呼叫時建立，之後重複使用）。

        連線以 `mode=ro` 開啟，誤下 INSERT/UPDATE 會直接報錯而不是默默繞過 writer。
        """
        self._ensure_open()
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
                isolation_level=None,
                check_same_thread=False,       # 只為了讓 close() 能從別的執行緒關它
                timeout=self.cfg.busy_timeout_ms / 1000.0,
            )
            conn.row_factory = sqlite3.Row
            self._apply_pragmas(conn, writer=False)
            if self.sqlite_vec_loaded:
                _try_load_sqlite_vec(conn)
            self._local.conn = conn
            with self._readers_lock:
                self._readers.append(conn)
        return conn

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self.reader().execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return self.reader().execute(sql, params).fetchone()

    def query_value(self, sql: str, params: Sequence[Any] | dict[str, Any] = (), default: Any = None) -> Any:
        row = self.reader().execute(sql, params).fetchone()
        return default if row is None else row[0]

    def iter_query(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = (), chunk: int = 2000
    ) -> Iterator[sqlite3.Row]:
        """大量結果用：分批 fetch，不一次全部載入記憶體。迭代期間請勿在同一執行緒再下別的查詢。"""
        cur = self.reader().execute(sql, params)
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                return
            yield from rows

    # ------------------------------------------------------------------ #
    # 寫入
    # ------------------------------------------------------------------ #
    def submit(self, fn: WriteFn[T]) -> "Future[T]":
        """排入一個寫入操作。`fn(conn)` 會在 writer 執行緒、交易內執行。

        Returns:
            Future：交易 COMMIT 成功後才會完成；`fn` 的回傳值就是結果，
            `fn` 丟的例外會原樣出現在 `future.result()`。

        佇列滿時會阻塞（背壓）。
        """
        self._ensure_open()
        if threading.current_thread() is self._writer:
            raise DatabaseError("不可在寫入操作內再提交寫入（會死結）；請把 SQL 寫在同一個 fn 裡")
        fut: "Future[T]" = Future()
        op = _Op(fn, fut)
        while True:
            try:
                self._wq.put(op, timeout=0.5)
                break
            except queue.Full:
                self._ensure_open()            # 等待期間被關閉 -> 丟例外而不是永遠卡住
        if self._closed.is_set() and (self._writer is None or not self._writer.is_alive()):
            # close() 與 submit() 競速：writer 已結束，這個操作不會有人執行
            self._fail_leftovers()
        return fut

    def write(self, fn: WriteFn[T], timeout: float | None = None) -> T:
        """`submit(fn).result()` 的簡寫：等到 commit 完成並回傳結果。"""
        return self.submit(fn).result(timeout)

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> "Future[int]":
        """單一 DML。Future 的結果：INSERT 為 lastrowid，其餘為受影響列數。"""
        is_insert = sql.lstrip()[:6].upper() == "INSERT"

        def op(conn: sqlite3.Connection) -> int:
            cur = conn.execute(sql, params)
            return int(cur.lastrowid or 0) if is_insert else cur.rowcount

        return self.submit(op)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any] | dict[str, Any]]) -> "Future[int]":
        """批次 DML（同一個交易）。`rows` 會先被實體化，避免在 writer 執行緒跑呼叫端的產生器。"""
        data = list(rows)

        def op(conn: sqlite3.Connection) -> int:
            return conn.executemany(sql, data).rowcount

        return self.submit(op)

    def flush(self, timeout: float | None = None) -> None:
        """等到「呼叫當下已在佇列裡的所有寫入」都 commit。"""
        self.submit(lambda conn: None).result(timeout)

    @property
    def pending_writes(self) -> int:
        return self._wq.qsize()

    # ------------------------------------------------------------------ #
    # writer 執行緒
    # ------------------------------------------------------------------ #
    def _writer_loop(self) -> None:
        conn = self._wconn
        assert conn is not None
        max_ops = max(1, self.cfg.write_batch_max_ops)
        stopping = False
        while not stopping:
            first = self._wq.get()
            if first is _STOP:
                break
            batch: list[_Op] = [first]
            # group commit：把此刻佇列裡已有的操作一併撈出來，合成一個交易
            while len(batch) < max_ops:
                try:
                    nxt = self._wq.get_nowait()
                except queue.Empty:
                    break
                if nxt is _STOP:
                    stopping = True
                    break
                batch.append(nxt)
            self._run_batch(conn, batch)
        self._fail_leftovers()

    def _fail_leftovers(self) -> None:
        """close() 之後才擠進佇列的操作：明確告知失敗，不讓呼叫端永遠等下去。"""
        while True:
            try:
                leftover = self._wq.get_nowait()
            except queue.Empty:
                return
            if isinstance(leftover, _Op) and not leftover.future.done():
                leftover.future.set_exception(DatabaseClosedError("資料庫已關閉"))

    def _run_batch(self, conn: sqlite3.Connection, batch: list[_Op]) -> None:
        try:
            self._begin(conn)
        except BaseException as e:  # noqa: BLE001
            for op in batch:
                if op.future.set_running_or_notify_cancel():
                    op.future.set_exception(e)
            return

        outcomes: list[tuple[_Op, Any, BaseException | None]] = []
        for op in batch:
            if not op.future.set_running_or_notify_cancel():
                continue                       # 呼叫端已取消
            conn.execute("SAVEPOINT op")
            try:
                result = op.fn(conn)
            except BaseException as e:  # noqa: BLE001 - 連 KeyboardInterrupt 也要回報給呼叫端
                try:
                    conn.execute("ROLLBACK TO op")
                    conn.execute("RELEASE op")
                except sqlite3.Error:
                    log.exception("回滾 savepoint 失敗")
                outcomes.append((op, None, e))
            else:
                conn.execute("RELEASE op")
                outcomes.append((op, result, None))

        try:
            conn.execute("COMMIT")
        except BaseException as e:  # noqa: BLE001
            log.exception("COMMIT 失敗，整批 %d 個操作回滾", len(batch))
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            for op, _result, _err in outcomes:
                op.future.set_exception(e)
            return

        self.commits += 1
        self.ops_written += len(outcomes)
        # commit 成功後才通知呼叫端
        for op, result, err in outcomes:
            if err is not None:
                op.future.set_exception(err)
            else:
                op.future.set_result(result)

    def _begin(self, conn: sqlite3.Connection) -> None:
        """BEGIN IMMEDIATE：一開始就拿寫入鎖。另一個行程佔著鎖時重試幾次。"""
        last: sqlite3.OperationalError | None = None
        for attempt in range(5):
            try:
                conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                    raise
                last = e
                time.sleep(0.1 * (attempt + 1))
        assert last is not None
        raise last

    # ------------------------------------------------------------------ #
    def _ensure_open(self) -> None:
        if not self._opened:
            raise DatabaseError("DBManager 尚未 open()")
        if self._closed.is_set():
            raise DatabaseClosedError("資料庫已關閉")


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """選配：載入 sqlite-vec 擴充。沒裝或載入失敗只記錄，不影響其他功能。

    載入後即可對現有的 BLOB 欄位直接下向量查詢，例如::

        SELECT photo_id, vec_distance_cosine(vector, :q) AS d
        FROM embeddings WHERE model_id = :m ORDER BY d LIMIT 50
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception as e:  # noqa: BLE001
        log.info("sqlite-vec 未啟用（%s）；向量檢索改用記憶體矩陣", e)
        return False
