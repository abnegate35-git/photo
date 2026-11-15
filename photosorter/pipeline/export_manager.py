"""非破壞性整理的最後一步：把資料庫裡的分類結果「實體化」成資料夾結構。

在呼叫 `ExportManager.execute()` 之前，本工具對原始檔案是**唯讀**的 —— 所有分類、標籤、
人臉命名、廢片標記都只存在 SQLite。匯出分三步，每一步都可以單獨檢視：

  1. `plan()`    —— Dry-run。算出每個檔案的目標路徑並解決檔名衝突，**不碰檔案系統、不寫資料庫**。
                    回傳的 `ExportPlan` 可以直接給 UI 顯示成預覽清單。
  2. `execute()` —— 照計畫執行，每一筆操作都記在 `export_logs`（交易日誌）。
  3. `rollback()`—— 依日誌倒著把一個批次還原。

四種模式
--------
  copy      複製。最安全，佔兩份空間。
  move      移動。原始位置不再有檔案；`photos.path` 會同步更新，回復時搬回原位。
  hardlink  硬連結。同一個磁碟區上不佔額外空間，兩個路徑是同一份資料
            （**編輯其中一個，另一個也會變**；刪除其中一個，另一個還在）。不能跨磁碟區。
  symlink   符號連結。可跨磁碟區，但原始檔搬走連結就斷。
            Windows 10/11 上建立符號連結需要「開發人員模式」或系統管理員權限。

安全守則（實作上的硬性規則，修改時不可破壞）
----------------------------------------------
  * **絕不覆寫既有檔案。** 衝突一律在 `plan()` 階段以改名或略過解決；執行時若目標在計畫之後
    才冒出來，該筆操作失敗，不覆寫。
  * **Write-ahead 日誌。** 先把 `state='pending'` 的日誌 commit，再動檔案，成功後改 `done`。
    任何時刻當機，留下的 `pending` 就是「可能做了一半」的操作，`recover()` 依檔案系統的實際
    狀態收尾。
  * 複製類操作先寫到同資料夾的暫存檔再 rename，目標路徑上不會出現寫到一半的檔案。
  * 跨磁碟區的 move = 複製 → 驗證（大小 + SHA-256）→ 才刪來源。驗證不過就保留來源。
  * 回復時先確認目標還是「我們當初建立的那個檔案」，被使用者改過或換掉的不刪。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from typing import Any, Callable, Iterable

from ..core.config import AppConfig
from ..core.events import EventBus, EventKind
from ..core.types import ConflictPolicy, ExportMode, ExportPlan, ExportResult, PlannedOp
from ..db.db_manager import DBManager
from ..db.repository import Repository
from ..utils.hashing import sha256_file

log = logging.getLogger(__name__)

_PART_SUFFIX = ".photosorter-part"
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

#: 路徑模板可用的欄位。模板以 "/" 分隔資料夾層級，例如 "{year}/{year}-{month}/{event}/{filename}"
TEMPLATE_FIELDS = (
    "year", "month", "day", "hour", "minute", "second", "date",
    "event", "person", "tag", "camera", "country", "city",
    "filename", "stem", "ext",
)


class ExportError(RuntimeError):
    pass


def sanitize_component(name: str, fallback: str = "_") -> str:
    """把任意字串變成合法的 Windows 檔名/資料夾名（單一層級）。"""
    s = _ILLEGAL.sub("_", str(name)).strip().rstrip(". ")
    if not s or s in (".", ".."):
        return fallback
    if s.split(".")[0].upper() in _RESERVED:
        s = "_" + s
    return s[:150]


def _key(path: str) -> str:
    """比較路徑是否相同用的鍵（Windows 不分大小寫）。"""
    return os.path.normcase(os.path.normpath(path))


def _volume(path: str) -> Any:
    """路徑所在的磁碟區識別。路徑還不存在時往上找到存在的那一層。"""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        dev = os.stat(p).st_dev
        if dev:
            return dev
    except OSError:
        pass
    return os.path.splitdrive(p)[0].upper()


def can_symlink() -> bool:
    """目前的使用者有沒有建立符號連結的權限（在系統暫存資料夾實測，不碰使用者的資料夾）。"""
    try:
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "t")
            with open(target, "wb"):
                pass
            os.symlink(target, os.path.join(d, "l"))
        return True
    except (OSError, NotImplementedError):
        return False


class ExportManager:
    def __init__(self, cfg: AppConfig, db: DBManager, repo: Repository, bus: EventBus | None = None) -> None:
        self._cfg = cfg
        self._db = db
        self._repo = repo
        self._bus = bus

    # ================================================================== #
    # 1. Dry-run
    # ================================================================== #
    def plan(
        self,
        dest_root: str | os.PathLike[str],
        *,
        photo_ids: Iterable[int] | None = None,
        mode: ExportMode | str = ExportMode.COPY,
        template: str | None = None,
        conflict_policy: ConflictPolicy | str | None = None,
    ) -> ExportPlan:
        """產生匯出計畫。不建立任何資料夾、不寫資料庫，可以放心反覆呼叫來預覽。

        Args:
            dest_root: 目標根目錄（可以還不存在）。
            photo_ids: 要匯出的照片；None = 全部未遺失、未標為廢片的照片。
            template: 路徑模板，欄位見 `TEMPLATE_FIELDS`；None 用 `export.path_template`。

        Raises:
            ExportError: 模板用了不存在的欄位。
        """
        mode = ExportMode(mode)
        policy = ConflictPolicy(conflict_policy or self._cfg.export.conflict_policy)
        template = template or self._cfg.export.path_template
        root = os.path.normpath(os.path.abspath(os.fspath(dest_root)))
        plan = ExportPlan(mode=mode, dest_root=root, template=template, conflict_policy=policy)

        if mode is ExportMode.SYMLINK and not can_symlink():
            plan.warnings.append("目前沒有建立符號連結的權限：請開啟 Windows「開發人員模式」或以系統管理員身分執行")
        dest_volume = _volume(root)
        planned: dict[str, bytes | None] = {}          # 本計畫已佔用的目標 -> 來源的 sha256

        for row in self._repo.fetch_export_rows(photo_ids):
            src = row["path"]
            if not os.path.isfile(src):
                plan.ops.append(PlannedOp(row["id"], src, None, "skip", "missing_source"))
                continue
            try:
                rel = self._render(template, row)
            except KeyError as e:
                raise ExportError(f"路徑模板用了不存在的欄位 {e}；可用：{', '.join(TEMPLATE_FIELDS)}") from None
            dst = os.path.join(root, *rel)
            if _key(dst) == _key(src):
                plan.ops.append(PlannedOp(row["id"], src, None, "skip", "same_path"))
                continue
            if mode is ExportMode.HARDLINK and _volume(src) != dest_volume:
                plan.ops.append(PlannedOp(row["id"], src, None, "skip", "cross_volume"))
                continue

            dst, note = self._resolve_conflict(src, row["sha256"], row["file_size"], dst, policy, planned)
            if dst is None:
                plan.ops.append(PlannedOp(row["id"], src, None, "skip", note))
                continue
            planned[_key(dst)] = row["sha256"]
            plan.ops.append(PlannedOp(row["id"], src, dst, mode.value, note))

        cross = sum(1 for op in plan.ops if op.note == "cross_volume")
        if cross:
            plan.warnings.append(f"{cross} 個檔案與目標不在同一個磁碟區，無法建立硬連結（已略過）")
        return plan

    def _render(self, template: str, row: sqlite3.Row) -> list[str]:
        """把模板套上一張照片的資料，回傳各層資料夾/檔名（已清理成合法名稱）。"""
        if row["taken_at"]:
            dt = datetime.strptime(row["taken_at"], "%Y-%m-%dT%H:%M:%S")
        else:
            dt = datetime.fromtimestamp(row["mtime_ns"] / 1e9)
        event = row["event_name"]
        if not event and row["event_start_ts"] is not None:
            day = time.strftime("%Y-%m-%d", time.gmtime(row["event_start_ts"]))
            event = f"{day} {row['location_label']}" if row["location_label"] else day
        stem, ext = os.path.splitext(row["file_name"])
        values = {
            "year": f"{dt.year:04d}", "month": f"{dt.month:02d}", "day": f"{dt.day:02d}",
            "hour": f"{dt.hour:02d}", "minute": f"{dt.minute:02d}", "second": f"{dt.second:02d}",
            "date": dt.strftime("%Y-%m-%d"),
            "event": event or "未分類事件",
            "person": row["primary_person"] or "未命名人物",
            "tag": row["primary_tag"] or "未分類",
            "camera": row["camera_model"] or "未知相機",
            "country": row["country_code"] or "未知地點",
            "city": row["city"] or "未知地點",
            "filename": row["file_name"], "stem": stem, "ext": ext.lstrip("."),
        }
        # 先清理各欄位的值（值裡的斜線不可以變成額外的資料夾層級），再依模板的 "/" 切層
        safe = {k: sanitize_component(v) for k, v in values.items()}
        rendered = template.format_map(safe)
        parts = [sanitize_component(p) for p in re.split(r"[\\/]+", rendered) if p.strip()]
        return parts or [sanitize_component(row["file_name"])]

    def _resolve_conflict(
        self, src: str, src_sha: bytes | None, src_size: int, dst: str, policy: ConflictPolicy,
        planned: dict[str, bytes | None],
    ) -> tuple[str | None, str]:
        """決定最終目標路徑。回傳 (路徑或 None, 說明)。None 代表這筆略過。"""
        key = _key(dst)
        on_disk = os.path.lexists(dst)
        in_plan = key in planned
        if not on_disk and not in_plan:
            return dst, ""
        if policy is ConflictPolicy.SKIP:
            return None, "exists"
        if policy is ConflictPolicy.SKIP_IDENTICAL:
            if in_plan and src_sha is not None and planned[key] == src_sha:
                return None, "identical"                # 同一批裡內容相同的另一份複本
            if on_disk and not in_plan and self._same_content(src, src_sha, src_size, dst):
                return None, "identical"
        # 改名：name_1.ext、name_2.ext …
        base, ext = os.path.splitext(dst)
        for i in range(1, 100000):
            candidate = f"{base}_{i}{ext}"
            if not os.path.lexists(candidate) and _key(candidate) not in planned:
                return candidate, "renamed"
        return None, "exists"

    @staticmethod
    def _same_content(src: str, src_sha: bytes | None, src_size: int, dst: str) -> bool:
        try:
            if os.path.getsize(dst) != src_size:
                return False
            return sha256_file(dst) == (src_sha or sha256_file(src))
        except OSError:
            return False

    # ================================================================== #
    # 2. 執行
    # ================================================================== #
    def execute(self, plan: ExportPlan, *, should_stop: Callable[[], bool] = lambda: False) -> ExportResult:
        """照計畫執行。同步（阻塞到做完）；GUI 請放在背景執行緒呼叫，進度走 EventBus。

        單筆失敗不會中斷整批；結果裡的 `errors` 列出每一筆失敗的原因。
        """
        ops = [op for op in plan.ops if op.dst is not None and op.action != "skip"]
        batch_id = self._db.execute(
            "INSERT INTO export_batches(mode, dest_root, template, conflict_policy, total) VALUES (?, ?, ?, ?, ?)",
            (plan.mode.value, plan.dest_root, plan.template, plan.conflict_policy.value, len(ops)),
        ).result()
        result = ExportResult(batch_id=batch_id, skipped=len(plan.ops) - len(ops))

        for seq, op in enumerate(ops):
            if should_stop():
                result.cancelled = True
                break
            assert op.dst is not None
            size = _safe_size(op.src)
            log_id = self._db.execute(
                "INSERT INTO export_logs(batch_id, seq, photo_id, action, src_path, dst_path, file_size) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (batch_id, seq, op.photo_id, op.action, op.src, op.dst, size),
            ).result()                                   # 已 commit：之後才動檔案（write-ahead）
            try:
                self._perform(op.action, op.src, op.dst)
                self._mark_done(log_id, op)
                result.done += 1
            except Exception as e:  # noqa: BLE001
                message = f"{type(e).__name__}: {e}"
                log.warning("匯出失敗 %s -> %s: %s", op.src, op.dst, message)
                self._db.execute(
                    "UPDATE export_logs SET state = 'failed', error = ?, executed_at = unixepoch() WHERE id = ?",
                    (message[:2000], log_id),
                ).result()
                result.failed += 1
                result.errors.append((op.src, message))
            if self._bus is not None:
                self._bus.publish(EventKind.EXPORT_PROGRESS, batch_id=batch_id, index=seq + 1,
                                  total=len(ops), src=op.src, dst=op.dst)

        state = "cancelled" if result.cancelled else "done"
        self._db.execute(
            "UPDATE export_batches SET state = ?, finished_at = unixepoch() WHERE id = ?", (state, batch_id)
        ).result()
        if self._bus is not None:
            self._bus.publish(EventKind.EXPORT_DONE, batch_id=batch_id, done=result.done,
                              skipped=result.skipped, failed=result.failed, cancelled=result.cancelled)
        return result

    def _perform(self, action: str, src: str, dst: str) -> None:
        if os.path.lexists(dst):
            raise ExportError("目標在計畫產生之後出現了同名檔案（不覆寫）")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if action == "hardlink":
            os.link(src, dst)
        elif action == "symlink":
            os.symlink(src, dst)
        elif action == "copy":
            self._copy_verified(src, dst, deep=False)
        elif action == "move":
            if _volume(src) == _volume(os.path.dirname(dst)):
                os.rename(src, dst)                      # 同磁碟區：原子操作，瞬間完成
            else:
                self._copy_verified(src, dst, deep=self._cfg.export.verify_copy)
                os.remove(src)                           # 驗證通過才刪來源
        else:
            raise ExportError(f"未知的操作：{action}")

    def _copy_verified(self, src: str, dst: str, *, deep: bool) -> None:
        """複製到暫存檔 → 驗證 → rename 成正式檔名。任何一步失敗都不會留下半成品。"""
        part = dst + _PART_SUFFIX
        try:
            shutil.copy2(src, part)
            if self._cfg.export.verify_copy and os.path.getsize(part) != os.path.getsize(src):
                raise ExportError("複製後大小不符")
            if deep and sha256_file(part) != sha256_file(src):
                raise ExportError("複製後 SHA-256 不符")
            os.rename(part, dst)                         # Windows 上目標已存在會失敗 → 不會覆寫
        except BaseException:
            try:
                if os.path.lexists(part):
                    os.remove(part)
            except OSError:
                pass
            raise

    def _mark_done(self, log_id: int, op: PlannedOp) -> None:
        """日誌改成 done；move 另外把 `photos.path` 改到新位置（同一個交易）。"""
        assert op.dst is not None
        dst = op.dst

        def fn(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE export_logs SET state = 'done', executed_at = unixepoch() WHERE id = ?", (log_id,))
            if op.action == "move":
                _set_photo_path(conn, op.photo_id, dst)

        try:
            self._db.write(fn)
        except sqlite3.IntegrityError as e:
            # 新路徑在資料庫裡已被另一列佔用（例如一筆標為遺失的舊紀錄）：把檔案搬回去，維持一致
            shutil.move(dst, op.src)
            raise ExportError(f"資料庫中已有相同路徑的紀錄，已還原：{e}") from e

    # ================================================================== #
    # 3. 回復
    # ================================================================== #
    def rollback(self, batch_id: int) -> ExportResult:
        """把一個批次倒著還原。回傳的 `done` 是成功還原的筆數，`failed` 是無法還原的筆數。

        無法還原的情況（會標成 `rollback_failed` 並保留檔案，不強行處理）：
          * 目標檔已被修改或換掉（不是我們當初建立的那個）。
          * move 的原始位置已經被別的檔案佔用。
        """
        batch = self._db.query_one("SELECT dest_root FROM export_batches WHERE id = ?", (batch_id,))
        if batch is None:
            raise ExportError(f"找不到匯出批次 {batch_id}")
        rows = self._db.query(
            "SELECT id, photo_id, action, src_path, dst_path, file_size FROM export_logs "
            "WHERE batch_id = ? AND state = 'done' ORDER BY seq DESC",
            (batch_id,),
        )
        result = ExportResult(batch_id=batch_id)
        for r in rows:
            try:
                self._undo(r)
            except Exception as e:  # noqa: BLE001
                message = f"{type(e).__name__}: {e}"
                self._db.execute(
                    "UPDATE export_logs SET state = 'rollback_failed', error = ? WHERE id = ?",
                    (message[:2000], r["id"]),
                ).result()
                result.failed += 1
                result.errors.append((r["dst_path"], message))
                continue
            result.done += 1
            _prune_empty_dirs(os.path.dirname(r["dst_path"]), batch["dest_root"])

        state = "rolled_back" if result.failed == 0 else "partial_rollback"
        self._db.execute(
            "UPDATE export_batches SET state = ?, finished_at = unixepoch() WHERE id = ?", (state, batch_id)
        ).result()
        return result

    def _undo(self, r: sqlite3.Row) -> None:
        src, dst, action = r["src_path"], r["dst_path"], r["action"]
        if action == "move":
            if not os.path.lexists(dst):
                raise ExportError("搬過去的檔案已不在目標位置")
            if os.path.lexists(src):
                raise ExportError("原始位置已被其他檔案佔用")
            os.makedirs(os.path.dirname(src), exist_ok=True)
            if _volume(dst) == _volume(os.path.dirname(src)):
                os.rename(dst, src)
            else:
                self._copy_verified(dst, src, deep=self._cfg.export.verify_copy)
                os.remove(dst)
        elif os.path.lexists(dst):
            if action == "symlink":
                ours = os.path.islink(dst)
            elif action == "hardlink" and os.path.exists(src):
                ours = os.path.samefile(src, dst)
            else:
                ours = r["file_size"] is None or os.path.getsize(dst) == r["file_size"]
            if not ours:
                raise ExportError("目標檔已被修改或取代，不刪除")
            os.remove(dst)
        # 目標本來就不在了 → 視為已還原

        def fn(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE export_logs SET state = 'rolled_back', rolled_back_at = unixepoch() WHERE id = ?",
                (r["id"],),
            )
            if action == "move" and r["photo_id"] is not None:
                _set_photo_path(conn, r["photo_id"], src)

        self._db.write(fn)

    # ================================================================== #
    # 當機後收尾
    # ================================================================== #
    def recover(self) -> int:
        """處理上次執行到一半就中斷的操作（`state='pending'` 的日誌）。程式啟動時呼叫一次。

        依檔案系統的實際狀態判斷每一筆是「其實做完了」還是「沒做成」，並清掉殘留的暫存檔。
        回傳處理的筆數。
        """
        rows = self._db.query(
            "SELECT id, photo_id, action, src_path, dst_path FROM export_logs WHERE state = 'pending' ORDER BY id"
        )
        for r in rows:
            src, dst, action = r["src_path"], r["dst_path"], r["action"]
            part = dst + _PART_SUFFIX
            if os.path.lexists(part):
                try:
                    os.remove(part)
                except OSError:
                    log.warning("無法清除暫存檔 %s", part)
            dst_ok, src_ok = os.path.lexists(dst), os.path.lexists(src)
            if action == "move" and dst_ok and src_ok:
                # 跨磁碟區 move 複製完、還沒刪來源：來源仍是完整的，撤掉目標那份
                os.remove(dst)
                dst_ok = False
            done = dst_ok

            def fn(conn: sqlite3.Connection, r: sqlite3.Row = r, done: bool = done) -> None:
                if done:
                    conn.execute(
                        "UPDATE export_logs SET state = 'done', executed_at = unixepoch() WHERE id = ?", (r["id"],))
                    if r["action"] == "move" and r["photo_id"] is not None:
                        _set_photo_path(conn, r["photo_id"], r["dst_path"])
                else:
                    conn.execute(
                        "UPDATE export_logs SET state = 'failed', error = 'interrupted' WHERE id = ?", (r["id"],))

            self._db.write(fn)
        self._db.execute(
            "UPDATE export_batches SET state = 'cancelled', finished_at = unixepoch() WHERE state = 'running'"
        ).result()
        return len(rows)


# --------------------------------------------------------------------------- #
def _set_photo_path(conn: sqlite3.Connection, photo_id: int, path: str) -> None:
    d, name = os.path.split(path)
    conn.execute(
        "UPDATE photos SET path = ?, dir_path = ?, file_name = ?, ext = ? WHERE id = ?",
        (path, d, name, os.path.splitext(name)[1].lstrip(".").lower(), photo_id),
    )


def _safe_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _prune_empty_dirs(start: str, stop_at: str) -> None:
    """回復後把因此變空的資料夾往上清掉，清到 `stop_at`（不含）為止。非空的資料夾不動。"""
    stop = _key(stop_at)
    current = start
    while _key(current) != stop and _key(current).startswith(stop):
        try:
            os.rmdir(current)                            # 只有空資料夾刪得掉
        except OSError:
            return
        current = os.path.dirname(current)
