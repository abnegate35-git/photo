"""非破壞性匯出與多視圖歸檔引擎：「單一實體主庫 + 硬連結多維度視圖」。

匯出根目錄底下的四層結構::

    <export_root>/
    ├── 01_Timeline_Master/                實體主庫（唯一一份實體資料）
    │   ├── {Year}/{YYYY-MM-DD}_{City}_{EventName}/
    │   └── _No_EXIF_Date/{FileMtimeYear}/     沒有 EXIF 拍攝日期的照片
    ├── 02_By_Person/                      人物視圖（硬連結，不佔額外空間）
    │   ├── {PersonName}/Solo_Portraits/       畫面裡只有這一張臉
    │   ├── {PersonName}/{Year}/               其餘
    │   └── _Group_Photos/                     畫面含 2 位以上已知人物（額外再連一份）
    ├── 03_By_Semantic_Tag/{TagName}/      語意標籤視圖（硬連結；一張照片可出現在多個標籤下）
    └── 99_Review_Staging/                 待清理隔離區（不進主庫、沒有視圖）
        ├── _Blurry_Suspects/                  模糊／被標為廢片
        ├── _Burst_Duplicates/                 連拍次佳照片
        └── _Screenshots_Downloads/            螢幕截圖

主庫的檔案依使用者選的主模式（COPY / MOVE / HARDLINK）從原始位置放進來；人物與標籤視圖一律是
**從主庫的新檔案**建立的硬連結，所以即使原始相簿在另一顆硬碟（主庫只能用 COPY），視圖仍然是
同磁碟區、零空間的硬連結。

檔名統一為 `{Date}_{Time}_{SubSec}_{DeviceOrTag}_{ShortHash}.{ext}`，例如
`20240501_101500_000_Zenfone9_a3f9c1.jpg`。ShortHash 是檔案 SHA-256 的前 6 碼，所以：
  * 內容不同的檔案幾乎不會撞名；真的撞了（前綴與 6 碼都相同但完整 SHA-256 不同）才加 `_v2`。
  * 重複匯出是冪等的：同一個檔案第二次匯出時目標已存在，直接略過。

Sidecar（Live Photo 的 .mov、RAW、.xmp、.aae…）與主照片是同一個不可分割的群組：
一起複製／搬移／硬連結，並沿用主照片的新檔名主幹，只保留各自的副檔名。

三個步驟（與第一階段的 `plan / execute / rollback` 相同的精神）：
  1. `preview_export()`  —— Dry-run：回傳動作清單與預估佔用空間，不碰檔案系統、不寫資料庫。
  2. `execute_export()`  —— 照預覽執行，每一筆操作都記在 `export_logs`。
  3. `rollback_batch()`  —— 依日誌倒著還原整個批次，最後清掉空資料夾。

本類別繼承 `pipeline.export_manager.ExportManager`（自訂模板的單層匯出），沿用它的底層安全機制：
先寫 `pending` 日誌再動檔案、複製先寫暫存檔再 rename、絕不覆寫既有檔案、跨磁碟區搬移先驗證再刪來源、
回復前確認目標仍是當初建立的那個檔案、程式啟動時以 `recover()` 收尾中斷的操作。

`export_logs` 的欄位對應：提示規格裡的 `op_type` 即既有欄位 `action`，`status` 即 `state`。
主檔（主庫或隔離區的那一筆）會記 `photo_id` 與完整 `sha256`；Sidecar 與視圖的 `photo_id` 留 NULL，
回復 MOVE 時才不會把 Sidecar 的路徑寫回 `photos.path`。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable

from ..db.db_manager import DBManager
from ..db.repository import Repository
from ..pipeline.export_manager import ExportError, _key, _set_photo_path
from ..pipeline.export_manager import ExportManager as TemplateExportManager
from ..pipeline.scanner import normalize_root
from ..utils.clustering import person_display_name
from ..utils.hashing import sha256_file
from .config import AppConfig
from .events import EventBus, EventKind
from .types import ExportMode, ExportResult

log = logging.getLogger(__name__)

# ---- 目錄名稱 ----
MASTER_DIR = "01_Timeline_Master"
PERSON_DIR = "02_By_Person"
TAG_DIR = "03_By_Semantic_Tag"
STAGING_DIR = "99_Review_Staging"
NO_DATE_DIR = "_No_EXIF_Date"
SOLO_DIR = "Solo_Portraits"
GROUP_DIR = "_Group_Photos"
STAGING_BLURRY = "_Blurry_Suspects"
STAGING_BURST = "_Burst_Duplicates"
STAGING_SCREENSHOT = "_Screenshots_Downloads"
LAYER_DIRS = (MASTER_DIR, PERSON_DIR, TAG_DIR, STAGING_DIR)
_FIXED_NAMES = frozenset((*LAYER_DIRS, NO_DATE_DIR, SOLO_DIR, GROUP_DIR,
                          STAGING_BLURRY, STAGING_BURST, STAGING_SCREENSHOT))   # 路徑過長時不可縮短的名稱

#: 與主照片同主檔名時視為附屬檔的副檔名
SIDECAR_EXTS = (".mov", ".mp4", ".xmp", ".arw", ".cr2", ".dng", ".aae")
_RAW_EXTS = {"arw", "cr2", "dng"}
_HEIF_EXTS = {"heic", "heif"}

MAX_PATH_LEN = 240                       # 完整絕對路徑必須短於此值
LAYOUT_ID = "four_layer_v1"              # 寫進 export_batches.template，用來辨識這種版面的批次
_MIN_DIR_LEN = 8                         # 路徑過長時資料夾名稱最短縮到幾個字元
_DEVICE_MAX = 24

_ILLEGAL = re.compile(r'[<>:"/\\|?*]')
_CONTROL = re.compile(r"[\x00-\x1f]")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")
_EXIF_IFD, _EXIF_SUBSEC_ORIGINAL = 0x8769, 0x9291

ROLE_MASTER, ROLE_STAGING, ROLE_SIDECAR = "master", "staging", "sidecar"
ROLE_PERSON_VIEW, ROLE_TAG_VIEW = "person_view", "tag_view"
_PRIMARY_ROLES = (ROLE_MASTER, ROLE_STAGING)
_VIEW_ROLES = (ROLE_PERSON_VIEW, ROLE_TAG_VIEW)


def clean_name(name: Any, fallback: str = "_") -> str:
    """把任意字串清成合法的 Windows 檔名／資料夾名（單一層級）。

    以 `_` 取代 `< > : " / \\ | ? *`，移除控制字元，去掉頭尾的空白與句點；
    另外避開 Windows 的保留裝置名稱（CON、NUL、COM1…）。
    """
    s = _CONTROL.sub("", _ILLEGAL.sub("_", str(name))).strip(" .")
    if not s:
        return fallback
    if s.split(".")[0].upper() in _RESERVED:
        s = "_" + s
    return s


def device_token(text: Any) -> str:
    """相機型號或標籤名稱 → 檔名裡的 `{DeviceOrTag}`：移除空白與特殊字元（`Zenfone 9` → `Zenfone9`）。

    只保留英數字、底線與連字號，確保這一段不會破壞檔名的欄位分隔。
    """
    return re.sub(r"[^0-9A-Za-z_\-]", "", str(text or ""))[:_DEVICE_MAX]


def _drive(path: str) -> str:
    """Windows 磁碟機代號（或 UNC 的 `\\\\server\\share`），統一大寫。"""
    return os.path.splitdrive(os.path.abspath(path))[0].upper()


@dataclass(slots=True)
class ExportAction:
    """預覽清單裡的一筆動作。同一個 `group` 的動作來自同一張照片，是一個不可分割的單位。"""

    group: int
    photo_id: int
    role: str                            # master | staging | sidecar | person_view | tag_view
    op_type: str                         # copy | move | hardlink | skip
    src: str
    dst: str | None                      # skip 且沒有對應目標時為 None
    bytes_added: int = 0                 # 這筆動作會在匯出磁碟區新增的佔用空間
    note: str = ""                       # 略過或特殊處理的原因
    sha256: bytes | None = None          # 只有主檔有值


@dataclass(slots=True)
class ExportPreview:
    """`preview_export()` 的結果：完整的動作清單與預估佔用空間。尚未動到任何檔案。"""

    export_root: str
    mode: ExportMode
    actions: list[ExportAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimated_bytes: int = 0             # 執行後匯出磁碟區預計新增的佔用空間
    free_bytes: int | None = None        # 匯出磁碟區目前的可用空間（查不到時為 None）

    def summary(self) -> dict[str, int]:
        """各類動作的筆數，鍵為 `role:op_type` 或 `role:skip:原因`。"""
        out: dict[str, int] = {}
        for a in self.actions:
            key = f"{a.role}:{a.op_type}" + (f":{a.note}" if a.op_type == "skip" and a.note else "")
            out[key] = out.get(key, 0) + 1
        return out

    def pending(self) -> list[ExportAction]:
        """實際會執行的動作（不含略過的）。"""
        return [a for a in self.actions if a.op_type != "skip" and a.dst is not None]


class ExportManager(TemplateExportManager):
    """四層多視圖歸檔。也保有父類別的 `plan / execute / rollback / recover`（自訂模板的單層匯出）。"""

    def __init__(self, cfg: AppConfig, db: DBManager, repo: Repository, bus: EventBus | None = None) -> None:
        super().__init__(cfg, db, repo, bus)
        self._heif_ready: bool | None = None

    # ================================================================== #
    # 1. Dry-run
    # ================================================================== #
    def preview_export(
        self,
        export_root: str | os.PathLike[str],
        *,
        mode: ExportMode | str = ExportMode.COPY,
        photo_ids: Iterable[int] | None = None,
        blur_threshold: float | None = None,
        include_views: bool = True,
        cross_volume_fallback: bool = True,
    ) -> ExportPreview:
        """算出完整的匯出動作清單與預估佔用空間。不建立資料夾、不寫資料庫，可以反覆呼叫。

        Args:
            export_root: 匯出根目錄（可以還不存在）。
            mode: 主庫與隔離區的放置方式：COPY、MOVE 或 HARDLINK。
            photo_ids: 只匯出這些照片；None = 資料庫裡所有未遺失的照片。
            blur_threshold: 給定時，`blur_score` 低於此值的照片也視為模糊。None（預設）只採用
                已被標為廢片的照片（`photos.is_rejected = 1`）—— 自動門檻需要先用真實照片校準。
            include_views: False 時只產生主庫與隔離區，不建立人物／標籤視圖。
            cross_volume_fallback: 主模式是 HARDLINK 但來源與匯出根目錄不在同一個磁碟機時，
                True（預設）改用 COPY 放進主庫；False 則略過該照片。

        Raises:
            ExportError: 模式不是 COPY / MOVE / HARDLINK。
        """
        mode = ExportMode(mode)
        if mode is ExportMode.SYMLINK:
            raise ExportError("多視圖歸檔的主模式只支援 copy、move、hardlink")
        # 與掃描器用同一套正規化（還原 8.3 短檔名與實際大小寫）。MOVE 之後 `photos.path` 會指到這底下，
        # 寫法若與下次掃描時不一致，同一個檔案會被當成另一份複本。
        root = normalize_root(export_root)
        preview = ExportPreview(export_root=root, mode=mode)
        root_drive = _drive(root)

        wanted = None if photo_ids is None else {int(p) for p in photo_ids}
        rows = [r for r in self._fetch_photos() if wanted is None or r["id"] in wanted]
        tags = self._fetch_tags()
        persons, face_counts = self._fetch_persons()
        exported = self._previous_exports(root)            # sha256 -> 先前匯出的主檔路徑
        # 目標路徑 -> 當初寫入的 sha256：判斷「同名檔案是不是同一個內容」時不必重讀檔案
        logged = {_key(dst): sha for sha, dst in exported.items()}
        claimed: dict[str, bytes] = {}                     # 本次預覽已佔用的目標路徑 -> sha256
        dir_cache: dict[str, dict[str, list[str]]] = {}

        # ---- 先決定附屬檔歸誰：同主檔名的檔案只能跟一個主照片走 ----
        ordered = sorted(rows, key=self._main_priority)
        db_paths = {_key(r["path"]): r["id"] for r in rows}
        sidecars_of: dict[int, list[str]] = {}
        consumed: set[str] = set()                         # 已被當成附屬檔的路徑
        absorbed: set[int] = set()                         # 本身在資料庫裡、但被當成附屬檔的照片
        for r in ordered:
            if _key(r["path"]) in consumed:
                absorbed.add(r["id"])
                continue
            found = []
            for path in self._find_sidecars(r["path"], dir_cache):
                k = _key(path)
                if k in consumed:
                    continue
                consumed.add(k)
                found.append(path)
                if k in db_paths:
                    absorbed.add(db_paths[k])
            sidecars_of[r["id"]] = found

        cross_fallbacks = 0
        for group, r in enumerate(ordered):
            pid, src = r["id"], r["path"]

            def skip(note: str, role: str = ROLE_MASTER) -> None:
                preview.actions.append(ExportAction(group, pid, role, "skip", src, None, note=note))

            if pid in absorbed:
                skip("exported_as_sidecar")
                continue
            if r["sha256"] is None or r["width"] is None:
                skip("not_processed")                      # 讀檔或解碼失敗的檔案：沒有可靠的指紋
                continue
            if not os.path.isfile(src):
                skip("missing_source")
                continue

            category = self._categorize(r, blur_threshold)
            role = ROLE_MASTER if category == "master" else ROLE_STAGING
            sha, size = bytes(r["sha256"]), int(r["file_size"])
            dt, has_exif_date = self._photo_datetime(r)
            photo_tags = tags.get(pid, [])
            stem_prefix = "_".join((
                dt.strftime("%Y%m%d_%H%M%S"),
                self._subsec(r),
                device_token(r["camera_model"]) or device_token(photo_tags[0] if photo_tags else "") or "Unknown",
            ))
            dir_parts = self._primary_dir(r, category, dt, has_exif_date)
            extras = sidecars_of.get(pid, [])
            longest_ext = max([r["ext"]] + [os.path.splitext(p)[1].lstrip(".") for p in extras], key=len)

            # ---- 主檔的操作方式 ----
            op_type, note = mode.value, ""
            if mode is ExportMode.HARDLINK and _drive(src) != root_drive:
                if not cross_volume_fallback:
                    skip("cross_volume", role)
                    continue
                op_type, note = "copy", "hardlink_fallback_copy"
                cross_fallbacks += 1

            # ---- 目標路徑：先前匯出過就維持原位，否則解決撞名 ----
            previous = exported.get(sha)
            if previous is not None and os.path.isfile(previous) and self._same_layer(previous, root, category):
                primary_dst, primary_note, primary_exists = previous, "already_exported", True
            else:
                resolved = self._resolve_target(root, dir_parts, stem_prefix, sha, size, r["ext"],
                                                longest_ext, claimed, logged)
                if resolved is None:
                    skip("path_too_long", role)
                    continue
                primary_dst, primary_note, primary_exists = resolved
            if _key(primary_dst) == _key(src):
                primary_note, primary_exists = "already_in_place", True

            if primary_note == "duplicate_content":
                # 同一批裡內容完全相同的另一份複本：主庫只留一份
                skip("duplicate_content", role)
                continue
            if primary_exists:
                preview.actions.append(ExportAction(group, pid, role, "skip", src, primary_dst,
                                                    note=primary_note, sha256=sha))
            else:
                added = size if op_type == "copy" or (op_type == "move" and _drive(src) != root_drive) else 0
                preview.actions.append(ExportAction(group, pid, role, op_type, src, primary_dst, added,
                                                    note or primary_note, sha))

            # ---- Sidecar：沿用主檔的新檔名主幹 ----
            dst_dir = os.path.dirname(primary_dst)
            new_stem = os.path.splitext(os.path.basename(primary_dst))[0]
            for side in extras:
                side_dst = os.path.join(dst_dir, new_stem + os.path.splitext(side)[1].lower())
                side_size = _size_of(side)
                if _key(side_dst) == _key(side) or os.path.lexists(side_dst):
                    preview.actions.append(ExportAction(group, pid, ROLE_SIDECAR, "skip", side, side_dst,
                                                        note="already_exported"))
                    continue
                claimed[_key(side_dst)] = sha
                added = side_size if op_type == "copy" or (op_type == "move" and _drive(side) != root_drive) else 0
                preview.actions.append(ExportAction(group, pid, ROLE_SIDECAR, op_type, side, side_dst, added))

            # ---- 視圖：只有主庫的照片有，一律從主庫的新檔案硬連結 ----
            if category == "master" and include_views:
                file_name = os.path.basename(primary_dst)
                for view_role, parts in self._view_dirs(r, dt, persons.get(pid, []), face_counts.get(pid, 0),
                                                         photo_tags):
                    view_dst = self._fit_path(root, parts, file_name)
                    if view_dst is None:
                        preview.actions.append(ExportAction(group, pid, view_role, "skip", primary_dst, None,
                                                            note="path_too_long"))
                        continue
                    k = _key(view_dst)
                    if k in claimed:
                        continue                           # 兩個標籤清理後落在同一個資料夾：只連一次
                    if os.path.lexists(view_dst):
                        same = primary_exists and _samefile(primary_dst, view_dst)
                        preview.actions.append(ExportAction(
                            group, pid, view_role, "skip", primary_dst, view_dst,
                            note="already_linked" if same else "view_conflict"))
                        continue
                    claimed[k] = sha
                    preview.actions.append(ExportAction(group, pid, view_role, "hardlink", primary_dst, view_dst))

        preview.estimated_bytes = sum(a.bytes_added for a in preview.actions)
        preview.free_bytes = _free_space(root)
        if cross_fallbacks:
            preview.warnings.append(
                f"{cross_fallbacks} 張照片與匯出根目錄不在同一個磁碟機，無法硬連結，主庫改用複製（會佔用空間）")
        if preview.free_bytes is not None and preview.estimated_bytes > preview.free_bytes:
            preview.warnings.append(
                f"空間不足：預估需要 {preview.estimated_bytes / 2**30:.2f} GB，"
                f"匯出磁碟區只剩 {preview.free_bytes / 2**30:.2f} GB")
        conflicts = sum(1 for a in preview.actions if a.note == "view_conflict")
        if conflicts:
            preview.warnings.append(f"{conflicts} 個視圖位置已有不是本工具建立的同名檔案，已略過（不覆寫）")
        too_long = sum(1 for a in preview.actions if a.note == "path_too_long")
        if too_long:
            preview.warnings.append(f"{too_long} 筆的完整路徑會超過 {MAX_PATH_LEN} 字元，已略過；請改用較短的匯出根目錄")
        return preview

    # ---- 預覽用的資料載入（整批查詢，不逐張查）----
    def _fetch_photos(self) -> list[sqlite3.Row]:
        return self._db.query(
            "SELECT p.id, p.path, p.file_name, p.ext, p.file_size, p.sha256, p.mtime_ns, p.width, p.taken_at, "
            "p.taken_src, p.camera_model, p.city, p.is_screenshot, p.is_rejected, p.reject_reason, p.blur_score, "
            "p.is_burst_duplicate, p.event_id, e.name AS event_name, e.start_ts AS event_start_ts, "
            "e.location_label AS event_location "
            "FROM photos p LEFT JOIN events e ON e.id = p.event_id WHERE p.is_missing = 0 ORDER BY p.id"
        )

    def _fetch_tags(self) -> dict[int, list[str]]:
        """photo_id -> 標籤名稱，分數最高的在前（手動標籤優先）。不含被使用者否決的。"""
        out: dict[int, list[str]] = {}
        for r in self._db.iter_query(
            "SELECT pt.photo_id, t.name FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            "WHERE pt.is_rejected = 0 ORDER BY pt.photo_id, (pt.source = 'manual') DESC, pt.score DESC, t.name"
        ):
            out.setdefault(r["photo_id"], []).append(r["name"])
        return out

    def _fetch_persons(self) -> tuple[dict[int, list[str]], dict[int, int]]:
        """回傳 (photo_id -> 畫面中已知人物的顯示名稱, photo_id -> 臉的總數)。隱藏的人物不列入。"""
        persons: dict[int, list[str]] = {}
        for r in self._db.iter_query(
            "SELECT DISTINCT f.photo_id, pe.id, pe.name FROM faces f JOIN persons pe ON pe.id = f.person_id "
            "WHERE f.is_ignored = 0 AND pe.is_hidden = 0 ORDER BY f.photo_id, pe.id"
        ):
            name = person_display_name(r["id"], r["name"])
            names = persons.setdefault(r["photo_id"], [])
            if name not in names:
                names.append(name)
        counts = {r[0]: r[1] for r in self._db.query(
            "SELECT photo_id, COUNT(*) FROM faces WHERE is_ignored = 0 GROUP BY photo_id")}
        return persons, counts

    def _previous_exports(self, root: str) -> dict[bytes, str]:
        """這個匯出根目錄先前成功匯出過的主檔：sha256 -> 當時的目標路徑（同一檔案取最近一次）。

        有了它，重複匯出時照片會留在第一次放的位置 —— 即使之後事件被重新命名、資料夾名稱變了，
        也不會在主庫裡多出一份。
        """
        out: dict[bytes, str] = {}
        for r in self._db.iter_query(
            "SELECT l.sha256, l.dst_path FROM export_logs l JOIN export_batches b ON b.id = l.batch_id "
            "WHERE b.template = ? AND b.dest_root = ? AND l.state = 'done' AND l.sha256 IS NOT NULL ORDER BY l.id",
            (LAYOUT_ID, root),
        ):
            out[bytes(r["sha256"])] = r["dst_path"]
        return out

    # ---- 分類與命名 ----
    @staticmethod
    def _main_priority(r: sqlite3.Row) -> tuple:
        """同主檔名有多個候選時誰當主照片：HEIC（Live Photo）最優先，其次一般影像，RAW 最後。"""
        d, name = os.path.split(r["path"])
        ext = r["ext"]
        rank = 0 if ext in _HEIF_EXTS else 2 if ext in _RAW_EXTS else 1
        return (_key(d), os.path.splitext(name)[0].lower(), rank, name.lower())

    @staticmethod
    def _find_sidecars(path: str, cache: dict[str, dict[str, list[str]]]) -> list[str]:
        """同資料夾、同主檔名、副檔名在 `SIDECAR_EXTS` 裡的檔案（不含主照片自己）。每個資料夾只列一次。"""
        d, name = os.path.split(path)
        listing = cache.get(d)
        if listing is None:
            listing = {}
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            stem, ext = os.path.splitext(entry.name)
                            if ext.lower() in SIDECAR_EXTS:
                                listing.setdefault(stem.lower(), []).append(entry.path)
            except OSError:
                pass
            cache[d] = listing
        stem = os.path.splitext(name)[0].lower()
        return sorted(p for p in listing.get(stem, ()) if _key(p) != _key(path))

    @staticmethod
    def _categorize(r: sqlite3.Row, blur_threshold: float | None) -> str:
        """回傳 'screenshot' / 'blurry' / 'burst' / 'master'。同時符合多項時依這個順序取第一個。"""
        if r["is_screenshot"]:
            return "screenshot"
        if r["is_rejected"]:
            return "burst" if r["reject_reason"] == "duplicate" else "blurry"
        if blur_threshold is not None and r["blur_score"] is not None and r["blur_score"] < blur_threshold:
            return "blurry"
        if r["is_burst_duplicate"]:
            return "burst"
        return "master"

    @staticmethod
    def _photo_datetime(r: sqlite3.Row) -> tuple[datetime, bool]:
        """照片的日期時間與「是否來自 EXIF」。

        `photos.taken_at` 在 Stage 1 已依 EXIF → 檔名裡的日期時間 → mtime 的順序決定好了。
        """
        if r["taken_at"]:
            try:
                return datetime.strptime(r["taken_at"], "%Y-%m-%dT%H:%M:%S"), r["taken_src"] == "exif"
            except ValueError:
                pass
        return datetime.fromtimestamp(r["mtime_ns"] / 1e9), False

    def _subsec(self, r: sqlite3.Row) -> str:
        """EXIF SubSecTimeOriginal 的前 3 碼（毫秒）；沒有就是 `000`。只有時間來自 EXIF 的照片才去讀檔頭。"""
        if r["taken_src"] != "exif":
            return "000"
        try:
            from PIL import Image

            if r["ext"] in _HEIF_EXTS and self._heif_ready is None:
                try:
                    import pillow_heif

                    pillow_heif.register_heif_opener()
                    self._heif_ready = True
                except Exception:  # noqa: BLE001
                    self._heif_ready = False
            with Image.open(r["path"]) as img:
                value = img.getexif().get_ifd(_EXIF_IFD).get(_EXIF_SUBSEC_ORIGINAL)
        except Exception:  # noqa: BLE001 - 讀不到 EXIF 不該讓匯出失敗
            return "000"
        digits = re.sub(r"\D", "", str(value or ""))
        return (digits + "000")[:3]

    def _primary_dir(self, r: sqlite3.Row, category: str, dt: datetime, has_exif_date: bool) -> list[str]:
        """主檔要放進哪個資料夾（相對於匯出根目錄的各層名稱）。"""
        if category == "screenshot":
            return [STAGING_DIR, STAGING_SCREENSHOT]
        if category == "blurry":
            return [STAGING_DIR, STAGING_BLURRY]
        if category == "burst":
            return [STAGING_DIR, STAGING_BURST]
        if not has_exif_date:
            return [MASTER_DIR, NO_DATE_DIR, str(datetime.fromtimestamp(r["mtime_ns"] / 1e9).year)]

        if r["event_id"] is not None and r["event_start_ts"] is not None:
            # 同一個事件的照片放在同一個資料夾，日期取事件的起始日（跨日的旅行不會被拆開）
            start = time.gmtime(r["event_start_ts"])
            year, date = f"{start.tm_year:04d}", time.strftime("%Y-%m-%d", start)
            name = (r["event_name"] or "").strip()
            if _DATE_PREFIX.match(name):
                folder = name                                # 自動產生的預設名稱本身就是 日期_城市_標籤
            else:
                city = (r["event_location"] or "").split(",")[0].strip() or (r["city"] or "")
                folder = "_".join(p for p in (date, city, name) if p)
        else:
            year = f"{dt.year:04d}"
            folder = "_".join(p for p in (dt.strftime("%Y-%m-%d"), r["city"] or "") if p)
        return [MASTER_DIR, year, clean_name(folder)]

    @staticmethod
    def _view_dirs(r: sqlite3.Row, dt: datetime, persons: list[str], face_count: int, tags: list[str]
                   ) -> list[tuple[str, list[str]]]:
        """這張照片要出現在哪些視圖資料夾。"""
        out: list[tuple[str, list[str]]] = []
        for name in persons:
            # 獨照：畫面裡只有一張臉，而且就是這個人
            sub = SOLO_DIR if face_count == 1 and len(persons) == 1 else f"{dt.year:04d}"
            out.append((ROLE_PERSON_VIEW, [PERSON_DIR, clean_name(name), sub]))
        if len(persons) >= 2:
            out.append((ROLE_PERSON_VIEW, [PERSON_DIR, GROUP_DIR]))
        for tag in tags:
            out.append((ROLE_TAG_VIEW, [TAG_DIR, clean_name(tag)]))
        return out

    @staticmethod
    def _same_layer(previous_dst: str, root: str, category: str) -> bool:
        """先前匯出的位置與現在的分類是否在同一層（主庫 vs 隔離區）。分類改變了就要重新放。"""
        layer = MASTER_DIR if category == "master" else STAGING_DIR
        return _key(previous_dst).startswith(_key(os.path.join(root, layer)) + os.sep)

    @staticmethod
    def _fit_path(root: str, parts: list[str], file_name: str) -> str | None:
        """組出完整路徑並確保長度 < `MAX_PATH_LEN`。

        過長時從最長的「動態」資料夾名稱（事件、人物、標籤）開始縮短；固定的結構名稱
        （`01_Timeline_Master`、`Solo_Portraits`…）與檔名不動。縮不下去回傳 None。
        """
        parts = list(parts)
        while True:
            full = os.path.join(root, *parts, file_name)
            excess = len(full) - (MAX_PATH_LEN - 1)
            if excess <= 0:
                return full
            shrinkable = [k for k, p in enumerate(parts) if p not in _FIXED_NAMES and len(p) > _MIN_DIR_LEN]
            if not shrinkable:
                return None
            i = max(shrinkable, key=lambda k: len(parts[k]))
            keep = max(_MIN_DIR_LEN, len(parts[i]) - excess)
            parts[i] = parts[i][:keep].rstrip(" ._") or parts[i][:_MIN_DIR_LEN]

    def _resolve_target(
        self, root: str, dir_parts: list[str], stem_prefix: str, sha: bytes, size: int, ext: str,
        longest_ext: str, claimed: dict[str, bytes], logged: dict[str, bytes],
    ) -> tuple[str, str, bool] | None:
        """決定主檔的最終路徑。回傳 (路徑, 說明, 是否已存在而不需要動作)；路徑過長回傳 None。

        同名（含相同的 6 碼 ShortHash）時：
          * 完整 SHA-256 也相同 → 這就是先前匯出的同一個檔案，略過（冪等）。
          * 完整 SHA-256 不同 → 真正的撞碼，在 ShortHash 後面加 `_v2`、`_v3`…
        """
        short = sha.hex()[:6]
        for version in range(1, 1000):
            stem = f"{stem_prefix}_{short}" + ("" if version == 1 else f"_v{version}")
            # 以群組裡最長的副檔名檢查長度，Sidecar 才不會比主檔多出幾個字元而超長
            probe = self._fit_path(root, dir_parts, f"{stem}.{longest_ext}")
            if probe is None:
                return None
            dst = os.path.join(os.path.dirname(probe), f"{stem}.{ext}")
            key = _key(dst)
            if key in claimed:
                if claimed[key] == sha:
                    return dst, "duplicate_content", True
                continue
            if os.path.lexists(dst):
                if self._is_same_file(dst, key, sha, size, logged):
                    return dst, "already_exported", True
                continue
            claimed[key] = sha
            return dst, ("" if version == 1 else f"hash_collision_v{version}"), False
        return None

    @staticmethod
    def _is_same_file(dst: str, key: str, sha: bytes, size: int, logged: dict[str, bytes]) -> bool:
        """目標位置已存在的檔案是不是同一個內容。先查匯出日誌；日誌沒有紀錄才比大小與 SHA-256。"""
        known = logged.get(key)
        if known is not None:
            return known == sha
        try:
            return os.path.getsize(dst) == size and sha256_file(dst) == sha
        except OSError:
            return False

    # ================================================================== #
    # 2. 執行
    # ================================================================== #
    def execute_export(self, preview: ExportPreview, *,
                       should_stop: Callable[[], bool] = lambda: False) -> ExportResult:
        """照預覽執行。同步（阻塞到做完）；GUI 請放在背景執行緒，進度走 EventBus。

        每張照片的「主檔 + 所有 Sidecar」是一個不可分割的群組：任何一筆失敗，該群組已完成的部分
        會立刻還原，整組記為失敗，不會留下只有照片沒有 Live Photo 影片的半套結果。
        視圖（硬連結）失敗只影響那一個連結，主檔照常保留。

        Raises:
            ExportError: 主模式是 HARDLINK，但匯出磁碟區的檔案系統不支援硬連結（例如 exFAT）。
        """
        root = preview.export_root
        actions = preview.pending()
        for layer in LAYER_DIRS:
            os.makedirs(os.path.join(root, layer), exist_ok=True)

        if any(a.op_type == "hardlink" for a in actions) and not _supports_hardlink(root):
            if any(a.op_type == "hardlink" and a.role not in _VIEW_ROLES for a in actions):
                raise ExportError(f"{root} 所在的檔案系統不支援硬連結，無法使用 hardlink 模式")
            log.warning("%s 所在的檔案系統不支援硬連結，略過人物與標籤視圖", root)
            actions = [a for a in actions if a.role not in _VIEW_ROLES]

        batch_id = self._db.execute(
            "INSERT INTO export_batches(mode, dest_root, template, conflict_policy, total) VALUES (?, ?, ?, ?, ?)",
            (preview.mode.value, root, LAYOUT_ID, "shorthash", len(actions)),
        ).result()
        result = ExportResult(batch_id=batch_id, skipped=len(preview.actions) - len(actions))

        groups: dict[int, list[ExportAction]] = {}
        for a in actions:
            groups.setdefault(a.group, []).append(a)

        seq = 0
        for index, (_group, items) in enumerate(sorted(groups.items()), 1):
            if should_stop():
                result.cancelled = True
                break
            critical = [a for a in items if a.role not in _VIEW_ROLES]       # 主檔 + Sidecar
            views = [a for a in items if a.role in _VIEW_ROLES]
            ordered = critical + views
            log_ids = self._log_pending(batch_id, seq, ordered)              # 已 commit，之後才動檔案
            seq += len(ordered)
            outcome: dict[int, str | None] = {}                              # log id -> None（成功）或錯誤訊息

            # ---- 不可分割的部分 ----
            performed: list[ExportAction] = []
            failure: str | None = None
            for a in critical:
                try:
                    self._perform(a.op_type, a.src, a.dst)                   # type: ignore[arg-type]
                    performed.append(a)
                except Exception as e:  # noqa: BLE001
                    failure = f"{type(e).__name__}: {e}"
                    log.warning("匯出失敗 %s -> %s: %s", a.src, a.dst, failure)
                    result.errors.append((a.src, failure))
                    break
            if failure is not None:
                self._revert(performed)
                for a, log_id in zip(ordered, log_ids):
                    outcome[log_id] = failure if a.role not in _VIEW_ROLES else "所屬群組匯出失敗"
                self._log_outcome(outcome, [])
                result.failed += len(ordered)
                self._publish_progress(batch_id, index, len(groups), items)
                continue

            # ---- 視圖：各自獨立 ----
            for a, log_id in zip(ordered, log_ids):
                if a.role not in _VIEW_ROLES:
                    outcome[log_id] = None
                    continue
                try:
                    self._perform(a.op_type, a.src, a.dst)                   # type: ignore[arg-type]
                    outcome[log_id] = None
                except Exception as e:  # noqa: BLE001
                    outcome[log_id] = f"{type(e).__name__}: {e}"
                    result.errors.append((a.src, outcome[log_id]))           # type: ignore[arg-type]

            moved = [(a.photo_id, a.dst) for a in critical if a.op_type == "move" and a.role in _PRIMARY_ROLES]
            try:
                self._log_outcome(outcome, moved)                            # type: ignore[arg-type]
            except sqlite3.IntegrityError as e:
                # 新路徑在資料庫裡已被另一列佔用：把檔案放回去，維持資料庫與硬碟一致
                self._revert([a for a in views if outcome.get(log_ids[ordered.index(a)]) is None] + performed)
                message = f"資料庫中已有相同路徑的紀錄，已還原：{e}"
                self._log_outcome({log_id: message for log_id in log_ids}, [])
                result.failed += len(ordered)
                result.errors.append((critical[0].src, message))
                self._publish_progress(batch_id, index, len(groups), items)
                continue
            result.done += sum(1 for v in outcome.values() if v is None)
            result.failed += sum(1 for v in outcome.values() if v is not None)
            self._publish_progress(batch_id, index, len(groups), items)

        self._db.execute(
            "UPDATE export_batches SET state = ?, finished_at = unixepoch() WHERE id = ?",
            ("cancelled" if result.cancelled else "done", batch_id),
        ).result()
        if self._bus is not None:
            self._bus.publish(EventKind.EXPORT_DONE, batch_id=batch_id, done=result.done,
                              skipped=result.skipped, failed=result.failed, cancelled=result.cancelled)
        return result

    def _log_pending(self, batch_id: int, first_seq: int, actions: list[ExportAction]) -> list[int]:
        """把一個群組的所有操作以 `pending` 寫入日誌（單一交易），回傳各筆的 log id。"""
        rows = [
            (batch_id, first_seq + i, a.photo_id if a.role in _PRIMARY_ROLES else None, a.op_type, a.src, a.dst,
             _size_of(a.src), a.sha256 if a.role in _PRIMARY_ROLES else None)
            for i, a in enumerate(actions)
        ]

        def op(conn: sqlite3.Connection) -> list[int]:
            return [
                int(conn.execute(
                    "INSERT INTO export_logs(batch_id, seq, photo_id, action, src_path, dst_path, file_size, sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row).lastrowid)
                for row in rows
            ]

        return self._db.write(op)

    def _log_outcome(self, outcome: dict[int, str | None], moved: list[tuple[int, str]]) -> None:
        """把一個群組的結果寫回日誌；MOVE 的主檔同時更新 `photos.path`（同一個交易）。"""

        def op(conn: sqlite3.Connection) -> None:
            for log_id, error in outcome.items():
                if error is None:
                    conn.execute(
                        "UPDATE export_logs SET state = 'done', executed_at = unixepoch() WHERE id = ?", (log_id,))
                else:
                    conn.execute(
                        "UPDATE export_logs SET state = 'failed', error = ?, executed_at = unixepoch() WHERE id = ?",
                        (error[:2000], log_id))
            for photo_id, dst in moved:
                _set_photo_path(conn, photo_id, dst)

        self._db.write(op)

    @staticmethod
    def _revert(performed: list[ExportAction]) -> None:
        """把同一個群組裡已經做完的操作倒著還原（群組中途失敗時用）。"""
        for a in reversed(performed):
            try:
                assert a.dst is not None
                if a.op_type == "move":
                    os.makedirs(os.path.dirname(a.src), exist_ok=True)
                    shutil.move(a.dst, a.src)
                elif os.path.lexists(a.dst):
                    os.remove(a.dst)
            except OSError:
                log.exception("還原 %s 失敗", a.dst)

    def _publish_progress(self, batch_id: int, index: int, total: int, items: list[ExportAction]) -> None:
        if self._bus is not None:
            self._bus.publish(EventKind.EXPORT_PROGRESS, batch_id=batch_id, index=index, total=total,
                              src=items[0].src, dst=items[0].dst)

    # ================================================================== #
    # 3. 回復
    # ================================================================== #
    def rollback_batch(self, batch_id: int) -> ExportResult:
        """把一個批次完整還原。

        依日誌的逆序處理：先刪視圖的硬連結，再處理 Sidecar，最後是主檔 ——
        COPY 與 HARDLINK 建立的檔案刪除，MOVE 的檔案搬回 `src_path`（並把 `photos.path` 改回去）。
        最後清掉因此變空的資料夾，包含四個頂層資料夾。

        不會強行處理的情況（該筆標為 `rollback_failed` 並保留檔案）：目標檔已被修改或換成別的檔案、
        MOVE 的原始位置已被其他檔案佔用。回傳的 `done` 是成功還原的筆數，`failed` 是無法還原的筆數。
        """
        result = self.rollback(batch_id)
        root = self._db.query_value("SELECT dest_root FROM export_batches WHERE id = ?", (batch_id,))
        if root:
            # 由下往上清掉四層結構裡所有空資料夾（含頂層）。os.rmdir 只刪得掉空的，有檔案的不受影響。
            for layer in LAYER_DIRS:
                for dirpath, _dirs, _files in os.walk(os.path.join(root, layer), topdown=False):
                    try:
                        os.rmdir(dirpath)
                    except OSError:
                        pass
        return result


# --------------------------------------------------------------------------- #
def _size_of(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _samefile(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _free_space(path: str) -> int | None:
    """路徑所在磁碟區的可用空間；路徑還不存在時往上找到存在的那一層。"""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


@dataclass(slots=True)
class ArchiveReport:
    """`verify_archive()` 的結果：四層結構實際上佔了多少空間、視圖是不是真的硬連結。"""

    export_root: str
    master_files: int = 0                # 01_Timeline_Master 裡的檔案數（含 Sidecar）
    staging_files: int = 0               # 99_Review_Staging 裡的檔案數
    view_links: int = 0                  # 02 / 03 視圖裡的檔案數
    broken_views: list[str] = field(default_factory=list)    # 不是主庫檔案硬連結的視圖檔（實體複本或孤兒）
    apparent_bytes: int = 0              # 把每個路徑的大小都加起來（檔案總管顯示的就是這個）
    physical_bytes: int = 0              # 實際佔用：同一份資料的多個硬連結只算一次

    @property
    def saved_bytes(self) -> int:
        """硬連結省下的空間。"""
        return self.apparent_bytes - self.physical_bytes

    @property
    def ok(self) -> bool:
        return not self.broken_views


def verify_archive(export_root: str | os.PathLike[str]) -> ArchiveReport:
    """檢查一個四層歸檔：人物與標籤視圖的每個檔案，是否都是主庫裡某個檔案的硬連結。

    以檔案系統的檔案識別碼（磁碟區 + 檔案索引）判斷兩個路徑是不是同一份資料，不比對內容，所以很快。
    只讀取、不修改任何東西。

    為什麼不能用檔案總管看：檔案總管的「大小」與「磁碟大小」都會把每個硬連結各算一次，
    所以整個匯出資料夾看起來會是主庫的好幾倍；真正的佔用是這裡的 `physical_bytes`。
    """
    root = normalize_root(export_root)
    report = ArchiveReport(export_root=root)
    physical: dict[tuple[int, int], int] = {}          # 檔案識別碼 -> 大小
    master_ids: set[tuple[int, int]] = set()

    def walk(layer: str):
        for dirpath, _dirs, files in os.walk(os.path.join(root, layer)):
            for name in files:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                yield path, (st.st_dev, st.st_ino), st.st_size

    for layer in (MASTER_DIR, STAGING_DIR):
        for _path, file_id, size in walk(layer):
            physical[file_id] = size
            report.apparent_bytes += size
            if layer == MASTER_DIR:
                master_ids.add(file_id)
                report.master_files += 1
            else:
                report.staging_files += 1
    for layer in (PERSON_DIR, TAG_DIR):
        for path, file_id, size in walk(layer):
            report.view_links += 1
            report.apparent_bytes += size
            physical[file_id] = size
            if file_id not in master_ids:
                report.broken_views.append(path)
    report.physical_bytes = sum(physical.values())
    return report


def _supports_hardlink(directory: str) -> bool:
    """實測這個資料夾所在的檔案系統能不能建立硬連結（NTFS 可以，exFAT / FAT32 不行）。"""
    try:
        with tempfile.TemporaryDirectory(dir=directory, prefix=".linkprobe-") as d:
            target = os.path.join(d, "t")
            with open(target, "wb"):
                pass
            os.link(target, os.path.join(d, "l"))
        return True
    except OSError:
        return False
