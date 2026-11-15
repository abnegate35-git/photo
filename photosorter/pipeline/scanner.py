"""Stage 0：走訪資料夾，與資料庫比對，只挑出「需要處理」的檔案。

對每個檔案只做一次 `stat`（Windows 上 `os.scandir` 已在列目錄時帶回大小與修改時間，
不需要額外的系統呼叫），資料庫端則是一次把指紋載入記憶體（十萬張實測約 0.3 秒）再逐一比對。
所以「什麼都沒變」的重掃成本幾乎只剩列目錄：

    路徑不在資料庫                         -> NEW
    路徑在，但 size 或 mtime 不同          -> CHANGED
    路徑在、沒變，狀態位元還欠             -> INCOMPLETE（中斷續掃就是靠這個）
    路徑在、沒變、上次失敗                 -> 略過（`scan.retry_errors = true` 時改為 INCOMPLETE）
    路徑在、沒變、全部做完                 -> 略過

「路徑改變但內容相同」的搬移判定需要 SHA-256，那要讀檔，所以放在解碼執行緒裡做
（`PipelineEngine._decode_job`），這裡只把它列為 NEW。

資料庫裡有、但這次沒看到的路徑會回報為 missing（由引擎標記，不刪除紀錄）。
**根目錄本身不存在或無法存取時（外接硬碟沒插、網路磁碟斷線）不會回報 missing** ——
否則拔掉硬碟重掃一次，整個照片庫都會被標成遺失。
"""
from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.config import ScanConfig
from ..core.status import ProcessStatus
from ..core.types import CandidateKind, FileCandidate
from ..db.repository import Repository

log = logging.getLogger(__name__)

_ERR = ProcessStatus.error_bits()


@dataclass(slots=True)
class ScanResult:
    candidates: list[FileCandidate] = field(default_factory=list)
    total_files: int = 0                       # 看到的影像檔總數
    unchanged: int = 0                         # 沒變且已完成，直接略過
    error_skipped: int = 0                     # 上次失敗、這次不重試
    missing: list[tuple[int, str]] = field(default_factory=list)   # 資料庫有、硬碟上已不在：(photo_id, path)
    reappeared_ids: list[int] = field(default_factory=list)    # 之前標為遺失、現在又出現
    unreachable_roots: list[str] = field(default_factory=list)
    stopped: bool = False


def normalize_root(root: str | os.PathLike[str]) -> str:
    """根目錄正規化成絕對路徑。`realpath` 在 Windows 上會還原實際的大小寫，
    避免 `d:\\photos` 與 `D:\\Photos` 被當成兩個不同的路徑而重複入庫。"""
    return os.path.normpath(os.path.realpath(os.fspath(root)))


class LibraryScanner:
    def __init__(self, cfg: ScanConfig, repo: Repository) -> None:
        self._cfg = cfg
        self._repo = repo
        exts = {e.lower().lstrip(".") for e in cfg.extensions}
        if cfg.enable_raw:
            exts |= {e.lower().lstrip(".") for e in cfg.raw_extensions}
        self._exts = exts
        self._exclude = {n.lower() for n in cfg.exclude_dir_names}

    def scan(
        self,
        roots: Sequence[str],
        required: ProcessStatus,
        *,
        should_stop: Callable[[], bool] = lambda: False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> ScanResult:
        """
        Args:
            roots: 已正規化的根目錄（`normalize_root`）。
            required: 這次執行要求完成的位元（`core.status.required_mask`）。
            should_stop: 回傳 True 時盡快結束；結果的 `stopped` 會是 True，且不回報 missing。
            on_progress: `fn(files_seen, to_process, current_dir)`，每進入一個資料夾呼叫一次。
        """
        result = ScanResult()
        reachable = []
        for r in roots:
            if os.path.isdir(r):
                reachable.append(r)
            else:
                log.warning("根目錄不存在或無法存取，略過：%s", r)
                result.unreachable_roots.append(r)

        index = self._repo.load_scan_index(reachable)
        seen: set[str] = set()
        retry = self._cfg.retry_errors

        for root in reachable:
            stack = [root]
            while stack:
                if should_stop():
                    result.stopped = True
                    return result
                current = stack.pop()
                if on_progress is not None:
                    on_progress(result.total_files, len(result.candidates), current)
                try:
                    entries = list(os.scandir(current))
                except OSError as e:
                    log.warning("無法讀取資料夾 %s: %s", current, e)
                    # 這個子樹讀不到：底下的檔案狀態未知，不能當成遺失
                    prefix = current.rstrip("\\/") + os.sep
                    seen.update(p for p in index if p.startswith(prefix))
                    continue
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=self._cfg.follow_symlinks):
                            if not self._skip_dir(entry):
                                stack.append(entry.path)
                            continue
                        if not entry.is_file(follow_symlinks=self._cfg.follow_symlinks):
                            continue
                        name = entry.name
                        ext = os.path.splitext(name)[1].lstrip(".").lower()
                        if ext not in self._exts:
                            continue
                        st = entry.stat(follow_symlinks=self._cfg.follow_symlinks)
                    except OSError as e:
                        log.debug("略過無法存取的項目 %s: %s", entry.path, e)
                        seen.add(entry.path)
                        continue
                    if self._cfg.skip_hidden and (name.startswith(".") or _is_hidden(st)):
                        seen.add(entry.path)
                        continue
                    if st.st_size < self._cfg.min_file_size:
                        seen.add(entry.path)
                        continue

                    path = entry.path
                    seen.add(path)
                    result.total_files += 1
                    known = index.get(path)
                    if known is None:
                        result.candidates.append(
                            FileCandidate(path, st.st_size, st.st_mtime_ns, CandidateKind.NEW))
                        continue
                    if known.is_missing:
                        result.reappeared_ids.append(known.photo_id)
                    if known.size != st.st_size or known.mtime_ns != st.st_mtime_ns:
                        kind = CandidateKind.CHANGED
                    elif (known.status & _ERR) and not retry:
                        result.error_skipped += 1
                        continue
                    elif (known.status & required) != required or (known.status & _ERR):
                        kind = CandidateKind.INCOMPLETE
                    else:
                        result.unchanged += 1
                        continue
                    result.candidates.append(FileCandidate(
                        path, st.st_size, st.st_mtime_ns, kind,
                        photo_id=known.photo_id, status=known.status, known_sha256=known.sha256,
                    ))

        # 這裡只是「候選」：其中有些其實是搬移，要等解碼執行緒比對 SHA-256 後才知道。
        # 引擎在所有檔案處理完之後才呼叫 mark_missing()，屆時已搬移的列路徑已變，不會被標記。
        result.missing = [(e.photo_id, p) for p, e in index.items() if p not in seen and not e.is_missing]
        # 依路徑排序：同資料夾的檔案連續處理，傳統硬碟的磁頭移動最少
        result.candidates.sort(key=lambda c: c.path)
        return result

    def _skip_dir(self, entry: os.DirEntry[str]) -> bool:
        name = entry.name
        if name.lower() in self._exclude:
            return True
        if self._cfg.skip_hidden:
            if name.startswith("."):
                return True
            try:
                return _is_hidden(entry.stat(follow_symlinks=False))
            except OSError:
                return True
        return False


def _is_hidden(st: os.stat_result) -> bool:
    """Windows 的隱藏/系統屬性。其他平台沒有這個欄位，回傳 False。"""
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))
