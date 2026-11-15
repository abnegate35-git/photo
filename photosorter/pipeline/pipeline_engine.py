"""多階段處理管線的調度引擎（Headless Core 的主體）。

執行緒模型
----------
::

    pipeline（調度執行緒，start() 建立）
      │  1. Stage 0 掃描（LibraryScanner）─ 與此同時 ─▶ inference 執行緒載入模型
      │  2. 依實際生效的 provider 決定 RuntimePlan（解碼執行緒數、batch 大小）
      │  3. 把 candidates 派給解碼執行緒池（以 semaphore 限制在途數量）
      ▼
    decode-N（ThreadPoolExecutor，生產者，N 條）
      │  讀檔 → SHA-256 → 搬移/重複判定 → 解碼一次 → 切分支 → Stage 1 寫入資料庫
      ▼
    [ queue.Queue(maxsize) ]  ←── 有界：滿了生產者就阻塞，記憶體用量有上限
      ▼
    inference（消費者，**唯一**一條）
      │  動態 batching（湊滿 batch_size 或等超過 batch_timeout）
      │  Stage 2：SigLIP vision → 影像向量 → 多標籤
      │  Stage 3：條件式人臉（文件/收據/截圖且無人物語意 → 略過）
      │  整個 batch 的結果 + 狀態位元 → 同一個資料庫交易
      ▼
    db-writer（DBManager 內部，唯一的寫入連線）

    全部結束後，pipeline 執行緒依序跑全域後處理：離線地理編碼 → 事件聚類 →（可掛載的其他步驟）

只有一條推論執行緒是刻意的：GPU / DirectML 上多條執行緒同時 `run()` 只會互相搶資源、
切換 context，總吞吐量反而下降；平行度交給 batch。

背壓鏈：推論慢 → 佇列滿 → 解碼執行緒卡在 put → semaphore 用完 → 調度執行緒停止派工。
任何一環變慢，上游都會自動等待，不會無限堆積記憶體。

中斷續掃（Checkpoint Resume）
-----------------------------
沒有獨立的 checkpoint 檔。每個階段的資料與 `photos.process_status` 的位元在同一個交易寫入，
所以不論是 `stop()`、當機還是斷電，重啟後 Stage 0 讀到的狀態一定與資料一致，
`required & ~status` 就是每張照片還欠的階段，從那裡接著做。

暫停 / 停止
-----------
  * `pause()`：調度與解碼執行緒在下一個檢查點停住；推論執行緒把手上的 batch 做完後停住。
    模型留在記憶體，`resume()` 立即繼續。
  * `stop()`：同樣在檢查點離開。已 commit 的結果全部保留，佇列裡還沒推論的照片丟棄
    （它們的 Stage 1 已存檔，下次續掃只補 AI 階段）。`stop()` 不會中途打斷一次 ONNX 推論。

所有對外通知都走 `EventBus`；本模組不 import 任何 GUI。
"""
from __future__ import annotations

import enum
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

from ..core.config import AppConfig
from ..core.events import EventBus, EventKind, Throttle
from ..core.hardware import CPU, PROVIDER_PRIORITY, plan_runtime, probe_hardware
from ..core.interfaces import (
    EventClusterer,
    FaceAnalyzer,
    ImageDecoder,
    ReverseGeocoder,
    TagScorer,
    VisionEmbedder,
)
from ..core.status import ProcessStatus, required_mask
from ..core.types import (
    CandidateKind,
    DecodedImageTask,
    FileCandidate,
    GateRole,
    InferenceResult,
    ModelInfo,
    RuntimePlan,
)
from ..db.db_manager import DBManager
from ..db.repository import Repository
from ..models.preprocess import stack_images
from .scanner import LibraryScanner, ScanResult, normalize_root

log = logging.getLogger(__name__)

_SENTINEL = object()                           # 佇列結尾標記：之後不會再有任務
_POLL_S = 0.2                                  # 阻塞等待時多久檢查一次停止旗標

_STAGE1 = ProcessStatus.stage1()
_STAGE2 = ProcessStatus.stage2()
_STAGE3 = ProcessStatus.stage3()
_AI = ProcessStatus.ai_bits()


class EngineState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINISHED = "finished"                      # 正常跑完
    STOPPED = "stopped"                        # 使用者中止（可續掃）
    FAILED = "failed"                          # 發生無法繼續的錯誤


@dataclass(slots=True)
class PipelineStats:
    """一次執行的統計。`PROGRESS` 事件的 payload 就是它的快照。"""

    files_seen: int = 0                        # Stage 0 看到的影像檔
    unchanged: int = 0                         # 沒變且已完成，直接略過
    to_process: int = 0                        # 需要處理的檔案數（進度條的分母）
    processed: int = 0                         # 已有結論的檔案數（進度條的分子，含下列各種結局）
    moved: int = 0                             # 判定為搬移，只更新路徑
    duplicates: int = 0                        # 內容與既有照片相同，直接複製 AI 結果
    stage1_done: int = 0
    inferred: int = 0                          # 走完推論的照片數
    tag_hits: int = 0
    faces_found: int = 0
    faces_skipped: int = 0                     # 條件式略過人臉的照片數
    errors: int = 0
    batches: int = 0
    queue_depth: int = 0
    elapsed_s: float = 0.0


def should_run_faces(is_screenshot: bool, roles: Iterable[GateRole]) -> bool:
    """Stage 3 的條件式判斷：這張照片要不要跑人臉偵測。

    規則（依優先序）：
      1. 命中任何「人物語意」標籤 → 跑（即使同時是截圖，例如視訊通話的截圖）。
      2. Stage 1 判定為螢幕截圖，或命中任何「文件/收據/截圖」類標籤 → 略過。
      3. 其餘 → 跑。寧可多跑也不要漏掉人臉；略過只是省算力的最佳化。
    """
    roles = set(roles)
    if GateRole.HAS_PERSON in roles:
        return True
    if is_screenshot or GateRole.SKIP_FACE in roles:
        return False
    return True


class PipelineEngine:
    """管線引擎。建構時注入所有元件（見 `pipeline/factory.py`），本身不知道任何具體模型。

    典型用法（GUI 與 CLI 相同）::

        engine = build_engine(cfg, db, repo, bus)
        engine.start([photo_folder])     # 立即返回，工作在背景執行緒進行
        ...                              # 由 bus 的事件得知進度
        engine.pause(); engine.resume()
        engine.stop()                    # 要求停止（不阻塞）
        engine.join()                    # 等到真的停下來

    所有公開方法都可以從任何執行緒呼叫。
    """

    def __init__(
        self,
        cfg: AppConfig,
        db: DBManager,
        repo: Repository,
        bus: EventBus,
        *,
        decoder: ImageDecoder,
        vision: VisionEmbedder | None = None,
        tagger: TagScorer | None = None,
        faces: FaceAnalyzer | None = None,
        geocoder: ReverseGeocoder | None = None,
        event_clusterer: EventClusterer | None = None,
        scanner: LibraryScanner | None = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._repo = repo
        self._bus = bus
        self._decoder = decoder
        self._vision = vision
        self._tagger = tagger
        self._faces = faces
        self._geocoder = geocoder
        self._event_clusterer = event_clusterer
        self._scanner = scanner or LibraryScanner(cfg.scan, repo)

        stages = list(cfg.pipeline.stages)
        if "siglip" in stages and (vision is None or tagger is None):
            raise ValueError("pipeline.stages 含 'siglip'，但沒有提供 vision / tagger 元件")
        if "faces" in stages and faces is None:
            raise ValueError("pipeline.stages 含 'faces'，但沒有提供 faces 元件")
        self._required = required_mask(stages)
        self._use_siglip = "siglip" in stages
        self._use_faces = "faces" in stages
        self._has_ai = bool(self._required & _AI)

        # ---- 控制 ----
        self._ctl = threading.RLock()              # 可重入：事件 callback 裡可以呼叫 pause()/stop()
        self._stop = threading.Event()             # set = 要求停止
        self._run_gate = threading.Event()         # set = 可以跑；clear = 暫停
        self._run_gate.set()
        self._state = EngineState.IDLE
        self._thread: threading.Thread | None = None
        self._consumer: threading.Thread | None = None

        # ---- 一次執行的狀態 ----
        self._models_ready = threading.Event()
        self._queue_ready = threading.Event()
        self._task_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._plan: RuntimePlan | None = None
        self._model_infos: list[ModelInfo] = []
        self._image_model_id: int | None = None
        self._face_model_id: int | None = None
        self._fatal: BaseException | None = None
        self._scan_id: int | None = None
        self._stats = PipelineStats()
        self._stats_lock = threading.Lock()
        self._t0 = 0.0
        self._progress_throttle = Throttle(cfg.pipeline.progress_interval_ms / 1000.0)
        self._post_steps: list[tuple[str, Callable[[], str | None]]] = []
        if cfg.pipeline.geocode and geocoder is not None:
            self._post_steps.append(("geocode", self._step_geocode))
        if cfg.pipeline.cluster_events and event_clusterer is not None:
            self._post_steps.append(("events", self._step_events))

    # ================================================================== #
    # 對外控制（任何執行緒）
    # ================================================================== #
    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def plan(self) -> RuntimePlan | None:
        """這次執行實際採用的資源配置；模型載入完成前為 None。"""
        return self._plan

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stats(self) -> PipelineStats:
        """目前統計的快照（複本）。"""
        with self._stats_lock:
            snap = PipelineStats(**asdict(self._stats))
        snap.queue_depth = self._task_q.qsize()
        snap.elapsed_s = round(time.monotonic() - self._t0, 2) if self._t0 else 0.0
        return snap

    def add_post_step(self, name: str, fn: Callable[[], str | None]) -> None:
        """掛載一個全域後處理步驟（例如人臉分群）。在所有照片處理完後、於調度執行緒依序執行。

        `fn` 回傳一段給 UI 顯示的摘要文字（或 None）；丟例外只會讓這個步驟失敗，不影響整體結果。
        長時間的步驟應定期檢查 `engine.stop_requested`。
        """
        self._post_steps.append((name, fn))

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def start(self, roots: Sequence[str | os.PathLike[str]]) -> None:
        """開始（或續掃）。立即返回。

        Raises:
            RuntimeError: 上一次執行還沒結束。
            ValueError: 沒有給根目錄。
        """
        if not roots:
            raise ValueError("至少要給一個根目錄")
        with self._ctl:
            if self.is_running:
                raise RuntimeError("管線正在執行中")
            self._stop.clear()
            self._run_gate.set()
            self._models_ready.clear()
            self._queue_ready.clear()
            self._fatal = None
            self._plan = None
            self._consumer = None
            self._stats = PipelineStats()
            self._t0 = time.monotonic()
            self._set_state(EngineState.RUNNING)
            self._thread = threading.Thread(
                target=self._run, args=([normalize_root(r) for r in roots],), name="pipeline", daemon=True
            )
            self._thread.start()

    def pause(self) -> None:
        with self._ctl:
            if self._state is EngineState.RUNNING:
                self._run_gate.clear()
                self._set_state(EngineState.PAUSED)

    def resume(self) -> None:
        with self._ctl:
            if self._state is EngineState.PAUSED:
                self._set_state(EngineState.RUNNING)
                self._run_gate.set()

    def stop(self) -> None:
        """要求停止。不阻塞；要等它真的停下來請接著呼叫 `join()`。可重複呼叫。"""
        with self._ctl:
            if not self.is_running:
                return
            self._stop.set()
            self._run_gate.set()                   # 把暫停中的執行緒放出來，讓它們看到停止旗標
            if self._state in (EngineState.RUNNING, EngineState.PAUSED):
                self._set_state(EngineState.STOPPING)

    def join(self, timeout: float | None = None) -> bool:
        """等管線結束。回傳是否已結束（False = 逾時）。不要在事件 callback 裡呼叫。"""
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    # ================================================================== #
    # 內部：狀態、事件、檢查點
    # ================================================================== #
    def _set_state(self, state: EngineState) -> None:
        previous, self._state = self._state, state
        if previous is not state:
            self._bus.publish(EventKind.STATE, state=state.value, previous=previous.value)

    def _bump(self, **deltas: int) -> None:
        with self._stats_lock:
            for name, d in deltas.items():
                setattr(self._stats, name, getattr(self._stats, name) + d)

    def _emit_progress(self, force: bool = False) -> None:
        if force or self._progress_throttle.ready():
            self._bus.publish(EventKind.PROGRESS, **asdict(self.stats()))

    def _checkpoint(self) -> bool:
        """暫停時在這裡等。回傳 False 代表應該停止。所有工作執行緒在每個單位工作前呼叫。"""
        while not self._run_gate.is_set():
            if self._stop.is_set():
                return False
            self._run_gate.wait(_POLL_S)
        return not self._stop.is_set()

    # ================================================================== #
    # 調度執行緒
    # ================================================================== #
    def _run(self, roots: list[str]) -> None:
        final = EngineState.FAILED
        error: str | None = None
        try:
            self._scan_id = self._repo.begin_scan(roots)
            if self._has_ai:
                # 模型載入與掃描同時進行：掃完的時候模型通常也好了
                self._consumer = threading.Thread(target=self._inference_main, name="inference", daemon=True)
                self._consumer.start()
            else:
                self._models_ready.set()

            scan = self._scan(roots)

            while not self._models_ready.wait(_POLL_S):
                if self._consumer is not None and not self._consumer.is_alive():
                    break
            if self._fatal is not None:
                raise self._fatal

            self._plan = self._make_plan()
            self._task_q = queue.Queue(maxsize=self._plan.queue_maxsize)
            self._queue_ready.set()
            log.info("RuntimePlan: %s", self._plan)

            if not self._stop.is_set():
                self._dispatch(scan.candidates, self._plan)

            if self._consumer is not None:
                self._offer(_SENTINEL)
                self._consumer.join()
            if self._fatal is not None:
                raise self._fatal
            self._db.flush()

            if not self._stop.is_set() and not scan.stopped:
                # 一定要在所有檔案處理完之後：被判定為搬移的列此時路徑已更新，不會被誤標
                self._repo.mark_missing(scan.missing)
                self._run_post_steps()
            final = EngineState.STOPPED if self._stop.is_set() else EngineState.FINISHED
        except BaseException as e:  # noqa: BLE001 - 任何錯誤都要轉成 FAILED 事件，不能讓執行緒默默死掉
            log.exception("管線失敗")
            error = f"{type(e).__name__}: {e}"
            self._bus.publish(EventKind.ERROR, where="pipeline", message=error)
        finally:
            # 確保推論執行緒一定會結束
            if final is EngineState.FAILED:
                self._stop.set()
            self._run_gate.set()
            self._queue_ready.set()
            if self._consumer is not None and self._consumer.is_alive():
                self._consumer.join(timeout=60)
            self._emit_progress(force=True)        # 節流可能吃掉最後一筆：結束前一定送出最終數字
            stats = asdict(self.stats())
            try:
                if self._scan_id is not None and self._db.is_open:
                    self._repo.finish_scan(self._scan_id, final.value, stats)
            except Exception:  # noqa: BLE001
                log.exception("寫入掃描紀錄失敗")
            with self._ctl:                        # 與 pause()/stop() 互斥，最終狀態不會被蓋掉
                self._set_state(final)
            self._bus.publish(
                EventKind.FINISHED, state=final.value, stats=stats, elapsed_s=stats["elapsed_s"], error=error
            )

    def _scan(self, roots: list[str]) -> ScanResult:
        throttle = Throttle(self._cfg.pipeline.progress_interval_ms / 1000.0)

        def on_progress(files_seen: int, to_process: int, current_dir: str) -> None:
            if throttle.ready():
                self._bus.publish(EventKind.SCAN_PROGRESS, files_seen=files_seen,
                                  to_process=to_process, current_dir=current_dir)

        scan = self._scanner.scan(roots, self._required, should_stop=lambda: not self._checkpoint(),
                                  on_progress=on_progress)
        self._bump(files_seen=scan.total_files, unchanged=scan.unchanged + scan.error_skipped,
                   to_process=len(scan.candidates))
        self._repo.clear_missing(scan.reappeared_ids)
        for r in scan.unreachable_roots:
            self._bus.publish(EventKind.ERROR, where="scan", message=f"根目錄不存在或無法存取：{r}")
        self._bus.publish(
            EventKind.SCAN_DONE, total_files=scan.total_files, to_process=len(scan.candidates),
            unchanged=scan.unchanged, error_skipped=scan.error_skipped, missing=len(scan.missing),
        )
        return scan

    def _make_plan(self) -> RuntimePlan:
        """依**實際生效**的 provider 決定資源配置。多個模型落在不同 provider 時取最慢的那個。"""
        providers = [m.provider for m in self._model_infos if m.kind != "text_embed" and m.provider]
        rank = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}
        provider = max(providers, key=lambda p: rank.get(p, len(rank))) if providers else CPU
        hw = probe_hardware()
        plan = plan_runtime(self._cfg.runtime, hw, provider)
        if not self._has_ai and self._cfg.runtime.decode_workers <= 0:
            # 只跑 Stage 1：沒有推論執行緒來搶 CPU，解碼可以用滿實體核心
            plan.decode_workers = max(2, min(8, hw.physical_cores))
        for comp in (self._vision, self._faces):
            setter = getattr(comp, "set_batch_size", None)
            if callable(setter):
                setter(plan.batch_size)            # 此時推論執行緒正在等 _queue_ready，不會同時使用元件
        return plan

    def _dispatch(self, candidates: list[FileCandidate], plan: RuntimePlan) -> None:
        """把 candidates 派給解碼執行緒池。在途數量以 semaphore 限制，候選清單再長也不會一次全塞進池子。"""
        slots = threading.BoundedSemaphore(plan.decode_workers * 2)
        pool = ThreadPoolExecutor(max_workers=plan.decode_workers, thread_name_prefix="decode")
        try:
            for cand in candidates:
                if not self._checkpoint():
                    break
                acquired = False
                while not acquired and not self._stop.is_set():
                    acquired = slots.acquire(timeout=_POLL_S)
                if not acquired:
                    break
                try:
                    fut = pool.submit(self._decode_job, cand)
                except BaseException:
                    slots.release()
                    raise
                fut.add_done_callback(lambda _f: slots.release())
        finally:
            pool.shutdown(wait=True, cancel_futures=self._stop.is_set())

    # ================================================================== #
    # 解碼執行緒（生產者）
    # ================================================================== #
    def _decode_job(self, cand: FileCandidate) -> None:
        if not self._checkpoint():
            return
        stage = "read"
        sha256: bytes | None = None
        try:
            raw = self._decoder.read(cand)
            sha256 = raw.sha256
            kind, photo_id, status = cand.kind, cand.photo_id, cand.status
            duplicate_of: int | None = None

            # ---- Stage 0（需要內容雜湊的部分）----
            if kind is CandidateKind.CHANGED:
                if cand.known_sha256 == sha256:
                    # mtime 被動過但內容沒變：更新指紋即可
                    assert photo_id is not None
                    self._repo.touch_file(photo_id, cand.size, cand.mtime_ns, self._scan_id)
                    kind = CandidateKind.INCOMPLETE
                else:
                    status = ProcessStatus.NONE     # 內容變了，全部重做
            elif kind is CandidateKind.NEW:
                matches = self._repo.find_by_sha256(sha256)
                for m in matches:
                    if not os.path.exists(m.path) and self._repo.relocate(
                            m.photo_id, m.path, cand, self._scan_id):
                        # 舊路徑已不存在、內容相同 → 檔案搬移：只改路徑，AI 結果全部沿用
                        kind, photo_id, status = CandidateKind.INCOMPLETE, m.photo_id, m.status
                        self._bump(moved=1)
                        break
                else:
                    done = [m for m in matches if (m.status & self._required) == self._required]
                    if done:
                        duplicate_of = done[0].photo_id   # 舊檔還在 → 這是另一份複本

            pending = self._required & ~status
            if kind is CandidateKind.INCOMPLETE and not pending:
                self._bump(processed=1)
                self._emit_progress()
                return

            # ---- 解碼一次 + Stage 1 ----
            stage = "decode"
            decode_for = (pending & _STAGE1) if duplicate_of is not None else pending
            decoded = self._decoder.decode(cand, raw, decode_for)
            del raw
            stage = "stage1"
            if kind is not CandidateKind.INCOMPLETE or (pending & _STAGE1) or (status & ProcessStatus.error_bits()):
                photo_id = self._repo.upsert_stage1(decoded.record, self._scan_id)
            assert photo_id is not None
            self._bump(stage1_done=1)

            if duplicate_of is not None:
                self._repo.clone_derived(duplicate_of, photo_id)
                self._bump(duplicates=1, processed=1)
                self._emit_progress()
                return

            ai_pending = pending & _AI
            if not ai_pending:
                self._bump(processed=1)
                self._emit_progress()
                return

            # ---- 交給推論執行緒（佇列滿時在這裡阻塞 = 背壓）----
            stage = "enqueue"
            self._offer(DecodedImageTask(
                photo_id=photo_id, path=cand.path, pending=ai_pending,
                width=decoded.record.width, height=decoded.record.height,
                is_screenshot=decoded.record.is_screenshot,
                siglip_input=decoded.siglip_input, face_input=decoded.face_input,
                face_letterbox=decoded.face_letterbox, work_image=decoded.work_image,
                decode_ms=decoded.decode_ms,
            ))
        except Exception as e:  # noqa: BLE001 - 單一檔案的錯誤不可中斷整條管線
            message = f"{type(e).__name__}: {e}"
            log.warning("處理失敗 %s -> %s: %s", cand.path, stage, message)
            pid: int | None = None
            try:
                pid = self._repo.record_decode_error(cand, sha256, f"{stage}: {message}", self._scan_id)
            except Exception:  # noqa: BLE001
                log.exception("寫入錯誤紀錄失敗 %s", cand.path)
            self._bump(errors=1, processed=1)
            self._bus.publish(EventKind.PHOTO_ERROR, photo_id=pid, path=cand.path, stage=stage, message=message)
            self._emit_progress()

    def _offer(self, item: Any) -> bool:
        """把任務放進有界佇列。滿了就等；停止或推論執行緒已結束則放棄並回傳 False。"""
        while True:
            if item is not _SENTINEL and self._stop.is_set():
                return False
            if self._consumer is None or not self._consumer.is_alive():
                return False
            try:
                self._task_q.put(item, timeout=_POLL_S)
                return True
            except queue.Full:
                continue

    # ================================================================== #
    # 推論執行緒（消費者）
    # ================================================================== #
    def _inference_main(self) -> None:
        components = [c for c in (self._vision if self._use_siglip else None,
                                  self._tagger if self._use_siglip else None,
                                  self._faces if self._use_faces else None) if c is not None]
        try:
            # ---- 載入模型（ONNX session 建立在使用它的這條執行緒上）----
            self._bus.publish(EventKind.MODEL, phase="loading", models=[], error=None)
            infos: list[ModelInfo] = []
            for comp in components:
                infos.extend(comp.load())
            if self._use_siglip and self._vision is not None:
                self._image_model_id = self._repo.ensure_model(
                    self._vision.model_name, "image_embed", self._vision.dim)
            if self._use_faces and self._faces is not None:
                self._face_model_id = self._repo.ensure_model(
                    self._faces.model_name, "face_embed", self._faces.dim)
            self._model_infos = infos
            self._bus.publish(EventKind.MODEL, phase="ready", models=[asdict(m) for m in infos], error=None)
        except BaseException as e:  # noqa: BLE001
            log.exception("模型載入失敗")
            self._fatal = e
            self._bus.publish(EventKind.MODEL, phase="failed", models=[], error=f"{type(e).__name__}: {e}")
            self._models_ready.set()
            self._close_components(components)
            return
        self._models_ready.set()

        try:
            while not self._queue_ready.wait(_POLL_S):
                if self._stop.is_set():
                    return
            plan = self._plan
            if plan is None:                       # 調度執行緒在建立 plan 前就失敗了
                return
            while True:
                batch, end = self._collect_batch(plan)
                if batch:
                    self._process_batch(batch)
                if end:
                    return
        except BaseException as e:  # noqa: BLE001
            log.exception("推論執行緒發生無法復原的錯誤")
            self._fatal = e
            self._stop.set()                       # 讓生產者不要再塞任務進來
            self._run_gate.set()
        finally:
            self._close_components(components)

    @staticmethod
    def _close_components(components: list[Any]) -> None:
        for comp in components:
            try:
                comp.close()
            except Exception:  # noqa: BLE001
                log.exception("關閉模型元件失敗")

    def _collect_batch(self, plan: RuntimePlan) -> tuple[list[DecodedImageTask], bool]:
        """動態 batching：從第一個任務到手開始計時，湊滿 `batch_size` 或超過 `batch_timeout` 就送出。

        Returns:
            (batch, end)：`end` 為 True 代表之後不會再有任務（收到結尾標記或被要求停止）。
            停止時手上還沒送出的任務直接丟棄。
        """
        batch: list[DecodedImageTask] = []
        deadline: float | None = None
        while len(batch) < plan.batch_size:
            if self._stop.is_set():
                return [], True
            if not self._run_gate.is_set():
                if batch:
                    break                          # 暫停：先把手上的做完，下次進來再停在這裡
                self._run_gate.wait(_POLL_S)
                continue
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            wait = _POLL_S if deadline is None else min(_POLL_S, deadline - now)
            try:
                item = self._task_q.get(timeout=max(0.001, wait))
            except queue.Empty:
                continue
            if item is _SENTINEL:
                return batch, True
            batch.append(item)
            if deadline is None:
                deadline = time.monotonic() + plan.batch_timeout_s
        return batch, False

    def _process_batch(self, batch: list[DecodedImageTask]) -> None:
        results = {t.photo_id: InferenceResult(photo_id=t.photo_id) for t in batch}
        roles: dict[int, set[GateRole]] = {}

        # ---- Stage 2：SigLIP 影像向量 + 多標籤 ----
        s2 = [t for t in batch if (t.pending & _STAGE2) and t.siglip_input is not None]
        if s2 and self._use_siglip:
            self._guarded("siglip", lambda tasks: self._stage2(tasks, results, roles), s2, results)

        # ---- Stage 3：條件式人臉（Stage 2 這一輪失敗的照片不做，留待下次）----
        s3 = [t for t in batch if (t.pending & _STAGE3) and t.face_input is not None
              and results[t.photo_id].error is None]
        if s3 and self._use_faces:
            self._stage3(s3, results, roles)

        # 結果與狀態位元寫進同一個交易；等 commit 完成才往下走
        ordered = [results[t.photo_id] for t in batch]
        self._repo.persist_inference(ordered, self._image_model_id, self._face_model_id)

        ok = [r for r in ordered if r.error is None]
        self._bump(
            processed=len(ordered), inferred=len(ok), errors=len(ordered) - len(ok), batches=1,
            tag_hits=sum(len(r.tags or ()) for r in ordered),
            faces_found=sum(len(r.faces or ()) for r in ordered),
            faces_skipped=sum(1 for r in ordered if r.faces_skipped),
        )
        for t, r in zip(batch, ordered):
            if r.error is not None:
                self._bus.publish(EventKind.PHOTO_ERROR, photo_id=r.photo_id, path=t.path,
                                  stage="inference", message=r.error)
            else:
                self._bus.publish(EventKind.PHOTO_DONE, photo_id=r.photo_id, path=t.path,
                                  tags=len(r.tags or ()), faces=len(r.faces or ()),
                                  faces_skipped=r.faces_skipped)
        self._emit_progress()

    def _guarded(
        self,
        stage: str,
        fn: Callable[[list[DecodedImageTask]], None],
        tasks: list[DecodedImageTask],
        results: dict[int, InferenceResult],
    ) -> None:
        """執行一個階段；整批失敗時逐張重試，把壞掉的那一張隔離出來，其他張照常完成。

        失敗只記在該照片的 `InferenceResult.error`，先前階段已算好的結果照樣會寫入資料庫。
        `NotImplementedError` 例外：那代表元件還是骨架（不是某張照片的問題），直接讓整條管線
        失敗並給出明確訊息，而不是把全庫的照片都標成推論錯誤。
        """

        def fail(task: DecodedImageTask, e: Exception) -> None:
            log.warning("%s 推論失敗 %s: %s", stage, task.path, e)
            r = results[task.photo_id]
            r.error = f"{stage}: {type(e).__name__}: {e}"

        try:
            fn(tasks)
            return
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001
            if len(tasks) == 1:
                fail(tasks[0], e)
                return
            log.warning("%s 整批失敗（%s），改為逐張重試", stage, e)
        for task in tasks:
            try:
                fn([task])
            except NotImplementedError:
                raise
            except Exception as e:  # noqa: BLE001
                fail(task, e)

    def _stage2(self, tasks: list[DecodedImageTask], results: dict[int, InferenceResult],
                roles: dict[int, set[GateRole]]) -> None:
        assert self._vision is not None and self._tagger is not None
        embeddings = self._vision.embed_images(stack_images([t.siglip_input for t in tasks]))
        hits = self._tagger.score(embeddings)
        for i, t in enumerate(tasks):
            r = results[t.photo_id]
            r.embedding = embeddings[i]
            r.tags = hits[i]
            r.status_set |= ProcessStatus.EMBEDDED | ProcessStatus.TAGGED
            roles[t.photo_id] = {self._tagger.gate_role(h.tag_id) for h in hits[i]}

    def _stage3(self, tasks: list[DecodedImageTask], results: dict[int, InferenceResult],
                roles: dict[int, set[GateRole]]) -> None:
        assert self._faces is not None
        # 續掃：Stage 2 上一輪已完成的照片，這一輪沒有重算標籤，從資料庫取回 gate 角色
        unknown = [t.photo_id for t in tasks if t.photo_id not in roles]
        if unknown and self._use_siglip:
            roles.update(self._repo.fetch_gate_roles(unknown))

        run: list[DecodedImageTask] = []
        for t in tasks:
            if should_run_faces(t.is_screenshot, roles.get(t.photo_id, ())):
                run.append(t)
            else:
                r = results[t.photo_id]
                r.faces_skipped = True
                r.status_set |= ProcessStatus.FACES | ProcessStatus.FACES_SKIPPED
        if not run:
            return

        def detect(subset: list[DecodedImageTask]) -> None:
            assert self._faces is not None
            for t, faces in zip(subset, self._faces.analyze(subset)):
                r = results[t.photo_id]
                r.faces = faces
                r.status_set |= ProcessStatus.FACES

        self._guarded("faces", detect, run, results)

    # ================================================================== #
    # 全域後處理（調度執行緒）
    # ================================================================== #
    def _run_post_steps(self) -> None:
        for name, fn in self._post_steps:
            if self._stop.is_set():
                return
            self._bus.publish(EventKind.POST_STEP, name=name, phase="start", detail=None)
            try:
                detail = fn()
            except Exception as e:  # noqa: BLE001 - 後處理失敗不影響已完成的逐張結果
                log.exception("後處理步驟 %s 失敗", name)
                self._bus.publish(EventKind.POST_STEP, name=name, phase="failed",
                                  detail=f"{type(e).__name__}: {e}")
            else:
                self._bus.publish(EventKind.POST_STEP, name=name, phase="done", detail=detail)

    def _step_geocode(self) -> str:
        """離線反向地理編碼。整批查一次（KD-tree 只建一次），不是每張照片各查一次。"""
        assert self._geocoder is not None
        rows = self._repo.fetch_ungeocoded()
        done = 0
        for i in range(0, len(rows), 5000):
            if self._stop.is_set():
                break
            part = rows[i : i + 5000]
            places = self._geocoder.lookup([(lat, lon) for _id, lat, lon in part])
            self._repo.save_places([(pid, place) for (pid, _lat, _lon), place in zip(part, places)])
            done += len(part)
        return f"{done} 張照片已轉換地點"

    def _step_events(self) -> str:
        assert self._event_clusterer is not None
        clusters = self._event_clusterer.cluster(self._repo.fetch_event_inputs())
        created = self._repo.replace_auto_events(clusters)
        return f"建立 {created} 個事件"
