"""三欄式主視窗：左側多維度導航樹、中央虛擬照片牆、右側屬性與匯出面板。

零阻塞的做法
------------
Tk 的主執行緒只做三件事：畫畫面、處理使用者操作、每 100 ms 輪詢兩個佇列。

    工作執行緒                          Tk 主執行緒
    ─────────────                      ─────────────
    PipelineEngine ──EventBus──▶ EventQueue ─┐
    _run_bg(fn) ────結果───────▶ _ui_queue ──┴─▶ self.after(100, self._poll_events) ─▶ 更新 widget

  * 所有會讀檔、查資料庫、跑模型或動檔案的事（掃描、分群、匯出、回復、載入清單與明細）都丟到
    背景執行緒；結果連同「要在主執行緒執行的 callback」放進 `_ui_queue`。
  * 工作執行緒**絕不**直接碰任何 widget。
  * 清單與明細的查詢帶有序號：使用者連續點了好幾個節點時，較舊的結果回來會被直接丟掉。

沒有模型權重的環境（或人臉階段還沒實作）不會崩潰：`factory.resolve_stages()` 只排入跑得起來的
階段，工具列的徽章顯示 `Mock / Weights Missing`，掃描、瀏覽、分群與匯出照常可用。
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from PIL import ImageTk

from ..core.events import Event, EventKind
from ..core.export_manager import ExportManager, ExportPreview
from ..core.hardware import CPU, CUDA, DML, probe_hardware, provider_candidates
from ..core.types import ExportResult
from ..db import queries_ui
from ..db.queries_ui import STAGING_LABELS, FaceInfo, PhotoDetail, PhotoItem, SidebarData
from ..pipeline.factory import AppContext, resolve_stages
from ..pipeline.pipeline_engine import PipelineEngine
from ..utils.clustering import EventClusterEngine, FaceClusterEngine
from .dialogs import ExportPreviewDialog, RenamePersonDialog, RollbackDialog, format_bytes
from .theme import PALETTE, ui_scale
from .thumbnail_grid import ThumbnailGrid, load_thumbnail

log = logging.getLogger(__name__)

EXPORT_MODES = ("COPY", "MOVE", "HARDLINK")
_PROVIDER_LABELS = {CUDA: "CUDA", DML: "DirectML", CPU: "CPU"}
_STATE_TEXT = {
    "idle": "待命", "running": "處理中", "paused": "已暫停", "stopping": "正在安全停止…",
    "finished": "完成", "stopped": "已停止（下次會從中斷處繼續）", "failed": "失敗",
}
_KV_LIBRARY, _KV_EXPORT_ROOT, _KV_EXPORT_MODE = "ui.library", "ui.export_root", "ui.export_mode"


class MainWindow(ttk.Frame):
    """主視窗的內容（放進 `tk.Tk()` 裡 `pack(fill="both", expand=True)`）。

    Args:
        ctx: `pipeline.factory.create_app()` 的結果。
        owns_context: True 時視窗關閉會一併 `ctx.close()`（正式啟動時為 True；測試自己管理 ctx）。
        poll_ms: 輪詢背景佇列的間隔。
        confirm: 詢問使用者「是否繼續」的函式 `(標題, 訊息) -> bool`；預設是 messagebox。
            測試會換成不跳視窗的版本。
    """

    def __init__(
        self,
        master: tk.Misc,
        ctx: AppContext,
        *,
        owns_context: bool = False,
        poll_ms: int = 100,
        confirm: Callable[[str, str], bool] | None = None,
    ) -> None:
        super().__init__(master)
        self.ctx = ctx
        self.exporter: ExportManager = ctx.exporter
        self._owns_context = owns_context
        self._poll_ms = poll_ms
        self.confirm = confirm or (lambda title, message: messagebox.askyesno(title, message, parent=self))
        self._scale = ui_scale(self)

        # ---- 狀態 ----
        self.engine: PipelineEngine | None = None
        self.engine_state = "idle"
        self.library: str | None = ctx.repo.kv_get(_KV_LIBRARY)
        self.sidebar: SidebarData | None = None
        self.current_node = "all"
        self.detail: PhotoDetail | None = None
        self.sidecars: list[str] = []
        self.last_preview: ExportPreview | None = None
        self.last_export: ExportResult | None = None
        self.skipped_stages: list[str] = []
        self.messages: list[tuple[str, str]] = []           # (level, text)：狀態列顯示過的訊息
        self.errors: list[str] = []
        self._job: str | None = None                        # 正在執行的獨佔背景工作（分群、匯出、回復）
        self._cancel = threading.Event()                    # 要求背景工作盡快結束（關閉視窗時）
        self._list_token = 0
        self._detail_token = 0
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._closing = False
        self._closed = False
        self._poll_id: str | None = None
        self._suppress_tree_event = False
        self.preview_dialog: ExportPreviewDialog | None = None
        self.rollback_dialog: RollbackDialog | None = None

        self._event_queue = ctx.bus.open_queue()
        self._ui_queue: "queue.Queue[tuple[Callable[..., None], tuple[Any, ...]]]" = queue.Queue()
        self._threads: list[threading.Thread] = []          # 還在跑的背景工作（它們的 callback 參考著這個視窗）

        self._build()
        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Destroy>", self._on_destroy)

        self._update_controls()
        self._refresh_badge()
        self.refresh_sidebar()
        self.select_node("all")
        self._poll_id = self.after(self._poll_ms, self._poll_events)

    # ================================================================== #
    # 版面
    # ================================================================== #
    def _build(self) -> None:
        s = self._scale
        pad = int(6 * s)
        ttk.Style(self).configure("Treeview", rowheight=int(24 * s))

        # ---- 頂部工具列 ----
        bar = ttk.Frame(self, padding=(pad, pad))
        bar.pack(side="top", fill="x")
        self.btn_choose = ttk.Button(bar, text="選擇相簿資料夾", command=self.choose_library)
        self.btn_scan = ttk.Button(bar, text="開始增量掃描", command=self.start_scan)
        self.btn_pause = ttk.Button(bar, text="暫停", command=self.toggle_pause)
        self.btn_stop = ttk.Button(bar, text="安全停止", command=self.stop_scan)
        self.btn_recluster = ttk.Button(bar, text="重新分群 (Re-cluster)", command=self.recluster)
        for b in (self.btn_choose, self.btn_scan, self.btn_pause, self.btn_stop, self.btn_recluster):
            b.pack(side="left", padx=(0, pad))
        self.badge_var = tk.StringVar(value="偵測硬體中…")
        self.badge = tk.Label(bar, textvariable=self.badge_var, background=PALETTE["panel"], foreground=PALETTE["muted"],
                              padx=int(10 * s), pady=int(3 * s), font=("Segoe UI", 9, "bold"))
        self.badge.pack(side="right")
        self.library_var = tk.StringVar(value=self.library or "（尚未選擇相簿資料夾）")
        ttk.Label(bar, textvariable=self.library_var, foreground=PALETTE["muted"]).pack(side="left", padx=pad)

        # ---- 底部狀態列 ----
        status = ttk.Frame(self, padding=(pad, int(3 * s)))
        status.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(status, mode="determinate", length=int(220 * s))
        self.progress.pack(side="left")
        self.status_var = tk.StringVar(value="待命")
        self.status_label = ttk.Label(status, textvariable=self.status_var)
        self.status_label.pack(side="left", padx=pad)

        # ---- 三欄 ----
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(side="top", fill="both", expand=True)
        self._panes = panes
        self._sidebar_width, self._inspector_width = int(300 * s), int(380 * s)

        left = ttk.Frame(panes)
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        # 欄寬給足並允許水平捲動：事件名稱與連拍標示很長，不可以被截斷到看不出內容
        self.tree.column("#0", width=int(520 * s), minwidth=int(520 * s), stretch=False)
        tree_scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        tree_xscroll = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll.set, xscrollcommand=tree_xscroll.set)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        tree_xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-Button-1>", self._on_tree_double_click)
        panes.add(left, weight=0)

        center = ttk.Frame(panes)
        self.header_var = tk.StringVar(value="")
        ttk.Label(center, textvariable=self.header_var, padding=(pad, int(4 * s))).pack(side="top", anchor="w")
        self.photo_grid = ThumbnailGrid(center, cache_dir=self.ctx.cfg.cache_dir, thumb_size=128,
                                        on_select=self._on_grid_select)
        self.photo_grid.pack(side="top", fill="both", expand=True)
        panes.add(center, weight=1)

        right = ttk.Frame(panes, padding=(pad, 0))
        self._build_inspector(right)
        panes.add(right, weight=0)
        self._sashes_placed = False
        panes.bind("<Configure>", self._place_sashes)

    def _place_sashes(self, _event: tk.Event | None = None) -> None:
        """第一次排版時把三欄切成：左右兩側固定寬度，中間的照片牆拿剩下的全部。之後使用者可以自己拖。"""
        width = self._panes.winfo_width()
        if self._sashes_placed or width <= 1:
            return
        left = min(self._sidebar_width, width // 4)
        right = min(self._inspector_width, width // 3)
        self._panes.sashpos(0, left)
        self._panes.sashpos(1, width - right)
        self._sashes_placed = True

    def _build_inspector(self, parent: ttk.Frame) -> None:
        """右側面板。由下往上先放匯出控制台與人臉清單，再放預覽圖，屬性表最後放並吃掉剩餘高度 ——
        視窗變矮時被壓縮的是屬性表（可以用滾輪捲動），匯出按鈕永遠看得到。"""
        s = self._scale
        pad = int(6 * s)

        # ---- 匯出控制台（最下方，永遠可見）----
        box = ttk.Labelframe(parent, text="非破壞性匯出與封存", padding=pad)
        box.pack(side="bottom", fill="x", pady=(pad, pad))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="模式").pack(side="left")
        self.mode_var = tk.StringVar(value=(self.ctx.repo.kv_get(_KV_EXPORT_MODE) or "COPY"))
        self.mode_box = ttk.Combobox(row, textvariable=self.mode_var, values=EXPORT_MODES, state="readonly", width=10)
        self.mode_box.pack(side="left", padx=pad)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(pad, 0))
        ttk.Label(row, text="目標").pack(side="left")
        self.dest_var = tk.StringVar(value=self.ctx.repo.kv_get(_KV_EXPORT_ROOT) or "")
        ttk.Entry(row, textvariable=self.dest_var).pack(side="left", fill="x", expand=True, padx=(pad, 0))
        ttk.Button(row, text="…", width=3, command=self.choose_export_root).pack(side="left", padx=(pad, 0))
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(pad, 0))
        self.btn_preview = ttk.Button(row, text="預覽匯出 (Dry-Run)", command=self.preview_export)
        self.btn_archive = ttk.Button(row, text="執行四層目錄歸檔", command=self.run_archive)
        self.btn_preview.pack(side="left", fill="x", expand=True)
        self.btn_archive.pack(side="left", fill="x", expand=True, padx=(pad, 0))
        self.btn_rollback = ttk.Button(box, text="⏪ 復原歷史整理 (Rollback)", command=self.open_rollback)
        self.btn_rollback.pack(fill="x", pady=(pad, 0))

        # ---- 人臉清單 ----
        faces = ttk.Labelframe(parent, text="畫面中的人臉（按兩下，或點預覽圖上的框來命名）", padding=pad)
        faces.pack(side="bottom", fill="x", pady=(pad, 0))
        self.face_list = ttk.Treeview(faces, columns=("score",), show="tree", height=2, selectmode="browse")
        self.face_list.column("#0", width=int(150 * s))
        self.face_list.column("score", width=int(56 * s), stretch=False)
        self.face_list.pack(side="left", fill="x", expand=True)
        self.face_list.bind("<Double-Button-1>", lambda _e: self.prompt_rename_face())
        self.btn_rename = ttk.Button(faces, text="命名並鎖定…", command=self.prompt_rename_face)
        self.btn_rename.pack(side="right", padx=(pad, 0))

        # ---- 預覽圖 ----
        self._preview_size = int(240 * s)
        self.preview_canvas = tk.Canvas(parent, width=self._preview_size, height=self._preview_size,
                                        background=PALETTE["panel"], highlightthickness=0)
        self.preview_canvas.pack(side="top", pady=(0, pad))

        # ---- 屬性表（吃掉剩餘高度）----
        table = ttk.Frame(parent)
        table.pack(side="top", fill="both", expand=True)
        self.info = ttk.Treeview(table, columns=("value",), show="tree", height=4, selectmode="none")
        self.info.column("#0", width=int(80 * s), stretch=False)
        self.info.column("value", width=int(640 * s), minwidth=int(640 * s), stretch=False)
        info_scroll = ttk.Scrollbar(table, orient="vertical", command=self.info.yview)
        info_xscroll = ttk.Scrollbar(table, orient="horizontal", command=self.info.xview)
        self.info.configure(yscrollcommand=info_scroll.set, xscrollcommand=info_xscroll.set)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.info.grid(row=0, column=0, sticky="nsew")
        info_scroll.grid(row=0, column=1, sticky="ns")
        info_xscroll.grid(row=1, column=0, sticky="ew")

    # ================================================================== #
    # 輪詢與背景工作
    # ================================================================== #
    def _poll_events(self) -> None:
        """每 `poll_ms` 毫秒在主執行緒執行一次：處理引擎事件與背景工作交回的結果。"""
        self._poll_id = None
        if self._closed:
            return
        try:
            for event in self._event_queue.drain(200):
                self._guarded(self._handle_event, event)
            for _ in range(50):
                try:
                    callback, args = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                self._guarded(callback, *args)
        finally:
            if not self._closed:
                self._poll_id = self.after(self._poll_ms, self._poll_events)

    def _guarded(self, fn: Callable[..., None], *args: Any) -> None:
        """執行一個 UI callback；出錯只記錄，不讓輪詢停掉。"""
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            log.exception("UI callback 失敗")
            self.errors.append(f"{getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}")

    def _run_bg(self, name: str, fn: Callable[[], Any], on_done: Callable[[Any], None],
                on_error: Callable[[BaseException], None] | None = None) -> None:
        """在背景執行緒跑 `fn()`，結果交回主執行緒的 `on_done(result)`。"""

        def worker() -> None:
            try:
                result = fn()
            except BaseException as e:  # noqa: BLE001
                log.exception("背景工作 %s 失敗", name)
                self._ui_queue.put((self._bg_failed, (name, e, on_error)))
            else:
                self._ui_queue.put((on_done, (result,)))

        thread = threading.Thread(target=worker, name=f"ui-{name}", daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()

    def _background_alive(self) -> bool:
        """還有背景工作沒結束。關閉視窗前必須等它們結束（見 `close()`）。"""
        self._threads = [t for t in self._threads if t.is_alive()]
        return bool(self._threads)

    def _bg_failed(self, name: str, error: BaseException, on_error: Callable[[BaseException], None] | None) -> None:
        if on_error is not None:
            on_error(error)
        else:
            self._notify(f"{name} 失敗：{type(error).__name__}: {error}", "err")

    def _start_job(self, name: str, label: str, fn: Callable[[], Any], on_done: Callable[[Any], None]) -> bool:
        """啟動一個獨佔的背景工作（同時間只能有一個，也不能和掃描同時進行）。"""
        if self._closing:
            return False
        if self.engine_running:
            self._notify("掃描進行中，請等它結束或先按「安全停止」", "warn")
            return False
        if self._job is not None:
            self._notify(f"「{self._job}」還在執行中", "warn")
            return False
        self._job = label
        self._notify(f"{label}…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._update_controls()

        def finish(callback: Callable[[Any], None], value: Any) -> None:
            self._job = None
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self._update_controls()
            callback(value)

        self._run_bg(name, fn, lambda result: finish(on_done, result),
                     lambda error: finish(lambda e: self._notify(f"{label}失敗：{type(e).__name__}: {e}", "err"), error))
        return True

    def _notify(self, text: str, level: str = "info") -> None:
        """在狀態列顯示一則訊息（不跳視窗）。"""
        self.messages.append((level, text))
        if level == "err":
            self.errors.append(text)
        self.status_var.set(text)
        color = {"err": PALETTE["err"], "warn": PALETTE["warn"], "ok": PALETTE["ok"]}.get(level, PALETTE["fg"])
        try:
            self.status_label.configure(foreground=color)
        except tk.TclError:
            pass

    @property
    def engine_running(self) -> bool:
        return self.engine is not None and self.engine.is_running

    @property
    def busy(self) -> bool:
        """有掃描或獨佔背景工作在進行。"""
        return self.engine_running or self._job is not None

    def _update_controls(self) -> None:
        running, paused = self.engine_running, self.engine_state == "paused"
        idle = not running and self._job is None and not self._closing

        def enable(widget: ttk.Widget, on: bool) -> None:
            widget.state(["!disabled"] if on else ["disabled"])

        enable(self.btn_choose, idle)
        enable(self.btn_scan, idle)
        enable(self.btn_pause, running and self.engine_state in ("running", "paused"))
        enable(self.btn_stop, running and self.engine_state != "stopping")
        enable(self.btn_recluster, idle)
        for b in (self.btn_preview, self.btn_archive, self.btn_rollback):
            enable(b, idle)
        self.btn_pause.configure(text="繼續" if paused else "暫停")

    # ================================================================== #
    # 引擎事件（主執行緒）
    # ================================================================== #
    def _handle_event(self, event: Event) -> None:
        d = event.data
        if event.kind is EventKind.STATE:
            self.engine_state = d["state"]
            self._notify(_STATE_TEXT.get(d["state"], d["state"]), "err" if d["state"] == "failed" else "info")
            self._update_controls()
        elif event.kind is EventKind.MODEL:
            if d["phase"] == "loading":
                self._notify("載入模型中…")
            elif d["phase"] == "ready":
                providers = sorted({m["provider"] for m in d["models"] if m["kind"] != "text_embed"})
                if providers:
                    self._set_badge("⚡ " + " / ".join(_PROVIDER_LABELS.get(p, p) for p in providers),
                                    PALETTE["ok"] if providers != [CPU] else PALETTE["fg"])
            else:
                self._set_badge("模型載入失敗", PALETTE["err"])
                self._notify(f"模型載入失敗：{d['error']}", "err")
        elif event.kind is EventKind.SCAN_PROGRESS:
            self._notify(f"掃描資料夾中… 已看到 {d['files_seen']} 個影像檔")
        elif event.kind is EventKind.SCAN_DONE:
            self.progress.configure(mode="determinate", maximum=max(1, d["to_process"]), value=0)
            self._notify(f"共 {d['total_files']} 個影像檔，需處理 {d['to_process']}，未變動 {d['unchanged']}")
        elif event.kind is EventKind.PROGRESS:
            self.progress.configure(maximum=max(1, d["to_process"]), value=d["processed"])
            self.status_var.set(f"{d['processed']} / {d['to_process']}　推論 {d['inferred']}　人臉 {d['faces_found']}　"
                                f"錯誤 {d['errors']}　{d['elapsed_s']:.0f} 秒")
        elif event.kind is EventKind.POST_STEP:
            if d["phase"] == "start":
                self._notify(f"後處理：{d['name']}…")
            elif d["phase"] == "failed":
                self._notify(f"後處理 {d['name']} 失敗：{d['detail']}", "err")
        elif event.kind is EventKind.ERROR:
            self._notify(f"{d['where']}：{d['message']}", "err")
        elif event.kind is EventKind.FINISHED:
            stats = d["stats"]
            if d["error"]:
                self._notify(f"掃描失敗：{d['error']}", "err")
            else:
                note = f"（略過：{'；'.join(self.skipped_stages)}）" if self.skipped_stages else ""
                self._notify(f"{_STATE_TEXT.get(d['state'], d['state'])}：處理 {stats['processed']} 張，"
                             f"錯誤 {stats['errors']}，{stats['elapsed_s']:.0f} 秒{note}",
                             "ok" if d["state"] == "finished" else "info")
            self.progress.configure(value=0)
            self._update_controls()
            self.refresh_all()
        elif event.kind is EventKind.EXPORT_PROGRESS:
            self.status_var.set(f"匯出中 {d['index']} / {d['total']}")

    # ================================================================== #
    # 工具列
    # ================================================================== #
    def choose_library(self) -> None:
        path = filedialog.askdirectory(parent=self, title="選擇相簿資料夾", initialdir=self.library or None)
        if path:
            self.set_library(path)

    def set_library(self, path: str) -> None:
        self.library = os.path.normpath(path)
        self.library_var.set(self.library)
        self.ctx.repo.kv_set(_KV_LIBRARY, self.library)

    def start_scan(self) -> bool:
        """開始（或續掃）。只排入這台電腦跑得起來的階段。回傳是否真的開始了。"""
        if self.busy or self._closing:
            return False
        if not self.library:
            self._notify("請先選擇相簿資料夾", "warn")
            return False
        stages, self.skipped_stages = resolve_stages(self.ctx.cfg)
        try:
            self.engine = self.ctx.build_engine(stages)
            self.engine.start([self.library])
        except Exception as e:  # noqa: BLE001
            self._notify(f"無法開始掃描：{type(e).__name__}: {e}", "err")
            return False
        self.progress.configure(mode="determinate", value=0)
        self._update_controls()
        return True

    def toggle_pause(self) -> None:
        if self.engine is None:
            return
        if self.engine_state == "paused":
            self.engine.resume()
        else:
            self.engine.pause()

    def stop_scan(self) -> None:
        """安全停止：手上的 batch 做完就停，已完成的結果都保留，下次從中斷處繼續。"""
        if self.engine is not None:
            self.engine.stop()

    def recluster(self) -> bool:
        """在背景重跑人臉分群與事件／連拍分群，完成後刷新導航樹。"""
        cfg, db = self.ctx.cfg, self.ctx.db

        def job() -> tuple[Any, Any, str | None]:
            events = EventClusterEngine.from_config(cfg, db).run()
            faces, face_error = None, None
            try:
                faces = FaceClusterEngine.from_config(cfg, db).run(should_stop=self._cancel.is_set)
            except ImportError as e:                         # 沒裝 scikit-learn
                face_error = f"人臉分群需要 scikit-learn：{e}"
            return events, faces, face_error

        def done(result: tuple[Any, Any, str | None]) -> None:
            events, faces, face_error = result
            text = f"重新分群完成：{events.events_created} 個事件、{events.burst_groups} 組連拍"
            if faces is not None:
                text += f"；人臉 {faces.total_faces} 張，新人物 {faces.new_persons}，歸入既有 {faces.assigned_to_existing}"
            self._notify(text if face_error is None else f"{text}；{face_error}", "ok" if face_error is None else "warn")
            self.refresh_all()

        return self._start_job("recluster", "重新分群", job, done)

    # ---- 硬體徽章 ----
    def _set_badge(self, text: str, color: str) -> None:
        self.badge_var.set(text)
        self.badge.configure(foreground=color)

    def _refresh_badge(self) -> None:
        cfg = self.ctx.cfg

        def probe() -> tuple[list[str], list[str], Any, list[str]]:
            stages, skipped = resolve_stages(cfg)
            hw = probe_hardware()
            return stages, skipped, hw, provider_candidates(cfg.runtime.providers, hw.available_providers)

        def show(result: tuple[list[str], list[str], Any, list[str]]) -> None:
            stages, skipped, hw, chain = result
            self.skipped_stages = skipped
            ai = [s for s in stages if s != "meta"]
            if not ai:
                # 沒有任何 AI 階段跑得起來：仍可掃描（EXIF/品質）、瀏覽、分群與匯出
                self._set_badge("Mock / Weights Missing", PALETTE["warn"])
            elif hw.ort_version is None or not chain:
                self._set_badge("onnxruntime 未安裝", PALETTE["err"])
            else:
                label = _PROVIDER_LABELS.get(chain[0], chain[0])
                self._set_badge(f"⚡ {label}" + ("（部分階段略過）" if skipped else ""),
                                PALETTE["ok"] if chain[0] != CPU else PALETTE["fg"])
            if skipped:
                self._notify("；".join(skipped), "warn")

        self._run_bg("probe", probe, show, lambda e: self._set_badge("硬體偵測失敗", PALETTE["err"]))

    # ================================================================== #
    # 左側導航樹
    # ================================================================== #
    def refresh_all(self) -> None:
        self.refresh_sidebar()
        self.select_node(self.current_node, keep_selection=True)

    def refresh_sidebar(self) -> None:
        self._run_bg("sidebar", lambda: queries_ui.load_sidebar(self.ctx.db), self._fill_sidebar)

    def _fill_sidebar(self, data: SidebarData) -> None:
        self.sidebar = data
        tree = self.tree
        opened = {iid for iid in self._all_nodes() if tree.item(iid, "open")} or {"events", "persons", "staging"}
        selected = self.current_node
        tree.delete(*tree.get_children())

        tree.insert("", "end", iid="all", text=f"🖼️ 全部照片 ({data.total})")
        tree.insert("", "end", iid="events", text=f"📅 時間與事件 (Events) ({data.in_events})")
        for e in data.events:
            iid = f"event:{'none' if e.id is None else e.id}"
            tree.insert("events", "end", iid=iid, text=f"{e.name} ({e.count})")
            for b in e.bursts:
                best = f"　⭐ Best Shot: {b.best_file}" if b.best_file else ""
                tree.insert(iid, "end", iid=f"burst:{b.group_id}",
                            text=f"連拍 burst_group_id={b.group_id} ({b.count} 張){best}")
        tree.insert("", "end", iid="persons", text=f"👤 人物群組 (Persons) ({len(data.persons)})")
        for p in data.persons:
            tree.insert("persons", "end", iid=f"person:{p.id}",
                        text=f"{'🔒 ' if p.is_locked else ''}{p.display_name} ({p.count})")
        tree.insert("", "end", iid="tags", text=f"🏷️ AI 語意標籤 (SigLIP Tags) ({len(data.tags)})")
        for t in data.tags:
            tree.insert("tags", "end", iid=f"tag:{t.id}", text=f"{t.display_name} ({t.count})")
        tree.insert("", "end", iid="locations", text=f"📍 拍攝地點 (Locations) ({len(data.locations)})")
        for c in data.locations:
            tree.insert("locations", "end", iid=f"loc:{c.code}", text=f"{c.code} ({c.count})")
            for city, n in c.cities:
                tree.insert(f"loc:{c.code}", "end", iid=f"city:{c.code}|{city}", text=f"{city} ({n})")
        tree.insert("", "end", iid="staging", text=f"🧹 待清理隔離區 (Staging) ({sum(data.staging.values())})")
        for key, label in STAGING_LABELS.items():
            tree.insert("staging", "end", iid=f"staging:{key}", text=f"{label} ({data.staging.get(key, 0)})")

        for iid in opened:
            if tree.exists(iid):
                tree.item(iid, open=True)
        if not tree.exists(selected):
            selected = "all"
        self._suppress_tree_event = True
        tree.selection_set(selected)
        tree.see(selected)
        self.after_idle(self._release_tree_event)
        self._update_header()

    def _release_tree_event(self) -> None:
        self._suppress_tree_event = False

    def _all_nodes(self, parent: str = "") -> list[str]:
        out: list[str] = []
        for iid in self.tree.get_children(parent):
            out.append(iid)
            out.extend(self._all_nodes(iid))
        return out

    def _on_tree_select(self, _event: tk.Event) -> None:
        if self._suppress_tree_event:
            return
        selection = self.tree.selection()
        if selection and selection[0] != self.current_node:
            self.select_node(selection[0])

    def _on_tree_double_click(self, event: tk.Event) -> None:
        iid = self.tree.identify_row(event.y)
        if iid.startswith("person:"):
            self.prompt_rename_person(int(iid.split(":", 1)[1]))

    def select_node(self, node: str, *, keep_selection: bool = False) -> None:
        """切換導航節點：在背景查出該節點的照片，再交給照片牆。"""
        self.current_node = node
        self._list_token += 1
        token = self._list_token

        def show(items: list[PhotoItem]) -> None:
            if token != self._list_token:
                return                                       # 使用者已經點了別的節點
            self.photo_grid.set_items(items, keep_selection=keep_selection)
            self._update_header()
            if not keep_selection or self.photo_grid.selected_item is None:
                self._clear_detail()

        self._run_bg("list", lambda: queries_ui.list_photos(self.ctx.db, node), show)

    def _update_header(self) -> None:
        """照片牆上方的標題：目前節點的名稱與張數。導航樹或清單任一方更新後都要重算。"""
        node = self.current_node
        label = self.tree.item(node, "text") if self.tree.exists(node) else ""
        count = f"{len(self.photo_grid.items)} 張"
        self.header_var.set(f"{label}　—　{count}" if label else count)

    # ================================================================== #
    # 右側屬性面板
    # ================================================================== #
    def _on_grid_select(self, item: PhotoItem | None) -> None:
        if item is None:
            self._clear_detail()
        else:
            self.show_photo(item.id)

    def show_photo(self, photo_id: int) -> None:
        """在背景載入一張照片的明細、Sidecar 清單與預覽圖。"""
        self._detail_token += 1
        token = self._detail_token
        db, size, cache_dir = self.ctx.db, self._preview_size, self.ctx.cfg.cache_dir

        def load() -> tuple[PhotoDetail | None, list[str], Any]:
            detail = queries_ui.photo_detail(db, photo_id)
            if detail is None:
                return None, [], None
            path = detail.row["path"]
            sidecars = [os.path.basename(p) for p in ExportManager._find_sidecars(path, {})]
            return detail, sidecars, load_thumbnail(path, detail.row["sha256"], size, cache_dir)

        def show(result: tuple[PhotoDetail | None, list[str], Any]) -> None:
            if token != self._detail_token:
                return
            detail, sidecars, image = result
            if detail is None:
                self._clear_detail()
                return
            self._show_detail(detail, sidecars, image)

        self._run_bg("detail", load, show)

    def _clear_detail(self) -> None:
        self._detail_token += 1
        self.detail, self.sidecars, self._preview_photo = None, [], None
        self.preview_canvas.delete("all")
        self.info.delete(*self.info.get_children())
        self.face_list.delete(*self.face_list.get_children())

    def _show_detail(self, detail: PhotoDetail, sidecars: list[str], image: Any) -> None:
        self.detail, self.sidecars = detail, sidecars
        r = detail.row

        # ---- 預覽圖與人臉框 ----
        canvas, size = self.preview_canvas, self._preview_size
        canvas.delete("all")
        if image is not None:
            self._preview_photo = ImageTk.PhotoImage(image, master=canvas)
            canvas.create_image(size // 2, size // 2, image=self._preview_photo)
            iw, ih = image.size
            ox, oy = (size - iw) // 2, (size - ih) // 2
            for face in detail.faces:
                x, y, w, h = face.bbox
                rect = canvas.create_rectangle(ox + x * iw, oy + y * ih, ox + (x + w) * iw, oy + (y + h) * ih,
                                               outline=PALETTE["ok"] if face.is_locked else PALETTE["accent"],
                                               width=2, fill="", activewidth=3)
                canvas.tag_bind(rect, "<Button-1>", lambda _e, f=face: self.prompt_rename_face(f))
        else:
            self._preview_photo = None
            canvas.create_text(size // 2, size // 2, text="無法載入預覽\n（檔案不存在或無法解碼）",
                               fill=PALETTE["muted"], justify="center")

        # ---- 屬性 ----
        def fmt(value: Any, pattern: str = "{}") -> str:
            return "" if value is None else pattern.format(value)

        camera = " ".join(x for x in (r["camera_make"], r["camera_model"]) if x)
        exposure = "　".join(x for x in (
            fmt(r["f_number"], "f/{:.1f}"),
            "" if not r["exposure_time"] else (f"1/{round(1 / r['exposure_time'])} 秒" if r["exposure_time"] < 1
                                               else f"{r['exposure_time']:.1f} 秒"),
            fmt(r["iso"], "ISO {}"), fmt(r["focal_length"], "{:.0f} mm"), r["lens_model"] or "") if x)
        place = "、".join(x for x in (r["city"], r["admin1"], r["country_code"]) if x)
        if r["gps_lat"] is not None:
            place += f"　({r['gps_lat']:.5f}, {r['gps_lon']:.5f})"
        burst = ""
        if r["burst_group_id"] is not None:
            burst = f"burst_group_id={r['burst_group_id']}　" + ("⭐ Best Shot" if r["is_best_shot"] else "次佳")
        flags = "、".join(label for label, on in (
            ("⭐ Best Shot", r["is_best_shot"]), ("連拍次佳", r["is_burst_duplicate"]), ("螢幕截圖", r["is_screenshot"]),
            ("廢片" + (f"（{r['reject_reason']}）" if r["reject_reason"] else ""), r["is_rejected"]),
            ("最愛", r["is_favorite"]), ("檔案遺失", r["is_missing"])) if on)
        tags = "　".join(
            f"{name} {score:.2f}" if score is not None else f"{name}（手動）" for name, score, _src in detail.tags)
        rows = [
            ("檔名", r["file_name"]), ("路徑", r["path"]),
            ("拍攝時間", fmt(r["taken_at"]) + (f"（來源：{r['taken_src']}）" if r["taken_src"] else "")),
            ("相機", camera), ("曝光", exposure),
            ("尺寸", f"{r['width']} × {r['height']}　{format_bytes(r['file_size'])}" if r["width"] else format_bytes(r["file_size"])),
            ("地點", place), ("事件", detail.event_name or ""),
            ("清晰度", fmt(r["blur_score"], "{:.1f}") + "（blur_score，越小越糊）" if r["blur_score"] is not None else ""),
            ("ShortHash", detail.short_hash),
            ("Sidecar", "、".join(sidecars) if sidecars else "無"),
            ("連拍", burst), ("標記", flags), ("標籤", tags),
        ]
        self.info.delete(*self.info.get_children())
        for key, value in rows:
            self.info.insert("", "end", text=key, values=(value,))

        self.face_list.delete(*self.face_list.get_children())
        for face in detail.faces:
            self.face_list.insert("", "end", iid=str(face.id), values=(f"{face.det_score:.2f}",),
                                  text=f"{'🔒 ' if face.is_locked else ''}{face.person_name}")

    # ---- 人物命名與鎖定 ----
    def prompt_rename_face(self, face: FaceInfo | None = None) -> RenamePersonDialog | None:
        """開啟命名對話框。`face=None` 時取人臉清單裡選取的那一張。"""
        if face is None and self.detail is not None:
            selection = self.face_list.selection()
            if selection:
                face = next((f for f in self.detail.faces if f.id == int(selection[0])), None)
            elif len(self.detail.faces) == 1:
                face = self.detail.faces[0]
        if face is None:
            self._notify("請先在人臉清單選一張臉", "warn")
            return None
        current = "" if face.person_id is None or face.person_name.startswith("Person_") else face.person_name
        return RenamePersonDialog(self, current, lambda name: self.rename_face(face.id, name))

    def prompt_rename_person(self, person_id: int) -> RenamePersonDialog | None:
        node = next((p for p in (self.sidebar.persons if self.sidebar else []) if p.id == person_id), None)
        current = node.display_name if node is not None and node.is_named else ""
        return RenamePersonDialog(self, current, lambda name: self.rename_person(person_id, name))

    def rename_person(self, person_id: int, name: str) -> None:
        """替人物改名並鎖定（`persons.is_locked = 1`），然後刷新畫面。"""
        self._run_bg("rename", lambda: queries_ui.rename_person(self.ctx.db, person_id, name),
                     lambda _r: self._after_rename(name))

    def rename_face(self, face_id: int, name: str) -> None:
        """替一張臉命名：已有人物就改名並鎖定；沒有歸屬就建立（或歸入同名的）人物。"""
        self._run_bg("rename", lambda: queries_ui.name_face(self.ctx.db, face_id, name),
                     lambda _r: self._after_rename(name))

    def _after_rename(self, name: str) -> None:
        self._notify(f"🔒 已命名為「{name}」並鎖定", "ok")
        self.refresh_sidebar()
        if self.detail is not None:
            self.show_photo(self.detail.row["id"])

    # ================================================================== #
    # 匯出與回復
    # ================================================================== #
    def choose_export_root(self) -> None:
        path = filedialog.askdirectory(parent=self, title="選擇匯出根目錄", initialdir=self.dest_var.get() or None)
        if path:
            self.dest_var.set(os.path.normpath(path))

    def _export_args(self) -> tuple[str, str] | None:
        dest = self.dest_var.get().strip()
        mode = self.mode_var.get().strip().lower()
        if not dest:
            self._notify("請先指定匯出根目錄", "warn")
            return None
        if mode.upper() not in EXPORT_MODES:
            self._notify(f"不支援的匯出模式：{mode}", "warn")
            return None
        self.ctx.repo.kv_set(_KV_EXPORT_ROOT, dest)
        self.ctx.repo.kv_set(_KV_EXPORT_MODE, mode.upper())
        return dest, mode

    def preview_export(self) -> bool:
        """Dry-Run：在背景算出動作清單與預估佔用空間，完成後開啟預覽視窗。不動任何檔案。"""
        args = self._export_args()
        if args is None:
            return False
        dest, mode = args

        def done(preview: ExportPreview) -> None:
            self.last_preview = preview
            pending = preview.pending()
            self._notify(f"預覽完成：將執行 {len(pending)} 筆，預估新增 {format_bytes(preview.estimated_bytes)}"
                         + (f"　⚠ {len(preview.warnings)} 則警告" if preview.warnings else ""),
                         "warn" if preview.warnings else "ok")
            self.preview_dialog = ExportPreviewDialog(self, preview, self.execute_preview)

        return self._start_job("preview", "預覽匯出", lambda: self.exporter.preview_export(dest, mode=mode), done)

    def run_archive(self) -> bool:
        """執行四層目錄歸檔：先預覽，使用者確認後才動檔案。"""
        args = self._export_args()
        if args is None:
            return False
        dest, mode = args

        def done(preview: ExportPreview) -> None:
            self.last_preview = preview
            self.execute_preview(preview)

        return self._start_job("preview", "計算匯出內容", lambda: self.exporter.preview_export(dest, mode=mode), done)

    def execute_preview(self, preview: ExportPreview) -> bool:
        pending = preview.pending()
        if not pending:
            self._notify("沒有需要匯出的項目（都已經在定位）", "ok")
            return False
        message = (f"即將以 {preview.mode.value.upper()} 模式匯出到\n{preview.export_root}\n\n"
                   f"共 {len(pending)} 筆操作，預估新增佔用空間 {format_bytes(preview.estimated_bytes)}。\n")
        if preview.mode.value == "move":
            message += "\nMOVE 會把原始檔案搬離目前的位置（可以用「復原歷史整理」搬回來）。\n"
        if preview.warnings:
            message += "\n" + "\n".join(f"⚠ {w}" for w in preview.warnings) + "\n"
        if not self.confirm("執行四層目錄歸檔", message + "\n要繼續嗎？"):
            self._notify("已取消匯出")
            return False

        def done(result: ExportResult) -> None:
            self.last_export = result
            text = f"匯出批次 {result.batch_id}：完成 {result.done}，略過 {result.skipped}，失敗 {result.failed}"
            self._notify(text + ("（已取消）" if result.cancelled else ""), "ok" if result.failed == 0 else "warn")
            self.refresh_all()                               # MOVE 之後照片的路徑會變

        return self._start_job("export", "匯出中", lambda: self.exporter.execute_export(
            preview, should_stop=self._cancel.is_set), done)

    def open_rollback(self) -> None:
        """讀取匯出歷史（`export_logs` 的 action / state 統計）並開啟復原視窗。"""

        def show(batches: list[queries_ui.ExportBatchInfo]) -> None:
            self.rollback_dialog = RollbackDialog(self, batches, self.rollback)

        self._run_bg("history", lambda: queries_ui.export_history(self.ctx.db), show)

    def rollback(self, batch_id: int) -> bool:
        if not self.confirm("復原歷史整理", f"要還原匯出批次 {batch_id} 嗎？\n\n"
                            "複製與硬連結建立的檔案會被刪除，搬移的檔案會搬回原位。"):
            return False

        def done(result: ExportResult) -> None:
            self._notify(f"批次 {batch_id}：已還原 {result.done} 筆，無法還原 {result.failed} 筆",
                         "ok" if result.failed == 0 else "warn")
            self.refresh_all()

        return self._start_job("rollback", f"還原批次 {batch_id}", lambda: self.exporter.rollback_batch(batch_id), done)

    # ================================================================== #
    # 關閉
    # ================================================================== #
    def close(self) -> None:
        """`WM_DELETE_WINDOW`：先發出停止訊號，等背景工作安全結束、資料庫寫入都 commit 之後才關視窗。

        期間介面仍然有回應（狀態列顯示「正在安全停止…」），不會凍結。

        「等所有背景執行緒結束才銷毀視窗」不只是為了資料安全：背景工作的 callback 參考著這個視窗，
        如果視窗銷毀後它們還活著，視窗物件（連同 Tcl 直譯器）最後可能在背景執行緒被回收，
        Tcl 會直接中止整個行程。見 `theme.releasing_tk()`。
        """
        if self._closing:
            return
        self._closing = True
        self._cancel.set()
        if self.engine_running:
            self.engine.stop()                               # type: ignore[union-attr]
        self._notify("正在安全停止，請稍候…")
        self._update_controls()
        self._await_shutdown(time.monotonic() + 120)

    def _await_shutdown(self, deadline: float) -> None:
        if (self.busy or self._background_alive()) and time.monotonic() < deadline:
            self.after(100, lambda: self._await_shutdown(deadline))
            return
        try:
            self.ctx.db.flush(timeout=30)                    # 佇列裡的寫入全部 commit
        except Exception:  # noqa: BLE001
            log.exception("關閉前 flush 失敗")
        self._teardown()
        if self._owns_context:
            self.ctx.close()
        try:
            self.winfo_toplevel().destroy()
        except tk.TclError:
            pass

    def _teardown(self) -> None:
        """停止輪詢、釋放佇列與縮圖執行緒。可重複呼叫。"""
        if self._closed:
            return
        self._closed = True
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        self.ctx.bus.close_queue(self._event_queue)
        self.photo_grid.shutdown()                           # 會等縮圖執行緒真的結束
        for thread in self._threads:                         # 正常關閉時這裡已經是空的；直接被銷毀時在此收尾
            thread.join(timeout=10)

    def _on_destroy(self, event: tk.Event) -> None:
        """視窗被直接銷毀（沒有經過 `close()`）時的保險：一樣發出停止訊號並收尾。"""
        if event.widget is not self or self._closed:
            return
        self._cancel.set()
        engine = self.engine
        self._teardown()
        if engine is not None and engine.is_running:
            engine.stop()
            if self._owns_context:
                engine.join(timeout=60)
        if self._owns_context:
            self.ctx.close()
