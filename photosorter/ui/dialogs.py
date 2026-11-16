"""對話框：人物命名、匯出預覽（Dry-Run 結果）、匯出歷史與復原。

對話框只負責顯示與收集輸入，真正的動作由建立它的人以 callback 傳入（都是 `MainWindow` 的方法），
所以對話框本身不碰資料庫、不開執行緒。全部是非模態的 `Toplevel`，不會卡住主視窗的事件輪詢。
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

from ..core.export_manager import ExportPreview
from ..db.queries_ui import ExportBatchInfo
from .theme import PALETTE, ui_scale

_ROLE_LABELS = {
    "master": "主庫", "staging": "隔離區", "sidecar": "附屬檔", "person_view": "人物視圖", "tag_view": "標籤視圖",
}
_STATE_LABELS = {
    "running": "執行中", "done": "完成", "cancelled": "已取消", "failed": "失敗",
    "rolled_back": "已還原", "partial_rollback": "部分還原",
}


def format_bytes(n: int | None) -> str:
    if n is None:
        return "未知"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


class _Dialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, title: str, size: tuple[int, int]) -> None:
        super().__init__(parent)
        scale = ui_scale(parent)
        self.title(title)
        self.configure(background=PALETTE["bg"])
        self.geometry(f"{int(size[0] * scale)}x{int(size[1] * scale)}")
        self.transient(parent.winfo_toplevel())
        self.bind("<Escape>", lambda _e: self.destroy())
        self._pad = int(10 * scale)

    def destroy(self) -> None:
        """關閉對話框，並在主執行緒立刻放掉它的 Tk 變數。

        `tk.Variable` 被回收時會呼叫 Tcl 把變數刪掉。對話框關閉後整個物件要靠循環垃圾回收器才收得掉，
        而回收器可能在背景執行緒執行 —— 那時才去呼叫 Tcl 就會跨執行緒。在這裡先把變數放掉，
        它們會當場（主執行緒）被釋放。
        """
        for name, value in list(vars(self).items()):
            if isinstance(value, tk.Variable):
                setattr(self, name, None)
        super().destroy()


class RenamePersonDialog(_Dialog):
    """輸入人物名稱。按下確定時呼叫 `on_submit(name)`；命名後該人物會被鎖定。"""

    def __init__(self, parent: tk.Misc, current: str, on_submit: Callable[[str], None]) -> None:
        super().__init__(parent, "人物命名", (380, 170))
        self._on_submit = on_submit
        body = ttk.Frame(self, padding=self._pad)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="名稱").pack(anchor="w")
        self.name_var = tk.StringVar(value=current)
        self.entry = ttk.Entry(body, textvariable=self.name_var)
        self.entry.pack(fill="x", pady=(4, 6))
        ttk.Label(body, text="🔒 命名後這個人物群組會被鎖定，重新分群不會覆蓋或拆散。",
                  foreground=PALETTE["muted"], wraplength=int(340 * ui_scale(parent))).pack(anchor="w")
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", side="bottom")
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="確定並鎖定", command=self.submit).pack(side="right", padx=(0, 6))
        self.entry.bind("<Return>", lambda _e: self.submit())
        self.entry.focus_set()
        self.entry.select_range(0, "end")

    def submit(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            return
        self.destroy()
        self._on_submit(name)


class ExportPreviewDialog(_Dialog):
    """Dry-Run 的結果：各類動作的筆數、預估佔用空間、警告，以及前幾百筆動作的明細。"""

    MAX_ROWS = 500

    def __init__(self, parent: tk.Misc, preview: ExportPreview, on_execute: Callable[[ExportPreview], None]) -> None:
        super().__init__(parent, "匯出預覽（Dry-Run，尚未動到任何檔案）", (980, 560))
        self.preview = preview
        self._on_execute = on_execute
        body = ttk.Frame(self, padding=self._pad)
        body.pack(fill="both", expand=True)

        pending = preview.pending()
        head = (f"匯出根目錄：{preview.export_root}\n模式：{preview.mode.value.upper()}　"
                f"將執行 {len(pending)} 筆，略過 {len(preview.actions) - len(pending)} 筆　"
                f"預估新增佔用空間 {format_bytes(preview.estimated_bytes)}（可用 {format_bytes(preview.free_bytes)}）")
        ttk.Label(body, text=head, justify="left").pack(anchor="w")
        self.summary_text = "　".join(f"{k} × {n}" for k, n in sorted(preview.summary().items()))
        ttk.Label(body, text=self.summary_text, foreground=PALETTE["muted"], justify="left",
                  wraplength=int(940 * ui_scale(parent))).pack(anchor="w", pady=(4, 0))
        for warning in preview.warnings:
            ttk.Label(body, text=f"⚠ {warning}", foreground=PALETTE["warn"], justify="left",
                      wraplength=int(940 * ui_scale(parent))).pack(anchor="w", pady=(4, 0))

        table = ttk.Frame(body)
        table.pack(fill="both", expand=True, pady=self._pad)
        columns = ("role", "op", "src", "dst")
        self.tree = ttk.Treeview(table, columns=columns, show="headings")
        for col, text, width in (("role", "類別", 80), ("op", "動作", 80), ("src", "來源", 380), ("dst", "目標", 400)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=int(width * ui_scale(parent)), stretch=col in ("src", "dst"))
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for a in pending[: self.MAX_ROWS]:
            self.tree.insert("", "end", values=(_ROLE_LABELS.get(a.role, a.role), a.op_type, a.src, a.dst))
        if len(pending) > self.MAX_ROWS:
            self.tree.insert("", "end", values=("…", "", f"（另有 {len(pending) - self.MAX_ROWS} 筆未列出）", ""))

        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="關閉", command=self.destroy).pack(side="right")
        self.execute_button = ttk.Button(buttons, text="執行四層目錄歸檔", command=self.execute)
        self.execute_button.pack(side="right", padx=(0, 6))
        if not pending:
            self.execute_button.state(["disabled"])

    def execute(self) -> None:
        self.destroy()
        self._on_execute(self.preview)


class RollbackDialog(_Dialog):
    """匯出歷史。每一列是一個批次，統計取自 `export_logs` 的 `action` 與 `state`。"""

    def __init__(self, parent: tk.Misc, batches: Sequence[ExportBatchInfo],
                 on_rollback: Callable[[int], None]) -> None:
        super().__init__(parent, "⏪ 復原歷史整理", (980, 420))
        self._on_rollback = on_rollback
        self.batches = {b.id: b for b in batches}
        body = ttk.Frame(self, padding=self._pad)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="選一個批次後按「還原」：複製與硬連結建立的檔案會被刪除，搬移的檔案會搬回原位，"
                             "最後清掉空資料夾。被你修改過的檔案不會被刪。",
                  foreground=PALETTE["muted"], justify="left",
                  wraplength=int(940 * ui_scale(parent))).pack(anchor="w")

        table = ttk.Frame(body)
        table.pack(fill="both", expand=True, pady=self._pad)
        columns = ("batch_id", "time", "mode", "state", "logs", "dest")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        for col, text, width in (("batch_id", "batch_id", 70), ("time", "時間", 130), ("mode", "模式", 70),
                                 ("state", "批次狀態", 80), ("logs", "action: state 筆數", 320),
                                 ("dest", "匯出根目錄", 280)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=int(width * ui_scale(parent)), stretch=col in ("logs", "dest"))
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for b in batches:
            self.tree.insert("", "end", iid=str(b.id), values=(
                b.id, time.strftime("%Y-%m-%d %H:%M", time.localtime(b.started_at)), b.mode,
                _STATE_LABELS.get(b.state, b.state), b.describe_counts(), b.dest_root))
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._update_button())

        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="關閉", command=self.destroy).pack(side="right")
        self.rollback_button = ttk.Button(buttons, text="⏪ 還原選取的批次", command=self.rollback)
        self.rollback_button.pack(side="right", padx=(0, 6))
        self._update_button()

    def selected_batch(self) -> ExportBatchInfo | None:
        selection = self.tree.selection()
        return self.batches.get(int(selection[0])) if selection else None

    def _update_button(self) -> None:
        batch = self.selected_batch()
        self.rollback_button.state(["!disabled"] if batch is not None and batch.can_rollback else ["disabled"])

    def rollback(self) -> None:
        batch = self.selected_batch()
        if batch is None or not batch.can_rollback:
            return
        self.destroy()
        self._on_rollback(batch.id)
