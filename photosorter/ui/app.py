"""GUI 啟動入口。

    python -m photosorter            （不帶參數）
    python -m photosorter gui [--config <路徑>]

啟動順序有講究：高 DPI 感知必須在建立第一個 Tk 視窗之前設定；資料庫在視窗出現之前就開好，
但模型要等使用者按下「開始增量掃描」才會在背景載入，所以視窗是秒開的。

結束時也有講究：Tk 主視窗（Tcl 直譯器）必須在主執行緒被釋放，見 `theme.releasing_tk()`。
所以所有指向視窗的參考都只活在 `_window_loop()` 的區域變數裡，函式返回後立刻在主執行緒回收。
"""
from __future__ import annotations

import gc
import logging
import os

from ..pipeline.factory import AppContext, create_app
from ..utils.logging_setup import setup_logging
from .main_window import MainWindow
from .theme import apply_theme, create_root, enable_high_dpi, ui_scale

log = logging.getLogger(__name__)

APP_TITLE = "photosorter — 本地離線照片智慧分類"


def run_gui(config_path: str | os.PathLike[str] | None = None) -> int:
    """建立主視窗並進入事件迴圈；視窗關閉後回傳 0。"""
    enable_high_dpi()
    ctx = create_app(config_path)
    setup_logging(ctx.cfg.log_dir, logging.INFO, console=False)
    gc_was_enabled = gc.isenabled()
    try:
        _window_loop(ctx)                  # 返回前已停用自動回收，視窗物件此刻成為垃圾
    finally:
        gc.collect()                       # 在主執行緒把 Tk 物件（含 Tcl 直譯器）收掉
        if gc_was_enabled:
            gc.enable()
        if ctx.db.is_open:                 # 正常關閉時 MainWindow 已經關了；這裡是啟動失敗時的保險
            ctx.close()
    return 0


def _window_loop(ctx: AppContext) -> None:
    root = create_root()
    root.title(APP_TITLE)
    theme = apply_theme(root)
    scale = ui_scale(root)
    width = min(int(1440 * scale), root.winfo_screenwidth() - 80)
    height = min(int(900 * scale), root.winfo_screenheight() - 120)
    root.geometry(f"{width}x{height}")
    root.minsize(int(960 * scale), int(600 * scale))
    log.info("GUI 啟動：theme=%s scale=%.2f db=%s", theme, scale, ctx.cfg.db_path)

    # owns_context=True：視窗關閉（含安全停止引擎）之後由 MainWindow 負責關資料庫
    MainWindow(root, ctx, owns_context=True).pack(fill="both", expand=True)
    try:
        root.mainloop()
    finally:
        # 從這裡到 run_gui() 的 gc.collect() 之間，不讓任何背景執行緒搶先回收視窗物件
        gc.disable()
