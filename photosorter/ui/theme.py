"""外觀：高 DPI 感知、深色主題（sv-ttk，沒裝時退回 ttk 的 clam 主題加深色配色）。"""
from __future__ import annotations

import contextlib
import gc
import logging
import sys
import tkinter as tk
from tkinter import ttk
from typing import Iterator

log = logging.getLogger(__name__)

#: 深色配色。Canvas 之類不吃 ttk 樣式的元件直接取用這裡的顏色。
PALETTE: dict[str, str] = {
    "bg": "#1c1c1c",          # 視窗底色
    "panel": "#252526",       # 面板、輸入框
    "cell": "#2d2d30",        # 照片牆的格子
    "border": "#3c3c3c",
    "fg": "#e6e6e6",
    "muted": "#9a9a9a",
    "accent": "#4c9aff",
    "select": "#2f6fde",
    "ok": "#3fb950",
    "warn": "#d29922",
    "err": "#f85149",
}


def enable_high_dpi() -> None:
    """宣告本程式能處理高 DPI，避免 Windows 把整個視窗當點陣圖放大而變模糊。

    必須在建立第一個 Tk 視窗**之前**呼叫。非 Windows 或舊版系統呼叫失敗時直接略過。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)      # Per-Monitor DPI Aware（Windows 8.1 以上）
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()           # 舊系統：System DPI Aware
    except Exception as e:  # noqa: BLE001 - 設不成只是比較模糊，不該讓程式起不來
        log.debug("無法設定 DPI 感知：%s", e)


def create_root(retries: int = 5) -> tk.Tk:
    """建立 Tk 主視窗，失敗時重試幾次。

    Windows 上 Tcl 啟動時要讀自己的腳本檔（`init.tcl`、`tk.tcl`）；防毒或 EDR 正好在掃描那些檔案時，
    偶爾會得到「Can't find a usable tk.tcl」。那是暫時性的，隔一下再試就會成功。
    重試用完仍失敗（例如真的沒有顯示器或 Tk 沒裝好）才把錯誤往上丟。
    """
    import time

    last: tk.TclError | None = None
    for attempt in range(max(1, retries)):
        try:
            return tk.Tk()
        except tk.TclError as e:
            last = e
            log.warning("建立 Tk 視窗失敗（第 %d 次）：%s", attempt + 1, str(e).splitlines()[0])
            time.sleep(0.2 * (attempt + 1))
    assert last is not None
    raise last


@contextlib.contextmanager
def releasing_tk() -> Iterator[None]:
    """把「銷毀 Tk 主視窗並放掉它最後的參考」這段程式包起來，確保 Tcl 直譯器在**目前的執行緒**被釋放。

    為什麼需要：Tcl 直譯器只能由建立它的執行緒釋放。Tk 的物件之間有循環參考，要靠 Python 的
    循環垃圾回收器才收得掉，而回收器會在「剛好觸發配置門檻的那條執行緒」上執行 —— 如果那是
    解碼或縮圖的背景執行緒，Tcl 會直接中止整個行程：
    `Tcl_AsyncDelete: async handler deleted by the wrong thread`。

    做法：區塊內先停用自動回收（其他執行緒就不會搶先回收），離開區塊時在目前的執行緒明確回收一次。
    呼叫端必須在主執行緒，而且要在區塊內放掉所有指向主視窗的參考（`del root`、把屬性設成 None）。
    區塊很短，停用自動回收不會造成記憶體累積。
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def ui_scale(widget: tk.Misc) -> float:
    """目前螢幕相對於 96 DPI 的縮放倍率。以像素指定的尺寸（縮圖、格子）要乘上它。"""
    try:
        return max(1.0, float(widget.winfo_fpixels("1i")) / 96.0)
    except tk.TclError:
        return 1.0


def apply_theme(root: tk.Misc) -> str:
    """套用深色主題。回傳實際使用的主題名稱：`"sv-ttk"` 或 `"clam-dark"`。"""
    try:
        import sv_ttk

        sv_ttk.set_theme("dark", root)
        name = "sv-ttk"
    except Exception as e:  # noqa: BLE001 - 沒安裝或載入失敗都退回內建主題
        log.info("sv-ttk 不可用（%s），改用 ttk clam 深色配色", e)
        _apply_clam_dark(root)
        name = "clam-dark"
    try:
        root.configure(background=PALETTE["bg"])                 # type: ignore[call-arg]
    except tk.TclError:
        pass
    return name


def _apply_clam_dark(root: tk.Misc) -> None:
    p = PALETTE
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=p["bg"], foreground=p["fg"], fieldbackground=p["panel"],
                    bordercolor=p["border"], darkcolor=p["bg"], lightcolor=p["bg"], troughcolor=p["panel"],
                    selectbackground=p["select"], selectforeground="#ffffff", insertcolor=p["fg"])
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabel", background=p["bg"], foreground=p["fg"])
    style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
    style.configure("TLabelframe.Label", background=p["bg"], foreground=p["muted"])
    style.configure("TButton", background=p["panel"], foreground=p["fg"], bordercolor=p["border"], padding=(10, 5))
    style.map("TButton",
              background=[("disabled", p["bg"]), ("pressed", p["select"]), ("active", p["border"])],
              foreground=[("disabled", p["muted"])])
    style.configure("TEntry", fieldbackground=p["panel"], foreground=p["fg"], bordercolor=p["border"])
    style.configure("TCombobox", fieldbackground=p["panel"], background=p["panel"], foreground=p["fg"],
                    arrowcolor=p["fg"], bordercolor=p["border"])
    style.map("TCombobox", fieldbackground=[("readonly", p["panel"])], foreground=[("readonly", p["fg"])])
    style.configure("Treeview", background=p["panel"], fieldbackground=p["panel"], foreground=p["fg"],
                    bordercolor=p["border"], rowheight=24)
    style.map("Treeview", background=[("selected", p["select"])], foreground=[("selected", "#ffffff")])
    style.configure("Treeview.Heading", background=p["bg"], foreground=p["muted"], bordercolor=p["border"])
    style.configure("TScrollbar", background=p["border"], troughcolor=p["bg"], bordercolor=p["bg"],
                    arrowcolor=p["muted"])
    # 內容不需要捲動時捲軸是 disabled 狀態，clam 預設會畫成亮灰色，在深色畫面上很刺眼
    style.map("TScrollbar", background=[("disabled", p["bg"]), ("active", p["muted"])],
              arrowcolor=[("disabled", p["bg"])])
    style.configure("TProgressbar", background=p["accent"], troughcolor=p["panel"], bordercolor=p["border"])
    style.configure("TPanedwindow", background=p["bg"])
    style.configure("TSeparator", background=p["border"])
    # 下拉選單的清單是傳統 Tk 元件，要另外設定
    root.option_add("*TCombobox*Listbox.background", p["panel"])
    root.option_add("*TCombobox*Listbox.foreground", p["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["select"])
