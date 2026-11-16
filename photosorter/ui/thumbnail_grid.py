"""中央虛擬照片牆：十萬張照片也只建立畫面上看得到的那幾十格。

三個機制：

  * **Viewport 虛擬滾動** —— Canvas 的捲動範圍依照片總數算出，但只替「可見的列 ± 2 列緩衝」
    建立 Canvas 物件；捲動或改變視窗大小時，離開範圍的格子立刻刪除。
  * **LRU 記憶體快取** —— 已解碼的 `PhotoImage` 放在 `OrderedDict` 裡，上限 300 張，
    超過就淘汰最久沒用到的。畫面上正在顯示的圖另有強參考，不會因為被淘汰而變成空白。
  * **背景載入** —— 縮圖的讀檔、解碼、縮小在 `ThreadPoolExecutor(max_workers=2)` 裡做，
    結果經佇列交回主執行緒；只有主執行緒建立 `PhotoImage` 與操作 Canvas。

縮圖另有磁碟快取：`<cache_dir>/thumbs/<邊長>/<sha256 前 2 碼>/<sha256>.jpg`。以內容雜湊命名，
所以檔案搬移或改名之後快取仍然有效，內容變了則自然換一個檔名。
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable, Generic, Hashable, Iterator, Sequence, TypeVar

from PIL import Image, ImageOps, ImageTk

from ..db.queries_ui import PhotoItem
from .theme import PALETTE, ui_scale

log = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

DEFAULT_CACHE_SIZE = 300
BUFFER_ROWS = 2
_HEIF_EXTS = {".heic", ".heif"}
_heif_lock = threading.Lock()
_heif_ready: bool | None = None


class LRUCache(Generic[K, V]):
    """以 `OrderedDict` 實作的 LRU 快取。只在主執行緒使用，不加鎖。"""

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        if maxsize < 1:
            raise ValueError("maxsize 至少要 1")
        self.maxsize = maxsize
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self.evictions = 0

    def get(self, key: K) -> V | None:
        """取值並把它標成「最近用過」。沒有回傳 None。"""
        try:
            self._data.move_to_end(key)
        except KeyError:
            return None
        return self._data[key]

    def put(self, key: K, value: V) -> None:
        """放入（或更新）一筆；超過上限時淘汰最久沒用到的。"""
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._data.clear()

    def keys(self) -> list[K]:
        """由最久沒用到 → 最近用過。"""
        return list(self._data.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)


# --------------------------------------------------------------------------- #
# 縮圖產生（在背景執行緒執行；不碰任何 Tk 物件）
# --------------------------------------------------------------------------- #
def _ensure_heif() -> None:
    global _heif_ready
    with _heif_lock:
        if _heif_ready is None:
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
                _heif_ready = True
            except Exception:  # noqa: BLE001
                _heif_ready = False


def thumb_cache_path(cache_dir: Path | None, sha256: bytes | None, size: int) -> Path | None:
    """縮圖在磁碟快取裡的位置。沒有快取資料夾或沒有雜湊（照片還沒處理完）時回傳 None。"""
    if cache_dir is None or not sha256:
        return None
    digest = bytes(sha256).hex()
    return Path(cache_dir) / "thumbs" / str(size) / digest[:2] / f"{digest}.jpg"


def make_thumbnail(path: str, size: int) -> Image.Image:
    """讀原圖並縮成長邊不超過 `size` 的 RGB 縮圖，已套用 EXIF 方向。"""
    if os.path.splitext(path)[1].lower() in _HEIF_EXTS:
        _ensure_heif()
    with Image.open(path) as img:
        img.draft("RGB", (size * 2, size * 2))           # JPEG：直接解成縮小版，快很多
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img.convert("RGB")


def load_thumbnail(path: str, sha256: bytes | None, size: int, cache_dir: Path | None) -> Image.Image | None:
    """取得縮圖：先看磁碟快取，沒有才從原圖產生並寫入快取。讀不到（檔案不在、損毀）回傳 None。"""
    cached = thumb_cache_path(cache_dir, sha256, size)
    if cached is not None and cached.is_file():
        try:
            with Image.open(cached) as img:
                return img.convert("RGB")
        except Exception:  # noqa: BLE001 - 快取檔壞了就重做
            pass
    try:
        thumb = make_thumbnail(path, size)
    except Exception as e:  # noqa: BLE001
        log.debug("無法產生縮圖 %s: %s", path, e)
        return None
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_name(f"{cached.name}.{threading.get_ident()}.tmp")
            thumb.save(tmp, "JPEG", quality=85)
            os.replace(tmp, cached)                      # 原子替換：別的執行緒不會讀到寫一半的檔
        except OSError as e:
            log.debug("無法寫入縮圖快取 %s: %s", cached, e)
    return thumb


# --------------------------------------------------------------------------- #
# 照片牆
# --------------------------------------------------------------------------- #
class ThumbnailGrid(ttk.Frame):
    """虛擬滾動的縮圖牆。

    用法::

        grid = ThumbnailGrid(parent, cache_dir=cfg.cache_dir, on_select=show_details)
        grid.set_items(items)            # list[PhotoItem]；可以是十萬筆

    所有公開方法都必須在 Tk 主執行緒呼叫。
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        cache_dir: Path | None = None,
        thumb_size: int = 144,
        on_select: Callable[[PhotoItem | None], None] | None = None,
        on_activate: Callable[[PhotoItem], None] | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        workers: int = 2,
    ) -> None:
        """
        Args:
            cache_dir: 縮圖磁碟快取的根目錄（`AppConfig.cache_dir`）；None = 不使用磁碟快取。
            thumb_size: 縮圖邊長（96 DPI 下的像素；高 DPI 螢幕會自動放大）。
            on_select: 選取改變時呼叫（主執行緒）；取消選取時參數為 None。
            on_activate: 在某一格上按兩下時呼叫。
            cache_size: 記憶體中最多保留幾張 `PhotoImage`。
        """
        super().__init__(master)
        scale = ui_scale(self)
        self.thumb_size = int(thumb_size * scale)
        self._pad = int(8 * scale)
        self._label_h = int(20 * scale)
        self.cell_w = self.thumb_size + self._pad * 2
        self.cell_h = self.thumb_size + self._pad * 2 + self._label_h
        self._cache_dir = cache_dir
        self._on_select = on_select
        self._on_activate = on_activate

        self.cache: LRUCache[int, ImageTk.PhotoImage] = LRUCache(cache_size)
        self._items: list[PhotoItem] = []
        self._index_of: dict[int, int] = {}                 # photo id -> 在 _items 裡的位置
        self._cells: dict[int, list[int]] = {}              # 位置 -> 這一格的 Canvas 物件 id
        self._cell_images: dict[int, ImageTk.PhotoImage] = {}   # 位置 -> 顯示中的圖（強參考）
        self._image_item: dict[int, int] = {}               # 位置 -> 圖片（或佔位方塊）的 Canvas 物件 id
        self._selected: int | None = None
        self._cols = 1
        self._margin = 0

        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="thumb")
        self._results: "queue.Queue[tuple[int, Image.Image | None, bool]]" = queue.Queue()
        self._pending: set[int] = set()                     # 正在背景載入的 photo id
        self._failed: set[int] = set()                      # 讀不到的 photo id（換一批照片時清空）
        self._wanted: frozenset[int] = frozenset()          # 目前可見範圍的 photo id（背景執行緒只讀）
        self._drain_id: str | None = None
        self._closed = False

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=PALETTE["bg"],
                                yscrollincrement=max(1, self.cell_h // 4), takefocus=1)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._on_scrollbar)
        self.canvas.configure(yscrollcommand=self._on_yview_changed)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", lambda _e: self._layout())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        for key, delta in (("<Left>", -1), ("<Right>", 1)):
            self.canvas.bind(key, lambda _e, d=delta: self._move_selection(d))
        self.canvas.bind("<Up>", lambda _e: self._move_selection(-self._cols))
        self.canvas.bind("<Down>", lambda _e: self._move_selection(self._cols))
        self.bind("<Destroy>", self._on_destroy)

    # ------------------------------------------------------------------ #
    # 公開介面
    # ------------------------------------------------------------------ #
    def set_items(self, items: Sequence[PhotoItem], *, keep_selection: bool = False) -> None:
        """換一批照片。記憶體快取保留（同一張照片在別的分類下也用得到）。"""
        previous = self.selected_item.id if (keep_selection and self.selected_item) else None
        self._items = list(items)
        self._index_of = {item.id: i for i, item in enumerate(self._items)}
        self._clear_cells()
        self._failed.clear()
        self._selected = self._index_of.get(previous) if previous is not None else None
        if not keep_selection:
            self.canvas.yview_moveto(0.0)
        self._layout()
        if previous is not None and self._selected is None and self._on_select is not None:
            self._on_select(None)

    @property
    def items(self) -> list[PhotoItem]:
        return self._items

    @property
    def selected_item(self) -> PhotoItem | None:
        return self._items[self._selected] if self._selected is not None and self._selected < len(self._items) else None

    def select_index(self, index: int | None, *, notify: bool = True) -> None:
        """選取第 `index` 格（None = 取消選取），必要時捲動到看得見的位置。"""
        if index is not None and not 0 <= index < len(self._items):
            return
        old, self._selected = self._selected, index
        for i in (old, index):
            if i is not None and i in self._cells:
                self._redraw_cell(i)
        if index is not None:
            self._scroll_into_view(index)
        if notify and self._on_select is not None:
            self._on_select(self.selected_item)

    def select_photo(self, photo_id: int, *, notify: bool = True) -> bool:
        index = self._index_of.get(photo_id)
        if index is None:
            return False
        self.select_index(index, notify=notify)
        return True

    def visible_range(self) -> tuple[int, int]:
        """目前有建立 Canvas 物件的位置範圍 `[first, last)`（含緩衝列）。"""
        return self._wanted_range()

    @property
    def rendered_count(self) -> int:
        """目前實際建立的格子數 —— 不論照片總數多少，都只會是畫面上放得下的量。"""
        return len(self._cells)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def shutdown(self) -> None:
        """停止背景載入並釋放資源。視窗關閉時會自動呼叫；可重複呼叫。

        會等縮圖執行緒**真的結束**才返回（還沒開始的工作直接取消，進行中的頂多是一張圖的解碼時間）。
        這不是可有可無的：縮圖工作參考著這個元件，如果視窗銷毀後它們還活著，Tk 物件可能在
        背景執行緒被回收而讓 Tcl 中止整個行程。見 `theme.releasing_tk()`。
        """
        if self._closed:
            return
        self._closed = True
        if self._drain_id is not None:
            try:
                self.after_cancel(self._drain_id)
            except tk.TclError:
                pass
            self._drain_id = None
        self._executor.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------ #
    # 版面與虛擬滾動
    # ------------------------------------------------------------------ #
    def _layout(self) -> None:
        if self._closed:
            return
        width = max(self.canvas.winfo_width(), self.cell_w)
        cols = max(1, width // self.cell_w)
        margin = (width - cols * self.cell_w) // 2
        rows = -(-len(self._items) // cols)
        if (cols, margin) != (self._cols, self._margin):
            self._cols, self._margin = cols, margin
            self._clear_cells()                              # 欄數變了，每一格的位置都不同
        self.canvas.configure(scrollregion=(0, 0, width, max(rows * self.cell_h, 1)))
        self._render()

    def _wanted_range(self) -> tuple[int, int]:
        """可見的列 ± `BUFFER_ROWS` 列所涵蓋的位置範圍。"""
        top = self.canvas.canvasy(0)
        height = max(self.canvas.winfo_height(), 1)
        first_row = max(0, int(top // self.cell_h) - BUFFER_ROWS)
        last_row = int((top + height) // self.cell_h) + BUFFER_ROWS
        first = min(len(self._items), first_row * self._cols)
        last = min(len(self._items), (last_row + 1) * self._cols)
        return first, last

    def _render(self) -> None:
        """讓 Canvas 上的格子與可見範圍一致：範圍外的刪掉，範圍內缺的補上。"""
        if self._closed:
            return
        first, last = self._wanted_range()
        for index in [i for i in self._cells if not first <= i < last]:
            self._drop_cell(index)
        self._wanted = frozenset(self._items[i].id for i in range(first, last))
        for index in range(first, last):
            if index not in self._cells:
                self._draw_cell(index)

    def _cell_origin(self, index: int) -> tuple[int, int]:
        row, col = divmod(index, self._cols)
        return self._margin + col * self.cell_w, row * self.cell_h

    def _draw_cell(self, index: int) -> None:
        item = self._items[index]
        x, y = self._cell_origin(index)
        pad, size = self._pad, self.thumb_size
        cx, cy = x + self.cell_w // 2, y + pad + size // 2
        selected = index == self._selected
        ids = [self.canvas.create_rectangle(
            x + 2, y + 2, x + self.cell_w - 2, y + self.cell_h - 2, width=2,
            fill=PALETTE["select"] if selected else PALETTE["bg"],
            outline=PALETTE["accent"] if selected else PALETTE["bg"])]

        photo = self.cache.get(item.id)
        if photo is not None:
            image_id = self.canvas.create_image(cx, cy, image=photo)
            self._cell_images[index] = photo
        else:
            image_id = self.canvas.create_rectangle(x + pad, y + pad, x + pad + size, y + pad + size,
                                                    fill=PALETTE["cell"], outline="")
            self._request(item)
        ids.append(image_id)
        self._image_item[index] = image_id

        ids.append(self.canvas.create_text(
            cx, y + pad + size + self._label_h // 2, text=_ellipsis(item.file_name, 22),
            fill="#ffffff" if selected else PALETTE["muted"], font=("Segoe UI", 8)))
        badge = _badge_text(item)
        if badge:
            ids.append(self.canvas.create_text(x + pad + 4, y + pad + 4, text=badge, anchor="nw",
                                               fill="#ffd866", font=("Segoe UI Emoji", 10)))
        self._cells[index] = ids

    def _drop_cell(self, index: int) -> None:
        for cid in self._cells.pop(index, ()):
            self.canvas.delete(cid)
        self._cell_images.pop(index, None)
        self._image_item.pop(index, None)

    def _redraw_cell(self, index: int) -> None:
        self._drop_cell(index)
        if 0 <= index < len(self._items):
            self._draw_cell(index)

    def _clear_cells(self) -> None:
        self.canvas.delete("all")
        self._cells.clear()
        self._cell_images.clear()
        self._image_item.clear()

    # ------------------------------------------------------------------ #
    # 背景載入
    # ------------------------------------------------------------------ #
    def _request(self, item: PhotoItem) -> None:
        if self._closed or item.id in self._pending or item.id in self._failed:
            return
        self._pending.add(item.id)
        self._executor.submit(self._load_job, item)
        if self._drain_id is None:
            self._drain_id = self.after(30, self._drain)

    def _load_job(self, item: PhotoItem) -> None:
        """背景執行緒：只做讀檔與縮圖，結果丟進佇列。絕不碰 Tk。"""
        image: Image.Image | None = None
        skipped = False
        try:
            # 快速捲動時，排到的工作可能早已捲出畫面：直接跳過，把時間留給看得到的
            if item.id in self._wanted:
                image = load_thumbnail(item.path, item.sha256, self.thumb_size, self._cache_dir)
            else:
                skipped = True
        except Exception:  # noqa: BLE001
            log.exception("載入縮圖失敗 %s", item.path)
        self._results.put((item.id, image, skipped))

    def _drain(self) -> None:
        """主執行緒：把背景載入好的縮圖變成 `PhotoImage` 並放上畫面。每次只處理一小批，不卡住介面。"""
        self._drain_id = None
        if self._closed:
            return
        for _ in range(24):
            try:
                photo_id, image, skipped = self._results.get_nowait()
            except queue.Empty:
                break
            self._pending.discard(photo_id)
            index = self._index_of.get(photo_id)
            visible = index is not None and index in self._cells
            if image is not None:
                self.cache.put(photo_id, ImageTk.PhotoImage(image, master=self.canvas))
                if visible:
                    self._redraw_cell(index)                 # type: ignore[arg-type]
            elif skipped:
                if visible:
                    self._request(self._items[index])        # type: ignore[index]  # 被跳過後又捲回來了
            else:
                self._failed.add(photo_id)                   # 讀不到的檔案：留著佔位方塊，不再重試
        # 上面重新請求時可能已經排好下一輪了；不要重複排，否則會有兩個輪詢同時在跑
        if self._drain_id is None and (self._pending or not self._results.empty()):
            self._drain_id = self.after(30, self._drain)

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #
    def _on_scrollbar(self, *args: Any) -> None:
        self.canvas.yview(*args)

    def _on_yview_changed(self, first: str, last: str) -> None:
        self._scrollbar.set(first, last)
        self._render()

    def _on_mousewheel(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
        return "break"

    def _index_at(self, event: tk.Event) -> int | None:
        x = self.canvas.canvasx(event.x) - self._margin
        y = self.canvas.canvasy(event.y)
        if x < 0 or x >= self._cols * self.cell_w or y < 0:
            return None
        index = int(y // self.cell_h) * self._cols + int(x // self.cell_w)
        return index if 0 <= index < len(self._items) else None

    def _on_click(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        index = self._index_at(event)
        if index is not None and index != self._selected:
            self.select_index(index)

    def _on_double_click(self, event: tk.Event) -> None:
        index = self._index_at(event)
        if index is not None and self._on_activate is not None:
            self._on_activate(self._items[index])

    def _move_selection(self, delta: int) -> str:
        if not self._items:
            return "break"
        current = self._selected if self._selected is not None else -delta if delta > 0 else 0
        self.select_index(min(len(self._items) - 1, max(0, current + delta)))
        return "break"

    def _scroll_into_view(self, index: int) -> None:
        row = index // self._cols
        top = self.canvas.canvasy(0)
        height = max(self.canvas.winfo_height(), 1)
        total = max(-(-len(self._items) // self._cols) * self.cell_h, 1)
        if row * self.cell_h < top:
            self.canvas.yview_moveto(row * self.cell_h / total)
        elif (row + 1) * self.cell_h > top + height:
            self.canvas.yview_moveto(max(0.0, ((row + 1) * self.cell_h - height) / total))

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self.shutdown()


def _badge_text(item: PhotoItem) -> str:
    """格子左上角的標記：⭐ 連拍最佳、⧉ 連拍次佳、✕ 廢片、♥ 最愛。"""
    parts = []
    if item.is_best_shot:
        parts.append("⭐")
    elif item.is_burst_duplicate:
        parts.append("⧉")
    if item.is_rejected:
        parts.append("✕")
    if item.is_favorite:
        parts.append("♥")
    return " ".join(parts)


def _ellipsis(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
