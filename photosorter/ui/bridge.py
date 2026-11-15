"""核心 -> Tkinter 的橋接。GUI 與引擎之間唯一需要的膠水。

Tkinter 不是執行緒安全的：**只有主執行緒可以碰 widget**。引擎的事件是在工作執行緒上發布的，
所以不能在 callback 裡直接更新畫面。這裡的做法是標準的「佇列 + 計時器輪詢」：

    工作執行緒 ──publish──▶ EventQueue（執行緒安全、不阻塞）
                                  ▲
    Tk 主執行緒 ──root.after(50ms)──┘ drain() → 呼叫 handler → 更新 widget

用法::

    ctx = create_app()
    engine = ctx.build_engine()
    root = tk.Tk()
    pump = TkEventPump(root, ctx.bus, on_event=view.handle_event)
    pump.start()
    ...
    engine.start([folder])           # 立即返回，不會卡住畫面
    root.mainloop()
    pump.stop(); engine.stop(); engine.join(); ctx.close()

GUI 層的規則見 HANDOFF.md §5。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from ..core.events import Event, EventBus, EventKind, EventQueue

log = logging.getLogger(__name__)


class TkEventPump:
    """定時把 EventQueue 裡的事件搬到 Tk 主執行緒上處理。

    本類別只用到 `widget.after()` / `after_cancel()`，不 import tkinter，
    所以也適用於 ttkbootstrap、sv-ttk、customtkinter 等任何以 Tk 為底的框架。
    """

    def __init__(
        self,
        widget: Any,
        bus: EventBus,
        on_event: Callable[[Event], None],
        *,
        interval_ms: int = 50,
        max_per_tick: int = 200,
        kinds: Iterable[EventKind] | None = None,
    ) -> None:
        """
        Args:
            widget: 任何 Tk widget（通常是 root）。
            on_event: 在 **Tk 主執行緒**上被呼叫，可以直接更新畫面。
            interval_ms: 輪詢間隔。50ms（每秒 20 次）對進度條已經很順。
            max_per_tick: 每次最多處理幾個事件，避免事件暴增時卡住畫面。
            kinds: 只接收這些種類的事件；None = 全部。
        """
        self._widget = widget
        self._bus = bus
        self._on_event = on_event
        self._interval = interval_ms
        self._max = max_per_tick
        self._queue: EventQueue = bus.open_queue(kinds=kinds)
        self._after_id: Any = None
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._tick()

    def stop(self) -> None:
        """停止輪詢並釋放佇列。視窗關閉前呼叫。"""
        self._running = False
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:  # noqa: BLE001 - 視窗可能已經被銷毀
                pass
            self._after_id = None
        self._bus.close_queue(self._queue)

    def _tick(self) -> None:
        if not self._running:
            return
        for event in self._queue.drain(self._max):
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001 - 一個事件處理失敗不該讓整個輪詢停掉
                log.exception("處理事件 %s 時發生錯誤", event.kind.value)
        self._after_id = self._widget.after(self._interval, self._tick)
