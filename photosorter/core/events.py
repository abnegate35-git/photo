"""引擎 -> 上層（GUI / CLI）的單向事件通道。這是核心與 UI 之間唯一的耦合點。

兩種接法，擇一或並用：

  1. **Event Queue（GUI 用）** —— `q = bus.open_queue()`，然後在 UI 執行緒用計時器
     （Tkinter 的 `root.after(50, poll)`）定期呼叫 `q.drain()`。引擎執行緒完全不碰 GUI。
  2. **Callback（CLI / 日誌用）** —— `bus.subscribe(fn)`。`fn` 會在**發布事件的那條
     工作執行緒**上被呼叫，所以必須很快返回，而且不可以碰 GUI 元件。

不論哪種接法，慢的訂閱者都不會拖慢管線：佇列滿了就丟掉進度類事件（下一筆進度
本來就會蓋掉它），狀態、錯誤、完成等關鍵事件永遠不丟。
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

log = logging.getLogger(__name__)


class EventKind(str, enum.Enum):
    # ---- 生命週期（關鍵，不丟）----
    STATE = "state"                  # data: state(EngineState.value), previous
    MODEL = "model"                  # data: phase('loading'|'ready'|'failed'), models[list[dict]], error
    SCAN_DONE = "scan_done"          # data: total_files, to_process, unchanged, missing
    POST_STEP = "post_step"          # data: name, phase('start'|'done'|'failed'), detail
    FINISHED = "finished"            # data: state, stats(dict), elapsed_s, error
    ERROR = "error"                  # data: where, message（非單張照片的錯誤）
    EXPORT_DONE = "export_done"      # data: batch_id, done, skipped, failed, cancelled

    # ---- 進度（可丟）----
    SCAN_PROGRESS = "scan_progress"  # data: files_seen, to_process, current_dir
    PROGRESS = "progress"            # data: PipelineStats 的快照（見 pipeline_engine.PipelineStats）
    PHOTO_DONE = "photo_done"        # data: photo_id, path, tags(int), faces(int), faces_skipped
    PHOTO_ERROR = "photo_error"      # data: photo_id, path, stage, message
    EXPORT_PROGRESS = "export_progress"   # data: batch_id, index, total, src, dst
    LOG = "log"                      # data: level, message


#: 佇列滿時可以直接丟掉的事件
DROPPABLE: frozenset[EventKind] = frozenset(
    {
        EventKind.SCAN_PROGRESS,
        EventKind.PROGRESS,
        EventKind.PHOTO_DONE,
        EventKind.EXPORT_PROGRESS,
        EventKind.LOG,
    }
)


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    data: Mapping[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Callback = Callable[[Event], None]


class EventQueue:
    """給 UI 輪詢的有界事件佇列。`put` 永不阻塞。"""

    def __init__(self, maxsize: int = 2000, kinds: Iterable[EventKind] | None = None) -> None:
        self._items: deque[Event] = deque()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._kinds = frozenset(kinds) if kinds is not None else None
        self.dropped = 0                     # 被丟掉的進度事件數（除錯用）

    def put(self, event: Event) -> None:
        if self._kinds is not None and event.kind not in self._kinds:
            return
        with self._lock:
            if len(self._items) >= self._maxsize and event.kind in DROPPABLE:
                self.dropped += 1
                return
            self._items.append(event)

    def drain(self, max_items: int = 500) -> list[Event]:
        """取出目前累積的事件（最多 `max_items` 筆）。UI 執行緒呼叫，不阻塞。"""
        out: list[Event] = []
        with self._lock:
            while self._items and len(out) < max_items:
                out.append(self._items.popleft())
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class EventBus:
    """執行緒安全的事件發布器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callbacks: list[tuple[Callback, frozenset[EventKind] | None]] = []
        self._queues: list[EventQueue] = []

    # ---- 訂閱 ----
    def subscribe(self, callback: Callback, kinds: Iterable[EventKind] | None = None) -> Callable[[], None]:
        """註冊 callback；回傳的函式用來取消訂閱。"""
        entry = (callback, frozenset(kinds) if kinds is not None else None)
        with self._lock:
            self._callbacks.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._callbacks:
                    self._callbacks.remove(entry)

        return unsubscribe

    def open_queue(self, maxsize: int = 2000, kinds: Iterable[EventKind] | None = None) -> EventQueue:
        q = EventQueue(maxsize, kinds)
        with self._lock:
            self._queues.append(q)
        return q

    def close_queue(self, q: EventQueue) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    # ---- 發布 ----
    def publish(self, kind: EventKind, **data: Any) -> None:
        """可從任何執行緒呼叫。訂閱者丟出的例外會被吞掉並記錄，不會影響管線。"""
        event = Event(kind, data)
        with self._lock:
            callbacks = list(self._callbacks)
            queues = list(self._queues)
        for q in queues:
            q.put(event)
        for cb, kinds in callbacks:
            if kinds is not None and kind not in kinds:
                continue
            try:
                cb(event)
            except Exception:  # noqa: BLE001 - 訂閱者的錯不可擴散
                log.exception("event subscriber failed on %s", kind.value)


class Throttle:
    """進度事件節流：`ready()` 距離上次回傳 True 超過 interval 才再回傳 True。執行緒安全。"""

    def __init__(self, interval_s: float) -> None:
        self._interval = interval_s
        self._last = 0.0
        self._lock = threading.Lock()

    def ready(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if now - self._last >= self._interval:
                self._last = now
                return True
            return False
