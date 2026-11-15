"""時間密度事件聚類：把照片依「拍攝時間的間隔」與「GPS 距離」切成一段一段的事件。

演算法（一維的 gap-based 分段，O(N log N)）：
  1. 依拍攝時間排序。
  2. 由前往後掃，遇到下列任一情況就切開，開始新的一段：
       a. 與前一張的時間差 > `events.gap_minutes`
       b. 這張與「本段最近一張有 GPS 的照片」距離 > `events.max_distance_km`
          （兩張都要有 GPS 才比；沒有 GPS 的照片只看時間）
  3. 張數少於 `events.min_photos` 的段不成立事件（那些照片的 `event_id` 留 NULL）。

這是全域後處理：整條管線跑完後執行一次，每次都對全庫重算（十萬張實測約 0.1 秒），
以 `Repository.replace_auto_events()` 取代所有自動事件；使用者編輯過的事件不會被動到。

輸入只包含時間可信的照片（EXIF 或檔名），見 `Repository.fetch_event_inputs()`。
"""
from __future__ import annotations

from typing import Sequence

from ..core.config import EventConfig
from ..core.types import EventCluster, EventInput
from ..utils.geo import haversine_km


class TimeGapEventClusterer:
    """`core.interfaces.EventClusterer` 的預設實作。"""

    def __init__(self, cfg: EventConfig) -> None:
        self._gap_s = max(1, cfg.gap_minutes) * 60
        self._max_km = cfg.max_distance_km
        self._min_photos = max(1, cfg.min_photos)

    def cluster(self, items: Sequence[EventInput]) -> list[EventCluster]:
        ordered = sorted(items, key=lambda it: (it.taken_ts, it.photo_id))
        clusters: list[EventCluster] = []
        segment: list[EventInput] = []
        last_gps: tuple[float, float] | None = None

        def flush() -> None:
            if len(segment) >= self._min_photos:
                clusters.append(self._make(segment))

        for it in ordered:
            has_gps = it.lat is not None and it.lon is not None
            if segment:
                split = it.taken_ts - segment[-1].taken_ts > self._gap_s
                if not split and has_gps and last_gps is not None and self._max_km > 0:
                    split = haversine_km(last_gps[0], last_gps[1], it.lat, it.lon) > self._max_km  # type: ignore[arg-type]
                if split:
                    flush()
                    segment = []
                    last_gps = None
            segment.append(it)
            if has_gps:
                last_gps = (it.lat, it.lon)  # type: ignore[assignment]
        if segment:
            flush()
        return clusters

    @staticmethod
    def _make(segment: list[EventInput]) -> EventCluster:
        gps = [(it.lat, it.lon) for it in segment if it.lat is not None and it.lon is not None]
        lat = lon = None
        if gps:
            # 簡單平均。跨越 180 度經線的事件中心點會不準，但只用於顯示，不影響分群。
            lat = sum(g[0] for g in gps) / len(gps)   # type: ignore[misc]
            lon = sum(g[1] for g in gps) / len(gps)   # type: ignore[misc]
        return EventCluster(
            photo_ids=[it.photo_id for it in segment],
            start_ts=segment[0].taken_ts,
            end_ts=segment[-1].taken_ts,
            center_lat=lat,
            center_lon=lon,
        )
