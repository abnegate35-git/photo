"""地理工具：距離計算與離線反向地理編碼。"""
from __future__ import annotations

import logging
import math
import threading
from typing import Sequence

from ..core.types import GeoPlace

log = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """兩個經緯度之間的大圓距離（公里）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


class OfflineGeocoder:
    """`reverse_geocoder` 的薄包裝（`core.interfaces.ReverseGeocoder`）。

    * 完全離線：資料是套件內附的 GeoNames cities1000（人口 > 1000 的城市），回傳最近的城市。
      海上或荒野的座標也會得到「最近的城市」，可能在很遠的地方 —— 這是資料集的性質。
    * 第一次查詢要載入資料並建 KD-tree（A 電腦實測 1.6 秒），之後 5000 筆只要 4 毫秒。所以
      **延遲到第一次 `lookup()` 才 import**，而且設計成「掃描結束後整批查一次」，
      不要每張照片各查一次。
    * 一律用 `mode=1`（單行程）。`mode=2` 會用 multiprocessing，在 Windows 上會重新 import
      主程式，打包成 exe 後還會不斷開新行程。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def lookup(self, coords: Sequence[tuple[float, float]]) -> list[GeoPlace]:
        if not coords:
            return []
        import reverse_geocoder as rg

        with self._lock:                       # 套件內部有全域單例，序列化呼叫比較安全
            results = rg.search([(float(a), float(b)) for a, b in coords], mode=1, verbose=False)
        return [
            GeoPlace(
                country_code=(r.get("cc") or None),
                admin1=(r.get("admin1") or None),
                city=(r.get("name") or None),
            )
            for r in results
        ]
