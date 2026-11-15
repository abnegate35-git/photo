"""Stage 1 中繼資料：EXIF 解析與螢幕截圖判定。

==========================  實作狀態：基準版  ==========================
`BasicMetadataExtractor` 是能用的基準實作（拍攝時間、相機、曝光參數、GPS、方向、檔名時間），
已有單元測試。接手後可擴充的項目見 HANDOFF.md §4.4（時區、MakerNote、XMP、影片…）。
要換成更完整的解析器，實作 `core.interfaces.MetadataExtractor` 後在 factory 注入即可。
=======================================================================
"""
from __future__ import annotations

import calendar
import logging
import os
import re
from datetime import datetime
from typing import Any

from ..core.config import ScreenshotConfig
from ..core.types import PhotoMeta, RawFile

log = logging.getLogger(__name__)

# EXIF tag id
_TAG_MAKE, _TAG_MODEL, _TAG_ORIENTATION, _TAG_DATETIME = 0x010F, 0x0110, 0x0112, 0x0132
_IFD_EXIF, _IFD_GPS = 0x8769, 0x8825
_EXIF_DT_ORIGINAL, _EXIF_DT_DIGITIZED = 0x9003, 0x9004
_EXIF_EXPOSURE, _EXIF_FNUMBER, _EXIF_ISO, _EXIF_FOCAL, _EXIF_LENS = 0x829A, 0x829D, 0x8827, 0x920A, 0xA434
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON, _GPS_ALT_REF, _GPS_ALT = 1, 2, 3, 4, 5, 6

# 檔名裡的時間：IMG_20230815_142355、2023-08-15 14.23.55、Screenshot_20230815-142355…
_NAME_DATETIME = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[-_ T.]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})(?!\d)"
)
# 只有日期：IMG-20230815-WA0001
_NAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)")


def local_naive_to_ts(dt: datetime) -> int:
    """把「沒有時區的當地時間」當成 UTC 換成 epoch 秒。只用於排序與計算間隔。"""
    return calendar.timegm(dt.timetuple())


def format_taken(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def parse_exif_datetime(value: Any) -> datetime | None:
    """EXIF 的 'YYYY:MM:DD HH:MM:SS'。相機沒設時間時常見全 0 或空白，一律視為沒有。"""
    if not isinstance(value, str):
        return None
    s = value.strip().replace("\x00", "")[:19]
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return dt if dt.year >= 1990 else None
    return None


def parse_filename_datetime(file_name: str) -> datetime | None:
    stem = os.path.splitext(file_name)[0]
    m = _NAME_DATETIME.search(stem)
    try:
        if m:
            return datetime(*(int(g) for g in m.groups()))
        m = _NAME_DATE.search(stem)
        if m:
            return datetime(*(int(g) for g in m.groups()))
    except ValueError:                          # 像日期但不是（例如 20231345）
        return None
    return None


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _clean_str(v: Any) -> str | None:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if not isinstance(v, str):
        return None
    s = v.replace("\x00", "").strip()
    return s or None


def _dms_to_degrees(dms: Any, ref: Any) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "replace")
    if isinstance(ref, str) and ref.strip().upper() in ("S", "W"):
        deg = -deg
    return deg


class BasicMetadataExtractor:
    """以 Pillow 讀 EXIF 的基準實作。無狀態，可被多條解碼執行緒同時呼叫。"""

    def extract(self, raw: RawFile, image: Any | None) -> PhotoMeta:
        meta = PhotoMeta()
        if image is not None:
            try:
                self._read_exif(image, meta)
            except Exception as e:  # noqa: BLE001 - EXIF 壞掉不該讓整張照片失敗
                log.debug("EXIF 解析失敗 %s: %s", raw.path, e)
        if meta.taken_at is None:
            dt = parse_filename_datetime(os.path.basename(raw.path))
            if dt is not None:
                meta.taken_at, meta.taken_ts, meta.taken_src = format_taken(dt), local_naive_to_ts(dt), "filename"
        return meta

    @staticmethod
    def _read_exif(image: Any, meta: PhotoMeta) -> None:
        exif = image.getexif()
        if not exif:
            return
        meta.camera_make = _clean_str(exif.get(_TAG_MAKE))
        meta.camera_model = _clean_str(exif.get(_TAG_MODEL))
        o = exif.get(_TAG_ORIENTATION)
        if isinstance(o, int) and 1 <= o <= 8:
            meta.orientation = o

        sub = exif.get_ifd(_IFD_EXIF)
        dt = (
            parse_exif_datetime(sub.get(_EXIF_DT_ORIGINAL))
            or parse_exif_datetime(sub.get(_EXIF_DT_DIGITIZED))
            or parse_exif_datetime(exif.get(_TAG_DATETIME))
        )
        if dt is not None:
            meta.taken_at, meta.taken_ts, meta.taken_src = format_taken(dt), local_naive_to_ts(dt), "exif"
        meta.exposure_time = _to_float(sub.get(_EXIF_EXPOSURE))
        meta.f_number = _to_float(sub.get(_EXIF_FNUMBER))
        meta.focal_length = _to_float(sub.get(_EXIF_FOCAL))
        meta.lens_model = _clean_str(sub.get(_EXIF_LENS))
        iso = sub.get(_EXIF_ISO)
        if isinstance(iso, (tuple, list)) and iso:
            iso = iso[0]
        if isinstance(iso, int) and iso > 0:
            meta.iso = iso

        gps = exif.get_ifd(_IFD_GPS)
        if gps:
            lat = _dms_to_degrees(gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF))
            lon = _dms_to_degrees(gps.get(_GPS_LON), gps.get(_GPS_LON_REF))
            # (0, 0) 幾乎一定是相機沒定位到時寫的預設值，不是真的在幾內亞灣
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180 \
                    and not (abs(lat) < 1e-9 and abs(lon) < 1e-9):
                meta.gps_lat, meta.gps_lon = lat, lon
                alt = _to_float(gps.get(_GPS_ALT))
                if alt is not None:
                    ref = gps.get(_GPS_ALT_REF)
                    if isinstance(ref, bytes):
                        ref = ref[0] if ref else 0
                    meta.gps_alt = -alt if ref == 1 else alt


class ScreenshotDetector:
    """以路徑與尺寸規則初判螢幕截圖（Stage 1，不用 AI）。

    判定為截圖的條件（任一成立）：
      1. 檔名或任一層資料夾名稱含有截圖關鍵字（`screenshot.name_patterns`）。
      2. 沒有相機 EXIF（Make/Model 都沒有）**且**長寬正好等於某個常見螢幕解析度。

    刻意保守：只憑長寬比（16:9、19.5:9…）不算，因為手機照片與下載的圖片也常是這些比例。
    漏掉的截圖還有 Stage 2 的 `screenshot` 標籤（SigLIP）可以補抓。
    """

    def __init__(self, cfg: ScreenshotConfig) -> None:
        self._patterns = [p.lower() for p in cfg.name_patterns if p]
        self._resolutions = {tuple(sorted(int(v) for v in r)) for r in cfg.screen_resolutions if len(r) == 2}

    def is_screenshot(self, path: str, width: int, height: int, meta: PhotoMeta) -> bool:
        lowered = path.lower()
        if any(p in lowered for p in self._patterns):
            return True
        has_camera = bool(meta.camera_make or meta.camera_model)
        return not has_camera and tuple(sorted((width, height))) in self._resolutions
