"""Single-Decode Multi-Resize：一張照片只讀檔一次、只解碼一次，在記憶體裡切出所有分支。

    硬碟 ──讀一次──▶ bytes ─┬─▶ SHA-256（Stage 0：搬移/重複判定）
                             ├─▶ Pillow 只讀檔頭：尺寸 + EXIF（Stage 1，不解碼像素）
                             └─▶ 解碼一次（JPEG 用 DCT 縮放直接解成 1/2、1/4、1/8）
                                    │ 套用 EXIF 方向
                                    ▼
                               工作圖（長邊 ≤ 1280，RGB uint8）
                                 ├─▶ 灰階（長邊 ≤ 1024）─▶ blur_score、dHash        ← 分支 1
                                 ├─▶ 640x640 letterbox ─▶ InsightFace 偵測           ← 分支 2
                                 ├─▶ 224x224 ─▶ SigLIP                               ← 分支 3
                                 └─▶ 工作圖本身 ─▶ 人臉對齊的來源（裁 112x112）

為什麼人臉對齊用工作圖而不是 640 那張：640 的畫面裡一張臉常常只有 30~60 像素，放大成
112x112 餵 ArcFace 特徵品質很差；從 1280 的工作圖裁，解析度多一倍，分群明顯較準。

JPEG 的 DCT 縮放（`cv2.IMREAD_REDUCED_COLOR_*` / Pillow 的 `draft()`）是最大的加速來源：
解碼器直接解出 1/2、1/4、1/8 尺寸，不必先還原全尺寸再縮小。A 電腦實測一張 6000x4000
（2400 萬像素）的 JPEG 走完整個 `decode()`：全尺寸解碼 342 ms，DCT 縮放 82 ms。

本類別無狀態，會被多條解碼執行緒同時呼叫。
"""
from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

from ..core.config import AppConfig
from ..core.interfaces import MetadataExtractor
from ..core.status import ProcessStatus
from ..core.types import DecodedImage, FileCandidate, PhotoMeta, RawFile, Stage1Record
from ..utils import imageops
from ..utils.hashing import sha256_bytes
from .metadata import BasicMetadataExtractor, ScreenshotDetector, format_taken, local_naive_to_ts

log = logging.getLogger(__name__)

_JPEG_EXTS = {"jpg", "jpeg", "jpe", "jfif"}
_HEIF_EXTS = {"heic", "heif", "avif"}


class DecodeError(Exception):
    """檔案損毀或格式不支援。引擎會把這張標成 ERR_DECODE 後繼續處理下一張。"""


def apply_orientation(img: np.ndarray, orientation: int | None) -> np.ndarray:
    """依 EXIF Orientation（1..8）把影像轉正。與 `PIL.ImageOps.exif_transpose` 結果相同。"""
    if orientation is None or orientation == 1:
        return img
    if orientation == 2:
        out = img[:, ::-1]
    elif orientation == 3:
        out = img[::-1, ::-1]
    elif orientation == 4:
        out = img[::-1]
    elif orientation == 5:
        out = img.transpose(1, 0, 2)
    elif orientation == 6:
        out = np.rot90(img, k=-1)
    elif orientation == 7:
        out = img[::-1, ::-1].transpose(1, 0, 2)
    elif orientation == 8:
        out = np.rot90(img, k=1)
    else:
        return img
    return np.ascontiguousarray(out)


class DefaultImageDecoder:
    """`core.interfaces.ImageDecoder` 的預設實作。"""

    def __init__(self, cfg: AppConfig, metadata: MetadataExtractor | None = None) -> None:
        self._cfg = cfg
        self._d = cfg.decode
        self._metadata = metadata or BasicMetadataExtractor()
        self._screenshot = ScreenshotDetector(cfg.screenshot)
        self._raw_exts = {e.lower() for e in cfg.scan.raw_extensions}
        self._heif_ready: bool | None = None

    # ------------------------------------------------------------------ #
    def read(self, cand: FileCandidate) -> RawFile:
        with open(cand.path, "rb") as f:
            data = f.read()
        return RawFile(path=cand.path, data=data, sha256=sha256_bytes(data))

    def decode(self, cand: FileCandidate, raw: RawFile, pending: ProcessStatus) -> DecodedImage:
        t0 = time.perf_counter()
        ext = os.path.splitext(cand.path)[1].lstrip(".").lower()
        if ext in self._raw_exts:
            rgb, meta = self._decode_raw(raw)
            full_h, full_w = rgb.shape[:2]
        else:
            pil = self._open_header(raw, ext)
            meta = self._metadata.extract(raw, pil)
            rgb = self._decode_pixels(raw, pil, ext)
            # pillow-heif 解碼時已依檔案內的旋轉資訊轉正，並把 EXIF Orientation 重設為 1
            # （以 1.8.1 實測方向 1/3/6/8 皆與 PIL.ImageOps.exif_transpose 一致）。
            # 這裡再明確排除一次，萬一日後版本不再重設 EXIF 也不會轉兩次。
            rotate = ext not in _HEIF_EXTS
            if rotate:
                rgb = apply_orientation(rgb, meta.orientation)
            if pil is not None:
                # 原圖尺寸取自檔頭（解碼出來的可能是 DCT 縮小版）
                full_w, full_h = pil.size
                if rotate and meta.orientation in (5, 6, 7, 8):
                    full_w, full_h = full_h, full_w
            else:
                full_h, full_w = rgb.shape[:2]

        if meta.taken_at is None:
            # 最後手段：檔案修改時間。標成 'mtime'，事件聚類不會採用它。
            dt = datetime.fromtimestamp(cand.mtime_ns / 1e9)
            meta.taken_at, meta.taken_ts, meta.taken_src = format_taken(dt), local_naive_to_ts(dt), "mtime"

        work = imageops.resize_long_edge(rgb, self._d.work_long_edge)
        del rgb
        gray = imageops.to_gray(imageops.resize_long_edge(work, self._d.blur_long_edge))
        record = Stage1Record(
            path=cand.path, file_size=cand.size, mtime_ns=cand.mtime_ns, sha256=raw.sha256,
            width=int(full_w), height=int(full_h), meta=meta,
            dhash=imageops.dhash64(gray),
            blur_score=imageops.laplacian_variance(gray),
            is_screenshot=self._screenshot.is_screenshot(cand.path, int(full_w), int(full_h), meta),
        )

        siglip_input = face_input = letterbox = work_image = None
        if pending & ProcessStatus.stage2():
            s = self._d.siglip_size
            # SigLIP 的前處理是直接壓成正方形（不保持長寬比、不裁切）
            siglip_input = np.ascontiguousarray(imageops.resize(work, s, s))
        if pending & ProcessStatus.stage3():
            face_input, letterbox = imageops.letterbox_square(work, self._d.face_det_size)
            work_image = np.ascontiguousarray(work)

        return DecodedImage(
            record=record, siglip_input=siglip_input, face_input=face_input,
            face_letterbox=letterbox, work_image=work_image,
            decode_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------ #
    def _ensure_heif(self) -> bool:
        if self._heif_ready is None:
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
                self._heif_ready = True
            except Exception as e:  # noqa: BLE001
                log.warning("pillow-heif 無法載入，HEIC/HEIF 將無法解碼：%s", e)
                self._heif_ready = False
        return self._heif_ready

    def _open_header(self, raw: RawFile, ext: str) -> Any | None:
        """用 Pillow 開檔（只讀檔頭，不解碼像素）。Pillow 不認得的格式回傳 None。"""
        from PIL import Image

        if ext in _HEIF_EXTS:
            self._ensure_heif()
        try:
            pil = Image.open(io.BytesIO(raw.data))
        except Exception:  # noqa: BLE001
            return None
        w, h = pil.size
        if w <= 0 or h <= 0 or w * h > self._d.max_pixels:
            raise DecodeError(f"影像尺寸異常：{w}x{h}")
        return pil

    def _decode_pixels(self, raw: RawFile, pil: Any | None, ext: str) -> np.ndarray:
        """解碼成 RGB uint8（尚未套用方向）。先試 OpenCV，不行再用 Pillow。"""
        target = self._d.work_long_edge
        cv2 = imageops._cv2()
        if cv2 is not None and ext not in _HEIF_EXTS:
            flag = cv2.IMREAD_COLOR
            if pil is not None and ext in _JPEG_EXTS and self._d.jpeg_draft:
                long_edge = max(pil.size)
                for factor, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                                        (2, cv2.IMREAD_REDUCED_COLOR_2)):
                    if long_edge / factor >= target:
                        flag = reduced
                        break
            # 方向由我們自己依 EXIF 處理（兩條解碼路徑行為一致），所以叫 OpenCV 不要自動轉
            bgr = cv2.imdecode(np.frombuffer(raw.data, dtype=np.uint8), flag | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is not None and bgr.ndim == 3 and bgr.shape[2] == 3:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if pil is None:
            raise DecodeError("不支援的影像格式或檔案已損毀")
        try:
            if ext in _JPEG_EXTS and self._d.jpeg_draft:
                pil.draft("RGB", (target, target))      # 只會縮到「不小於」要求的尺寸
            return np.asarray(pil.convert("RGB"))
        except Exception as e:  # noqa: BLE001
            raise DecodeError(f"解碼失敗：{e}") from e

    def _decode_raw(self, raw: RawFile) -> tuple[np.ndarray, PhotoMeta]:
        """RAW（DNG/CR2/NEF/ARW…）。

        TODO（選配，見 HANDOFF.md §4.4）：用 `rawpy` 取內嵌的 JPEG 預覽
        （`rawpy.imread(io.BytesIO(data)).extract_thumb()`）—— 比完整 demosaic 快兩個數量級，
        對分類與人臉已經足夠。EXIF 從預覽 JPEG 或 `exifread` 讀。
        `scan.enable_raw` 預設為 false，所以正常情況不會走到這裡。
        """
        raise DecodeError("RAW 格式尚未支援（scan.enable_raw 需搭配 rawpy 實作）")
