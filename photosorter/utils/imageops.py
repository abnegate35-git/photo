"""影像基本運算。優先用 OpenCV（快），沒裝時退回 Pillow / NumPy。

退回路徑只是為了讓沒有 OpenCV 的環境（例如只裝了 numpy + pillow 的開發機）也能跑測試；
正式環境請安裝 requirements.txt。兩條路徑的數值不保證逐位元相同（內插與邊界處理不同），
所以 `blur_score` 的門檻要在**正式環境**上校準。

所有函式的影像輸入輸出都是 `uint8`、HWC、RGB（灰階為 HW）。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..core.types import LetterboxInfo


@lru_cache(maxsize=1)
def _cv2() -> Any | None:
    try:
        import cv2

        cv2.setNumThreads(1)       # 平行度由解碼執行緒池負責，OpenCV 內部不要再開執行緒
        return cv2
    except Exception:  # noqa: BLE001
        return None


def have_cv2() -> bool:
    return _cv2() is not None


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """縮放到指定大小。縮小用面積平均（抗鋸齒），放大用雙三次。"""
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    shrinking = width < w or height < h
    cv2 = _cv2()
    if cv2 is not None:
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
        return cv2.resize(img, (width, height), interpolation=interp)
    from PIL import Image

    resample = Image.Resampling.BOX if shrinking else Image.Resampling.BICUBIC
    return np.asarray(Image.fromarray(img).resize((width, height), resample))


def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """等比縮到長邊不超過 `long_edge`；本來就比較小則原樣回傳（不放大）。"""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return img
    s = long_edge / m
    return resize(img, max(1, round(w * s)), max(1, round(h * s)))


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    cv2 = _cv2()
    if cv2 is not None:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ITU-R BT.601，與 OpenCV 相同的係數
    g = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    return np.clip(g + 0.5, 0, 255).astype(np.uint8)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian：影像銳利度指標，越小越模糊。

    分數與解析度有關，呼叫端必須先把影像正規化到固定長邊（`decode.blur_long_edge`），
    不同相機的照片才能用同一個門檻比較。
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    cv2 = _cv2()
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    g = gray.astype(np.float64)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def letterbox_square(img: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    """等比縮放後貼在 `size x size` 黑色畫布的**左上角**（InsightFace SCRFD 的前處理慣例）。

    回傳的 `LetterboxInfo` 用來把偵測結果換回原圖座標。小圖會被放大到填滿長邊。
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    nw, nh = min(nw, size), min(nh, size)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resize(img, nw, nh)
    return canvas, LetterboxInfo(scale=scale, content_w=nw, content_h=nh)


def dhash64(gray: np.ndarray) -> int:
    """64-bit difference hash，回傳**有號** 64-bit 整數（可直接存進 SQLite INTEGER）。

    演算法：縮成 9x8，每列相鄰像素比大小得 8 個 bit，共 64 bit，列優先、高位在前。
    對縮放、輕微壓縮與亮度變化穩定；兩個 hash 的漢明距離 ≤ 約 6~10 可視為近似重複
    （門檻需校準）。與 `imagehash.dhash()` 同一演算法，但縮圖內插不同，兩者的值不可混用比較。
    """
    small = resize(gray, 9, 8).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = int.from_bytes(np.packbits(bits).tobytes(), "big")
    return value - (1 << 64) if value >= (1 << 63) else value


def hamming64(a: int, b: int) -> int:
    """兩個（有號或無號）64-bit hash 的漢明距離。"""
    return ((a ^ b) & 0xFFFFFFFFFFFFFFFF).bit_count()
