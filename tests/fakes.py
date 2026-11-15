"""測試用的假元件與測試資料產生器。

假元件實作 `core.interfaces` 的 Protocol，讓整條管線可以在沒有權重、沒有 GPU 的機器上
端到端執行。它們的輸出是輸入的確定性函數，所以測試可以斷言具體數值。
接手實作真實模型時，這些假元件同時是「介面該長什麼樣」的最小範例。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from photosorter.core.config import AppConfig, load_config
from photosorter.core.hardware import CPU
from photosorter.core.types import DecodedImageTask, FaceDetection, GeoPlace, ModelInfo

EMBED_DIM = 16
FACE_DIM = 8


def make_config(root: Path, **overrides: Any) -> AppConfig:
    """指到暫存資料夾的設定：資料庫在 `root/data`，向量維度縮小以加快測試。"""
    base: dict[str, Any] = {
        "siglip": {"embed_dim": EMBED_DIM, "default_threshold": 0.5},
        "face": {"embed_dim": FACE_DIM},
        "runtime": {"decode_workers": 3, "batch_size": 4, "batch_timeout_ms": 30, "queue_maxsize": 8},
        "pipeline": {"progress_interval_ms": 0},
        "scan": {"min_file_size": 1},
    }
    for key, value in overrides.items():
        base.setdefault(key, {}).update(value)
    return load_config(root=root, overrides=base)


def make_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (120, 120, 120),
    size: tuple[int, int] = (320, 240),
    taken: str | None = "2024:05:01 10:00:00",
    camera: str | None = "TestCam X1",
    orientation: int | None = None,
    gps: tuple[float, float] | None = None,
    seed: int = 0,
) -> Path:
    """產生一張帶 EXIF 的 JPEG。`color` 決定假模型的輸出，`seed` 讓內容（與 SHA-256）不同。"""
    rng = np.random.default_rng(seed)
    arr = np.empty((size[1], size[0], 3), dtype=np.uint8)
    arr[:] = color
    arr[::8, ::8] = rng.integers(0, 255, size=arr[::8, ::8].shape, dtype=np.uint8)   # 一點紋理
    img = Image.fromarray(arr)
    exif = Image.Exif()
    if camera:
        exif[0x010F] = "TestMake"
        exif[0x0110] = camera
    if orientation:
        exif[0x0112] = orientation
    if taken:
        exif.get_ifd(0x8769)[0x9003] = taken
    if gps:
        lat, lon = gps
        g = exif.get_ifd(0x8825)
        g[1] = "N" if lat >= 0 else "S"
        g[2] = _dms(abs(lat))
        g[3] = "E" if lon >= 0 else "W"
        g[4] = _dms(abs(lon))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90, exif=exif)
    return path


def _dms(deg: float) -> tuple[float, float, float]:
    d = int(deg)
    m = int((deg - d) * 60)
    s = round((deg - d - m / 60) * 3600, 4)
    return (float(d), float(m), float(s))


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / max(1e-12, float(np.linalg.norm(v)))).astype(np.float32)


class FakeVision:
    """影像向量 = 平均顏色決定的單位向量。紅色系的圖在第 0 維最大，綠 1，藍 2。"""

    model_name = "fake-vision"

    def __init__(self, dim: int = EMBED_DIM, delay_s: float = 0.0, fail_on_red: int | None = None) -> None:
        self.dim = dim
        self.delay_s = delay_s
        self.fail_on_red = fail_on_red         # 平均紅色值等於此數的圖會讓推論丟例外
        self.batch_sizes: list[int] = []
        self.loaded_on: str | None = None
        self.closed = False

    def load(self) -> list[ModelInfo]:
        self.loaded_on = threading.current_thread().name
        return [ModelInfo(self.model_name, "image_embed", self.dim, CPU, "<fake>")]

    def embed_images(self, batch: np.ndarray) -> np.ndarray:
        assert batch.dtype == np.uint8 and batch.ndim == 4 and batch.shape[-1] == 3
        assert threading.current_thread().name == self.loaded_on, "推論必須在載入模型的那條執行緒上"
        self.batch_sizes.append(batch.shape[0])
        if self.delay_s:
            time.sleep(self.delay_s)
        means = batch.reshape(batch.shape[0], -1, 3).mean(axis=1)
        if self.fail_on_red is not None and any(abs(m[0] - self.fail_on_red) < 3 for m in means):
            raise RuntimeError("fake vision failure")
        out = np.zeros((batch.shape[0], self.dim), dtype=np.float32)
        out[:, :3] = means / 255.0
        out[:, 3] = 0.05
        return np.stack([_unit(v) for v in out])

    def close(self) -> None:
        self.closed = True


class FakeTextEncoder:
    """文字向量：prompt 裡出現 red/green/blue 就指向對應的維度。"""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.calls = 0

    def load(self) -> list[ModelInfo]:
        return [ModelInfo("fake-text", "text_embed", self.dim, CPU, "<fake>")]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, word in enumerate(("red", "green", "blue")):
                if word in t.lower():
                    out[i, j] = 1.0
            if not out[i].any():
                out[i, 5] = 1.0
        return np.stack([_unit(v) for v in out])

    def close(self) -> None:
        pass


class FakeFaces:
    """每張圖回傳一張臉；記錄被分析過的 photo_id，讓測試能確認「條件式略過」真的沒有呼叫。"""

    model_name = "fake-faces"

    def __init__(self, dim: int = FACE_DIM, not_implemented: bool = False) -> None:
        self.dim = dim
        self.not_implemented = not_implemented
        self.analyzed: list[int] = []

    def load(self) -> list[ModelInfo]:
        return [ModelInfo(self.model_name, "face_embed", self.dim, CPU, "<fake>")]

    def analyze(self, tasks: Sequence[DecodedImageTask]) -> list[list[FaceDetection]]:
        if self.not_implemented:
            raise NotImplementedError("fake skeleton")
        out = []
        for t in tasks:
            assert t.face_input is not None and t.face_letterbox is not None and t.work_image is not None
            assert t.face_input.shape[:2] == (640, 640)
            self.analyzed.append(t.photo_id)
            emb = np.zeros(self.dim, dtype=np.float32)
            emb[t.photo_id % self.dim] = 1.0
            out.append([FaceDetection(bbox=(0.25, 0.25, 0.5, 0.5), det_score=0.9,
                                      landmarks=np.full((5, 2), 0.5, dtype=np.float32), embedding=emb)])
        return out

    def close(self) -> None:
        pass


class FakeGeocoder:
    def __init__(self) -> None:
        self.calls = 0

    def lookup(self, coords: Sequence[tuple[float, float]]) -> list[GeoPlace]:
        self.calls += 1
        return [GeoPlace("TW" if lat > 0 else "AU", "Region", f"City{int(lat)}") for lat, _lon in coords]


def rename(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)
