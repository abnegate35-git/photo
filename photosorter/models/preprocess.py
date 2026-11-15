"""Batch 矩陣前處理框架：把管線裡流通的 `uint8 / NHWC / RGB` 影像轉成模型要的張量。

管線側永遠只產生「縮放到模型輸入尺寸的 uint8 RGB」，數值正規化集中在這裡一次向量化完成。
好處：佇列裡放 uint8（float32 的 1/4 記憶體），而且每個模型的前處理參數只出現在一個地方。

各模型的參數（必須與訓練時一致，改錯不會報錯只會讓準確度默默變差）：

    模型      縮放方式                         正規化                      通道
    SigLIP    直接壓成 224x224（不裁切）        (x/255 - 0.5) / 0.5         RGB
    SCRFD     等比縮放貼左上角，補 0 到 640     (x - 127.5) / 128           RGB
    ArcFace   5 點相似變換對齊到 112x112        (x - 127.5) / 127.5         RGB

InsightFace 原始程式吃的是 OpenCV 的 BGR 影像並在 `blobFromImage(swapRB=True)` 轉成 RGB，
所以模型實際的輸入是 RGB —— 我們的管線本來就是 RGB，不需要再換通道。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    """`y = (x * scale - mean) / std`，x 為 0..255 的像素值。"""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    scale: float = 1.0
    swap_rb: bool = False                      # True = 模型要 BGR


SIGLIP_SPEC = PreprocessSpec(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), scale=1.0 / 255.0)
SCRFD_SPEC = PreprocessSpec(mean=(127.5, 127.5, 127.5), std=(128.0, 128.0, 128.0))
ARCFACE_SPEC = PreprocessSpec(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))


def stack_images(images: Sequence[np.ndarray]) -> np.ndarray:
    """把同尺寸的 (H, W, 3) uint8 影像疊成 (N, H, W, 3)。尺寸不一致會丟 ValueError。"""
    if not images:
        raise ValueError("空的影像清單")
    return np.stack(images, axis=0)


def to_model_input(batch: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """(N, H, W, 3) uint8 RGB -> (N, 3, H, W) float32，C-contiguous。

    整批一次向量化運算；先轉軸再正規化，讓結果直接是連續記憶體（ONNX Runtime 可零複製接手）。
    """
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError(f"預期 (N, H, W, 3)，收到 {batch.shape}")
    if batch.dtype != np.uint8:
        raise ValueError(f"預期 uint8，收到 {batch.dtype}")
    if spec.swap_rb:
        batch = batch[..., ::-1]
    x = np.ascontiguousarray(batch.transpose(0, 3, 1, 2), dtype=np.float32)
    # 合併成一次乘加：y = x * a + b
    a = (spec.scale / np.asarray(spec.std, dtype=np.float32)).reshape(1, 3, 1, 1)
    b = (-np.asarray(spec.mean, dtype=np.float32) / np.asarray(spec.std, dtype=np.float32)).reshape(1, 3, 1, 1)
    x *= a
    x += b
    return x


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """沿 `axis` 做 L2 正規化，回傳 float32。零向量維持為零。"""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """數值穩定的 sigmoid（大的負數不會溢位）。"""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
