"""InsightFace buffalo_l：SCRFD 人臉偵測 + ArcFace 特徵（Stage 3）。

==========================  實作狀態：骨架  ==========================
已完成：兩個 session 的建立（含 provider 降級與暖機）、前處理參數、輸入輸出契約、
        座標換算工具 `letterbox_to_normalized()`。
未完成：`InsightFaceAnalyzer.analyze()` 內標示 TODO 的三個函式 —— 這些直接解讀模型輸出張量，
        必須拿著真實權重對照參考實作驗證，A 電腦沒有權重所以刻意不寫（沒驗證過的張量解碼
        看起來能跑但結果是錯的，比沒寫更糟）。
接手方式見 HANDOFF.md §4.2；驗證基準是 `pip install insightface` 的
`FaceAnalysis(name="buffalo_l")` 對同一張圖的輸出。
=====================================================================

模型規格（buffalo_l）
---------------------
det_10g.onnx（SCRFD-10GF，含 5 點關鍵點）
  * 輸入：`(1, 3, 640, 640)` float32，**batch 維度寫死為 1**（所以偵測只能一張一張跑，
    `OnnxModel` 會自動偵測並逐張送）。前處理見 `preprocess.SCRFD_SPEC`。
  * 輸出：9 個張量，依 stride (8, 16, 32) 分三組，順序是
    `[score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]`。
    每個位置 2 個 anchor，所以 stride 8 有 (640/8)^2 * 2 = 12800 列。
      - score: (A, 1)  已是 sigmoid 後的機率
      - bbox:  (A, 4)  到 anchor 中心的距離 (left, top, right, bottom)，**單位是 stride**，要乘回去
      - kps:   (A, 10) 5 個點相對 anchor 中心的位移 (dx, dy)，同樣要乘 stride
  * anchor 中心：`(x * stride, y * stride)`，列優先，每個位置重複 2 次。
  * 後處理：分數門檻 `face.det_threshold` -> 解碼 -> NMS（IoU `face.nms_threshold`）。

w600k_r50.onnx（ArcFace ResNet-50，WebFace600K）
  * 輸入：`(N, 3, 112, 112)` float32，動態 batch。前處理見 `preprocess.ARCFACE_SPEC`。
  * 輸出：`(N, 512)`，**尚未正規化**，存檔前要 L2 正規化。
  * 輸入必須是「對齊後」的臉：用 5 個關鍵點對 `ARCFACE_TEMPLATE` 求相似變換
    （旋轉 + 等比縮放 + 平移，不含剪切），再 warpAffine 成 112x112。
    直接裁 bbox 不對齊會讓辨識率大幅下降。
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..core.config import AppConfig
from ..core.types import DecodedImageTask, FaceDetection, LetterboxInfo, ModelInfo
from .onnx_runtime import ModelLoadError, OnnxModel
from .preprocess import ARCFACE_SPEC, SCRFD_SPEC, l2_normalize, to_model_input

log = logging.getLogger(__name__)

#: SCRFD 的特徵圖 stride 與每個位置的 anchor 數
SCRFD_STRIDES: tuple[int, ...] = (8, 16, 32)
SCRFD_NUM_ANCHORS = 2

#: ArcFace 對齊模板：112x112 影像上 5 個關鍵點的標準位置（左眼、右眼、鼻尖、左嘴角、右嘴角）。
#: 取自 insightface/utils/face_align.py 的 `arcface_dst`。
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ARCFACE_SIZE = 112


def letterbox_to_normalized(points: np.ndarray, lb: LetterboxInfo) -> np.ndarray:
    """把偵測輸入（letterbox 畫布）上的像素座標換成原圖正規化座標 0..1。

    Args:
        points: (..., 2) 的 (x, y)，單位是 640x640 畫布上的像素。
    """
    out = np.asarray(points, dtype=np.float32).copy()
    out[..., 0] /= max(1, lb.content_w)
    out[..., 1] /= max(1, lb.content_h)
    return np.clip(out, 0.0, 1.0)


class InsightFaceAnalyzer:
    """`core.interfaces.FaceAnalyzer` 的 buffalo_l 實作。"""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        f = cfg.face
        self._det = OnnxModel(cfg.weight(f.detector.file), f.detector, cfg.runtime)
        self._rec = OnnxModel(cfg.weight(f.recognizer.file), f.recognizer, cfg.runtime)
        self._batch_size = 0

    @property
    def model_name(self) -> str:
        return self._cfg.face.model_name

    @property
    def dim(self) -> int:
        return self._cfg.face.embed_dim

    def load(self) -> list[ModelInfo]:
        size = self._cfg.decode.face_det_size
        det = self._det.open(np.zeros((1, 3, size, size), dtype=np.float32))
        n_out = len(det.output_names)
        if n_out != 9:
            raise ModelLoadError(
                f"{det.model_path.name} 有 {n_out} 個輸出，預期 9 個（3 個 stride x score/bbox/kps）。"
                "不含關鍵點的 SCRFD 變體無法做 ArcFace 對齊"
            )
        rec = self._rec.open(np.zeros((1, 3, ARCFACE_SIZE, ARCFACE_SIZE), dtype=np.float32))
        return [
            ModelInfo(self.model_name, "face_detect", None, det.provider, str(det.model_path)),
            ModelInfo(self.model_name, "face_embed", self.dim, rec.provider, str(rec.model_path)),
        ]

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def close(self) -> None:
        self._det.close()
        self._rec.close()

    # ------------------------------------------------------------------ #
    def analyze(self, tasks: Sequence[DecodedImageTask]) -> list[list[FaceDetection]]:
        """偵測 -> 對齊 -> 取特徵。流程已定，三個 TODO 函式待實作。

        效能設計（實作時請維持）：
          * 偵測逐張（模型 batch 寫死為 1）。
          * 對齊後的臉**跨照片收集成一個大 batch** 再送 ArcFace 一次，不要每張照片各送一次。
        """
        f = self._cfg.face
        results: list[list[FaceDetection]] = []
        crops: list[np.ndarray] = []            # 所有照片的對齊臉，(112, 112, 3) uint8 RGB
        owners: list[tuple[int, int]] = []      # crops[k] 屬於 results[i][j]

        for i, task in enumerate(tasks):
            assert task.face_input is not None and task.face_letterbox is not None
            x = to_model_input(task.face_input[None], SCRFD_SPEC)
            outputs = self._det.run_raw(x)
            boxes, scores, kps = decode_scrfd(outputs, f.det_threshold, f.nms_threshold)   # TODO
            faces: list[FaceDetection] = []
            for j in range(min(len(boxes), f.max_faces_per_photo)):
                xyxy = letterbox_to_normalized(boxes[j].reshape(2, 2), task.face_letterbox).reshape(-1)
                lm = letterbox_to_normalized(kps[j], task.face_letterbox)
                face = FaceDetection(
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])),
                    det_score=float(scores[j]),
                    landmarks=lm,
                )
                faces.append(face)
                if task.work_image is not None and self._face_px(face, task.work_image) >= f.min_face_px:
                    crops.append(align_face(task.work_image, lm))                            # TODO
                    owners.append((i, len(faces) - 1))
            results.append(faces)

        if crops:
            x = to_model_input(np.stack(crops), ARCFACE_SPEC)
            emb = l2_normalize(self._rec.run_batched(x, self._batch_size))
            for (i, j), e in zip(owners, emb):
                results[i][j].embedding = e
        return results

    @staticmethod
    def _face_px(face: FaceDetection, work_image: np.ndarray) -> float:
        """臉在工作圖上的短邊像素數。太小的臉特徵不可靠，只記錄位置不取特徵。"""
        h, w = work_image.shape[:2]
        return min(face.bbox[2] * w, face.bbox[3] * h)


def is_implemented() -> bool:
    """人臉階段是否已經實作完成（`decode_scrfd` 與 `align_face` 不再是骨架）。

    GUI 與 `factory.resolve_stages()` 用它決定要不要把 `faces` 階段排進管線：還是骨架時直接
    略過這個階段，而不是讓整條管線跑到一半以 `NotImplementedError` 失敗。
    以空輸入試呼叫來判斷 —— 骨架會丟 `NotImplementedError`；實作完成後對空輸入不論回傳
    空結果或丟別的例外，都視為已實作。
    """
    try:
        decode_scrfd([], 0.5, 0.4)
        align_face(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((5, 2), dtype=np.float32))
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001
        return True
    return True


# --------------------------------------------------------------------------- #
# TODO（B 電腦實作，需以真實權重驗證）
# --------------------------------------------------------------------------- #
def decode_scrfd(outputs: Sequence[np.ndarray], det_threshold: float, nms_threshold: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 SCRFD 的 9 個輸出解成偵測結果。

    Returns:
        boxes:  (K, 4) float32，640x640 畫布上的像素座標 (x1, y1, x2, y2)
        scores: (K,)   float32，由高到低排序
        kps:    (K, 5, 2) float32，畫布上的像素座標

    實作步驟（細節見模組 docstring 與 insightface/model_zoo/scrfd.py 的 `forward` / `detect`）：
      1. 對每個 stride s（第 idx 組）：`scores = outputs[idx]`、`bbox = outputs[idx + 3] * s`、
         `kps = outputs[idx + 6] * s`。
      2. 產生 anchor 中心（可依 (高, 寬, stride) 快取）。
      3. 取 `scores >= det_threshold` 的列；`distance2bbox`：
         `x1 = cx - l, y1 = cy - t, x2 = cx + r, y2 = cy + b`；關鍵點 `px = cx + dx, py = cy + dy`。
      4. 三個 stride 的結果合併，依分數排序後做 NMS。

    驗證：與 insightface 的 `det_model.detect(img, input_size=(640, 640))` 比對，
    框座標誤差應 < 1 像素。
    """
    raise NotImplementedError("SCRFD 輸出解碼尚未實作（見 HANDOFF.md §4.2）")


def align_face(work_image: np.ndarray, landmarks_norm: np.ndarray) -> np.ndarray:
    """依 5 個關鍵點把臉對齊成 ArcFace 的輸入。

    Args:
        work_image: (h, w, 3) uint8 RGB 工作圖。
        landmarks_norm: (5, 2) 原圖正規化座標；乘上 (w, h) 得到工作圖上的像素座標。
    Returns:
        (112, 112, 3) uint8 RGB。

    實作：以最小平方法求「來源 5 點 -> `ARCFACE_TEMPLATE`」的相似變換（Umeyama；
    `cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)` 或 `skimage.transform.SimilarityTransform`），
    再 `cv2.warpAffine(work_image, M, (112, 112), borderValue=0)`。

    驗證：與 `insightface.utils.face_align.norm_crop(img, kps)` 的輸出逐像素比對。
    """
    raise NotImplementedError("人臉對齊尚未實作（見 HANDOFF.md §4.2）")
