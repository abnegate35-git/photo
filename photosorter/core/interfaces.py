"""介面契約（`typing.Protocol`）。

`PipelineEngine` 只依賴這裡的 Protocol，不 import 任何具體模型類別；具體實作由
`pipeline/factory.py` 組裝後注入。因此：
  * 測試用假實作（`tests/fakes.py`）就能在沒有權重、沒有 GPU 的機器上跑完整條管線。
  * 換模型（SigLIP2、別的人臉模型）只要實作同一個 Protocol，引擎不必改。

執行緒契約（實作者必須遵守）：
  * `VisionEmbedder` / `TagScorer` / `FaceAnalyzer` 的 `load()` 與推論方法**只會被推論執行緒
    呼叫**，且一次只有一個呼叫在進行。實作不需要加鎖，也不可以自己再開執行緒搶 GPU。
  * `ImageDecoder` / `MetadataExtractor` 會被多個解碼執行緒**同時**呼叫，必須無狀態或自行保護。
  * 所有方法都不可以碰 GUI。
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from .status import ProcessStatus
from .types import (
    DecodedImage,
    DecodedImageTask,
    EventCluster,
    EventInput,
    FaceDetection,
    FileCandidate,
    GateRole,
    GeoPlace,
    ModelInfo,
    PhotoMeta,
    RawFile,
    TagHit,
)


# --------------------------------------------------------------------------- #
# 解碼端（多執行緒）
# --------------------------------------------------------------------------- #
@runtime_checkable
class ImageDecoder(Protocol):
    def read(self, cand: FileCandidate) -> RawFile:
        """把檔案整個讀進記憶體並算出 sha256。只做 I/O 與雜湊，不解碼。

        Raises:
            OSError: 檔案讀不到（被刪、權限、網路磁碟斷線）。
        """
        ...

    def decode(self, cand: FileCandidate, raw: RawFile, pending: ProcessStatus) -> DecodedImage:
        """解碼一次，產生 Stage 1 結果與各 AI 分支。

        Args:
            pending: 這張還欠的階段。不含 Stage 2 就不必產生 `siglip_input`，
                不含 Stage 3 就不必產生 `face_input` / `work_image`。

        Raises:
            DecodeError: 檔案損毀或格式不支援。引擎會把這張標成 ERR_DECODE 後繼續。
        """
        ...


@runtime_checkable
class MetadataExtractor(Protocol):
    def extract(self, raw: RawFile, image: Any | None) -> PhotoMeta:
        """讀 EXIF。

        Args:
            image: 已開啟（尚未解碼像素）的 `PIL.Image.Image`；格式不被 Pillow 支援時為 None。

        讀不到的欄位留 None；不可丟例外讓整張照片失敗（中繼資料壞掉不該擋住分類）。
        """
        ...


# --------------------------------------------------------------------------- #
# 推論端（單一執行緒）
# --------------------------------------------------------------------------- #
@runtime_checkable
class Loadable(Protocol):
    def load(self) -> list[ModelInfo]:
        """建立 ONNX session（含 provider 降級）。由推論執行緒在開始前呼叫一次。

        Raises:
            ModelLoadError: 權重不存在，或所有 provider 都建立失敗。
        """
        ...

    def close(self) -> None:
        """釋放 session。可重複呼叫。"""
        ...


@runtime_checkable
class VisionEmbedder(Loadable, Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_images(self, batch: np.ndarray) -> np.ndarray:
        """
        Args:
            batch: (N, S, S, 3) uint8 RGB。
        Returns:
            (N, dim) float32，每列已 L2 正規化。
        """
        ...


@runtime_checkable
class TextEmbedder(Loadable, Protocol):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Returns: (N, dim) float32，每列已 L2 正規化。"""
        ...


@runtime_checkable
class TagScorer(Loadable, Protocol):
    def score(self, embeddings: np.ndarray) -> list[list[TagHit]]:
        """對一批影像向量算出超過門檻的標籤。

        Args:
            embeddings: (N, dim) float32 L2 正規化。
        Returns:
            長度 N 的清單；每個元素是該張照片命中的標籤（可為空清單）。
        """
        ...

    def gate_role(self, tag_id: int) -> GateRole:
        """標籤在 Stage 3 條件式判斷裡的角色。"""
        ...


@runtime_checkable
class FaceAnalyzer(Loadable, Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def analyze(self, tasks: Sequence[DecodedImageTask]) -> list[list[FaceDetection]]:
        """偵測 + 對齊 + 取特徵。

        每個 task 的 `face_input`（letterbox 後的偵測輸入）、`face_letterbox`、
        `work_image` 保證不為 None。回傳清單與 `tasks` 等長、同順序；沒有臉回空清單。
        bbox / landmarks 必須換算成原圖正規化座標（見 `LetterboxInfo`）。
        """
        ...


# --------------------------------------------------------------------------- #
# 後處理
# --------------------------------------------------------------------------- #
@runtime_checkable
class FaceClusterer(Protocol):
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (N, 512) float32 L2 正規化。
        Returns:
            (N,) int 標籤；-1 = 雜訊（不屬於任何人）。
        """
        ...


@runtime_checkable
class ReverseGeocoder(Protocol):
    def lookup(self, coords: Sequence[tuple[float, float]]) -> list[GeoPlace]:
        """批次離線查詢 (lat, lon) -> 國家/城市。回傳與輸入等長。"""
        ...


@runtime_checkable
class EventClusterer(Protocol):
    def cluster(self, items: Sequence[EventInput]) -> list[EventCluster]:
        """依時間密度（與 GPS 距離）分群。輸入不需預先排序。"""
        ...


# --------------------------------------------------------------------------- #
# 向量檢索
# --------------------------------------------------------------------------- #
@runtime_checkable
class VectorIndex(Protocol):
    """向量檢索的抽象。預設實作是記憶體內矩陣（`db/vector_index.py`）；
    之後要換 sqlite-vec / FAISS 只要實作這個 Protocol。"""

    def __len__(self) -> int: ...

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        """新增或覆寫。vectors: (N, dim) float32 L2 正規化。"""
        ...

    def remove(self, ids: Sequence[int]) -> None: ...

    def search(self, query: np.ndarray, top_k: int = 50, min_score: float | None = None
               ) -> list[tuple[int, float]]:
        """回傳 [(id, cosine similarity)]，依分數由高到低。"""
        ...

    def get(self, id_: int) -> np.ndarray | None: ...
