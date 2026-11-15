"""模組之間傳遞的封包（純資料，不含行為）。

慣例（所有實作都必須遵守，介面契約的一部分）：
  * 影像陣列一律 `uint8`、`HWC`、**RGB**、已套用 EXIF 方向（人是正的）。
    正規化、轉 NCHW、轉 float 是模型類別自己的事（`models/preprocess.py`）。
  * 向量一律 `float32`、一維、**已 L2 正規化**，所以 cosine similarity 就是內積。
  * 座標一律以「套用方向後的原圖」為基準、正規化到 0..1，與解析度無關。
  * 時間戳是整數 epoch 秒。`taken_ts` 把相機的當地時間當成 UTC 來換算
    （EXIF 通常沒有時區），只用來排序與算間隔，不可拿去跟 `mtime` 比。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .status import ProcessStatus


# --------------------------------------------------------------------------- #
# Stage 0
# --------------------------------------------------------------------------- #
class CandidateKind(enum.Enum):
    NEW = "new"                # 路徑不在資料庫
    CHANGED = "changed"        # 路徑在資料庫，但 size 或 mtime 不同
    INCOMPLETE = "incomplete"  # 檔案沒變，但還欠某些階段（中斷續掃）


@dataclass(slots=True)
class FileCandidate:
    """Stage 0 掃描後「需要處理」的一個檔案。沒變動且已完成的檔案不會產生 candidate。"""

    path: str                              # 絕對路徑（os.path.normpath 後）
    size: int
    mtime_ns: int
    kind: CandidateKind
    photo_id: int | None = None            # NEW 時為 None
    status: ProcessStatus = ProcessStatus.NONE   # 資料庫裡目前的狀態
    known_sha256: bytes | None = None      # 資料庫裡目前的 sha256


@dataclass(slots=True)
class RawFile:
    """讀進記憶體的檔案內容。整個檔案只從硬碟讀這一次，雜湊與解碼共用同一份 bytes。"""

    path: str
    data: bytes
    sha256: bytes                          # 32 bytes


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PhotoMeta:
    """Stage 1 的中繼資料。所有欄位都可為 None（讀不到就留空，不可猜）。"""

    taken_at: str | None = None            # 'YYYY-MM-DDTHH:MM:SS'，相機當地時間
    taken_ts: int | None = None
    taken_src: str | None = None           # 'exif' | 'filename' | 'mtime'
    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    iso: int | None = None
    f_number: float | None = None
    exposure_time: float | None = None     # 秒
    focal_length: float | None = None      # mm
    gps_lat: float | None = None           # 十進位度，南緯為負
    gps_lon: float | None = None           # 十進位度，西經為負
    gps_alt: float | None = None
    orientation: int | None = None         # EXIF Orientation 原始值 1..8


@dataclass(slots=True)
class Stage1Record:
    """要寫進 `photos` 的 Stage 1 結果。"""

    path: str
    file_size: int
    mtime_ns: int
    sha256: bytes
    width: int                             # 套用方向後的原圖寬
    height: int
    meta: PhotoMeta
    dhash: int                             # 有號 64-bit（SQLite INTEGER 可直接存）
    blur_score: float                      # Variance of Laplacian，越小越糊
    is_screenshot: bool


@dataclass(slots=True)
class LetterboxInfo:
    """把工作圖放進方形偵測輸入時的幾何資訊，用來把偵測座標換回原圖。

    工作圖等比縮放 `scale` 倍後貼在方形畫布的左上角，其餘補 0。
    偵測輸入上的點 (x, y) 對應到原圖正規化座標 (x / content_w, y / content_h)。
    """

    scale: float
    content_w: int                         # 畫布上實際有內容的寬（像素）
    content_h: int


@dataclass(slots=True)
class DecodedImage:
    """`ImageDecoder.decode()` 的產出：Stage 1 結果 + 給 AI 的各個分支（此時還沒有 photo_id）。"""

    record: Stage1Record
    siglip_input: np.ndarray | None
    face_input: np.ndarray | None
    face_letterbox: LetterboxInfo | None
    work_image: np.ndarray | None
    decode_ms: float = 0.0


@dataclass(slots=True)
class DecodedImageTask:
    """生產者（解碼執行緒）交給消費者（推論執行緒）的封包。

    一張照片解碼一次後在記憶體裡切出的三個分支都在這裡。大小約
    work(≤1280x960x3≈3.7MB) + face(640x640x3≈1.2MB) + siglip(0.15MB)，
    乘上 `runtime.queue_maxsize` 就是佇列的記憶體上限。
    """

    photo_id: int
    path: str
    pending: ProcessStatus                 # 這張還欠哪些 AI 階段（消費者只做這些）
    width: int
    height: int
    is_screenshot: bool
    siglip_input: np.ndarray | None        # (S, S, 3) uint8 RGB；不需要 Stage 2 時為 None
    face_input: np.ndarray | None          # (D, D, 3) uint8 RGB letterbox；不需要 Stage 3 時為 None
    face_letterbox: LetterboxInfo | None
    work_image: np.ndarray | None          # (h, w, 3) uint8 RGB，長邊 ≤ work_long_edge；人臉對齊從這張裁
    decode_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Stage 2 / 3
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TagHit:
    tag_id: int
    score: float                           # sigmoid 機率 0..1


@dataclass(slots=True)
class FaceDetection:
    """一張臉。bbox 與 landmarks 都是原圖正規化座標。"""

    bbox: tuple[float, float, float, float]    # (x, y, w, h)，0..1
    det_score: float
    landmarks: np.ndarray | None = None        # (5, 2) float32：左眼、右眼、鼻尖、左嘴角、右嘴角
    embedding: np.ndarray | None = None        # (512,) float32 L2 正規化；臉太小不取特徵時為 None
    quality: float | None = None               # 選配：臉部清晰度/大小綜合分數，供挑代表頭像


@dataclass(slots=True)
class InferenceResult:
    """消費者對一張照片的產出，會在**同一個交易**裡連同狀態位元一起寫入資料庫。"""

    photo_id: int
    embedding: np.ndarray | None = None        # (768,) float32；本輪沒跑 Stage 2 時為 None
    tags: list[TagHit] | None = None           # None = 本輪沒跑標籤；[] = 跑了但沒有命中
    faces: list[FaceDetection] | None = None   # None = 本輪沒跑人臉；[] = 跑了但沒有臉
    faces_skipped: bool = False                # 條件式略過（文件/截圖且無人物語意）
    status_set: ProcessStatus = ProcessStatus.NONE   # 要 OR 進 process_status 的位元（只含已完成的階段）
    error: str | None = None                   # 有值代表某個階段失敗；已完成階段的結果照樣寫入，另設 ERR_INFER


# --------------------------------------------------------------------------- #
# 標籤 / 地理 / 事件
# --------------------------------------------------------------------------- #
class GateRole(str, enum.Enum):
    """標籤在 Stage 3 條件式判斷裡扮演的角色（`tags.gate_role`）。"""

    NONE = "none"
    SKIP_FACE = "skip_face"        # 文件、收據、螢幕截圖…：命中就傾向略過人臉
    HAS_PERSON = "has_person"      # 人物語意：命中就一定跑人臉（優先於 SKIP_FACE）


@dataclass(slots=True)
class TagSpec:
    """一個可由 SigLIP 自動判定的標籤（`tags` 表的一列）。"""

    id: int
    name: str
    prompt: str                            # 餵給 text encoder 的完整英文句子
    threshold: float
    gate_role: GateRole = GateRole.NONE
    text_vector: np.ndarray | None = None  # (768,) float32；None = 尚未編碼


@dataclass(slots=True)
class GeoPlace:
    country_code: str | None = None        # ISO 3166-1 alpha-2
    admin1: str | None = None              # 省/州/縣市
    city: str | None = None


@dataclass(slots=True)
class EventInput:
    """事件聚類的輸入：一張照片的時間與位置。"""

    photo_id: int
    taken_ts: int
    lat: float | None = None
    lon: float | None = None


@dataclass(slots=True)
class EventCluster:
    photo_ids: list[int]
    start_ts: int
    end_ts: int
    center_lat: float | None = None
    center_lon: float | None = None


# --------------------------------------------------------------------------- #
# 匯出
# --------------------------------------------------------------------------- #
class ExportMode(str, enum.Enum):
    COPY = "copy"
    MOVE = "move"
    HARDLINK = "hardlink"
    SYMLINK = "symlink"


class ConflictPolicy(str, enum.Enum):
    SKIP_IDENTICAL = "skip_identical"   # 目標已有內容相同的檔案 -> 略過；內容不同 -> 改名
    RENAME = "rename"                   # 一律改名（name_1.jpg、name_2.jpg…）
    SKIP = "skip"                       # 目標已存在就略過


@dataclass(slots=True)
class PlannedOp:
    """匯出計畫裡的一筆操作（dry-run 的輸出，尚未動到任何檔案）。"""

    photo_id: int
    src: str
    dst: str | None                     # None = 這筆不會執行（見 note）
    action: str                         # 'copy' | 'move' | 'hardlink' | 'symlink' | 'skip'
    note: str = ""                      # 'renamed' / 'identical' / 'exists' / 'missing_source' / …


@dataclass(slots=True)
class ExportPlan:
    mode: ExportMode
    dest_root: str
    template: str
    conflict_policy: ConflictPolicy
    ops: list[PlannedOp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for op in self.ops:
            key = op.action if not op.note else f"{op.action}:{op.note}"
            out[key] = out.get(key, 0) + 1
        return out


@dataclass(slots=True)
class ExportResult:
    batch_id: int
    done: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)   # (src, 訊息)


# --------------------------------------------------------------------------- #
# 執行期
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HardwareInfo:
    """啟動時探測到的硬體（`core.hardware.probe_hardware()`）。"""

    logical_cores: int
    physical_cores: int
    total_ram_gb: float | None
    ort_version: str | None
    available_providers: list[str]


@dataclass(slots=True)
class RuntimePlan:
    """把「使用者意圖」與「探測結果」合起來後，這次執行實際採用的數值。"""

    provider: str
    decode_workers: int
    intra_op_threads: int
    batch_size: int
    queue_maxsize: int
    batch_timeout_s: float


@dataclass(slots=True)
class ModelInfo:
    """載入完成的模型資訊，會登記進 `models` 表並隨事件送給 UI。"""

    name: str
    kind: str                              # 'image_embed' | 'text_embed' | 'face_detect' | 'face_embed'
    dim: int | None
    provider: str
    path: str
    extra: dict[str, Any] = field(default_factory=dict)
