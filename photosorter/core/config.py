"""設定模型與載入。

**預設值只有一個來源：本檔的 dataclass。** `config.example.json` 是由
`python -m photosorter dump-config` 從這裡產生的範本（測試會檢查兩者同步）。
各電腦把要覆寫的鍵寫進專案根目錄的 `config.json`（已列入 .gitignore），只需寫差異。

載入優先序（後者覆蓋前者）：dataclass 預設 < `config.json` < `load_config(overrides=...)`。
設定檔位置：`load_config(path)` 參數 > 環境變數 `PHOTOSORTER_CONFIG` > `<專案根>/config.json`。

硬體相關的值（provider、執行緒數、batch 大小）在這裡**只存「使用者意圖」**：
`"auto"` 或 `0` 代表「執行時再探測」。實際數值由 `core.hardware.plan_runtime()`
在 B 電腦啟動 pipeline 的那一刻決定，絕不寫回設定檔。
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import ENV_CONFIG, project_root, resolve_path


class ConfigError(ValueError):
    """設定檔內容不合法（未知的鍵、型別錯誤…）。"""


# --------------------------------------------------------------------------- #
# 各區塊
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    """所有路徑。相對路徑相對於專案根目錄；載入後由 `AppConfig.resolve()` 轉成絕對 `Path`。"""

    db_path: str = "data/photos.db"
    weights_dir: str = "weights"
    cache_dir: str = "data/cache"
    log_dir: str = "data/logs"


@dataclass
class DbConfig:
    """SQLite PRAGMA 與寫入佇列。"""

    cache_size_kb: int = 65536          # 每個連線的 page cache（64 MB）
    mmap_size_mb: int = 256
    busy_timeout_ms: int = 5000
    wal_autocheckpoint_pages: int = 1000
    journal_size_limit_mb: int = 64
    synchronous: str = "NORMAL"         # WAL 下 NORMAL 即可保證資料庫不損毀
    write_queue_size: int = 4096        # 寫入佇列上限（滿了會對呼叫端產生背壓）
    write_batch_max_ops: int = 256      # 單一交易最多合併幾個寫入操作
    enable_sqlite_vec: bool = False     # True 時嘗試載入 sqlite-vec 擴充（需另外 pip install）


@dataclass
class RuntimeConfig:
    """執行期資源。`"auto"` / `0` = 啟動時探測（Lazy Hardware Probing）。"""

    providers: Any = "auto"             # "auto" 或明確清單，例如 ["DmlExecutionProvider","CPUExecutionProvider"]
    device_id: int = 0
    decode_workers: int = 0             # 0 = 依實體核心數決定
    intra_op_threads: int = 0           # 0 = 依 provider 與核心數決定
    queue_maxsize: int = 64             # 解碼 -> 推論的 bounded queue
    batch_size: int = 0                 # 0 = 依 provider 決定（CUDA 16 / DML 8 / CPU 4）
    batch_timeout_ms: int = 200         # 湊不滿一個 batch 時最多等多久
    warmup: bool = True                 # 建立 session 後先跑一次假資料，提早抓出 provider 執行期錯誤


@dataclass
class ScanConfig:
    extensions: list[str] = field(
        default_factory=lambda: [
            "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "heic", "heif",
        ]
    )
    raw_extensions: list[str] = field(
        default_factory=lambda: ["dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2"]
    )
    enable_raw: bool = False            # RAW 需要 rawpy（未列入 requirements），預設關閉
    follow_symlinks: bool = False
    skip_hidden: bool = True
    exclude_dir_names: list[str] = field(
        default_factory=lambda: ["$RECYCLE.BIN", "System Volume Information", ".thumbnails", "@eaDir"]
    )
    min_file_size: int = 1024           # 小於此大小視為圖示/縮圖，不入庫
    retry_errors: bool = False          # True 時重掃會再試一次先前失敗的檔案


@dataclass
class DecodeConfig:
    """Single-Decode Multi-Resize 的三個分支尺寸。"""

    work_long_edge: int = 1280          # 中解析度工作圖的長邊；模糊度與人臉對齊都取自這張
    blur_long_edge: int = 1024          # 計算 Laplacian 變異數前先正規化到此長邊，分數才能跨相機比較
    face_det_size: int = 640
    siglip_size: int = 224
    jpeg_draft: bool = True             # JPEG 以 DCT 縮放直接解到 1/2、1/4、1/8，省下大部分解碼時間
    max_pixels: int = 250_000_000       # 超過視為異常檔（decompression bomb 防護）


@dataclass
class ScreenshotConfig:
    name_patterns: list[str] = field(
        default_factory=lambda: [
            "screenshot", "screen shot", "screen_shot", "screencapture", "screen capture",
            "螢幕擷取畫面", "螢幕快照", "螢幕截圖", "截圖", "擷取", "snipaste", "snip_",
        ]
    )
    # 沒有相機 EXIF 且長寬正好等於這些解析度（任一方向）才算；單靠長寬比誤判太多
    screen_resolutions: list[list[int]] = field(
        default_factory=lambda: [
            [1920, 1080], [2560, 1440], [3840, 2160], [1366, 768], [1536, 864], [1920, 1200],
            [2560, 1600], [2880, 1800], [3440, 1440],
            [1170, 2532], [1179, 2556], [1284, 2778], [1290, 2796], [1125, 2436], [1242, 2688],
            [1080, 1920], [1080, 2340], [1080, 2400], [1440, 3120], [1440, 3200],
        ]
    )


@dataclass
class OnnxIOConfig:
    """一個 ONNX 模型的檔案與節點名稱。節點名稱留在設定檔，換匯出版本不必改程式。"""

    file: str = ""                      # 相對於 paths.weights_dir
    input_name: str = ""                # 空字串 = 取模型的第一個輸入
    output_name: str = ""               # 空字串 = 取模型的第一個輸出
    fixed_batch: int = 0                # 0 = 動態 batch；N = 模型的 batch 維度寫死為 N


@dataclass
class SiglipConfig:
    model_name: str = "siglip-base-patch16-224"
    vision: OnnxIOConfig = field(
        default_factory=lambda: OnnxIOConfig(
            file="siglip/vision_model.onnx", input_name="pixel_values", output_name="pooler_output"
        )
    )
    text: OnnxIOConfig = field(
        default_factory=lambda: OnnxIOConfig(
            file="siglip/text_model.onnx", input_name="input_ids", output_name="pooler_output"
        )
    )
    tokenizer_file: str = "siglip/tokenizer.json"
    embed_dim: int = 768
    max_text_len: int = 64              # SigLIP 訓練時固定 padding 到 64，推論必須一致
    pad_token_id: int = 1
    mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    # prob = sigmoid(logit_scale * cos_sim + logit_bias)。兩個值是模型權重的一部分，
    # 換模型必須從該 checkpoint 重新讀出（見 weights/README.md）。
    logit_scale: float = 117.3308
    logit_bias: float = -12.9324
    prompt_template: str = "a photo of {}."
    default_threshold: float = 0.2      # 初始值，需用實際照片校準；個別標籤可在 tags.threshold 覆寫


@dataclass
class FaceConfig:
    model_name: str = "buffalo_l"
    detector: OnnxIOConfig = field(
        default_factory=lambda: OnnxIOConfig(file="buffalo_l/det_10g.onnx", fixed_batch=1)
    )
    recognizer: OnnxIOConfig = field(
        default_factory=lambda: OnnxIOConfig(file="buffalo_l/w600k_r50.onnx")
    )
    embed_dim: int = 512
    det_threshold: float = 0.5
    nms_threshold: float = 0.4
    min_face_px: int = 32               # 在工作圖上小於此邊長的臉不取特徵（特徵品質太差會污染分群）
    max_faces_per_photo: int = 32
    # 分群（utils/clustering.py 的 FaceClusterEngine）
    cluster_algorithm: str = "dbscan"   # "dbscan" | "hdbscan"
    cluster_eps: float = 0.38           # DBSCAN 的 cosine distance 半徑
    cluster_min_samples: int = 3


@dataclass
class EventConfig:
    gap_minutes: int = 360              # 相鄰兩張照片時間差小於此值（6 小時）-> 同一事件
    max_distance_km: float = 30.0       # 兩張都有 GPS 且距離超過此值 -> 也切開
    min_photos: int = 3                 # 少於此張數的群不成立事件（event_id 留 NULL）


@dataclass
class PipelineConfig:
    stages: list[str] = field(default_factory=lambda: ["meta", "siglip", "faces"])
    geocode: bool = True
    cluster_events: bool = True
    progress_interval_ms: int = 150     # 進度事件節流


@dataclass
class ExportConfig:
    path_template: str = "{year}/{year}-{month}/{filename}"
    conflict_policy: str = "skip_identical"   # "skip_identical" | "rename" | "skip"
    verify_copy: bool = True            # Move/Copy 後比對大小；跨磁碟 Move 另比對 sha256 才刪來源


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    db: DbConfig = field(default_factory=DbConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    screenshot: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    siglip: SiglipConfig = field(default_factory=SiglipConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    events: EventConfig = field(default_factory=EventConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # 以下由 load_config() 填入，不出現在 JSON
    root: Path = field(default_factory=project_root, repr=False, compare=False)
    source_file: Path | None = field(default=None, repr=False, compare=False)

    # ---- 已解析的絕對路徑（其他模組只用這些，不自己拼路徑）----
    @property
    def db_path(self) -> Path:
        return resolve_path(self.paths.db_path, self.root)

    @property
    def weights_dir(self) -> Path:
        return resolve_path(self.paths.weights_dir, self.root)

    @property
    def cache_dir(self) -> Path:
        return resolve_path(self.paths.cache_dir, self.root)

    @property
    def log_dir(self) -> Path:
        return resolve_path(self.paths.log_dir, self.root)

    def weight(self, relative: str) -> Path:
        """`weights_dir` 底下某個模型檔的絕對路徑。"""
        return resolve_path(relative, self.weights_dir)

    def to_dict(self) -> dict[str, Any]:
        """可序列化成 JSON 的內容（不含 root / source_file）。"""
        data = dataclasses.asdict(self)
        data.pop("root", None)
        data.pop("source_file", None)
        return data


_NON_JSON_FIELDS = {"root", "source_file"}
_FREEFORM_FIELDS = {"providers"}        # 可以是 "auto" 也可以是清單


# --------------------------------------------------------------------------- #
# 載入
# --------------------------------------------------------------------------- #
def _apply(obj: Any, data: dict[str, Any], where: str) -> None:
    """把 dict 遞迴套到 dataclass 實例上；遇到未知的鍵直接報錯（防止拼錯字被默默忽略）。"""
    if not isinstance(data, dict):
        raise ConfigError(f"{where or '<root>'} 必須是 JSON 物件")
    names = {f.name for f in dataclasses.fields(obj)} - _NON_JSON_FIELDS
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"設定檔有未知的鍵：{', '.join(where + k for k in unknown)}")
    for key, value in data.items():
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _apply(current, value, f"{where}{key}.")
        elif key in _FREEFORM_FIELDS:
            if not isinstance(value, (str, list)):
                raise ConfigError(f"{where}{key} 必須是字串或陣列")
            setattr(obj, key, value)
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where}{key} 必須是 true/false")
            setattr(obj, key, value)
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where}{key} 必須是整數")
            setattr(obj, key, value)
        elif isinstance(current, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where}{key} 必須是數字")
            setattr(obj, key, float(value))
        elif isinstance(current, list):
            if not isinstance(value, list):
                raise ConfigError(f"{where}{key} 必須是陣列")
            setattr(obj, key, value)
        elif isinstance(current, str) and not isinstance(value, str):
            raise ConfigError(f"{where}{key} 必須是字串")
        else:
            setattr(obj, key, value)


def find_config_file(path: str | os.PathLike[str] | None = None, root: Path | None = None) -> Path | None:
    """找出要用的設定檔；都沒有就回傳 None（純預設值執行）。"""
    root = root or project_root()
    if path:
        p = resolve_path(path, root)
        if not p.is_file():
            raise ConfigError(f"找不到設定檔：{p}")
        return p
    env = os.environ.get(ENV_CONFIG)
    if env:
        p = resolve_path(env, root)
        if not p.is_file():
            raise ConfigError(f"環境變數 {ENV_CONFIG} 指向的設定檔不存在：{p}")
        return p
    default = root / "config.json"
    return default if default.is_file() else None


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> AppConfig:
    """載入設定。

    Args:
        path: 設定檔路徑；None 時依序找 `PHOTOSORTER_CONFIG`、`<root>/config.json`。
        overrides: 最後套用的覆寫（與 JSON 相同的巢狀結構），給 CLI 參數與測試用。
        root: 專案根目錄；None 時用 `project_root()`。測試用它指到暫存資料夾。
    """
    root = (root or project_root()).resolve()
    cfg = AppConfig(root=root)
    file = find_config_file(path, root)
    if file is not None:
        try:
            data = json.loads(file.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"{file} 不是合法的 JSON：{e}") from e
        _apply(cfg, data, "")
        cfg.source_file = file
    if overrides:
        _apply(cfg, overrides, "")
    return cfg


def dump_default_config() -> str:
    """`config.example.json` 的內容（純預設值）。"""
    return json.dumps(AppConfig().to_dict(), ensure_ascii=False, indent=2) + "\n"
