"""組裝：把設定、資料庫與各個具體元件接成可用的應用程式物件。

這是**唯一**把介面（`core.interfaces`）與具體實作綁在一起的地方。GUI 與 CLI 都從這裡拿
現成的物件，不自己 new 模型類別；要換實作（例如更完整的 EXIF 解析器、別的人臉模型）只改這裡。
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

from ..core.config import AppConfig, load_config
from ..core.events import EventBus
from ..core.export_manager import ExportManager
from ..db.db_manager import DBManager
from ..db.repository import Repository
from ..utils.clustering import EventClusterEngine, FaceClusterEngine
from .decoder import DefaultImageDecoder
from .pipeline_engine import PipelineEngine


@dataclass
class AppContext:
    """一個執行中的應用程式實例所需的全部核心物件。GUI 啟動時建立一個，結束時 `close()`。"""

    cfg: AppConfig
    db: DBManager
    repo: Repository
    bus: EventBus
    exporter: ExportManager          # 四層多視圖歸檔（preview_export / execute_export / rollback_batch）

    def build_engine(self, stages: list[str] | None = None) -> PipelineEngine:
        """建立管線引擎。`stages` 給定時覆寫 `pipeline.stages`（只影響這個引擎，不改 `self.cfg`）。"""
        cfg = self.cfg
        if stages is not None:
            cfg = copy.deepcopy(self.cfg)
            cfg.pipeline.stages = list(stages)
        return build_engine(cfg, self.db, self.repo, self.bus)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "AppContext":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def create_app(
    config_path: str | os.PathLike[str] | None = None, *, overrides: dict[str, Any] | None = None
) -> AppContext:
    """載入設定、開啟資料庫（不存在就建立）、收尾上次中斷的匯出。不載入任何模型。"""
    cfg = load_config(config_path, overrides=overrides)
    db = DBManager(cfg.db_path, cfg.db).open()
    try:
        repo = Repository(db)
        bus = EventBus()
        exporter = ExportManager(cfg, db, repo, bus)
        exporter.recover()
    except BaseException:
        db.close()
        raise
    return AppContext(cfg=cfg, db=db, repo=repo, bus=bus, exporter=exporter)


def resolve_stages(cfg: AppConfig) -> tuple[list[str], list[str]]:
    """`pipeline.stages` 裡這台電腦現在**真的跑得起來**的階段。

    Returns:
        (可執行的階段, 被略過的原因)。原因是給使用者看的文字，每個被略過的階段一則。

    會被略過的情況：模型權重檔不存在；或人臉階段的輸出解碼還是骨架。
    只檢查檔案是否存在，不載入模型、不探測硬體。CLI 的 `scan` 不使用這個函式（缺權重就明確失敗）；
    GUI 用它讓沒有權重的環境仍然可以掃描、瀏覽與匯出，只是少了 AI 分類。
    """
    stages: list[str] = []
    skipped: list[str] = []
    for stage in cfg.pipeline.stages:
        if stage == "siglip":
            s = cfg.siglip
            missing = [p.name for p in (cfg.weight(s.vision.file), cfg.weight(s.text.file),
                                        cfg.weight(s.tokenizer_file)) if not p.is_file()]
            if missing:
                skipped.append(f"語意標籤（siglip）：缺少權重 {', '.join(missing)}")
                continue
        elif stage == "faces":
            f = cfg.face
            missing = [p.name for p in (cfg.weight(f.detector.file), cfg.weight(f.recognizer.file))
                       if not p.is_file()]
            if missing:
                skipped.append(f"人臉（faces）：缺少權重 {', '.join(missing)}")
                continue
            from ..models.face import is_implemented

            if not is_implemented():
                skipped.append("人臉（faces）：偵測輸出解碼尚未實作（models/face.py）")
                continue
        stages.append(stage)
    return stages, skipped


def build_engine(cfg: AppConfig, db: DBManager, repo: Repository, bus: EventBus) -> PipelineEngine:
    """依 `pipeline.stages` 建立管線引擎。

    這裡只建立物件，**不載入模型、不探測硬體** —— 那些發生在 `engine.start()` 之後的推論執行緒上。
    模型類別在函式內才 import，所以只跑 Stage 1（`stages = ["meta"]`）時連 onnxruntime 都不需要。

    全域後處理的順序（逐張照片都處理完之後依序執行）：
      geocode → events → bursts → face_clusters
    事件命名要用到城市，所以地理編碼排在事件之前。
    """
    stages = set(cfg.pipeline.stages)
    vision = tagger = faces = geocoder = None

    if "siglip" in stages:
        from ..models.siglip import SiglipTagger, SiglipVisionEncoder, seed_default_tags

        seed_default_tags(repo, cfg.siglip.prompt_template)
        vision = SiglipVisionEncoder(cfg)
        tagger = SiglipTagger(cfg, repo)
    if "faces" in stages:
        from ..models.face import InsightFaceAnalyzer

        faces = InsightFaceAnalyzer(cfg)
    if cfg.pipeline.geocode:
        from ..utils.geo import OfflineGeocoder

        geocoder = OfflineGeocoder()

    engine = PipelineEngine(
        cfg, db, repo, bus,
        decoder=DefaultImageDecoder(cfg),
        vision=vision,
        tagger=tagger,
        faces=faces,
        geocoder=geocoder,
        # 事件分群改由下面的 EventClusterEngine 負責（它還會產生預設事件名稱並偵測連拍），
        # 所以不使用引擎內建的 EventClusterer 步驟。
        event_clusterer=None,
    )

    if cfg.pipeline.cluster_events:
        events = EventClusterEngine.from_config(cfg, db)

        def step_events() -> str:
            r = events.cluster_events()
            return f"建立 {r.events_created} 個事件（{r.photos_in_events} 張照片）"

        def step_bursts() -> str:
            r = events.detect_bursts()
            return f"{r.burst_groups} 組連拍，{r.burst_duplicates} 張標為次佳"

        engine.add_post_step("events", step_events)
        engine.add_post_step("bursts", step_bursts)

    if "faces" in stages:
        clusterer = FaceClusterEngine.from_config(cfg, db)

        def step_faces() -> str:
            r = clusterer.run(should_stop=lambda: engine.stop_requested)
            return f"{r.total_faces} 張臉：歸入既有人物 {r.assigned_to_existing}，新人物 {r.new_persons}，未成群 {r.noise_faces}"

        engine.add_post_step("face_clusters", step_faces)
    return engine
