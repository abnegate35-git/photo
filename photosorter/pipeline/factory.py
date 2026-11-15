"""組裝：把設定、資料庫與各個具體元件接成可用的應用程式物件。

這是**唯一**把介面（`core.interfaces`）與具體實作綁在一起的地方。GUI 與 CLI 都從這裡拿
現成的物件，不自己 new 模型類別；要換實作（例如更完整的 EXIF 解析器、別的人臉模型）只改這裡。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..core.config import AppConfig, load_config
from ..core.events import EventBus
from ..db.db_manager import DBManager
from ..db.repository import Repository
from .decoder import DefaultImageDecoder
from .event_clustering import TimeGapEventClusterer
from .export_manager import ExportManager
from .pipeline_engine import PipelineEngine


@dataclass
class AppContext:
    """一個執行中的應用程式實例所需的全部核心物件。GUI 啟動時建立一個，結束時 `close()`。"""

    cfg: AppConfig
    db: DBManager
    repo: Repository
    bus: EventBus
    exporter: ExportManager

    def build_engine(self) -> PipelineEngine:
        return build_engine(self.cfg, self.db, self.repo, self.bus)

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


def build_engine(cfg: AppConfig, db: DBManager, repo: Repository, bus: EventBus) -> PipelineEngine:
    """依 `pipeline.stages` 建立管線引擎。

    這裡只建立物件，**不載入模型、不探測硬體** —— 那些發生在 `engine.start()` 之後的推論執行緒上。
    模型類別在函式內才 import，所以只跑 Stage 1（`stages = ["meta"]`）時連 onnxruntime 都不需要。
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
        event_clusterer=TimeGapEventClusterer(cfg.events) if cfg.pipeline.cluster_events else None,
    )
    # TODO（B 電腦）：人臉分群實作完成後在這裡掛上，例如
    #   engine.add_post_step("face_clusters", lambda: run_face_clustering(cfg, repo))
    return engine
