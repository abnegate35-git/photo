"""人臉無監督分群（把 `faces` 歸成 `persons`）。

==========================  實作狀態：骨架  ==========================
已完成：介面、資料來源（`Repository.load_face_matrix`）、寫回資料庫時必須遵守的規則。
未完成：`DbscanFaceClusterer.cluster()` 與 `assign_persons()` —— 參數需要在真實照片庫上調，
        A 電腦沒有人臉資料。接手方式見 HANDOFF.md §4.3。
=====================================================================

設計要點
--------
  * 特徵已 L2 正規化，cosine distance = 1 - 內積。同一人的距離通常 < 0.4~0.5（需校準）。
  * DBSCAN：`DBSCAN(eps=face.cluster_eps, min_samples=face.cluster_min_samples, metric="cosine")`。
    幾萬張臉以內直接算即可；更多時先用 `sklearn.neighbors.NearestNeighbors(radius=eps)` 建稀疏
    鄰接圖再 `metric="precomputed"`，避免 N^2 的距離矩陣吃光記憶體。
  * HDBSCAN：`sklearn.cluster.HDBSCAN(min_cluster_size=…, metric="cosine")`（scikit-learn ≥ 1.3 內建，
    不需要另外安裝 hdbscan 套件）。不需要 eps，對密度不均的資料較穩，但較慢。
  * 分群在整條管線結束後跑一次（全域後處理），不在逐張照片的迴圈裡。
"""
from __future__ import annotations

import numpy as np

from ..core.config import FaceConfig
from ..db.repository import Repository


class DbscanFaceClusterer:
    """`core.interfaces.FaceClusterer` 的預設實作（待完成）。"""

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (N, 512) float32 L2 正規化。
        Returns:
            (N,) int 群標籤，-1 = 雜訊。N = 0 時回傳空陣列。
        """
        raise NotImplementedError("人臉分群尚未實作（見 HANDOFF.md §4.3）")


def assign_persons(repo: Repository, face_ids: np.ndarray, labels: np.ndarray) -> int:
    """把分群結果寫回資料庫。回傳新建立的 person 數。

    必須遵守的規則（使用者的決定永遠優先，重跑分群不可以把人工整理的成果洗掉）：
      1. `faces.assign_src = 'manual'` 的臉不可改 `person_id`。
      2. 一個群裡如果已有屬於某 person 的臉（不論自動或手動），整群併入那個 person，不另開新的；
         群裡同時出現多個既有 person 時，併入臉數最多的那個，**不要自動合併 person**
         （合併只能由使用者在 UI 決定）。
      3. 全新的群才 `INSERT INTO persons`；`name` 留 NULL 等使用者命名。
      4. 雜訊（-1）的臉 `person_id` 維持 NULL，`cluster_id = -1`。
      5. 更新 `persons.face_count`、`centroid`（成員特徵平均後再正規化）、
         `cover_face_id`（建議取 det_score 與臉面積都高、正面的那張）。
      6. 以上全部放在同一個 `repo.db.write(fn)` 交易裡。
    """
    raise NotImplementedError("分群結果寫回尚未實作（見 HANDOFF.md §4.3）")
