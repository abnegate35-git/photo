"""`photos.process_status` 的位元定義 —— 中斷續掃（checkpoint resume）的唯一依據。

設計原則：
  * 一個位元代表「某階段的結果已經 commit 進資料庫」。位元只在該階段的資料與狀態
    寫在同一個交易裡時才會被設上，所以任何時刻被砍掉行程都不會出現「狀態說做完、
    資料卻沒有」的情況。
  * 續掃不需要額外的 checkpoint 檔：重新掃描時 Stage 0 讀出每張照片的狀態，
    `required & ~status` 就是還欠的階段。
  * 位元值一旦發布就不可更動或重複使用（資料庫裡存的是整數）。新增階段請往高位加。
"""
from __future__ import annotations

from enum import IntFlag


class ProcessStatus(IntFlag):
    NONE = 0

    # ---- Stage 1 ----
    HASHED = 1 << 0          # sha256 已算
    META = 1 << 1            # 尺寸 / EXIF / 截圖判定
    QUALITY = 1 << 2         # dHash + blur_score

    # ---- Stage 2 ----
    EMBEDDED = 1 << 3        # SigLIP 影像向量已存
    TAGGED = 1 << 4          # 自動標籤已寫入（可為 0 個標籤）

    # ---- Stage 3 ----
    FACES = 1 << 5           # 人臉階段已有結論（跑過，或被條件式略過）
    FACES_SKIPPED = 1 << 6   # 資訊用：上面的結論是「略過」

    # ---- 全域後處理（非逐張階段，僅供 UI 顯示；不列入 required）----
    GEOCODED = 1 << 7        # GPS -> 國家/城市 已查

    # ---- 錯誤（放在同一欄位，一次查詢即可分辨；重跑成功後清除）----
    ERR_DECODE = 1 << 16     # 讀檔或解碼失敗
    ERR_INFER = 1 << 17      # 推論失敗

    @classmethod
    def stage1(cls) -> "ProcessStatus":
        return cls.HASHED | cls.META | cls.QUALITY

    @classmethod
    def stage2(cls) -> "ProcessStatus":
        return cls.EMBEDDED | cls.TAGGED

    @classmethod
    def stage3(cls) -> "ProcessStatus":
        return cls.FACES

    @classmethod
    def ai_bits(cls) -> "ProcessStatus":
        return cls.stage2() | cls.stage3()

    @classmethod
    def error_bits(cls) -> "ProcessStatus":
        return cls.ERR_DECODE | cls.ERR_INFER


#: 設定檔 `pipeline.stages` 的名稱 -> 該階段要求的完成位元
STAGE_BITS: dict[str, ProcessStatus] = {
    "meta": ProcessStatus.stage1(),
    "siglip": ProcessStatus.stage2(),
    "faces": ProcessStatus.stage3(),
}


def required_mask(stages: list[str] | tuple[str, ...]) -> ProcessStatus:
    """把啟用的階段名稱換成「完成時必須具備的位元」。Stage 1 永遠必須。"""
    mask = ProcessStatus.stage1()
    for name in stages:
        try:
            mask |= STAGE_BITS[name]
        except KeyError:
            raise ValueError(
                f"未知的 pipeline stage: {name!r}（可用：{sorted(STAGE_BITS)}）"
            ) from None
    return mask
