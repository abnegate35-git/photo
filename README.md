# photosorter

100% 本地離線運作的個人照片智慧分類與管理工具 —— **核心引擎**（Headless Core）。

照片不會離開這台電腦：語意標籤（SigLIP）、人臉（InsightFace）、地點（離線 GeoNames）全部在本機用 ONNX Runtime 推論，結果存在一個 SQLite 檔案裡。在你明確執行「匯出」之前，原始照片完全不會被修改或移動。

## 目前狀態

這個 repo 是底層架構：資料庫、多階段管線、介面契約、ONNX Runtime 層、匯出引擎，以及命令列介面。GUI 還沒有做。

| | |
|---|---|
| 已完成並有測試 | 資料庫層、管線調度（中斷續掃、暫停、背壓）、Stage 0/1、事件聚類、離線地理編碼、匯出與回復、provider 自動降級 |
| 寫完但尚未對真實權重驗證 | SigLIP 影像/文字編碼與多標籤判定 |
| 骨架 | 人臉偵測輸出解碼與對齊、人臉分群、RAW 解碼 |
| 尚未開始 | Tkinter GUI、縮圖、近似重複分群 |

接手開發請讀 [`HANDOFF.md`](HANDOFF.md)；設計理由在 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 安裝

需要 Windows 10 (1903 以上) 或 Windows 11、Python 3.13 64-bit。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m photosorter doctor
```

`requirements.txt` 安裝 DirectML 版的 onnxruntime（任何 DX12 顯示卡皆可加速，沒有顯示卡會自動用 CPU）。NVIDIA CUDA 環境改用 `requirements-cuda.txt`，純 CPU 用 `requirements-cpu.txt`。

模型權重不在 repo 裡，請依 [`weights/README.md`](weights/README.md) 放進 `weights/`。

## 使用

```powershell
# 環境檢查：硬體、可用的 provider、相依套件、權重
python -m photosorter doctor --load-models

# 掃描。可隨時 Ctrl+C，下次執行會從中斷的地方接著做
python -m photosorter scan D:\Photos
python -m photosorter scan D:\Photos --stages meta          # 只做 EXIF/品質，不需要任何模型
python -m photosorter scan D:\Photos --stages meta,siglip   # 加上語意標籤

python -m photosorter stats

# 依分類結果整理成資料夾。預設只預覽，加 --execute 才會動檔案
python -m photosorter export E:\Sorted --mode hardlink --template "{year}/{event}/{filename}"
python -m photosorter export E:\Sorted --mode copy --execute
python -m photosorter rollback 1
```

設定：在專案根目錄建立 `config.json`，只寫要覆寫的項目（完整清單見 `config.example.json`）。所有路徑都相對於專案根目錄；執行緒數、batch 大小、execution provider 預設在啟動時依這台電腦的硬體自動決定。

## 測試

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

測試不需要模型權重，也不需要顯示卡。
