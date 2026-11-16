# photosorter

100% 本地離線運作的個人照片智慧分類與管理工具：Headless 核心引擎、命令列介面與 Tkinter 桌面介面。

照片不會離開這台電腦：語意標籤（SigLIP）、人臉（InsightFace）、地點（離線 GeoNames）全部在本機用 ONNX Runtime 推論，結果存在一個 SQLite 檔案裡。在你明確執行「匯出」之前，原始照片完全不會被修改或移動。

## 目前狀態

核心引擎（資料庫、多階段管線、分群、四層歸檔）、命令列介面與 Tkinter 桌面介面都在這個 repo。目前還沒有接上真實的模型權重與真實照片實測過 —— 那是下一步，見 [`HANDOFF_TO_PC_B.md`](HANDOFF_TO_PC_B.md)。

| | |
|---|---|
| 已完成並有測試 | 資料庫層、管線調度（中斷續掃、暫停、背壓）、Stage 0/1、離線地理編碼、provider 自動降級、人臉分群、事件分群、連拍偵測、四層多視圖歸檔與回復、桌面介面（三欄式主視窗、虛擬滾動照片牆、匯出與復原） |
| 寫完但尚未對真實權重驗證 | SigLIP 影像/文字編碼與多標籤判定 |
| 骨架 | 人臉偵測輸出解碼與對齊（完成前資料庫裡不會有人臉，人臉分群是空跑）、RAW 解碼 |
| 尚未開始 | 以圖搜圖/以文搜圖的介面、在 GUI 裡管理標籤、人物合併/拆分 |

分群與歸檔的各個門檻（相似度、時間間隔、連拍判定）目前是初始值，還沒有用真實照片校準。

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

### 桌面介面

```powershell
python -m photosorter            # 不帶參數就是啟動 GUI
```

左側是多維度導航樹（事件與連拍、人物、語意標籤、地點、待清理隔離區），中間是照片牆，右側是照片屬性與匯出控制台。沒有模型權重也能開：右上角會顯示 `Mock / Weights Missing`，掃描只做 EXIF 與品質分析，瀏覽、分群與匯出照常可用。

### 命令列

```powershell
# 環境檢查：硬體、可用的 provider、相依套件、權重
python -m photosorter doctor --load-models

# 掃描。可隨時 Ctrl+C，下次執行會從中斷的地方接著做
python -m photosorter scan D:\Photos
python -m photosorter scan D:\Photos --stages meta          # 只做 EXIF/品質，不需要任何模型
python -m photosorter scan D:\Photos --stages meta,siglip   # 加上語意標籤

python -m photosorter stats

# 四層多視圖歸檔。預設只預覽（含預估佔用空間），加 --execute 才會動檔案
python -m photosorter archive E:\Sorted --mode copy
python -m photosorter archive E:\Sorted --mode copy --execute
python -m photosorter rollback 1                       # 還原整個批次

# 另一種：依自訂模板匯出成單層結構
python -m photosorter export E:\Flat --mode hardlink --template "{year}/{event}/{filename}"
```

`archive` 產生的結構：

```
E:\Sorted\
├── 01_Timeline_Master\2024\2024-05-01_Taipei_food\     實體主庫（唯一一份實體資料）
├── 02_By_Person\Alice\2024\                            人物視圖（硬連結，不佔額外空間）
├── 03_By_Semantic_Tag\food\                            標籤視圖（硬連結）
└── 99_Review_Staging\_Blurry_Suspects\ …               模糊、連拍次佳、螢幕截圖
```

檔名統一為 `日期_時間_毫秒_相機或標籤_雜湊前6碼.副檔名`，例如 `20240501_101500_000_Zenfone9_a3f9c1.jpg`；Live Photo 的影片、RAW、XMP 等附屬檔會跟著主照片一起走並共用新檔名。

設定：在專案根目錄建立 `config.json`，只寫要覆寫的項目（完整清單見 `config.example.json`）。所有路徑都相對於專案根目錄；執行緒數、batch 大小、execution provider 預設在啟動時依這台電腦的硬體自動決定。

## 測試

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

測試不需要模型權重，也不需要顯示卡。GUI 的測試會真的開視窗（執行時會短暫跳出幾個），沒有顯示器的環境會自動略過那一組。
