# 電腦 B 跨機銜接與 E2E 測試指南

這份文件給兩個讀者：把程式從 A 電腦帶到 **B 電腦（Windows 10、有 GPU 與真實照片）** 的你，以及在 B 電腦上接手的 **Antigravity**。

- 規則、模組現況、已知限制：[`HANDOFF.md`](HANDOFF.md)
- 設計理由：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 權重檔的來源與要核對的設定：[`weights/README.md`](weights/README.md)

## 0. 到 B 電腦之前先知道的三件事

1. **A 電腦從頭到尾沒有模型權重，也沒有真實照片。** 三個階段的 127 項測試全部通過，但用的是假模型、合成的人臉向量與程式產生的圖片。所以 B 電腦的第一次 E2E 不是「確認一切正常」，而是**第一次讓真實模型與真實照片進來**，預期會有需要調整的地方。
2. **人臉偵測還不能用。** `photosorter/models/face.py` 的輸出解碼與對齊仍是骨架。在它完成之前：GUI 會自動略過人臉階段；CLI 的 `scan` 預設階段會失敗，要加 `--stages meta,siglip`。人物群組、`02_By_Person` 視圖在那之前都會是空的。
3. **SigLIP 的數值正確性沒有驗證過。** 程式路徑用測試模型走通了，但節點名稱、tokenizer、`logit_scale` / `logit_bias` 都要在 B 電腦對真實權重核對（§4 的任務 1）。

---

## 1. A 電腦 Push 前的檢查清單

```powershell
cd photo
git status --short                       # 確認要提交的檔案
git ls-files | Select-String -Pattern '\.(db|sqlite|onnx)$|__pycache__|config\.json$'   # 應該沒有任何輸出
git check-ignore -v weights/siglip/vision_model.onnx data/photos.db config.json         # 三個都應該顯示被哪條規則忽略
```

`.gitignore` 排除的東西（都不該出現在 `git status`）：

| 類別 | 規則 | 說明 |
|---|---|---|
| 模型權重 | `weights/**`（只放行 `README.md`、`.gitkeep`）、`*.onnx`、`*.onnx.data` | 體積大、有授權條款，B 電腦自己下載 |
| 資料庫與快取 | `*.db`、`*.db-wal`、`*.db-shm`、`*.sqlite*`、`data/**`（只放行 `.gitkeep`） | 照片庫內容是個人資料；縮圖快取在 `data/cache/thumbs` |
| 本機設定 | `config.json`、`config.local.json`、`.env` | 各電腦路徑與硬體不同；範本是 `config.example.json` |
| Python | `__pycache__/`、`*.py[cod]`、`.venv/`、`.pytest_cache/` | |
| 打包產物 | `build/`、`dist/` | |

提交與推送：

```powershell
git add -A
git commit -m "Stage 2-3: clustering, four-layer archive export, Tkinter GUI"
git push
```

### 模組結構總覽

```
photosorter/
├── __main__.py             入口：不帶參數 = GUI；子命令 = CLI
├── core/
│   ├── config.py types.py interfaces.py events.py status.py hardware.py paths.py   基礎（第一階段）
│   └── export_manager.py   四層多視圖歸檔 ExportManager、verify_archive()（第二階段）
├── db/
│   ├── schema.sql db_manager.py repository.py vector_index.py                       資料庫層（schema v2）
│   └── queries_ui.py       GUI 專用查詢：導航樹計數、照片清單、人物命名鎖定、匯出歷史（第三階段）
├── models/
│   ├── onnx_runtime.py preprocess.py siglip.py                                      ONNX 層
│   └── face.py             ★ 骨架：decode_scrfd()、align_face() 待實作
├── pipeline/
│   ├── pipeline_engine.py scanner.py decoder.py metadata.py                         管線
│   ├── export_manager.py   自訂模板的單層匯出（四層歸檔的底層）
│   ├── event_clustering.py 最小版事件聚類（介面參考實作）
│   └── factory.py          組裝；resolve_stages() 決定哪些階段跑得起來
├── utils/
│   ├── clustering.py       FaceClusterEngine、EventClusterEngine（第二階段）
│   └── imageops.py hashing.py geo.py logging_setup.py
└── ui/                     （第三階段）
    ├── app.py              run_gui()
    ├── main_window.py      三欄式主視窗
    ├── thumbnail_grid.py   虛擬滾動照片牆、LRU 快取
    ├── dialogs.py          人物命名、匯出預覽、復原歷史
    ├── theme.py            高 DPI、sv-ttk / clam 深色、Tk 視窗的安全釋放
    └── bridge.py           TkEventPump（給其他以 Tk 為底的介面用；主視窗自己輪詢）
tests/                      127 項；不需要權重、不需要 GPU（GUI 測試需要能開視窗）
```

---

## 2. 電腦 B 環境初始化

需要：Windows 10 1903 以上、Python 3.13 64-bit、Git。

```powershell
git clone https://github.com/abnegate35-git/photo.git
cd photo
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m photosorter doctor
```

- `pytest` 應該是 **127 passed, 1 skipped**。略過的那一項是符號連結測試，需要開啟「開發人員模式」才會跑。GUI 測試會短暫跳出幾個視窗，屬正常現象。
- `requirements.txt` 裝的是 DirectML 版 onnxruntime（任何 DX12 顯示卡）。B 電腦是 NVIDIA 且 CUDA 12 / cuDNN 9 已裝好時，改用 `requirements-cuda.txt`；換之前先 `pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu`。**`requirements-cuda.txt` 與 `requirements-cpu.txt` 在 A 電腦只確認過能解析，沒有實際安裝執行過。**
- `doctor` 會列出 Windows build（DirectML 需要 ≥ 18362）、可用的 execution provider、長路徑是否開啟、符號連結權限、缺哪些權重。

### 模型權重的放置位置

權重放在**專案根目錄的 `weights/`**（與 `photosorter/` 套件資料夾同一層，不是在它裡面）：

```
photo\
├── photosorter\
└── weights\
    ├── siglip\
    │   ├── vision_model.onnx
    │   ├── text_model.onnx
    │   └── tokenizer.json
    └── buffalo_l\
        ├── det_10g.onnx
        └── w600k_r50.onnx
```

來源見 [`weights/README.md`](weights/README.md)。**那些下載位置在 A 電腦沒有實際驗證過**，以 `doctor` 的結果為準。放好之後：

```powershell
.venv\Scripts\python -m photosorter doctor --load-models
```

要確認的輸出：

| 看什麼 | 期望 |
|---|---|
| 五個權重檔 | 都是 `OK` 並顯示檔案大小 |
| 每個模型的 `provider=` | `DmlExecutionProvider` 或 `CUDAExecutionProvider`；若是 `CPUExecutionProvider`，下面一行會寫 GPU provider 被淘汰的原因 |
| SigLIP vision 的 `inputs` / `outputs` | 輸入名稱與 `config.siglip.vision.input_name`（預設 `pixel_values`）一致；輸出要有 `pooler_output` |
| SCRFD 的 `outputs` | 9 個輸出 |

節點名稱不一樣時，在專案根目錄建 `config.json` 覆寫（只寫差異），例如：

```json
{ "siglip": { "vision": { "output_name": "image_embeds" } } }
```

---

## 3. 第一次端到端整合測試 SOP

### 3.1 準備測試照片

**先複製一份出來測，不要直接對原始相簿操作。** 整個流程除了 MOVE 模式以外都不會動原始檔，但第一次跑真實資料，留一份備份是基本動作。

```powershell
mkdir D:\pstest\library
# 把挑好的 20~30 張照片（連同它們的附屬檔）複製進 D:\pstest\library，可以分幾個子資料夾
```

每一類照片都對應一個要驗證的功能：

| 準備什麼 | 數量 | 驗證的功能 | 需要的條件 |
|---|---|---|---|
| 同一場景 3 秒內連拍的幾張 | 3~5 | 連拍群組、Best Shot | Stage 1 |
| 同一天、同一地點的一批（要有 GPS） | 5~8 | 事件分群、離線地理編碼、預設事件名稱 | Stage 1 |
| 另一天或另一個城市的一批 | 4~6 | 事件切分（每個事件至少 3 張才成立） | Stage 1 |
| 風景、食物、建築等不同主題 | 5~6 | SigLIP 語意標籤 | SigLIP 權重 |
| 人像：獨照、兩人合照、多人合照，同一個人出現在多張 | 6~8 | 人臉、人物分群、人物視圖 | **人臉階段實作完成後** |
| Live Photo（`IMG_xxxx.HEIC` + `IMG_xxxx.MOV`）或 RAW+JPG（同主檔名） | 1~2 組 | Sidecar 綁定與重新命名 | Stage 1 |
| 手機螢幕截圖 | 1~2 | 截圖判定 → 隔離區 | Stage 1 |
| 一張明顯手震模糊的 | 1 | `blur_score`（觀察數值，預設不會自動進隔離區） | Stage 1 |
| 沒有 EXIF 的圖（通訊軟體收到的、下載的） | 1~2 | `_No_EXIF_Date` | Stage 1 |
| 把其中一張照片複製一份改個檔名 | 1 | 重複內容偵測 | Stage 1 |

記錄測試前的狀態，之後用來證明「還原得一模一樣」：

```powershell
Get-ChildItem D:\pstest\library -Recurse -File | Get-FileHash -Algorithm SHA256 |
    Select-Object Hash, Path | Export-Csv D:\pstest\before.csv -NoTypeInformation
(Get-PSDrive D).Free            # 記下可用空間（位元組）
```

### 3.2 CLI 全流程

以下指令都在專案根目錄執行，`python` 指 `.venv\Scripts\python`。

**(1) 掃描**

```powershell
python -m photosorter scan D:\pstest\library --stages meta            # 沒有權重也能跑
python -m photosorter scan D:\pstest\library --stages meta,siglip     # 有 SigLIP 權重之後；只會補做 Stage 2
python -m photosorter stats
```

期望：

- 第一次掃描結束狀態是 `finished`，`錯誤 0`（故意放壞檔才會有錯誤）。
- 後處理依序出現 `geocode`、`events`、`bursts`，並報告事件數與連拍組數。
- **再跑一次同樣的指令**：`需處理 0`，幾乎瞬間結束（增量掃描）。
- 掃描中按 Ctrl+C 再重跑：從中斷處接著做，不重做已完成的。
- 加了 `siglip` 之後，`[模型]` 那幾行顯示實際的 provider。

用資料庫直接確認（任何 SQLite 工具，或下面這一行）：

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/photos.db'); [print(r) for r in c.execute('SELECT file_name, taken_src, city, is_screenshot, burst_group_id, is_best_shot, round(blur_score,1) FROM photos ORDER BY taken_ts')]"
```

**(2) 自訂模板匯出：`plan` → `execute`**

```powershell
python -m photosorter plan D:\pstest\flat --mode copy --template "{year}/{year}-{month}/{filename}"
python -m photosorter execute D:\pstest\flat --mode copy --template "{year}/{year}-{month}/{filename}"
```

期望：`plan` 只列出動作、`D:\pstest\flat` 不會被建立；`execute` 之後才有檔案，並印出批次編號。

**(3) 四層歸檔：`archive`**

```powershell
python -m photosorter archive D:\pstest\sorted --mode hardlink                 # 預覽
python -m photosorter archive D:\pstest\sorted --mode hardlink --execute       # 執行
```

預覽要看的：

- `master:hardlink`、`staging:hardlink`、`sidecar:hardlink`、`person_view:hardlink`、`tag_view:hardlink` 各幾筆，數量與你準備的照片對得上。
- `預估新增佔用空間` 是 `0.000 GB`（來源與目標在同一個磁碟機時，硬連結不佔空間）。
- 沒有 `[警告]`。若出現「不在同一個磁碟機」，代表主庫會改用複製 —— 想驗證零空間就把目標換到與照片同一顆磁碟。

執行後檢查目錄：

```powershell
tree D:\pstest\sorted /F | more
```

| 位置 | 期望 |
|---|---|
| `01_Timeline_Master\{年}\{日期}_{城市}_{標籤}\` | 每個事件一個資料夾；Sidecar 與主照片在一起、主檔名相同 |
| `01_Timeline_Master\_No_EXIF_Date\{年}\` | 沒有 EXIF 日期的那幾張 |
| `02_By_Person\` | 人臉階段完成前是空的 |
| `03_By_Semantic_Tag\{標籤}\` | 有跑 SigLIP 才有內容 |
| `99_Review_Staging\_Screenshots_Downloads\` | 螢幕截圖 |
| `99_Review_Staging\_Burst_Duplicates\` | 每組連拍除了 Best Shot 以外的照片 |
| 檔名 | `日期_時間_毫秒_相機或標籤_雜湊前6碼.副檔名`，例如 `20240501_101500_000_Zenfone9_a3f9c1.jpg` |

**(4) 驗證硬連結與零額外空間**

```powershell
python -m photosorter verify-archive D:\pstest\sorted
(Get-PSDrive D).Free            # 與 3.1 記下的數字比較
```

- `verify-archive` 最後一行應該是「所有視圖都是主庫檔案的硬連結（零額外空間）」。它也會列出「各路徑大小加總」與「實際佔用空間」的差。
- HARDLINK 模式下，可用空間與匯出前相比應該幾乎不變（只多了資料夾與連結本身的中繼資料）。這個數字 A 電腦沒有量過，請記下實際的差值。
- **不要用檔案總管的「內容 → 大小」判斷**：它會把每個硬連結各算一次，數字會是實際的好幾倍。
- 想親眼確認某個檔案：`fsutil hardlink list "D:\pstest\sorted\01_Timeline_Master\...\某檔案.jpg"` 會列出指向同一份資料的所有路徑（應該包含原始照片與各視圖）。

再跑一次 `archive ... --execute`：應該是 `完成 0`（冪等，已匯出的不會重做）。

**(5) Rollback**

```powershell
python -m photosorter rollback <批次編號>
Get-ChildItem D:\pstest\sorted -Recurse -File | Measure-Object           # Count 應該是 0
Get-ChildItem D:\pstest\library -Recurse -File | Get-FileHash -Algorithm SHA256 |
    Select-Object Hash, Path | Export-Csv D:\pstest\after.csv -NoTypeInformation
Compare-Object (Get-Content D:\pstest\before.csv) (Get-Content D:\pstest\after.csv)   # 沒有輸出 = 完全相同
```

**(6) COPY 與 MOVE**

同樣的流程各跑一次：

- `--mode copy`：`預估新增佔用空間` 等於主庫與隔離區的檔案總和；視圖仍然不佔空間。原始照片完全不動。
- `--mode move`：**放在最後測**。執行後 `D:\pstest\library` 裡的照片會消失（搬進主庫），`rollback` 後必須全部回到原位，`Compare-Object` 仍然沒有輸出。

### 3.3 GUI 全流程

```powershell
python -m photosorter            # 或 python -m photosorter gui
```

| 步驟 | 操作 | 期望 |
|---|---|---|
| 1 | 看右上角徽章 | 有權重：`⚡ DirectML`（或 CUDA / CPU），人臉未實作時後面有「（部分階段略過）」。沒有權重：`Mock / Weights Missing`，不會崩潰 |
| 2 | 「選擇相簿資料夾」→ `D:\pstest\library`，「開始增量掃描」 | 進度條前進；期間視窗可以拖動、可以點別的節點，不會卡住 |
| 3 | 掃描中按「暫停」→「繼續」→「安全停止」，再按一次「開始增量掃描」 | 暫停時數字不動；停止後狀態列顯示已停止；重新開始會從中斷處接著做 |
| 4 | 左側樹展開「時間與事件」 | 每個事件有張數；連拍群組顯示 `burst_group_id=…` 與 `⭐ Best Shot: 檔名` |
| 5 | 點事件、標籤、地點、隔離區各節點 | 中央照片牆立刻切換；縮圖陸續出現；Best Shot 有 ⭐、連拍次佳有 ⧉ |
| 6 | 在「全部照片」快速拖動捲軸 | 捲動順暢，記憶體不會持續成長（只載入可見範圍，快取上限 300 張） |
| 7 | 點一張 Live Photo 或 RAW+JPG 的主照片 | 右側「Sidecar」列出附屬檔；顯示 EXIF、清晰度、ShortHash、標籤與信心度 |
| 8 | （人臉完成後）點預覽圖上的人臉框，輸入名字 | 左側該人物前面出現 🔒；`persons.is_locked` 變成 1 |
| 9 | 右下選模式 `HARDLINK`、目標 `D:\pstest\sorted_gui`，按「預覽匯出 (Dry-Run)」 | 跳出預覽視窗：筆數、預估空間、明細；此時目標資料夾還不存在 |
| 10 | 預覽視窗按「執行四層目錄歸檔」→ 確認 | 狀態列顯示批次編號與完成筆數 |
| 11 | `python -m photosorter verify-archive D:\pstest\sorted_gui` | 同 3.2 (4) |
| 12 | 「⏪ 復原歷史整理 (Rollback)」→ 選該批次 →「還原選取的批次」 | 表格的「action: state 筆數」欄顯示各動作的 done / failed；還原後目標資料夾清空 |
| 13 | 「重新分群 (Re-cluster)」 | 狀態列顯示事件數、連拍組數；已鎖定的人物不變 |
| 14 | 掃描進行中直接按視窗的 ✕ | 狀態列顯示「正在安全停止」，稍候視窗關閉；重新開啟後資料完好、可續掃 |

日誌在 `data\logs\photosorter.log`。

### 3.4 結果記錄表

跑完把這張表填好，交給 Antigravity 當作調整的依據：

| 項目 | 結果 | 備註（實際數字、異常現象） |
|---|---|---|
| `pytest` | | |
| `doctor --load-models`：五個模型的 provider | | |
| Stage 1：時間、GPS、城市是否正確 | | 特別注意 HEIC |
| 事件切分是否符合直覺 | | 幾個事件、哪裡切錯 |
| 連拍群組與 Best Shot 是否合理 | | 漏抓 / 誤抓幾組 |
| 螢幕截圖判定 | | 漏抓 / 誤判 |
| SigLIP 標籤是否合理 | | 哪些標籤太鬆 / 太緊 |
| 模糊照片的 `blur_score` 與清楚照片的差距 | | 記下幾張的數值 |
| 掃描速度（張 / 秒）、記憶體尖峰 | | |
| 四層目錄結構與檔名 | | |
| `verify-archive`：零額外空間 | | |
| Rollback 後 `Compare-Object` 無差異（三種模式） | | |
| GUI 各步驟 | | |

---

## 4. 交給 Antigravity 的後續任務

每個任務都先貼這段共同前言：

> 先讀專案根目錄的 `AGENTS.md`、`HANDOFF.md`（尤其 §3「不可違反的規則」）與 `HANDOFF_TO_PC_B.md`。一律使用專案的 `.venv`。不要把 PyTorch、transformers、insightface 裝進這個環境；需要它們產生參考答案時另開一個虛擬環境。改完跑 `.venv\Scripts\python -m pytest`，必須全部通過。沒有用真實權重或真實照片驗證過的結論，不要寫成已驗證。

### 任務 1 —— 驗證 SigLIP 的數值正確性（最優先，其他標籤相關的調整都以它為前提）

> 依 `HANDOFF.md` §4.1 驗證 `photosorter/models/siglip.py`。在另一個裝有 torch 與 transformers 的虛擬環境，用 `google/siglip-base-patch16-224` 對 `<測試照片資料夾>` 裡 5 張照片與 `DEFAULT_TAGS` 的全部 prompt 算出參考的影像向量、文字向量、`logit_scale.exp()`、`logit_bias` 與 `sigmoid(logits_per_image)`，存成 `.npz`。然後在本專案比對 `SiglipTextEncoder.embed_texts`、`DefaultImageDecoder` + `SiglipVisionEncoder.embed_images`、`SiglipTagger.probabilities`。回報：文字向量與影像向量各自的 cosine similarity、機率的最大絕對誤差、`config` 裡哪些值需要修改（節點名稱、`logit_scale`、`logit_bias`、`pad_token_id`）。文字向量對不上時先比 token id。把比對寫成 `tests/` 裡的測試，沒有權重時 skip。

### 任務 2 —— 校準 SigLIP 標籤門檻

> 任務 1 通過後才做。對 `<真實照片資料夾>`（至少 200 張）跑 `python -m photosorter scan <資料夾> --stages meta,siglip`，然後用 `SiglipTagger.probabilities()` 對已存的影像向量算出完整的機率矩陣。對每個預設標籤輸出機率的分布（各百分位數）與機率最高的 10 張照片路徑，讓我人工看過之後決定門檻。依我的決定更新 `siglip.default_threshold` 與個別標籤的 `tags.threshold`，再用 `rescore_library()` 全庫重標（不需要重跑 vision encoder）。不要自己決定門檻。

### 任務 3 —— 實作人臉偵測的輸出解碼與對齊

> 依 `HANDOFF.md` §4.2 實作 `photosorter/models/face.py` 的 `decode_scrfd()` 與 `align_face()`，規格寫在該檔的 docstring。驗證基準：在另一個虛擬環境用 `insightface` 的 `FaceAnalysis(name="buffalo_l")` 對 `<人像照片>` 取 bbox、關鍵點與 embedding，與我們的輸出比對（框與關鍵點的像素誤差、同一張臉 embedding 的 cosine similarity）。完成後 `photosorter.models.face.is_implemented()` 會回傳 True，GUI 與 `factory.resolve_stages()` 就會自動把 `faces` 階段排進去；確認 `python -m photosorter scan <資料夾>`（預設階段）能跑完，`faces` 表有資料，人臉分群後處理會產生 `persons`。

### 任務 4 —— 校準人臉分群

> 任務 3 完成後才做。對 `<有很多人像的資料夾>` 掃描後，用不同的 `face.cluster_eps`（0.30~0.50）與 `FaceClusterEngine(assign_threshold=...)`（0.55~0.75）各跑一次 `FaceClusterEngine.run(full_recluster=True)`，每組參數輸出：人物群組數、未成群的臉數、最大的 5 個群各取 6 張臉的裁切圖存成對照圖。讓我看過之後決定參數，再寫回 `config`。注意 `full_recluster=True` 不會動已命名或 `is_locked = 1` 的人物；測試時請用還沒有人工整理過的資料庫（另外指定 `paths.db_path`）。

### 任務 5 —— 模糊度門檻

> 對 `<真實照片資料夾>` 掃描後，從 `photos` 表取出 `blur_score` 的分布，列出分數最低的 30 張與最高的 10 張照片路徑給我看。依我選定的門檻，在 CLI 的 `archive` 加上預設值（目前要自己傳 `--blur-threshold`），並在 GUI 的匯出控制台加一個可調整的欄位。預設仍然不自動把照片標成廢片（`is_rejected` 只由使用者決定）。

### 任務 6 —— 權重下載腳本

> 寫 `scripts/download_weights.py`：把 `weights/README.md` 列的五個檔案下載到 `AppConfig.weights_dir` 底下對應的位置（路徑一律由 `load_config()` 取得，不可寫死）。先實際確認每個下載位置存在並記下檔案大小與 SHA-256，寫進腳本做完整性檢查；已存在且雜湊相符就略過。buffalo_l 是 zip，只取 `det_10g.onnx` 與 `w600k_r50.onnx`。下載完自動執行 `python -m photosorter doctor --load-models` 的檢查並回報。把驗證過的來源與雜湊更新回 `weights/README.md`，移除「A 電腦沒有驗證過」的註記。

### 任務 7 —— 事件與連拍參數

> 依我在 `HANDOFF_TO_PC_B.md` §3.4 記錄的問題調整：事件切太碎或太粗就調 `events.gap_minutes`、`events.max_distance_km`、`EventClusterEngine(trip_gap_hours=...)`；連拍漏抓或誤抓就調 `burst_seconds`、`burst_hamming`。每次調整後跑 `EventClusterEngine.from_config(cfg, db).run()` 並列出事件清單（名稱、張數、起訖時間）與連拍群組讓我確認。目前 `trip_gap_hours`、`burst_seconds`、`burst_hamming` 只能從建構子傳入；需要從 `config.json` 調整的話，在 `core/config.py` 新增欄位（純新增，並重新產生 `config.example.json`）。

### 給 Antigravity 的注意事項

- **Tk 主視窗只能在主執行緒被回收。** 背景執行緒若在視窗銷毀後還參考著它，Tcl 會直接中止整個行程（`Tcl_AsyncDelete: async handler deleted by the wrong thread`）。A 電腦已經踩過並修掉：`MainWindow.close()` 會等所有背景執行緒結束，`theme.releasing_tk()` 在主執行緒明確回收。新增任何背景工作都要走 `MainWindow._run_bg()` / `_start_job()`，不要自己開會參考到 widget 的執行緒。測試裡建立的每個 Tk 視窗都要在 `releasing_tk()` 區塊內銷毀。
- 工作執行緒不可以碰任何 widget；結果一律經 `_ui_queue` 交回主執行緒。
- GUI 用到的 SQL 寫在 `photosorter/db/queries_ui.py`，不要散在介面程式裡。
- 資料庫結構變更一律走 `db_manager.MIGRATIONS`，並同步更新 `schema.sql` 與 `SCHEMA_VERSION`（v1 → v2 就是這樣做的）。
