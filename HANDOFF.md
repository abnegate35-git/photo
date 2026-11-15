# 交接規格書

給在 **B 電腦（Windows 10）** 接手的開發模型（Claude Opus 5.5 或 Antigravity）。

A 電腦完成了底層架構：資料庫、管線調度、介面契約、ONNX Runtime 層、匯出引擎，並以測試驗證。你要做的是：把模型接上真實權重並驗證、實作人臉的輸出解碼與分群、做 Tkinter GUI，最後在 B 電腦上實際分類照片。

設計的理由在 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)；每個模組的用法在該檔案開頭的 docstring。本文件只講**現況、規則、待辦與呼叫方式**。

---

## 1. 第一個小時

```powershell
git clone <repo> photo ; cd photo
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m pytest                      # 應該全部通過（symlink 那一項沒權限時會略過）
.venv\Scripts\python -m photosorter doctor          # 看這台電腦的硬體、provider、缺哪些權重
```

- NVIDIA 顯示卡且 CUDA 12 / cuDNN 9 已裝好 → 改裝 `requirements-cuda.txt`；沒有可用顯示卡 → `requirements-cpu.txt`。三種 onnxruntime 互斥，換之前先 `pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu`。
- 依 [`weights/README.md`](weights/README.md) 放好權重，再跑 `python -m photosorter doctor --load-models`，確認每個模型實際落在哪個 provider，以及節點名稱是否與 `config.json` 一致。
- 沒有權重也能先跑 Stage 1：`python -m photosorter scan <照片資料夾> --stages meta`。
- 需要改設定就在專案根目錄建 `config.json`，只寫要覆寫的鍵（範本是 `config.example.json`）。`config.json` 不進版控。

---

## 2. 現況

### 2.1 已完成並驗證

| 模組 | 內容 | 驗證 |
|---|---|---|
| `core/` | 設定、路徑、型別、介面、事件、硬體探測 | `test_units.py` |
| `db/db_manager.py` | 單一寫入執行緒、group commit、savepoint 隔離、唯讀連線、遷移 | `test_db_manager.py` |
| `db/repository.py`、`schema.sql` | 全部業務 SQL 與 DDL | 透過管線與匯出測試 |
| `db/vector_index.py` | 記憶體內 cosine 檢索 | `test_units.py` |
| `models/onnx_runtime.py`、`preprocess.py` | session 建立、provider 降級、batch 框架 | `test_onnx_runtime.py`（真的 DirectML session、極小測試模型） |
| `pipeline/pipeline_engine.py` | 調度、有界佇列、動態 batching、暫停/停止、續掃、錯誤隔離 | `test_pipeline_engine.py`（假模型） |
| `pipeline/scanner.py`、`decoder.py` | Stage 0、Single-Decode Multi-Resize、Stage 1 | 管線測試 + 實際影像檔手動驗證（JPEG 方向與 DCT 縮放、HEIC、PNG、WebP、灰階、16-bit） |
| `pipeline/event_clustering.py` | 時間 + GPS 事件聚類 | `test_units.py`、管線測試 |
| `pipeline/export_manager.py` | dry-run、四種模式、衝突解決、回復、當機收尾 | `test_export_manager.py`（symlink 在 A 電腦無權限，**未實測**） |
| `utils/geo.py` | 離線反向地理編碼包裝 | 以真實座標手動驗證 |
| `ui/bridge.py` | `TkEventPump` | 以假 widget 測試；**沒有接過真的 Tk 視窗** |

### 2.2 程式寫完但沒對真實模型驗證

| 模組 | 狀態 |
|---|---|
| `models/siglip.py` | 流程完整，且用測試模型走過完整的程式路徑。但 A 電腦沒有權重，**數值正確性未驗證**。最可能出錯：節點名稱、tokenizer 的處理、`logit_scale` / `logit_bias`。→ §4.1 |
| `pipeline/metadata.py` | 基準版可用（時間、相機、曝光、GPS、方向、檔名時間），只用程式產生的 EXIF 測過，**沒有用真實相機/手機的照片驗證**。→ §4.4 |

### 2.3 骨架（呼叫會丟 `NotImplementedError`）

| 位置 | 待實作 |
|---|---|
| `models/face.py`：`decode_scrfd()`、`align_face()` | SCRFD 輸出解碼 + NMS、5 點對齊。→ §4.2 |
| `models/clustering.py`：`DbscanFaceClusterer.cluster()`、`assign_persons()` | 分群與寫回。→ §4.3 |
| `pipeline/decoder.py`：`_decode_raw()` | RAW 支援（選配）。→ §4.4 |

**重要**：預設的 `pipeline.stages` 是 `["meta", "siglip", "faces"]`。在 §4.2 完成之前，跑到第一個需要人臉的 batch 時管線會以 FAILED 結束，訊息裡有 `NotImplementedError`（這是刻意的：快速失敗，不會把照片標成錯誤）。在那之前請用 `--stages meta,siglip`，或在 `config.json` 設 `{"pipeline": {"stages": ["meta", "siglip"]}}`。之後加回 `faces` 重掃，只會補 Stage 3。

### 2.4 尚未開始

GUI、縮圖快取、近似重複分群（`photos.dup_group_id`、`phash` 欄位已預留）、廢片自動標記、人物合併/拆分的操作。

---

## 3. 不可違反的規則

這些是架構的承重牆。違反任何一條，測試不一定會立刻失敗，但會在大量照片或長時間執行時出問題。

1. **路徑零硬編碼。** 所有路徑由 `AppConfig` 的屬性取得（`cfg.db_path`、`cfg.weight(...)`、`cfg.cache_dir`）。不要寫磁碟代號、不要用目前工作目錄、不要 `os.path.join(__file__, ...)`。`test_units.py` 有一項會掃原始碼抓絕對路徑。
2. **硬體數值不寫進設定檔。** `providers`、`decode_workers`、`batch_size` 的預設是 `"auto"` / `0`。實際值由 `plan_runtime()` 在啟動時決定。不要把 B 電腦探測到的值寫回 `config.example.json`。
3. **不自己開 SQLite 連線。** 讀用 `db.query*()`（任何執行緒皆可），寫用 `db.write()` / `db.submit()` 或 `Repository` 的方法。寫入的 `fn(conn)` 裡只放 SQL，不做 I/O、不 `BEGIN`/`COMMIT`、不在裡面再呼叫 `db.write()`。
4. **狀態位元與資料同一個交易。** 新增階段時，資料與 `process_status` 的位元必須在同一個 `db.write(fn)` 裡寫入，否則續掃會出錯。位元值發布後不可更動；新位元往高位加，並更新 `StatusTest`。
5. **只有推論執行緒可以呼叫模型。** `VisionEmbedder` / `TagScorer` / `FaceAnalyzer` 的 `load()` 與推論方法只會在 inference 執行緒被呼叫。不要在別的執行緒建 ONNX session、不要在元件裡自己開執行緒。GUI 需要即時推論（例如以文搜圖）時另外建一個獨立的 encoder 實例，而且不要與掃描同時進行。
6. **核心不 import GUI，GUI 不在工作執行緒碰 widget。** 核心 → UI 只走 `EventBus`。GUI 用 `TkEventPump` 在主執行緒收事件。
7. **使用者的決定永遠優先於自動結果。** 自動流程不可覆蓋：`photo_tags.source='manual'` 或 `is_rejected=1`、`faces.assign_src='manual'`、`events.is_user_edited=1`、`photos.is_rejected / is_favorite / rating`。
8. **不動原始檔。** 除了 `ExportManager.execute()` / `rollback()`，任何程式碼都不可以寫入、改名、刪除使用者的照片。匯出絕不覆寫既有檔案。
9. **影像與向量的慣例**（`core/types.py` 開頭）：影像一律 `uint8`、HWC、RGB、已套用 EXIF 方向；向量一律 `float32`、已 L2 正規化；座標一律是原圖正規化 0..1。
10. **改預設設定後**執行 `python -m photosorter dump-config -o config.example.json`（測試會檢查兩者同步）。**改 schema** 不要動 `schema.sql` 裡已發布的結構，在 `db_manager.MIGRATIONS` 加一筆（新資料庫另外同步更新 `schema.sql` 並遞增 `SCHEMA_VERSION`）。

---

## 4. 待辦（依建議順序）

### 4.1 SigLIP 接上真實權重並驗證

目標：確認我們算出的影像向量、文字向量、標籤機率與 HuggingFace 參考實作一致。

1. `doctor --load-models` 看 vision / text 模型的輸入輸出名稱與形狀，必要時改 `config.json` 的 `siglip.vision.*`、`siglip.text.*`。vision 的輸出必須是 `(N, 768)`（`SiglipVisionEncoder.load()` 會檢查）。
2. 在**另一個**有 PyTorch 與 transformers 的環境產生參考答案（不要把 torch 裝進本專案的 venv）：對 3～5 張照片算 `model.get_image_features(...)`，對預設標籤的 prompt 算 `model.get_text_features(...)`（processor 要用 `padding="max_length"`），各自 L2 正規化後連同 `model.logit_scale.exp().item()`、`model.logit_bias.item()` 存成 `.npz`。
3. 在本專案比對：
   - **文字向量**：`SiglipTextEncoder.embed_texts(prompts)` 與參考的 cosine 應該非常接近 1。差很多的話先比 token id（`tokenizer.encode(...)` 對 HF 的 `input_ids`）：常見原因是小寫/標點的正規化、padding 沒補到 64、pad id 不對。如果 `tokenizer.json` 的行為與 HF 的 `SiglipTokenizer` 不一致，改用 `sentencepiece` 載入 `spiece.model`（需加進 requirements）。
   - **影像向量**：`DefaultImageDecoder` → `SiglipVisionEncoder.embed_images`。我們的縮圖路徑（DCT 縮小 → 面積平均縮到 224）與 HF 的 processor（原圖雙三次縮到 224）不完全相同，向量不會逐位元相等，但 cosine 應該很高。明顯偏低代表通道順序、正規化或輸出節點取錯。
   - **機率**：用參考的 `logit_scale` / `logit_bias` 更新設定後，`SiglipTagger.probabilities()` 應與 HF 的 `torch.sigmoid(logits_per_image)` 一致。
4. **校準門檻。** `siglip.default_threshold = 0.2` 只是初始值，沒有依據。拿一批真實照片看各標籤的機率分布（`SiglipTagger.probabilities()` 回傳完整矩陣），決定預設門檻，個別標籤用 `tags.threshold` 覆寫。SigLIP 的機率是各標籤獨立的，正確的配對也不一定接近 1。
5. 把這個比對寫成 `tests/` 裡的測試，沒有權重時 `skip`。

之後使用者改標籤不必重掃：`SiglipTagger.load()` 後呼叫 `rescore_library(tagger, repo)`。

### 4.2 人臉：SCRFD 解碼與對齊

`models/face.py` 的 `InsightFaceAnalyzer.analyze()` 流程已寫好，缺兩個函式；規格（輸出張量的順序、stride、anchor、對齊模板常數）都在該檔的 docstring。

1. 實作 `decode_scrfd(outputs, det_threshold, nms_threshold)`。
2. 實作 `align_face(work_image, landmarks_norm)`。
3. **驗證基準**：另開環境 `pip install insightface`，用 `FaceAnalysis(name="buffalo_l")` 對同一批照片取 bbox、kps、embedding，與我們的輸出比對（框與關鍵點的誤差、同一張臉的 embedding cosine）。注意 insightface 吃的是 BGR 原圖並自己做 letterbox；我們是從工作圖產生 640 輸入，座標要先換回同一個基準再比。
4. 已經處理好的細節：偵測模型的 batch 維度寫死為 1（`OnnxModel` 自動偵測並逐張送）；對齊臉跨照片收集成一個大 batch 送 ArcFace；太小的臉（`face.min_face_px`）只記位置不取特徵。
5. 可以之後再考慮：合照裡的小臉在 640 的輸入上偵測不到時，把工作圖切塊重疊偵測。

寫完後 `FakeFaces`（`tests/fakes.py`）是介面的最小範例；管線測試已涵蓋條件式略過與續掃。

### 4.3 人臉分群

1. `DbscanFaceClusterer.cluster()`：`sklearn.cluster.DBSCAN(metric="cosine")` 或 `HDBSCAN`（scikit-learn 內建）。參數 `face.cluster_eps`、`face.cluster_min_samples` 的預設值沒有依據，要用真實照片調。
2. `assign_persons()`：寫回規則列在該函式的 docstring（手動指派不可覆蓋、群裡已有 person 就併入、不自動合併 person）。
3. 在 `pipeline/factory.py` 標示 TODO 的地方用 `engine.add_post_step("face_clusters", ...)` 掛上，並在 `EventKind.POST_STEP` 回報。
4. 新照片的增量指派可以先比對 `persons.centroid`，夠近就直接指派，剩下的才進分群。

### 4.4 中繼資料與格式

- 用 B 電腦上真實的手機/相機照片確認 `BasicMetadataExtractor` 的結果（時間、GPS、方向），特別是 HEIC。
- 可擴充：`OffsetTimeOriginal`（時區）、鏡頭資訊、XMP、Live Photo 配對、影片。要換實作就寫一個符合 `MetadataExtractor` 的類別，在 `factory.py` 傳給 `DefaultImageDecoder(cfg, metadata=...)`。
- RAW：`scan.enable_raw` 預設關閉。要支援就實作 `_decode_raw()`（建議用 `rawpy` 取內嵌預覽），並把 `rawpy` 加進 requirements。
- `ScreenshotDetector` 的關鍵字與解析度清單在 `config.screenshot`，可依 B 電腦的照片調整。

### 4.5 品質功能（欄位已預留）

- 近似重複：`photos.dhash` 已有值，用 `utils.imageops.hamming64()` 找距離小的配對，分群後寫 `dup_group_id`。門檻要校準。
- 廢片建議：依 `blur_score` 提出候選，由使用者確認後才寫 `is_rejected`。`blur_score` 的門檻同樣要用真實照片決定。

### 4.6 GUI

見下一節。

---

## 5. GUI 接線

### 5.1 啟動與關閉

```python
from photosorter.pipeline.factory import create_app
from photosorter.ui.bridge import TkEventPump
from photosorter.core.events import EventKind

ctx = create_app()                         # 載入設定、開資料庫、收尾上次中斷的匯出；不載入模型
engine = ctx.build_engine()                # 只建立物件，不探測硬體

root = tk.Tk()
pump = TkEventPump(root, ctx.bus, on_event=view.handle_event)   # handle_event 在 Tk 主執行緒被呼叫
pump.start()

engine.start([folder])                     # 立即返回
engine.pause(); engine.resume(); engine.stop()                   # 都不阻塞

def on_close():
    pump.stop()
    engine.stop()
    engine.join(timeout=60)                # 等手上的 batch 做完
    ctx.close()
    root.destroy()
```

### 5.2 事件（`core/events.py` 的 `EventKind`，payload 欄位寫在各成員旁的註解）

| 事件 | 用途 |
|---|---|
| `STATE` | 啟用/停用按鈕。狀態：`idle / running / paused / stopping / finished / stopped / failed` |
| `MODEL` | 「載入模型中…」與載入後顯示每個模型用的 provider |
| `SCAN_PROGRESS` → `SCAN_DONE` | 掃描中顯示不確定進度；`SCAN_DONE` 之後才知道總數 |
| `PROGRESS` | 進度條：`processed / to_process`；另有 `inferred`、`faces_found`、`errors`、`queue_depth`、`elapsed_s` |
| `PHOTO_DONE` / `PHOTO_ERROR` | 即時更新列表、錯誤清單 |
| `POST_STEP` | 後處理階段（地理編碼、事件、分群） |
| `FINISHED` | 一定是最後一個事件；`state`、`stats`、`error` |
| `EXPORT_PROGRESS` / `EXPORT_DONE` | 匯出進度 |

進度類事件在 UI 太慢時會被丟棄（下一筆會蓋掉它），狀態與完成事件不會丟。

### 5.3 哪些呼叫會阻塞

| 呼叫 | 行為 | 在 GUI 裡 |
|---|---|---|
| `engine.start/pause/resume/stop` | 立即返回 | 直接呼叫 |
| `db.query*()` | 同步讀取；每條執行緒自己的連線，不會被寫入擋住 | 小查詢直接呼叫；大量結果放背景執行緒，結果用 `root.after(0, ...)` 交回 |
| `Repository` 的寫入、`db.write()` | 等 commit（通常幾毫秒） | 單筆操作直接呼叫即可 |
| `engine.join()` | 阻塞到結束 | 只在關閉視窗時用，或加 timeout |
| `exporter.plan()` | 會讀檔案系統，照片多時較慢 | 背景執行緒 |
| `exporter.execute()` / `rollback()` | 阻塞到做完 | 背景執行緒；`execute(plan, should_stop=...)` 可取消 |
| `SiglipTagger.load()`、`rescore_library()` | 載入模型、全庫運算 | 背景執行緒，且不要與掃描同時進行 |

### 5.4 讀寫範例

```python
# 讀：任何執行緒
rows = ctx.db.query(
    "SELECT id, path, taken_at, blur_score FROM photos WHERE event_id = ? ORDER BY taken_ts", (event_id,))

# 使用者手動加標籤 / 否決自動標籤
ctx.db.execute(
    "INSERT INTO photo_tags(photo_id, tag_id, source) VALUES (?, ?, 'manual') "
    "ON CONFLICT(photo_id, tag_id) DO UPDATE SET source = 'manual', is_rejected = 0", (pid, tid)).result()
ctx.db.execute("UPDATE photo_tags SET is_rejected = 1 WHERE photo_id = ? AND tag_id = ?", (pid, tid)).result()

# 人臉命名（指派必須標成 manual，重新分群才不會洗掉）
ctx.db.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id)).result()
ctx.db.execute("UPDATE faces SET person_id = ?, assign_src = 'manual' WHERE id = ?", (person_id, face_id)).result()

# 事件改名（標成已編輯，重新聚類才不會覆蓋）
ctx.db.execute("UPDATE events SET name = ?, is_user_edited = 1 WHERE id = ?", (name, event_id)).result()

# 新增自動標籤（prompt 是完整英文句子）→ 重標全庫
ctx.repo.upsert_tag("snow", "a photo of snow.", display_name="雪景")

# 匯出：先預覽，確認後執行，可回復
plan = ctx.exporter.plan(dest, mode="hardlink", template="{year}/{event}/{filename}")
plan.warnings; plan.summary(); plan.ops         # 給使用者看
result = ctx.exporter.execute(plan)             # 背景執行緒
ctx.exporter.rollback(result.batch_id)
```

GUI 會用到的查詢請加在 `db/repository.py`（或新增 `db/queries_ui.py`），不要把 SQL 散在 UI 程式碼裡。

### 5.5 以圖搜圖 / 以文搜圖

```python
from photosorter.db.vector_index import InMemoryVectorIndex
from photosorter.models.siglip import SiglipTextEncoder

s = ctx.cfg.siglip
model_id = ctx.repo.ensure_model(s.model_name, "image_embed", s.embed_dim)
index = InMemoryVectorIndex.from_arrays(*ctx.repo.load_embedding_matrix(model_id, s.embed_dim))
index.search(index.get(photo_id), top_k=50)                     # 以圖搜圖

text_encoder = SiglipTextEncoder(ctx.cfg)                       # 在背景執行緒 load()，且固定由同一條執行緒使用
text_encoder.load()
index.search(text_encoder.embed_texts(["a dog on the beach"])[0], top_k=50)   # 以文搜圖
```

索引在 GUI 啟動時於背景建立；掃描期間收到 `PHOTO_DONE` 可以用 `index.add()` 增量更新。

### 5.6 縮圖

核心沒有做縮圖。建議：存在 `cfg.cache_dir / "thumbs"` 底下，以 `sha256` 命名（內容不變就不必重做，搬移後也還有效），用小型執行緒池延遲產生，不要存進資料庫。人臉頭像用 `faces.bbox_*`（原圖正規化座標）從照片裁出。

### 5.7 套件

`ttkbootstrap` 或 `sv-ttk` 擇一，版本寫進新的 `requirements-gui.txt`。GUI 程式放在 `photosorter/ui/`。

---

## 6. Windows 10 注意事項

- **DirectML** 需要 Windows 10 1903（build 18362）以上與支援 DX12 的顯示卡驅動。不符合時不會報錯，`create_session()` 會降級到 CPU；`doctor` 會顯示 build 編號與實際的 provider。
- **符號連結**需要開啟「開發人員模式」或以系統管理員執行。`ExportManager.plan()` 會偵測並在 `plan.warnings` 提醒。symlink 模式在 A 電腦沒有權限、沒有實測過，B 電腦請跑 `tests/test_export_manager.py` 確認。
- **硬連結**只能在同一個 NTFS 磁碟區內。exFAT / FAT32 的外接硬碟不支援，該筆操作會失敗並記錄原因。
- **長路徑**：超過 260 字元的路徑需要系統開啟 `LongPathsEnabled`（`doctor` 會顯示）。
- **主控台編碼**：繁體中文 Windows 的主控台是 cp950。CLI 的輸出已處理；若自行把含中文的輸出重導到檔案，請設 `PYTHONUTF8=1`。
- **資料庫放本機磁碟**，不要放網路磁碟（WAL 需要共享記憶體）。
- **打包**（之後若要用 PyInstaller）：`photosorter/db/schema.sql` 要列入 datas；打包後專案根目錄是 exe 所在資料夾（`core/paths.py`）。A 電腦上曾遇過 `--noconsole` 的 exe 被防毒自動清除，B 電腦若有同樣情況改用 console 模式。

---

## 7. 已知限制與未決事項

- `taken_ts` 把相機的當地時間當成 UTC 換算，只適合排序與算間隔。跨時區旅行的照片順序正確，但與 `mtime` 不可互比。
- 事件聚類只採用時間來自 EXIF 或檔名的照片；時間來自 `mtime` 的照片不會被分到任何事件。
- 一個路徑一列：內容相同的兩個檔案是兩列（AI 結果直接複製，不重跑）。
- 從 `scan.extensions` 移除某個副檔名後重掃，該類檔案會被標成遺失（紀錄仍在）。
- 根目錄路徑以 `os.path.realpath` 正規化。同一個資料夾透過不同的網路路徑或磁碟代號掃描會被當成不同的照片（會由 SHA-256 判定為複本，但仍是兩列）。
- 每次 `engine.start()` 都重新載入模型，結束時釋放。常駐模型可以省下重複掃描時的載入時間，但會一直佔用記憶體；目前選擇釋放。
- 一個資料庫同時只應該有一個引擎在跑。
- 沒有處理影片與 Live Photo。
- 需要用真實照片校準、目前只是初始值的參數：`siglip.default_threshold`、`face.det_threshold`、`face.min_face_px`、`face.cluster_eps`、`face.cluster_min_samples`、`events.gap_minutes`、`events.max_distance_km`、blur 與 dHash 的門檻。
- 佇列加解碼的實際記憶體尖峰沒有量過（估算見 ARCHITECTURE §3.2）。
