# 架構說明

本文件說明核心引擎「為什麼這樣設計」。每個模組「怎麼用」寫在該檔案開頭的 docstring；接手實作要做什麼寫在 [`HANDOFF.md`](../HANDOFF.md)。

## 1. 目錄結構與職責

```
photo/
├── photosorter/                 Python 套件（Headless Core，不 import 任何 GUI）
│   ├── __main__.py              CLI：doctor / scan / stats / archive / export / rollback / dump-config
│   ├── core/                    零相依的基礎：設定、型別、介面、事件、硬體探測
│   │   ├── paths.py             專案根目錄與路徑解析（唯一決定檔案位置的地方）
│   │   ├── config.py            設定 dataclass 與載入（預設值的唯一來源）
│   │   ├── status.py            process_status 位元定義
│   │   ├── types.py             模組間傳遞的封包（DecodedImageTask、InferenceResult…）
│   │   ├── interfaces.py        Protocol 介面契約與執行緒契約
│   │   ├── events.py            EventBus / EventQueue（核心 → UI 的唯一通道）
│   │   ├── hardware.py          延遲硬體探測、provider 降級順序、RuntimePlan
│   │   └── export_manager.py    四層多視圖歸檔（第二階段；應用層服務，見下方說明）
│   ├── db/
│   │   ├── schema.sql           完整 DDL
│   │   ├── db_manager.py        單一寫入執行緒 + thread-local 唯讀連線
│   │   ├── repository.py        所有業務 SQL
│   │   ├── vector_index.py      記憶體內 cosine 檢索
│   │   └── queries_ui.py        GUI 專用查詢（導航樹計數、照片清單、人物命名鎖定、匯出歷史）
│   ├── models/                  只有這裡會碰 onnxruntime
│   │   ├── onnx_runtime.py      session 建立、provider 自動降級、OnnxModel
│   │   ├── preprocess.py        batch 前處理框架與各模型的正規化參數
│   │   ├── siglip.py            vision / text encoder、多標籤判定、全庫重標
│   │   └── face.py              SCRFD + ArcFace（骨架）
│   ├── pipeline/
│   │   ├── pipeline_engine.py   調度引擎：佇列、batching、暫停/停止、續掃
│   │   ├── scanner.py           Stage 0：走訪資料夾並與資料庫比對
│   │   ├── decoder.py           Single-Decode Multi-Resize + Stage 1
│   │   ├── metadata.py          EXIF 解析、截圖判定
│   │   ├── event_clustering.py  最小版事件聚類（EventClusterer 介面的參考實作）
│   │   ├── export_manager.py    自訂模板的單層匯出，也是四層歸檔的底層：dry-run / 執行 / 回復
│   │   └── factory.py           組裝（介面與具體實作唯一綁定的地方）
│   ├── utils/                   imageops、hashing、geo、logging_setup
│   │   └── clustering.py        人臉分群、事件分群、連拍偵測（第二階段；應用層服務）
│   └── ui/                      桌面介面（第三階段）
│       ├── app.py               run_gui()：啟動入口
│       ├── main_window.py       三欄式主視窗；每 100 ms 輪詢背景佇列
│       ├── thumbnail_grid.py    虛擬滾動照片牆、LRU 快取、背景載入縮圖
│       ├── dialogs.py           人物命名、匯出預覽、復原歷史
│       ├── theme.py             高 DPI、sv-ttk / clam 深色、Tk 視窗的安全釋放
│       └── bridge.py            TkEventPump（給其他以 Tk 為底的介面用）
├── tests/                       unittest（pytest 也能跑）；fakes.py 是假模型
├── docs/ARCHITECTURE.md         本文件
├── weights/                     模型權重（不進版控）
├── data/                        資料庫、快取、日誌（不進版控）
├── config.example.json          設定範本（由程式產生）
├── requirements*.txt
├── HANDOFF.md                   交接規格書
└── AGENTS.md / CLAUDE.md        給接手的 AI agent 的入口
```

相依方向是單向的：`core` ← `db` ← `models` ← `pipeline` ← `ui`。`core` 的基礎模組不 import 其他任何子套件；`pipeline_engine.py` 只認得 `core.interfaces` 裡的 Protocol，不 import `models/` 的具體類別（由 `factory.py` 注入）。

**兩個例外**：`core/export_manager.py` 與 `utils/clustering.py` 是第二階段依指定位置放的應用層服務，它們會讀寫資料庫，所以相依方向與所在的資料夾不同 —— 兩者都依賴 `db/`，`core/export_manager.py` 還繼承了 `pipeline/export_manager.py`。請把它們當成位在相依鏈頂端（與 `pipeline/factory.py` 同層）的模組：基礎模組（`core/config.py`、`core/types.py`、`db/`…）不可以反過來 import 它們。

## 2. 資料流

```
                         ┌────────────────────────── pipeline 執行緒（調度）──────────────────────────┐
 engine.start(roots) ──▶ │ Stage 0 掃描 ──▶ candidates ──▶ 派工（semaphore 限流）   …   全域後處理     │
                         └───────┬──────────────────────────────┬───────────────────────────▲─────────┘
                                 │（同時）                       ▼                           │
                                 │              ┌──────── decode-N 執行緒（生產者）────────┐  │
                                 │              │ 讀檔一次 → SHA-256 → 搬移/重複判定       │  │
                                 │              │ 解碼一次 → 工作圖 → 三個分支             │  │
                                 │              │ Stage 1 寫入資料庫                       │  │
                                 │              └──────────────────┬──────────────────────┘  │
                                 │                                 ▼ DecodedImageTask        │
                                 │                    ┌─ queue.Queue(maxsize) ─┐  有界       │
                                 ▼                    └────────────┬───────────┘             │
                       ┌──────────────── inference 執行緒（唯一的消費者）─────────────────┐   │
                       │ 載入模型（provider 降級）                                        │   │
                       │ 動態 batching → Stage 2 SigLIP → Stage 3 條件式人臉              │───┘
                       │ 整個 batch 的結果 + 狀態位元 → 一個交易                          │
                       └──────────────────────────────┬───────────────────────────────────┘
                                                      ▼
                                        db-writer 執行緒（唯一的寫入連線）
```

各階段做的事：

| 階段 | 位置 | 內容 | 完成位元 |
|---|---|---|---|
| Stage 0 | `scanner.py` + `_decode_job` | size/mtime 比對；SHA-256 相同且舊路徑已不存在 → 搬移，只改路徑 | — |
| Stage 1 | `decoder.py` | 尺寸、EXIF、截圖判定、dHash、blur_score | `HASHED｜META｜QUALITY` |
| Stage 2 | `_stage2` | SigLIP 影像向量、多標籤 | `EMBEDDED｜TAGGED` |
| Stage 3 | `_stage3` | 條件式人臉偵測與特徵 | `FACES`（略過時另加 `FACES_SKIPPED`） |
| 後處理 | `_run_post_steps` | 依序：離線地理編碼 → 事件分群與命名 → 連拍偵測 → 人臉分群 | `GEOCODED` |

## 3. 關鍵設計決定

### 3.1 一張照片只讀一次、只解碼一次

`decoder.read()` 把整個檔案讀進記憶體，SHA-256 與解碼共用同一份 bytes。解碼結果先縮成「工作圖」（長邊 ≤ 1280），之後的分支全部從工作圖產生：

- 灰階（長邊 ≤ 1024）→ `blur_score` 與 `dHash`。先正規化解析度再算 Laplacian 變異數，不同相機的照片才能用同一個門檻比較。
- 640×640 letterbox → 人臉偵測。
- 224×224 → SigLIP。
- 工作圖本身 → 人臉對齊的來源。640 的畫面裡一張臉常常只有 30～60 像素，從 1280 的圖裁對齊臉，ArcFace 的特徵品質明顯較好。

JPEG 用 DCT 縮放直接解到 1/2、1/4、1/8，是最大的加速來源（實測 2400 萬像素：342 ms → 82 ms）。

佇列裡放的是 `uint8`，正規化成 float 是模型類別在推論前一次向量化完成，佇列的記憶體只有 float32 的四分之一。

### 3.2 有界佇列與背壓

每個 `DecodedImageTask` 最多約 5 MB（工作圖 1280×960×3 = 3.7、偵測輸入 640×640×3 = 1.2、SigLIP 輸入 0.15）。預設 `queue_maxsize = 64`，佇列上限約 320 MB，另外每條解碼執行緒手上還有一份檔案內容與解碼中的暫存。整體的實際尖峰沒有量過，B 電腦第一次跑大量照片時請觀察記憶體；吃緊就調小 `runtime.queue_maxsize` 或 `decode.work_long_edge`。

背壓是一路往上傳的：推論慢 → 佇列滿 → 解碼執行緒卡在 `put` → semaphore 用完 → 調度執行緒停止派工。任何一環變慢，上游都會等，不會堆積。

### 3.3 只有一條推論執行緒

GPU / DirectML 上多條執行緒同時 `run()` 只會互相搶資源，總吞吐量反而下降。平行度交給 batch：從第一個任務到手開始計時，湊滿 `batch_size` 或超過 `batch_timeout_ms` 就送出。ONNX session 建立在使用它的那條執行緒上，結束時釋放。

### 3.4 中斷續掃不需要 checkpoint 檔

每個階段的資料與 `photos.process_status` 的位元**寫在同一個交易裡**。所以不論正常停止、當機還是斷電，資料庫裡「位元說做完了」與「資料真的在」永遠一致。重啟後 Stage 0 讀出每張照片的狀態，`required & ~status` 就是它還欠的階段。

這也讓「之後才開啟某個階段」變得很自然：先只跑 `["meta"]`，日後改成 `["meta","siglip","faces"]`，重掃時只會補 Stage 2/3，不重做 Stage 1。

位元值存在資料庫裡，**發布後不可更動**（`tests/test_units.py::StatusTest` 鎖住了數值）。

### 3.5 SQLite：單一寫入者

WAL 模式允許一個寫入者與多個讀取者並行，但同時只能有一個寫入者。所有寫入都排進一個佇列，由專用的 writer 執行緒合併成交易（group commit）：閒置時延遲接近零，忙碌時自動變成大批次。每個操作各自一個 SAVEPOINT，一個失敗不影響同批的其他操作；Future 在 COMMIT 成功後才完成。讀取走各執行緒自己的唯讀連線，不會被寫入擋住。

細節與使用規則在 `db/db_manager.py` 開頭。

### 3.6 SigLIP 多標籤與「不重跑 vision encoder 的重標」

文字向量在啟動時算好並快取在 `tags.text_vector`；prompt 一改，trigger 會把快取作廢。掃描時只跑 vision encoder，`prob = sigmoid(logit_scale · cos + logit_bias)`，每個標籤獨立判定。

影像向量存在 `embeddings`，所以新增或修改標籤後，`rescore_library()` 對全庫做一次矩陣乘法就能重標，不必再碰照片檔案。使用者手動加的標籤與否決過的標籤不會被自動重標覆蓋。

### 3.7 條件式人臉（Stage 3 的閘門）

規則集中在 `pipeline_engine.should_run_faces()`，而且是資料驅動的：每個標籤有 `gate_role`（`none` / `skip_face` / `has_person`）。

1. 命中任何 `has_person` 標籤 → 跑。
2. Stage 1 判定為截圖，或命中任何 `skip_face` 標籤（文件、收據、截圖、白板）→ 略過。
3. 其餘 → 跑。

要調整哪些類別略過人臉，改 `tags.gate_role` 即可，不必改程式。

### 3.8 階段失敗的隔離

- 單一檔案讀不到或解不開：記一列 `ERR_DECODE`，管線繼續。重掃時預設不再重試（`scan.retry_errors` 可開）。
- 一個 batch 推論失敗：逐張重試，把壞掉的那張隔離出來標 `ERR_INFER`，其他張照常完成；該張先前階段已完成的結果仍會保存。
- 元件丟 `NotImplementedError`（骨架還沒實作）：整條管線立即以 FAILED 結束並給出明確訊息，不會把全庫的照片都標成推論錯誤。
- 後處理步驟失敗：只有那個步驟失敗，逐張的結果不受影響。

### 3.9 硬體延遲探測

設定檔只存「意圖」：`providers: "auto"`、`decode_workers: 0`、`batch_size: 0`。實際數值在 pipeline 啟動時才決定：

1. 推論執行緒建立 session，依 CUDA → DirectML → CPU 逐一嘗試。「建立成功但被靜默退回 CPU」與「第一次推論才失敗」都視為該 provider 不可用，繼續往下試。
2. 調度執行緒依**實際生效**的 provider 與實體核心數算出 `RuntimePlan`（解碼執行緒數、ORT 執行緒數、batch 大小）。

`python -m photosorter doctor --load-models` 可以在不掃描的情況下看到這台電腦上的結果。

### 3.10 非破壞性

在 `ExportManager.execute()` 之前，本工具對原始檔案是唯讀的。匯出有三步：`plan()`（dry-run，不碰檔案系統也不寫資料庫）→ `execute()`（每筆操作先寫日誌再動檔案）→ `rollback()`。絕不覆寫既有檔案。安全守則列在 `pipeline/export_manager.py` 開頭。

### 3.11 分群（第二階段）

`utils/clustering.py` 的兩個引擎都是全域後處理，直接讀寫資料庫，可以重複執行。

**人臉**（`FaceClusterEngine`）預設是增量的：只處理還沒指派的臉。先以目前的成員算出每個既有人物的中心向量，新的臉與中心向量的 cosine similarity 達 0.65 就直接歸入；沒命中的才跑 `DBSCAN(eps=0.38, min_samples=3, metric="cosine")` 產生新人物。這樣使用者整理過的人物不會因為重跑而變動，DBSCAN 的 O(N²) 也只作用在新的臉上。`full_recluster=True` 會把未保護的自動群組解散重分；已命名、已鎖定的人物與手動指派的臉在任何模式下都不動，而且寫入的 SQL 本身帶保護條件。

**事件**（`EventClusterEngine`）是相鄰照片的分段：時間差小於 6 小時就同一段；超過的話，如果還在同一個城市或距離小於 30 公里、而且空檔不到 36 小時，視為同一趟行程。後面這條規則在常駐城市（有照片的天數最多的城市）不適用，否則每天在家拍照會被串成一個長達數月的事件。分完之後替自動事件產生預設名稱 `YYYY-MM-DD_{City}_{TopTag}`。

**連拍**：相鄰兩張時間差小於 3 秒且 dHash 漢明距離不超過 4 就串成一組，以清晰度與人臉偵測信心度評分選出最佳照片。

### 3.12 四層多視圖歸檔（第二階段）

`core/export_manager.py` 把分類結果實體化成「單一實體主庫 + 硬連結視圖」：

```
<export_root>/
├── 01_Timeline_Master/{Year}/{日期}_{City}_{EventName}/    唯一一份實體資料
├── 02_By_Person/{PersonName}/…                             硬連結
├── 03_By_Semantic_Tag/{TagName}/                           硬連結
└── 99_Review_Staging/_Blurry_Suspects｜_Burst_Duplicates｜_Screenshots_Downloads/
```

- 視圖一律**從主庫的新檔案**建立硬連結，所以和主庫永遠在同一個磁碟區：即使原始相簿在另一顆硬碟、主庫只能用複製，視圖仍然不佔空間。
- 檔名是 `{Date}_{Time}_{SubSec}_{DeviceOrTag}_{ShortHash}.{ext}`。ShortHash 是 SHA-256 的前 6 碼，讓重複匯出可以直接認出「這個檔案已經在了」而略過；前綴與 6 碼都相同但完整 SHA-256 不同時才加 `_v2`。
- 判斷「目標位置的同名檔案是不是同一個內容」先查 `export_logs`（主檔那一筆記有完整 SHA-256），日誌沒有紀錄才比大小與重新計算雜湊。同一份日誌也讓照片在重複匯出時留在第一次放的位置，事件改名不會在主庫多出一份。
- 每張照片的主檔與它的 Sidecar（Live Photo 影片、RAW、XMP…）是一個不可分割的群組：一起執行、共用新檔名主幹，任何一筆失敗就整組還原。視圖失敗只影響那一個連結。
- 它繼承第一階段的匯出引擎，所以 write-ahead 日誌、暫存檔再 rename、絕不覆寫、回復前的身分確認、當機後的 `recover()` 都是同一套。`export_logs` 裡只有主檔記 `photo_id`，Sidecar 與視圖留 NULL，回復 MOVE 時才不會把 Sidecar 的路徑寫回 `photos.path`。

### 3.13 桌面介面（第三階段）

主執行緒只做三件事：畫畫面、處理使用者操作、每 100 ms 輪詢兩個佇列。

```
工作執行緒                                  Tk 主執行緒
PipelineEngine ──EventBus──▶ EventQueue ─┐
_run_bg(fn) ────結果───────▶ _ui_queue ──┴─▶ self.after(100, self._poll_events) ─▶ 更新 widget
縮圖執行緒 x2 ──PIL 影像──▶ 佇列 ───────────▶ ThumbnailGrid._drain ─▶ PhotoImage、Canvas
```

- **零阻塞**：掃描、分群、匯出、回復，以及每一次清單與明細查詢，都在背景執行緒；工作執行緒不碰任何 widget。
- **虛擬滾動**：照片牆的 Canvas 只替「可見的列 ± 2 列」建立物件，照片總數不影響建立的物件數。
  解碼後的 `PhotoImage` 放在上限 300 張的 LRU；畫面上正在顯示的圖另有強參考，不會因為被淘汰而變空白。
- **沒有權重也能用**：`factory.resolve_stages()` 只排入跑得起來的階段（權重存在、人臉階段已實作），
  徽章顯示 `Mock / Weights Missing`。CLI 的 `scan` 則維持「缺權重就明確失敗」。
- **關閉**：`WM_DELETE_WINDOW` 先發停止訊號，用 `after()` 輪詢等引擎與所有背景執行緒結束、資料庫寫入 flush 之後才銷毀視窗；
  等待期間介面仍有回應。

**Tk 視窗只能在主執行緒被回收。** Tcl 直譯器必須由建立它的執行緒釋放，而 Tk 物件之間有循環參考，要靠 Python 的
循環垃圾回收器才收得掉 —— 回收器會在「剛好觸發配置門檻的那條執行緒」執行。如果那是解碼或縮圖的執行緒，
Tcl 會直接中止整個行程。所以：關閉前等所有可能參考到視窗的背景執行緒結束；`ui/app.py` 在事件迴圈結束後
停用自動回收、放掉視窗的參考、在主執行緒明確回收一次（`theme.releasing_tk()` 是同一件事的 context manager 版）。

## 4. 資料庫

完整 DDL 在 [`photosorter/db/schema.sql`](../photosorter/db/schema.sql)。

```
scan_sessions ◀── photos ──▶ events
                    │
        ┌───────────┼────────────┬──────────────┐
        ▼           ▼            ▼              ▼
   embeddings   photo_tags     faces ──▶ persons      export_logs ──▶ export_batches
        │           │            │
        ▼           ▼            ▼
      models       tags        models
```

| 資料表 | 用途 | 備註 |
|---|---|---|
| `photos` | 一個路徑一列：檔案指紋、EXIF、品質、使用者標記、`process_status`、連拍標記（v2：`burst_group_id`、`is_best_shot`、`is_burst_duplicate`） | 刻意不放 BLOB，瀏覽/篩選的掃描才會快 |
| `embeddings` | SigLIP 768d 向量 | 獨立一張表；`(photo_id, model_id)` 唯一，換模型可並存 |
| `tags` / `photo_tags` | 標籤定義（含 prompt、文字向量快取、`gate_role`）與關聯 | `source` 區分自動/手動，`is_rejected` 記錄使用者的否決 |
| `faces` / `persons` | 人臉框、偵測分數、512d 向量、所屬人物 | `assign_src='manual'` 的指派、已命名或 `is_locked=1`（v2）的人物不可被重新分群覆蓋 |
| `events` | 時間/地點聚類事件 | `is_user_edited=1` 的事件不可被重新聚類覆蓋 |
| `export_batches` / `export_logs` | 匯出批次與逐筆交易日誌 | write-ahead：先寫 `pending` 再動檔案 |
| `models` | 向量是哪個模型算的 | 防止換模型後誤用舊向量 |
| `scan_sessions` / `kv` | 掃描歷史、雜項設定 | |

向量 BLOB 一律是 `float32` little-endian、已 L2 正規化 —— 這也是 sqlite-vec 的原生格式，載入擴充後可以直接下 `vec_distance_cosine(vector, :q)`。

### 索引

| 索引 | 服務的查詢 |
|---|---|
| `photos.path` UNIQUE | Stage 0 依根目錄做前綴範圍載入（`path >= ? AND path < ?`） |
| `idx_photos_sha256` | Stage 0 以內容找搬移/重複 |
| `idx_photos_taken`、`idx_photos_event(event_id, taken_ts)` | 時間軸、事件內依時間排序 |
| `idx_photos_dir` | 資料夾瀏覽 |
| `idx_photos_dhash`（部分索引） | 近似重複 |
| `idx_photos_place(country_code, city)`（部分索引） | 地點瀏覽 |
| `idx_photos_gps_todo`、`_rejected`、`_shot`、`_missing`（部分索引） | 命中列很少的篩選，索引極小 |
| `idx_embeddings_model(model_id, photo_id)` | 依模型整批載入向量 |
| `idx_photo_tags_tag(tag_id, score DESC)` | 某標籤的照片依分數排序 |
| `idx_faces_photo`、`idx_faces_person`、`idx_faces_pending` | 照片的臉、人物的臉、待分群的臉 |
| `idx_export_logs_batch(batch_id, seq)`、`_open` | 回復時倒序走訪、找未完成的操作 |

`process_status` 沒有建索引：位元遮罩查詢用不到 B-tree，而且這張表沒有 BLOB，十萬列的全表掃描實測約 9 ms。

### PRAGMA

| PRAGMA | 值 | 原因 |
|---|---|---|
| `journal_mode` | `WAL` | 讀寫互不阻塞 |
| `synchronous` | `NORMAL` | WAL 下不會損毀資料庫，寫入快數倍；斷電最多丟最後幾個交易（重掃即可補回） |
| `foreign_keys` | `ON` | 每個連線都要開；刪照片時向量、標籤、人臉一併清除 |
| `busy_timeout` | 5000 ms | 另一個行程佔著鎖時等待而不是立刻失敗 |
| `cache_size` | −65536（64 MB） | 每個連線的 page cache |
| `mmap_size` | 256 MB | 讀取走記憶體映射 |
| `temp_store` | `MEMORY` | 排序與暫存表不落地 |
| `wal_autocheckpoint` | 1000 頁 | WAL 檔不會無限長大 |
| `journal_size_limit` | 64 MB | checkpoint 後截短 WAL |
| 關閉時 | `PRAGMA optimize`、`wal_checkpoint(TRUNCATE)` | 更新統計資訊、收乾淨 WAL |

資料庫必須放在本機磁碟：網路磁碟不支援 WAL 需要的共享記憶體。

## 5. 向量檢索

預設是 `InMemoryVectorIndex`：把 `embeddings` 整批載入成一個 NumPy 矩陣，查詢是一次矩陣乘法加 `argpartition`。精確檢索，不需要近似索引。

十萬張規模的實測數字（A 電腦，14 核心，合成資料）：

| 操作 | 耗時 |
|---|---|
| 向量檢索 top-50（100k × 768，矩陣 293 MB） | 7 ms |
| 從資料庫 BLOB 載入 10 萬筆向量建立索引 | 1.3 s |
| Stage 0 載入 10 萬張的檔案指紋 | 0.3 s |
| `photos` 全表掃描（位元遮罩條件） | 9 ms |
| 事件聚類 10 萬張 | 0.1 s |
| 資料庫檔案大小（10 萬張含向量，不含人臉） | 424 MB |

以圖搜圖用影像向量當 query；以文搜圖把文字送進 `SiglipTextEncoder` 得到向量再查同一個索引（影像與文字在同一個向量空間）。

之後要換 sqlite-vec 或 FAISS，實作 `core.interfaces.VectorIndex` 即可。

## 6. 驗證狀態

| 項目 | 驗證方式 |
|---|---|
| DBManager、Repository | `tests/test_db_manager.py`：併發寫入、savepoint 隔離、讀寫不互擋、關閉競速 |
| 管線調度、續掃、暫停、背壓、錯誤隔離、Stage 0 搬移/重複判定 | `tests/test_pipeline_engine.py`：假模型 + 真實的執行緒、佇列、資料庫 |
| Session 建立與 provider 降級、batch 框架、SigLIP 編碼器的程式路徑 | `tests/test_onnx_runtime.py`：測試當場產生的極小 ONNX 模型，在 A 電腦以真的 DirectML session 執行 |
| 解碼器（方向、DCT 縮放、HEIC、PNG/WebP/灰階/16-bit）、離線地理編碼 | 單元測試 + A 電腦以實際影像檔手動驗證 |
| ExportManager | `tests/test_export_manager.py`：四種模式、衝突、不覆寫、回復、當機收尾（symlink 在 A 電腦因無權限而略過） |
| **SigLIP 對真實權重的數值正確性** | **未驗證**（A 電腦沒有權重） |
| 人臉分群、事件分群、連拍（`utils/clustering.py`） | `tests/test_stage2_clustering_export.py`：合成的 512 維特徵與時間軸；保護已命名/鎖定/手動指派、增量歸入、重跑冪等。**參數未以真實照片校準** |
| 四層多視圖歸檔（`core/export_manager.py`） | 同上測試檔：真實檔案操作；COPY / MOVE / HARDLINK 匯出後回復，照片庫逐位元組相同；Sidecar 綁定、撞名、冪等、群組失敗還原。跨磁碟機只以模擬測試 |
| 桌面介面（`ui/`、`db/queries_ui.py`） | `tests/test_stage3_gui_smoke.py`：真的建立視窗並驅動 —— 導航樹計數、每個節點、照片明細、人物命名鎖定、匯出預覽/執行/回復、重新分群、沒有權重時掃描、掃描中關閉視窗後續掃、虛擬滾動與 LRU。另以截圖人工檢視 sv-ttk 與 clam 深色兩種主題的版面。**沒有用真實照片庫與真實模型操作過** |
| schema v1 → v2 遷移 | 同上測試檔：把資料庫降回 v1 再開啟，欄位補上且資料不變 |
| **InsightFace 輸出解碼、人臉對齊** | **未實作** |
