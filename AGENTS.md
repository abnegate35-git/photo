# AGENTS

本地離線照片分類工具：核心引擎、CLI 與 Tkinter 桌面介面。程式架構與三個階段的功能已在 A 電腦完成並以測試驗證，但**還沒有接上真實的模型權重與真實照片**；你是在 B 電腦接手實測與調校的人。

**開始之前先讀 [`HANDOFF.md`](HANDOFF.md)**（現況、不可違反的規則、待辦、GUI 接線方式）與 [`HANDOFF_TO_PC_B.md`](HANDOFF_TO_PC_B.md)（B 電腦的環境初始化、E2E 測試步驟、接下來的任務）。設計理由在 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 指令

```powershell
.venv\Scripts\python -m pytest                                   # 全部測試，不需要權重或顯示卡
.venv\Scripts\python -m photosorter doctor --load-models         # 這台電腦的硬體、provider、權重、節點名稱
.venv\Scripts\python -m photosorter                                 # 啟動桌面介面
.venv\Scripts\python -m photosorter scan <資料夾> --stages meta,siglip
.venv\Scripts\python -m photosorter archive <匯出根目錄> [--execute]   # 四層歸檔（預設只預覽）
.venv\Scripts\python -m photosorter verify-archive <匯出根目錄>        # 視圖是否為硬連結、實際佔用空間
.venv\Scripts\python -m photosorter dump-config -o config.example.json   # 改了預設設定之後
```

一律用專案自己的 `.venv`。不要把 PyTorch、transformers、insightface 裝進這個環境；需要它們產生參考答案時另開一個環境。

## 規則摘要（完整版在 HANDOFF.md §3）

- 路徑一律由 `AppConfig` 取得，不寫死、不依賴目前工作目錄。
- provider、執行緒數、batch 大小在執行時探測，不寫進版控的設定。
- 不自己開 SQLite 連線：讀用 `db.query*()`，寫用 `db.write()` 或 `Repository`。
- 資料與 `process_status` 位元寫在同一個交易；位元值不可更動。
- 模型只在 inference 執行緒上使用；核心不 import GUI；工作執行緒不碰任何 widget，GUI 只在主執行緒輪詢佇列。
- 自動流程不可覆蓋使用者的手動決定（手動標籤、否決的標籤、手動指派的人臉、編輯過的事件）。
- 只有 `ExportManager` 可以動使用者的照片檔，而且絕不覆寫。
- Tk 主視窗只能在主執行緒被回收：GUI 的背景工作一律走 `MainWindow._run_bg()`，測試裡的視窗在 `theme.releasing_tk()` 內銷毀。
- 沒有用真實權重或真實照片驗證過的東西，不要在文件或回報裡寫成已驗證。

## 目錄

`photosorter/core`（設定、型別、介面、事件）、`db`（SQLite 與 SQL）、`models`（ONNX）、`pipeline`（引擎、掃描、解碼、匯出）、`utils`、`ui`（GUI 放這裡）、`tests`（`fakes.py` 是各介面的最小範例）。
