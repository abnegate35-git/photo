# 模型權重

這個資料夾的內容**不進版控**（`.gitignore` 只放行本檔與 `.gitkeep`）。B 電腦 clone 之後要自己把權重放進來。

放好後執行 `python -m photosorter doctor --load-models`，它會列出每個檔案是否存在、實際生效的 provider，以及模型真正的輸入輸出節點名稱與形狀。

## 需要的檔案

```
weights/
├── siglip/
│   ├── vision_model.onnx      SigLIP ViT-B/16 224 的 vision encoder
│   ├── text_model.onnx        同一個 checkpoint 的 text encoder
│   └── tokenizer.json         HuggingFace `tokenizers` 格式
└── buffalo_l/
    ├── det_10g.onnx           SCRFD-10GF 人臉偵測（含 5 點關鍵點）
    └── w600k_r50.onnx         ArcFace ResNet-50 人臉特徵
```

路徑與檔名可以在 `config.json` 改（`siglip.vision.file`、`face.detector.file`…，都相對於 `paths.weights_dir`）。

## 來源

> **以下來源在 A 電腦沒有實際下載驗證過**（A 電腦只負責架構）。它們是依模型的公開發布位置寫的，B 電腦下載後請以 `doctor --load-models` 的輸出為準。

**SigLIP**：模型是 `google/siglip-base-patch16-224`。需要 ONNX 匯出版本，兩種取得方式：

- 直接下載別人匯出好的：HuggingFace 上的 `Xenova/siglip-base-patch16-224`（`onnx/vision_model.onnx`、`onnx/text_model.onnx`、`tokenizer.json`）。
- 自己匯出：在另一個有 PyTorch 的環境執行 `optimum-cli export onnx --model google/siglip-base-patch16-224 <輸出資料夾>`。

**buffalo_l**：InsightFace 官方 release 的 `buffalo_l.zip`（`https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`），解壓後取 `det_10g.onnx` 與 `w600k_r50.onnx`，其餘三個檔（2D/3D 關鍵點、性別年齡）用不到。

**授權**：buffalo_l 的預訓練權重僅限非商業研究用途。個人整理自己的照片沒有問題，但不可以把權重連同工具散布或商用。

## 換模型或換匯出版本時必須核對的設定

| 設定 | 預設值 | 怎麼確認 |
|---|---|---|
| `siglip.vision.input_name` / `output_name` | `pixel_values` / `pooler_output` | `doctor --load-models` 會印出實際名稱。輸出必須是 `(N, 768)` 的 pooled 向量，不是 `last_hidden_state` |
| `siglip.text.input_name` / `output_name` | `input_ids` / `pooler_output` | 同上 |
| `siglip.logit_scale` | `117.3308` | HF checkpoint 的 `model.logit_scale.exp().item()`（存的是 log，要取 exp） |
| `siglip.logit_bias` | `-12.9324` | `model.logit_bias.item()` |
| `siglip.max_text_len` / `pad_token_id` | `64` / `1` | `tokenizer.model_max_length`、`tokenizer.pad_token_id` |
| `siglip.mean` / `std` | `0.5` / `0.5` | `preprocessor_config.json` 的 `image_mean` / `image_std` |
| `face.detector.fixed_batch` | `1` | 模型輸入第 0 維是整數就代表寫死；`OnnxModel` 也會自動偵測 |

`logit_scale` 與 `logit_bias` 是模型權重的一部分，**不同 checkpoint 的值不同**（例如 SigLIP 2 或其他解析度的版本）。這兩個值錯了程式不會報錯，只會讓所有標籤的機率整體偏高或偏低。預設值是依 `siglip-base-patch16-224` 填的，B 電腦請用上表的方法從實際的 checkpoint 讀出來核對一次。

驗證步驟見 `HANDOFF.md` §4.1。
