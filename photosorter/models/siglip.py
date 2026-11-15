"""SigLIP：影像/文字編碼與多標籤判定（Stage 2）。

運作方式
--------
  * **啟動時**：把每個標籤的英文 prompt 送進 text encoder，得到 768 維文字向量，寫回
    `tags.text_vector` 快取。之後只要 prompt 與模型沒變就不會再算，text encoder 用完即釋放
    （它佔幾百 MB，掃描期間用不到）。
  * **掃描時**：每張照片只跑 vision encoder 一次得到 768 維影像向量，然後
    `prob = sigmoid(logit_scale * (img · txt) + logit_bias)`，對每個標籤**各自獨立**判定
    （不是 softmax，分數不互相競爭，一張照片可以同時是「海邊」「日落」「多人合照」）。
  * 影像向量會存進 `embeddings`，所以之後新增或修改標籤**不需要重跑 vision encoder**，
    用 `rescore_library()` 對全庫做一次矩陣乘法即可。

實作狀態（交接時請看 HANDOFF.md §4.1）
--------------------------------------
本檔的程式流程是完整的，也用測試當場產生的極小 ONNX 模型走過整條程式路徑
（`tests/test_onnx_runtime.py`）。但 **A 電腦沒有 SigLIP 權重，數值正確性從未對真實模型驗證**。
接手後第一件事是照 HANDOFF.md §4.1，用 HuggingFace 參考實作比對影像向量、文字向量與最終機率。
最容易出錯的地方：節點名稱、tokenizer 的處理、`logit_scale` / `logit_bias`。
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..core.config import AppConfig
from ..core.types import GateRole, ModelInfo, TagHit, TagSpec
from ..db.repository import Repository
from .onnx_runtime import ModelLoadError, OnnxModel
from .preprocess import SIGLIP_SPEC, PreprocessSpec, l2_normalize, sigmoid, to_model_input

log = logging.getLogger(__name__)


class SiglipVisionEncoder:
    """`core.interfaces.VisionEmbedder` 的 SigLIP 實作。"""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        s = cfg.siglip
        self._model = OnnxModel(cfg.weight(s.vision.file), s.vision, cfg.runtime)
        self._spec = PreprocessSpec(mean=tuple(s.mean), std=tuple(s.std), scale=SIGLIP_SPEC.scale)  # type: ignore[arg-type]
        self._batch_size = 0

    @property
    def model_name(self) -> str:
        return self._cfg.siglip.model_name

    @property
    def dim(self) -> int:
        return self._cfg.siglip.embed_dim

    def load(self) -> list[ModelInfo]:
        size = self._cfg.decode.siglip_size
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        handle = self._model.open(dummy)
        out = self._model.run_raw(dummy)[0]
        if out.ndim != 2 or out.shape[1] != self.dim:
            raise ModelLoadError(
                f"{handle.model_path.name} 的輸出形狀是 {out.shape}，預期 (N, {self.dim})。"
                "請確認 siglip.vision.output_name 指到 pooled 向量（不是 last_hidden_state）"
            )
        return [ModelInfo(self.model_name, "image_embed", self.dim, handle.provider, str(handle.model_path))]

    def set_batch_size(self, batch_size: int) -> None:
        """由引擎在決定 RuntimePlan 後呼叫；DirectML 會把不足的 batch 補滿以固定輸入形狀。"""
        self._batch_size = batch_size

    def embed_images(self, batch: np.ndarray) -> np.ndarray:
        x = to_model_input(batch, self._spec)
        return l2_normalize(self._model.run_batched(x, self._batch_size))

    def close(self) -> None:
        self._model.close()


class SiglipTextEncoder:
    """`core.interfaces.TextEmbedder` 的 SigLIP 實作。只在標籤的文字向量需要（重新）計算時載入。"""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        s = cfg.siglip
        self._model = OnnxModel(cfg.weight(s.text.file), s.text, cfg.runtime)
        self._tokenizer = None

    def load(self) -> list[ModelInfo]:
        s = self._cfg.siglip
        tok_path = self._cfg.weight(s.tokenizer_file)
        if not tok_path.is_file():
            raise ModelLoadError(f"找不到 tokenizer：{tok_path}（請依 weights/README.md 下載）")
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        # SigLIP 訓練時一律 padding 到固定長度 64 且沒有 attention mask，
        # 推論若改成「補到這批最長」會得到不同的向量。
        tok.enable_truncation(max_length=s.max_text_len)
        tok.enable_padding(
            length=s.max_text_len, pad_id=s.pad_token_id,
            pad_token=tok.id_to_token(s.pad_token_id) or "</s>",
        )
        self._tokenizer = tok
        dummy = np.full((1, s.max_text_len), s.pad_token_id, dtype=np.int64)
        handle = self._model.open(dummy)
        return [ModelInfo(s.model_name, "text_embed", s.embed_dim, handle.provider, str(handle.model_path))]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self._tokenizer is None:
            raise RuntimeError("SiglipTextEncoder 尚未 load()")
        # SigLIP 的 tokenizer 以小寫訓練。標點的處理由 tokenizer.json 的 normalizer 決定。
        enc = self._tokenizer.encode_batch([t.strip().lower() for t in texts])
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        out = self._model.run_batched(ids, batch_size=16)
        if out.ndim != 2 or out.shape[1] != self._cfg.siglip.embed_dim:
            raise ModelLoadError(f"text encoder 的輸出形狀是 {out.shape}，預期 (N, {self._cfg.siglip.embed_dim})")
        return l2_normalize(out)

    def close(self) -> None:
        self._model.close()
        self._tokenizer = None


def siglip_probabilities(image_emb: np.ndarray, text_emb: np.ndarray, logit_scale: float,
                         logit_bias: float) -> np.ndarray:
    """SigLIP 的配對機率。

    Args:
        image_emb: (N, D) L2 正規化。
        text_emb: (T, D) L2 正規化。
        logit_scale: 已取過 exp 的值（HF checkpoint 存的是 log，要先 `exp()`）。
    Returns:
        (N, T) float32，每格是「這張圖符合這段文字」的獨立機率。
    """
    return sigmoid(logit_scale * (image_emb @ text_emb.T) + logit_bias)


class SiglipTagger:
    """`core.interfaces.TagScorer`：持有標籤的文字向量矩陣，對影像向量做多標籤判定。"""

    def __init__(self, cfg: AppConfig, repo: Repository, text_encoder: SiglipTextEncoder | None = None) -> None:
        self._cfg = cfg
        self._repo = repo
        self._text_encoder = text_encoder
        self._tags: list[TagSpec] = []
        self._matrix = np.empty((0, cfg.siglip.embed_dim), dtype=np.float32)
        self._thresholds = np.empty(0, dtype=np.float32)
        self._roles: dict[int, GateRole] = {}

    @property
    def tags(self) -> list[TagSpec]:
        return list(self._tags)

    def load(self) -> list[ModelInfo]:
        """載入標籤；缺文字向量的才啟動 text encoder 補算並寫回資料庫，算完立刻釋放。"""
        s = self._cfg.siglip
        text_model_id = self._repo.ensure_model(s.model_name, "text_embed", s.embed_dim)
        tags = self._repo.load_auto_tags(text_model_id, s.default_threshold, s.embed_dim)
        missing = [t for t in tags if t.text_vector is None]
        infos: list[ModelInfo] = []
        if missing:
            encoder = self._text_encoder or SiglipTextEncoder(self._cfg)
            infos = encoder.load()
            try:
                vectors = encoder.embed_texts([t.prompt for t in missing])
            finally:
                encoder.close()
            for t, v in zip(missing, vectors):
                t.text_vector = v
            self._repo.save_text_vectors(text_model_id, {t.id: t.text_vector for t in missing})  # type: ignore[misc]
            log.info("已編碼 %d 個標籤的文字向量", len(missing))
        self._set_tags(tags)
        return infos

    def _set_tags(self, tags: list[TagSpec]) -> None:
        self._tags = tags
        self._roles = {t.id: t.gate_role for t in tags}
        if tags:
            self._matrix = np.stack([t.text_vector for t in tags]).astype(np.float32)  # type: ignore[misc]
            self._thresholds = np.asarray([t.threshold for t in tags], dtype=np.float32)
        else:
            self._matrix = np.empty((0, self._cfg.siglip.embed_dim), dtype=np.float32)
            self._thresholds = np.empty(0, dtype=np.float32)

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, D) -> (N, T) 機率矩陣，欄順序同 `self.tags`。給門檻校準工具與 UI 用。"""
        s = self._cfg.siglip
        return siglip_probabilities(embeddings, self._matrix, s.logit_scale, s.logit_bias)

    def score(self, embeddings: np.ndarray) -> list[list[TagHit]]:
        n = embeddings.shape[0]
        if not self._tags:
            return [[] for _ in range(n)]
        probs = self.probabilities(embeddings)
        hit = probs >= self._thresholds[None, :]
        out: list[list[TagHit]] = []
        for i in range(n):
            cols = np.flatnonzero(hit[i])
            out.append([TagHit(self._tags[j].id, float(probs[i, j])) for j in cols])
        return out

    def gate_role(self, tag_id: int) -> GateRole:
        return self._roles.get(tag_id, GateRole.NONE)

    def close(self) -> None:
        pass


def rescore_library(tagger: SiglipTagger, repo: Repository, chunk: int = 4096) -> int:
    """標籤定義改變後的全庫重標：只用已存的影像向量，不重跑 vision encoder。

    呼叫前 `tagger.load()` 必須已完成（文字向量已是最新）。回傳重標的照片數。
    使用者手動加的標籤與否決過的標籤不受影響（見 `repository._replace_auto_tags`）。
    """
    s = tagger._cfg.siglip
    image_model_id = repo.ensure_model(s.model_name, "image_embed", s.embed_dim)
    ids, matrix = repo.load_embedding_matrix(image_model_id, s.embed_dim)
    for start in range(0, len(ids), chunk):
        hits = tagger.score(matrix[start : start + chunk])
        repo.replace_auto_tags_bulk({
            int(pid): [(h.tag_id, h.score) for h in hs]
            for pid, hs in zip(ids[start : start + chunk], hits)
        })
    return int(len(ids))


#: 第一次建立資料庫時植入的預設標籤：(name, 中文顯示名稱, 英文 prompt 的主體, gate_role)。
#: prompt 會套用 `siglip.prompt_template`；主體以 "=" 開頭代表已是完整句子，不套模板。
DEFAULT_TAGS: list[tuple[str, str, str, GateRole]] = [
    # ---- 人物語意：命中就一定跑人臉 ----
    ("people", "人物", "people", GateRole.HAS_PERSON),
    ("portrait", "人像", "=a portrait photo of a person.", GateRole.HAS_PERSON),
    ("selfie", "自拍", "=a selfie.", GateRole.HAS_PERSON),
    ("group_photo", "合照", "=a group photo of several people.", GateRole.HAS_PERSON),
    ("baby", "嬰幼兒", "a baby", GateRole.HAS_PERSON),
    # ---- 非人物：命中且沒有人物語意就略過人臉 ----
    ("document", "文件", "=a photo of a printed document or a page of text.", GateRole.SKIP_FACE),
    ("receipt", "收據發票", "=a photo of a receipt or an invoice.", GateRole.SKIP_FACE),
    ("screenshot", "螢幕截圖", "=a screenshot of a phone or computer screen.", GateRole.SKIP_FACE),
    ("whiteboard", "白板筆記", "=a photo of a whiteboard with handwriting.", GateRole.SKIP_FACE),
    # ---- 一般分類 ----
    ("food", "美食", "food", GateRole.NONE),
    ("landscape", "風景", "a natural landscape", GateRole.NONE),
    ("beach", "海邊", "a beach", GateRole.NONE),
    ("mountain", "山", "mountains", GateRole.NONE),
    ("city", "城市街景", "a city street", GateRole.NONE),
    ("building", "建築", "a building", GateRole.NONE),
    ("night", "夜景", "=a photo taken at night.", GateRole.NONE),
    ("sunset", "日出日落", "a sunset", GateRole.NONE),
    ("flower", "花卉植物", "flowers", GateRole.NONE),
    ("cat", "貓", "a cat", GateRole.NONE),
    ("dog", "狗", "a dog", GateRole.NONE),
    ("car", "車輛", "a car", GateRole.NONE),
    ("indoor", "室內", "=a photo taken indoors.", GateRole.NONE),
]


def seed_default_tags(repo: Repository, prompt_template: str) -> int:
    """標籤表是空的才植入預設標籤（不會蓋掉使用者自己建立或修改過的）。回傳植入的數量。"""
    if repo.count_tags() > 0:
        return 0
    for name, display, body, role in DEFAULT_TAGS:
        prompt = body[1:] if body.startswith("=") else prompt_template.format(body)
        repo.upsert_tag(name, prompt, display_name=display, gate_role=role)
    return len(DEFAULT_TAGS)
