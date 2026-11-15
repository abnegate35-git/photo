"""用真的 onnxruntime session 驗證 provider 降級與 batch 框架。

模型是測試當場用 `onnx` 套件產生的極小計算圖（不需要下載任何權重），但走的是與正式模型
完全相同的程式路徑：`create_session` → `OnnxModel` → `SiglipVisionEncoder` / `SiglipTextEncoder`。
這台電腦有哪個 provider 就實際用哪個跑（DirectML 或 CPU）。

需要 `onnx`（列在 requirements-dev.txt）；沒裝時整組略過。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime  # noqa: F401
    from onnx import TensorProto, helper, numpy_helper
except Exception:  # noqa: BLE001
    onnx = None

from photosorter.core.config import OnnxIOConfig
from photosorter.core.hardware import CPU, probe_hardware, provider_candidates
from photosorter.models.onnx_runtime import ModelLoadError, OnnxModel, create_session
from photosorter.models.preprocess import l2_normalize

from .fakes import make_config

DIM = 6


def _save(graph, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def make_vision_model(path: Path, size: int, batch: object = "N") -> tuple[Path, np.ndarray]:
    """(N, 3, S, S) --空間平均--> (N, 3) --x W--> (N, DIM)。回傳 (路徑, W)。"""
    w = np.random.default_rng(0).normal(size=(3, DIM)).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["pooled", "W"], ["pooler_output"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_vision",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [batch, 3, size, size])],
        [helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, [batch, DIM])],
        [numpy_helper.from_array(w, "W")],
    )
    return _save(graph, path), w


def make_text_model(path: Path, seq_len: int) -> tuple[Path, np.ndarray]:
    """(N, L) int64 token id --Cast--> float --x W--> (N, DIM)。"""
    w = np.random.default_rng(1).normal(size=(seq_len, DIM)).astype(np.float32)
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["ids_f", "W"], ["pooler_output"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_text",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["N", seq_len])],
        [helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["N", DIM])],
        [numpy_helper.from_array(w, "W")],
    )
    return _save(graph, path), w


@unittest.skipIf(onnx is None, "需要 onnx 與 onnxruntime")
class OnnxRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = make_config(self.root, siglip={"embed_dim": DIM}, decode={"siglip_size": 8})
        self.rt = self.cfg.runtime
        self.model_path, self.w = make_vision_model(self.cfg.weight("siglip/vision_model.onnx"), 8)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _expected(self, x: np.ndarray) -> np.ndarray:
        return x.mean(axis=(2, 3)) @ self.w

    # ------------------------------------------------------------------ #
    def test_auto_picks_first_working_provider_in_the_chain(self) -> None:
        hw = probe_hardware()
        chain = provider_candidates("auto", hw.available_providers)
        self.assertEqual(chain[-1], CPU)
        handle = create_session(self.model_path, self.rt)
        self.assertIn(handle.provider, chain)
        # 排在生效 provider 之前的每一個都必須留下被淘汰的原因
        self.assertEqual([p for p, _ in handle.rejected], chain[: chain.index(handle.provider)])
        self.assertEqual(handle.session.get_providers()[0], handle.provider)
        self.assertEqual((handle.input_names, handle.output_names), (["pixel_values"], ["pooler_output"]))
        self.assertEqual(handle.input_shapes["pixel_values"][1:], [3, 8, 8])
        x = np.random.default_rng(3).normal(size=(2, 3, 8, 8)).astype(np.float32)
        out = handle.session.run(None, {"pixel_values": x})[0]
        np.testing.assert_allclose(out, self._expected(x), rtol=1e-3, atol=1e-4)

    def test_explicit_cpu_and_unavailable_preference(self) -> None:
        self.rt.providers = [CPU]
        self.assertEqual(create_session(self.model_path, self.rt).provider, CPU)
        self.rt.providers = ["NoSuchExecutionProvider"]
        self.assertEqual(create_session(self.model_path, self.rt).provider, CPU)

    def test_warmup_failure_falls_back_down_the_chain(self) -> None:
        """模擬「session 建得起來、第一次推論才失敗」的 provider：必須降級而不是把錯誤往上丟。"""
        attempts: list[str] = []

        def flaky_warmup(session) -> None:
            provider = session.get_providers()[0]
            attempts.append(provider)
            if provider != CPU:
                raise RuntimeError("simulated driver failure")

        handle = create_session(self.model_path, self.rt, warmup=flaky_warmup)
        self.assertEqual(handle.provider, CPU)
        self.assertEqual(attempts[-1], CPU)
        self.assertEqual([p for p, _ in handle.rejected], attempts[:-1])
        for _provider, reason in handle.rejected:
            self.assertIn("simulated driver failure", reason)

    def test_all_providers_failing_raises_with_every_reason(self) -> None:
        def always_fail(session) -> None:
            raise RuntimeError("nope")

        with self.assertRaises(ModelLoadError) as ctx:
            create_session(self.model_path, self.rt, warmup=always_fail)
        self.assertIn(CPU, str(ctx.exception))

    def test_missing_weights_and_wrong_node_names_give_actionable_errors(self) -> None:
        with self.assertRaises(ModelLoadError) as ctx:
            create_session(self.root / "weights" / "nope.onnx", self.rt)
        self.assertIn("nope.onnx", str(ctx.exception))
        model = OnnxModel(self.model_path, OnnxIOConfig(input_name="pixel_values", output_name="image_embeds"), self.rt)
        with self.assertRaises(ModelLoadError) as ctx:
            model.open()
        self.assertIn("pooler_output", str(ctx.exception))       # 訊息列出實際可用的節點名稱

    # ------------------------------------------------------------------ #
    def test_run_batched_chunks_and_pads(self) -> None:
        model = OnnxModel(self.model_path, OnnxIOConfig(output_name="pooler_output"), self.rt)
        model.open(np.zeros((1, 3, 8, 8), dtype=np.float32))
        x = np.random.default_rng(4).normal(size=(7, 3, 8, 8)).astype(np.float32)
        expected = self._expected(x)
        for kwargs in ({}, {"batch_size": 3}, {"batch_size": 4, "pad_partial": True},
                       {"batch_size": 16, "pad_partial": True}, {"batch_size": 2, "pad_partial": False}):
            out = model.run_batched(x, **kwargs)
            self.assertEqual(out.shape, (7, DIM), kwargs)
            np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4, err_msg=str(kwargs))
        with self.assertRaises(RuntimeError):
            OnnxModel(self.model_path, OnnxIOConfig(), self.rt).run_raw(x)

    def test_fixed_batch_model_is_detected_and_fed_one_at_a_time(self) -> None:
        """buffalo_l 的 det_10g 就是這種：batch 維度寫死為 1，設定檔沒寫也要自動偵測。"""
        path, w = make_vision_model(self.root / "weights" / "fixed.onnx", 8, batch=1)
        model = OnnxModel(path, OnnxIOConfig(), self.rt)
        model.open(np.zeros((1, 3, 8, 8), dtype=np.float32))
        self.assertEqual(model._fixed_batch, 1)
        x = np.random.default_rng(5).normal(size=(5, 3, 8, 8)).astype(np.float32)
        np.testing.assert_allclose(model.run_batched(x, batch_size=4), x.mean(axis=(2, 3)) @ w, rtol=1e-3, atol=1e-4)

    # ------------------------------------------------------------------ #
    def test_siglip_vision_encoder_end_to_end(self) -> None:
        from photosorter.models.siglip import SiglipVisionEncoder

        enc = SiglipVisionEncoder(self.cfg)
        infos = enc.load()
        self.assertEqual((infos[0].kind, infos[0].dim), ("image_embed", DIM))
        enc.set_batch_size(4)
        batch = np.random.default_rng(6).integers(0, 255, size=(5, 8, 8, 3), dtype=np.uint8)
        out = enc.embed_images(batch)
        self.assertEqual((out.shape, out.dtype), ((5, DIM), np.float32))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        x = (batch.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - 0.5) / 0.5
        np.testing.assert_allclose(out, l2_normalize(self._expected(x)), rtol=1e-3, atol=1e-4)
        enc.close()

    def test_siglip_vision_encoder_rejects_wrong_output_dim(self) -> None:
        from photosorter.models.siglip import SiglipVisionEncoder

        self.cfg.siglip.embed_dim = DIM + 1
        with self.assertRaises(ModelLoadError) as ctx:
            SiglipVisionEncoder(self.cfg).load()
        self.assertIn(str(DIM + 1), str(ctx.exception))

    def test_siglip_text_encoder_pads_to_fixed_length(self) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers

        from photosorter.models.siglip import SiglipTextEncoder

        seq = self.cfg.siglip.max_text_len
        vocab = {"<unk>": 0, "</s>": 1, "a": 2, "photo": 3, "of": 4, "cat": 5, "dog": 6}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok_path = self.cfg.weight(self.cfg.siglip.tokenizer_file)
        tok.save(str(tok_path))
        _path, w = make_text_model(self.cfg.weight(self.cfg.siglip.text.file), seq)

        enc = SiglipTextEncoder(self.cfg)
        enc.load()
        out = enc.embed_texts(["A photo of CAT", "a photo of a dog dog dog"])
        ids = np.full((2, seq), 1, dtype=np.float32)             # 其餘位置一律是 pad id 1
        ids[0, :4] = [2, 3, 4, 5]                                # 大寫會被轉小寫
        ids[1, :7] = [2, 3, 4, 2, 6, 6, 6]
        np.testing.assert_allclose(out, l2_normalize(ids @ w), rtol=1e-3, atol=1e-4)
        enc.close()


if __name__ == "__main__":
    unittest.main()
