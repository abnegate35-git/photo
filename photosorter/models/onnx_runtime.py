"""ONNX Runtime session 的建立與 provider 自動降級。全專案唯一 import onnxruntime 的地方
（`core.hardware` 只讀取可用清單）。

降級鏈：`CUDAExecutionProvider` -> `DmlExecutionProvider` -> `CPUExecutionProvider`

「provider 出現在 `get_available_providers()`」只代表這個 onnxruntime 發行版**編進了**它，
不代表真的能用。實際會遇到三種失敗，這裡全部處理：

  1. 建立 session 時直接丟例外（模型有該 provider 不支援的運算子、裝置編號錯誤…）。
  2. **靜默退回**：CUDA 的 DLL（cuDNN、cuBLAS…）缺少時，onnxruntime 只印一行警告就改用
     CPU，session 照樣建立成功。所以建立後一定要看 `session.get_providers()[0]`
     是不是我們要的那個；不是就視為失敗，繼續試鏈上的下一個（例如 DirectML）。
  3. 建立成功但第一次 `run()` 才失敗（DirectML 在某些驅動上的行為）。所以建立後用假資料
     跑一次暖機，失敗就降級。暖機同時把第一次推論的初始化成本移到載入階段。

硬體探測是延遲的：這個模組在 import 時不會碰 onnxruntime，要到 `create_session()` 才載入。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.config import OnnxIOConfig, RuntimeConfig
from ..core.hardware import CPU, CUDA, DML, plan_runtime, probe_hardware, provider_candidates

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """權重不存在，或鏈上所有 provider 都無法建立可用的 session。"""


@dataclass(slots=True)
class SessionHandle:
    session: Any                               # onnxruntime.InferenceSession
    provider: str                              # 實際生效的 provider
    model_path: Path
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, list[Any]]         # 動態維度是字串或 None
    load_ms: float = 0.0
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (provider, 被淘汰的原因)


WarmupFn = Callable[[Any], None]


def _session_options(ort: Any, provider: str, intra_threads: int) -> Any:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, intra_threads)
    so.inter_op_num_threads = 1
    so.log_severity_level = 3                  # 只留 error；降級原因由我們自己記錄
    if provider == DML:
        # DirectML 的硬性要求：不可用 memory pattern，必須循序執行。
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def _provider_options(provider: str, device_id: int) -> dict[str, Any]:
    if provider == CUDA:
        return {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",   # 不要以 2 的次方預佔視訊記憶體
            "cudnn_conv_algo_search": "HEURISTIC",          # 避免每種輸入形狀都做一次耗時的窮舉搜尋
        }
    if provider == DML:
        return {"device_id": device_id}
    return {}


def create_session(model_path: Path, rt: RuntimeConfig, warmup: WarmupFn | None = None) -> SessionHandle:
    """建立 session，依降級鏈逐一嘗試，回傳第一個**真的能跑**的。

    Args:
        model_path: `.onnx` 檔的絕對路徑。
        rt: 執行期設定（provider 偏好、裝置編號、執行緒數）。
        warmup: `fn(session)`，用假資料跑一次推論；丟例外代表這個 provider 不能用。
            `rt.warmup` 為 False 時不會呼叫。

    Raises:
        ModelLoadError: 檔案不存在、onnxruntime 無法載入，或所有 provider 都失敗
            （訊息裡列出每個 provider 的失敗原因）。
    """
    if not model_path.is_file():
        raise ModelLoadError(f"找不到模型權重：{model_path}（請依 weights/README.md 下載）")
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        raise ModelLoadError(f"onnxruntime 無法載入：{e}") from e

    hw = probe_hardware()
    chain = provider_candidates(rt.providers, hw.available_providers)
    if not chain:
        raise ModelLoadError(
            f"沒有可用的 execution provider（偏好={rt.providers!r}，可用={hw.available_providers}）"
        )

    rejected: list[tuple[str, str]] = []
    for provider in chain:
        intra = plan_runtime(rt, hw, provider).intra_op_threads
        t0 = time.perf_counter()
        try:
            providers: list[Any] = [(provider, _provider_options(provider, rt.device_id))]
            if provider != CPU:
                providers.append(CPU)          # 個別不支援的運算子可以退回 CPU 執行
            session = ort.InferenceSession(
                str(model_path), sess_options=_session_options(ort, provider, intra), providers=providers
            )
            active = session.get_providers()
            if not active or active[0] != provider:
                raise RuntimeError(f"onnxruntime 靜默退回 {active}（驅動或相依的 DLL 不齊全）")
            if warmup is not None and rt.warmup:
                warmup(session)
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}".splitlines()[0][:300]
            log.warning("%s 無法用 %s：%s", model_path.name, provider, reason)
            rejected.append((provider, reason))
            continue

        handle = SessionHandle(
            session=session,
            provider=provider,
            model_path=model_path,
            input_names=[i.name for i in session.get_inputs()],
            output_names=[o.name for o in session.get_outputs()],
            input_shapes={i.name: list(i.shape) for i in session.get_inputs()},
            load_ms=(time.perf_counter() - t0) * 1000.0,
            rejected=rejected,
        )
        log.info("%s 已載入：provider=%s，%.0f ms", model_path.name, provider, handle.load_ms)
        return handle

    detail = "; ".join(f"{p}: {r}" for p, r in rejected)
    raise ModelLoadError(f"{model_path.name} 在所有 provider 上都載入失敗 -> {detail}")


class OnnxModel:
    """單輸入、取單一輸出的 ONNX 模型包裝。具體模型（SigLIP、SCRFD、ArcFace）繼承它。

    子類別負責：前處理（uint8 影像 -> 模型輸入張量）與後處理（輸出張量 -> 業務結果）。
    這裡負責：session 生命週期、節點名稱解析、固定 batch 模型的切塊與補齊。

    只會在推論執行緒上使用，不需要加鎖。
    """

    def __init__(self, model_path: Path, io: OnnxIOConfig, rt: RuntimeConfig) -> None:
        self.model_path = model_path
        self.io = io
        self.rt = rt
        self.handle: SessionHandle | None = None
        self._input_name = ""
        self._output_names: list[str] | None = None
        self._fixed_batch = 0

    # ---- 生命週期 ----
    def open(self, dummy_input: np.ndarray | None = None) -> SessionHandle:
        """建立 session。`dummy_input` 是暖機用的假輸入（形狀、dtype 與真實輸入相同）。"""
        if self.handle is not None:
            return self.handle

        def warmup(session: Any) -> None:
            if dummy_input is None:
                return
            name = self.io.input_name or session.get_inputs()[0].name
            session.run(None, {name: dummy_input})

        handle = create_session(self.model_path, self.rt, warmup if dummy_input is not None else None)
        self._input_name = self._resolve(self.io.input_name, handle.input_names, "輸入")
        out = self.io.output_name
        self._output_names = [self._resolve(out, handle.output_names, "輸出")] if out else None
        # 模型的 batch 維度若寫死（例如 buffalo_l 的 det_10g 是 1），就算設定檔沒寫也自動偵測
        first_dim = handle.input_shapes.get(self._input_name, [None])[0]
        self._fixed_batch = self.io.fixed_batch or (first_dim if isinstance(first_dim, int) else 0)
        self.handle = handle
        return handle

    def close(self) -> None:
        self.handle = None                     # InferenceSession 沒有 close()，放掉參考即釋放

    @property
    def provider(self) -> str:
        return self.handle.provider if self.handle else ""

    def _resolve(self, wanted: str, available: list[str], what: str) -> str:
        if not wanted:
            return available[0]
        if wanted not in available:
            raise ModelLoadError(
                f"{self.model_path.name} 沒有名為 {wanted!r} 的{what}節點（實際：{available}）；"
                "請修改 config.json 對應的 input_name / output_name"
            )
        return wanted

    # ---- 推論 ----
    def run_raw(self, x: np.ndarray) -> list[np.ndarray]:
        """原樣送進 session。回傳設定指定的那個輸出（清單長度 1），或未指定時的全部輸出。"""
        if self.handle is None:
            raise RuntimeError(f"{self.model_path.name} 尚未 open()")
        return self.handle.session.run(self._output_names, {self._input_name: x})

    def run_batched(self, x: np.ndarray, batch_size: int = 0, pad_partial: bool | None = None) -> np.ndarray:
        """對 (N, …) 的輸入分塊推論，回傳第一個輸出沿 batch 維度串接的結果 (N, …)。

        Args:
            batch_size: 每塊大小；0 = 整批一次送。模型的 batch 維度寫死時一律以模型為準。
            pad_partial: 最後一塊不足時是否補零補到滿。None = 自動（固定 batch 模型，或 DirectML）。
                DirectML 每遇到一種新的輸入形狀就要重新編譯計算圖，形狀固定才有穩定的速度。
        """
        n = x.shape[0]
        if n == 0:
            raise ValueError("空的 batch")
        size = self._fixed_batch or batch_size or n
        if pad_partial is None:
            pad_partial = bool(self._fixed_batch) or (self.provider == DML and batch_size > 0)
        outs: list[np.ndarray] = []
        for i in range(0, n, size):
            chunk = x[i : i + size]
            real = chunk.shape[0]
            if real < size and pad_partial:
                pad = np.zeros((size - real, *chunk.shape[1:]), dtype=chunk.dtype)
                chunk = np.concatenate([chunk, pad], axis=0)
            outs.append(self.run_raw(np.ascontiguousarray(chunk))[0][:real])
        return outs[0] if len(outs) == 1 else np.concatenate(outs, axis=0)
