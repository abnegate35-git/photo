"""硬體延遲探測（Lazy Hardware Probing）。

**import 這個模組不會探測任何東西**，也不會 import onnxruntime。只有在
`probe_hardware()` 第一次被呼叫時（也就是 B 電腦真正要啟動 pipeline 的那一刻）才探測，
結果快取在行程內。A、B 兩台電腦硬體不同也不需要改任何設定。
"""
from __future__ import annotations

import logging
import os
import threading

from .config import RuntimeConfig
from .types import HardwareInfo, RuntimePlan

log = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"

#: 自動模式的降級順序
PROVIDER_PRIORITY: tuple[str, ...] = (CUDA, DML, CPU)

_lock = threading.Lock()
_cached: HardwareInfo | None = None


def probe_hardware(refresh: bool = False) -> HardwareInfo:
    """探測 CPU、記憶體與 onnxruntime 可用的 provider。結果會快取。"""
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached

        logical = os.cpu_count() or 1
        physical = max(1, logical // 2)
        ram_gb: float | None = None
        try:
            import psutil

            physical = psutil.cpu_count(logical=False) or physical
            ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
        except Exception:  # noqa: BLE001 - psutil 是選配
            log.debug("psutil 不可用，以邏輯核心數的一半估計實體核心數")

        ort_version: str | None = None
        providers: list[str] = []
        try:
            import onnxruntime as ort

            ort_version = ort.__version__
            providers = list(ort.get_available_providers())
        except Exception as e:  # noqa: BLE001
            log.warning("onnxruntime 無法載入：%s", e)

        _cached = HardwareInfo(
            logical_cores=logical,
            physical_cores=physical,
            total_ram_gb=ram_gb,
            ort_version=ort_version,
            available_providers=providers,
        )
        return _cached


def provider_candidates(preference: object, available: list[str]) -> list[str]:
    """依使用者意圖與實際可用的 provider，排出要依序嘗試的清單。

    Args:
        preference: `"auto"`，或明確的 provider 名稱清單（照給定順序嘗試）。
        available: `onnxruntime.get_available_providers()` 的結果。

    清單結尾一定有 CPU，所以只要 onnxruntime 裝得起來就一定有得用。
    注意「available」只代表這個 onnxruntime 發行版編進了該 provider，不代表驅動/函式庫
    齊全 —— 真正能不能用要到建立 session 時才知道，見 `models.onnx_runtime.create_session`。
    """
    if preference == "auto" or not preference:
        wanted = list(PROVIDER_PRIORITY)
    elif isinstance(preference, str):
        wanted = [preference]
    else:
        wanted = [str(p) for p in preference]  # type: ignore[union-attr]
    out = [p for p in wanted if p in available]
    if CPU not in out and CPU in available:
        out.append(CPU)
    return out


def plan_runtime(rt: RuntimeConfig, hw: HardwareInfo, provider: str) -> RuntimePlan:
    """把設定裡的 `0` / `"auto"` 換成這台電腦上的實際數值。

    原則：
      * GPU provider：解碼是瓶頸，解碼執行緒盡量多；ORT 的 CPU 執行緒只服務少數退回 CPU 的運算子。
      * CPU provider：推論與解碼搶同一批核心，各分一半，避免超額訂閱。
    """
    phys = max(1, hw.physical_cores)
    on_gpu = provider in (CUDA, DML)

    decode_workers = rt.decode_workers
    if decode_workers <= 0:
        decode_workers = max(2, min(8, phys - 1)) if on_gpu else max(1, min(6, phys // 2))

    intra = rt.intra_op_threads
    if intra <= 0:
        intra = min(2, phys) if on_gpu else max(1, phys - decode_workers)

    batch = rt.batch_size
    if batch <= 0:
        batch = {CUDA: 16, DML: 8}.get(provider, 4)

    return RuntimePlan(
        provider=provider,
        decode_workers=decode_workers,
        intra_op_threads=intra,
        batch_size=batch,
        queue_maxsize=max(batch, rt.queue_maxsize),
        batch_timeout_s=max(0.0, rt.batch_timeout_ms / 1000.0),
    )
