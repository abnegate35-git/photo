"""日誌設定。函式庫模組只 `logging.getLogger(__name__)`，不自行設定 handler；
由進入點（CLI 或 GUI）呼叫一次 `setup_logging()`。"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(threadName)-12s %(name)s: %(message)s"


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO, console: bool = True) -> None:
    """設定 root logger。`log_dir` 給定時另外寫輪替檔 `photosorter.log`（5 MB x 3）。"""
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(_FORMAT)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "photosorter.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
