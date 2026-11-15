"""專案根目錄與路徑解析。

全專案唯一允許「決定檔案位置」的地方。其他模組一律拿 `AppConfig` 裡已解析好的
`pathlib.Path`，不可自行拼接、不可寫死磁碟代號。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_HOME = "PHOTOSORTER_HOME"
ENV_CONFIG = "PHOTOSORTER_CONFIG"


def project_root() -> Path:
    """回傳專案根目錄（`config.example.json`、`weights/`、`data/` 所在處）。

    優先序：
      1. 環境變數 `PHOTOSORTER_HOME`
      2. PyInstaller 打包後：exe 所在資料夾
      3. 原始碼執行：本檔案往上兩層（photosorter/core/paths.py -> 專案根）
    """
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | os.PathLike[str], base: Path | None = None) -> Path:
    """把設定檔裡的路徑字串轉成絕對路徑。

    相對路徑一律相對於 `base`（預設為專案根目錄），不是相對於目前工作目錄 ——
    這樣從任何位置啟動、或打包成 exe 後行為都一致。支援 `~` 與環境變數展開。
    """
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    p = Path(expanded)
    if not p.is_absolute():
        p = (base or project_root()) / p
    return Path(os.path.normpath(p))
