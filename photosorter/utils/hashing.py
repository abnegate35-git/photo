"""內容雜湊。"""
from __future__ import annotations

import hashlib
import os

_CHUNK = 1 << 20


def sha256_bytes(data: bytes | bytearray | memoryview) -> bytes:
    """回傳 32 bytes 的 digest（資料庫存 BLOB，不存十六進位字串，省一半空間）。"""
    return hashlib.sha256(data).digest()


def sha256_file(path: str | os.PathLike[str]) -> bytes:
    """串流計算，不把整個檔案讀進記憶體。匯出驗證用。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()
