"""Headless 命令列介面。核心不需要 GUI 就能完整運作，這裡同時是 GUI 接線方式的活範例。

    python -m photosorter doctor [--load-models]     環境檢查（新電腦 clone 後先跑這個）
    python -m photosorter scan <資料夾>... [--stages meta,siglip,faces]
    python -m photosorter stats
    python -m photosorter export <目標資料夾> [--mode copy|move|hardlink|symlink] [--template T] [--execute]
    python -m photosorter rollback <batch_id>
    python -m photosorter dump-config -o config.example.json    由程式裡的預設值重新產生設定範本

共通選項：`--config <路徑>`（預設讀專案根目錄的 config.json，沒有就全用預設值）。
"""
from __future__ import annotations

import argparse
import importlib
import logging
import platform
import sys
from typing import Any

from .core.config import dump_default_config, load_config
from .core.events import Event, EventKind
from .core.hardware import probe_hardware, provider_candidates
from .utils.logging_setup import setup_logging


def _print(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:                 # 舊版主控台的字碼頁印不出某些字元
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace) -> int:
    """列出這台電腦的實際狀況。所有硬體相關的決定都在這裡看得到，不需要猜。"""
    cfg = load_config(args.config)
    problems = 0
    _print(f"專案根目錄   {cfg.root}")
    _print(f"設定檔       {cfg.source_file or '(無，全部使用預設值)'}")
    _print(f"資料庫       {cfg.db_path}")
    _print(f"Python       {platform.python_version()} ({platform.architecture()[0]})")
    _print(f"作業系統     {platform.platform()}")

    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        ok = build >= 18362
        _print(f"Windows build {build}  DirectML 需要 >= 18362 (Windows 10 1903)：{'OK' if ok else '不符合，將使用 CPU'}")
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                long_paths = winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
        except OSError:
            long_paths = False
        _print(f"長路徑支援   {'已開啟' if long_paths else '未開啟（路徑超過 260 字元的照片會讀不到，見 HANDOFF.md §6）'}")

    hw = probe_hardware()
    _print(f"CPU          {hw.physical_cores} 實體核心 / {hw.logical_cores} 邏輯核心，記憶體 {hw.total_ram_gb} GB")
    _print(f"onnxruntime  {hw.ort_version or '未安裝'}")
    _print(f"可用 provider {hw.available_providers}")
    chain = provider_candidates(cfg.runtime.providers, hw.available_providers)
    _print(f"降級順序     {' -> '.join(chain) or '(無)'}")
    if not hw.ort_version:
        problems += 1

    _print("\n相依套件")
    for module, purpose, required in [
        ("numpy", "陣列運算", True), ("cv2", "影像解碼/縮放", True), ("PIL", "EXIF 與備援解碼", True),
        ("pillow_heif", "HEIC/HEIF", False), ("tokenizers", "SigLIP 文字編碼", True),
        ("sklearn", "人臉分群", True), ("reverse_geocoder", "離線地理編碼", False),
        ("imagehash", "pHash（選配）", False), ("psutil", "硬體探測", False),
        ("sqlite_vec", "sqlite-vec（選配）", False),
    ]:
        try:
            m = importlib.import_module(module)
            _print(f"  OK    {module:<17} {getattr(m, '__version__', '')}  {purpose}")
        except Exception as e:  # noqa: BLE001
            _print(f"  {'缺少' if required else '未裝'}  {module:<17} {purpose}  ({type(e).__name__})")
            problems += int(required)

    _print("\n模型權重")
    weights = [
        ("SigLIP vision", cfg.weight(cfg.siglip.vision.file)),
        ("SigLIP text", cfg.weight(cfg.siglip.text.file)),
        ("SigLIP tokenizer", cfg.weight(cfg.siglip.tokenizer_file)),
        ("SCRFD 偵測", cfg.weight(cfg.face.detector.file)),
        ("ArcFace 特徵", cfg.weight(cfg.face.recognizer.file)),
    ]
    for label, path in weights:
        if path.is_file():
            _print(f"  OK    {label:<17} {path}  ({path.stat().st_size / 2**20:.1f} MB)")
        else:
            _print(f"  缺少  {label:<17} {path}")
            problems += 1

    from .pipeline.export_manager import can_symlink

    _print(f"\n符號連結權限 {'有' if can_symlink() else '沒有（匯出的 symlink 模式需要開發人員模式或系統管理員）'}")

    if args.load_models:
        _print("\n實際建立 session（確認每個模型最後落在哪個 provider）")
        from .models.onnx_runtime import ModelLoadError, create_session

        for label, path in weights:
            if path.suffix != ".onnx":
                continue
            try:
                h = create_session(path, cfg.runtime)
                note = "".join(f"\n          {p} 被淘汰：{why}" for p, why in h.rejected)
                _print(f"  OK    {label:<17} provider={h.provider}  {h.load_ms:.0f} ms"
                       f"\n          inputs={h.input_shapes}  outputs={h.output_names}{note}")
            except ModelLoadError as e:
                _print(f"  失敗  {label:<17} {e}")
                problems += 1

    _print(f"\n結果：{'沒有發現問題' if problems == 0 else f'{problems} 個項目需要處理'}")
    return 0 if problems == 0 else 1


# --------------------------------------------------------------------------- #
def cmd_scan(args: argparse.Namespace) -> int:
    from .pipeline.factory import create_app

    overrides: dict[str, Any] = {}
    if args.stages:
        overrides["pipeline"] = {"stages": [s.strip() for s in args.stages.split(",") if s.strip()]}
    if args.retry_errors:
        overrides["scan"] = {"retry_errors": True}

    with create_app(args.config, overrides=overrides) as ctx:
        setup_logging(ctx.cfg.log_dir, logging.INFO, console=args.verbose)

        def on_event(e: Event) -> None:        # 在工作執行緒上被呼叫：只做列印，不做耗時的事
            d = e.data
            if e.kind is EventKind.STATE:
                _print(f"[狀態] {d['previous']} -> {d['state']}")
            elif e.kind is EventKind.MODEL:
                if d["phase"] == "ready":
                    for m in d["models"]:
                        _print(f"[模型] {m['kind']:<12} {m['name']}  provider={m['provider']}")
                elif d["phase"] == "failed":
                    _print(f"[模型] 載入失敗：{d['error']}")
            elif e.kind is EventKind.SCAN_DONE:
                _print(f"[掃描] 共 {d['total_files']} 個影像檔，需處理 {d['to_process']}，"
                       f"未變動 {d['unchanged']}，遺失 {d['missing']}")
            elif e.kind is EventKind.PROGRESS:
                _print(f"[進度] {d['processed']}/{d['to_process']}  推論 {d['inferred']}  "
                       f"人臉 {d['faces_found']}  錯誤 {d['errors']}  佇列 {d['queue_depth']}  {d['elapsed_s']:.0f}s")
            elif e.kind is EventKind.PHOTO_ERROR:
                _print(f"[錯誤] {d['path']}  {d['stage']}: {d['message']}")
            elif e.kind is EventKind.POST_STEP and d["phase"] != "start":
                _print(f"[後處理] {d['name']} {d['phase']}：{d['detail']}")
            elif e.kind is EventKind.ERROR:
                _print(f"[錯誤] {d['where']}: {d['message']}")

        ctx.bus.subscribe(on_event)
        engine = ctx.build_engine()
        engine.start(args.roots)
        try:
            while not engine.join(timeout=0.5):
                pass
        except KeyboardInterrupt:
            _print("\n收到 Ctrl+C，正在安全停止（已完成的結果都會保留，下次可續掃）…")
            engine.stop()
            engine.join()
        stats = engine.stats()
        _print(f"\n結束狀態：{engine.state.value}，處理 {stats.processed}/{stats.to_process}，"
               f"搬移 {stats.moved}，重複 {stats.duplicates}，錯誤 {stats.errors}，耗時 {stats.elapsed_s:.1f}s")
        return 0 if engine.state.value in ("finished", "stopped") else 1


def cmd_stats(args: argparse.Namespace) -> int:
    from .pipeline.factory import create_app

    with create_app(args.config) as ctx:
        for key, value in ctx.repo.library_stats().items():
            _print(f"{key:<14} {value}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .pipeline.factory import create_app

    with create_app(args.config) as ctx:
        plan = ctx.exporter.plan(args.dest, mode=args.mode, template=args.template)
        for w in plan.warnings:
            _print(f"[警告] {w}")
        for key, n in sorted(plan.summary().items()):
            _print(f"  {key:<24} {n}")
        for op in plan.ops[: args.preview]:
            _print(f"  {op.action:<9}{op.note:<16}{op.src}  ->  {op.dst or '-'}")
        if not args.execute:
            _print("\n以上是預覽（dry-run），沒有動到任何檔案。加上 --execute 才會真的執行。")
            return 0
        result = ctx.exporter.execute(plan)
        _print(f"\n批次 {result.batch_id}：完成 {result.done}，略過 {result.skipped}，失敗 {result.failed}"
               f"（回復：python -m photosorter rollback {result.batch_id}）")
        for src, message in result.errors[:20]:
            _print(f"  失敗 {src}: {message}")
        return 0 if result.failed == 0 else 1


def cmd_rollback(args: argparse.Namespace) -> int:
    from .pipeline.factory import create_app

    with create_app(args.config) as ctx:
        result = ctx.exporter.rollback(args.batch_id)
        _print(f"批次 {result.batch_id}：已還原 {result.done}，無法還原 {result.failed}")
        for path, message in result.errors[:20]:
            _print(f"  {path}: {message}")
        return 0 if result.failed == 0 else 1


def cmd_dump_config(args: argparse.Namespace) -> int:
    text = dump_default_config()
    if args.output:
        # 一律 UTF-8、LF：不經過主控台的字碼頁（cp950 會把 JSON 裡的中文寫壞）
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        _print(f"已寫入 {args.output}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="photosorter", description="本地離線照片智慧分類（核心引擎 CLI）")
    parser.add_argument("--config", help="設定檔路徑（預設：專案根目錄的 config.json）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="檢查環境、硬體、相依套件與模型權重")
    p.add_argument("--load-models", action="store_true", help="實際建立 ONNX session，確認生效的 provider")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("scan", help="掃描並處理資料夾（可隨時 Ctrl+C，下次自動續掃）")
    p.add_argument("roots", nargs="+", help="照片根目錄")
    p.add_argument("--stages", help="覆寫 pipeline.stages，例如 meta 或 meta,siglip")
    p.add_argument("--retry-errors", action="store_true", help="重試上次失敗的檔案")
    p.add_argument("-v", "--verbose", action="store_true", help="同時把日誌印到主控台")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("stats", help="照片庫統計")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="依分類結果匯出（預設只預覽）")
    p.add_argument("dest", help="目標根目錄")
    p.add_argument("--mode", default="copy", choices=["copy", "move", "hardlink", "symlink"])
    p.add_argument("--template", help="路徑模板，例如 {year}/{event}/{filename}")
    p.add_argument("--preview", type=int, default=20, help="預覽時列出前幾筆")
    p.add_argument("--execute", action="store_true", help="真的執行（不加就是 dry-run）")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("rollback", help="還原一個匯出批次")
    p.add_argument("batch_id", type=int)
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("dump-config", help="印出預設設定 JSON")
    p.add_argument("-o", "--output", help="寫入檔案（UTF-8）；更新 config.example.json 時用")
    p.set_defaults(func=cmd_dump_config)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
