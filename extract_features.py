#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量特征提取脚本

对 ckpt_dir 下每个 checkpoint 调用 extract_features_only.py，
将特征保存到 output_root/{model_id}/ 目录下，供后续 PKL 模式的
run_unified_with_classification_timed.py 消费。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CKPT_EXTS = (".pth", ".ckpt", ".pt")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_extractor_script() -> str:
    """
    定位 extract_features_only.py。
    搜索顺序：
      1. 当前工作目录 (extract_features_only.py)
      2. utils/ 子目录 (utils/extract_features_only.py)
    若都找不到则抛出 RuntimeError，提示用户。
    """
    candidates = [
        Path("extract_features_only.py"),
        Path("utils") / "extract_features_only.py",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise RuntimeError(
        "找不到 extract_features_only.py。\n"
        "请确保以下任一路径存在:\n"
        "  ./extract_features_only.py\n"
        "  ./utils/extract_features_only.py\n"
        "当前工作目录: " + str(Path.cwd())
    )


def find_all_checkpoints(root_dir: str, recursive: bool = True) -> list[Path]:
    """递归查找所有 checkpoint 文件，按路径排序去重。"""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    ckpts: list[Path] = []
    for ext in CKPT_EXTS:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        ckpts.extend(root.glob(pattern))

    return sorted(set(ckpts))


def make_model_id(
    ckpt_path: Path | str,
    use_relative_path: bool = True,
    base_dir: str | None = None,
) -> str:
    """
    生成模型唯一标识符。
    use_relative_path=True 时保留目录层级（路径分隔符替换为 _），
    否则只用文件 stem。
    """
    ckpt_path = Path(ckpt_path)
    if use_relative_path and base_dir:
        try:
            rel = ckpt_path.relative_to(Path(base_dir))
            return str(rel.with_suffix("")).replace("/", "_").replace("\\", "_")
        except ValueError:
            pass
    return ckpt_path.stem


# ---------------------------------------------------------------------------
# single-model extraction
# ---------------------------------------------------------------------------

def extract_features_for_one_model(
    ckpt_path: Path | str,
    cfg_path: str,
    data_list_file: str,
    output_dir: Path,
    extractor_script: str,
    bg_sample_num: int | None = None,
    fv_sample_num: int | None = None,
    boundary_sample_num: int = 0,
    num_workers: int = 4,
    device: str = "cuda",
    skip_existing: bool = True,
    timeout_sec: int = 36000,
) -> tuple[bool, Path, str | None]:
    """
    调用 extract_features_only.py 提取单个模型的特征。

    Returns
    -------
    success : bool
    output_dir : Path
    error_msg : str or None
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "feature_metadata.json"

    # 幂等跳过
    if skip_existing and metadata_path.exists():
        print("  ✓ feature_metadata.json 已存在，跳过")
        return True, output_dir, None

    cmd = [
        sys.executable,
        extractor_script,
        "--cfg",              str(cfg_path),
        "--data_list_file",   str(data_list_file),
        "--model_path",       str(ckpt_path),
        "--model_name",       output_dir.name,
        "--feature_save_dir", str(output_dir),
        "--num_workers",      str(num_workers),
        "--device",           device,           # FIX: 之前此参数未传入 subprocess
    ]

    if bg_sample_num is not None:
        cmd += ["--bg_sample_num", str(bg_sample_num)]
    if fv_sample_num is not None:
        cmd += ["--fv_sample_num", str(fv_sample_num)]
    if boundary_sample_num:
        cmd += ["--boundary_sample_num", str(boundary_sample_num)]

    # 保留 per-model 日志文件，便于事后排查失败原因
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "extraction.log"

    try:
        print(f"  → 开始提取  (log: {log_path})")
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )

        if result.returncode == 0 and metadata_path.exists():
            print("  ✓ 提取成功")
            return True, output_dir, None

        # 失败：读取日志末尾方便定位
        tail = _read_log_tail(log_path, lines=40)
        error_msg = f"returncode={result.returncode}\n--- log tail ---\n{tail}"
        print(f"  ✗ 提取失败 (returncode={result.returncode})")
        print(f"    查看完整日志: {log_path}")
        return False, output_dir, error_msg

    except subprocess.TimeoutExpired:
        error_msg = f"超时 (>{timeout_sec}s)"
        print(f"  ✗ {error_msg}")
        return False, output_dir, error_msg
    except Exception as exc:
        error_msg = str(exc)
        print(f"  ✗ 异常: {error_msg}")
        return False, output_dir, error_msg


def _read_log_tail(log_path: Path, lines: int = 40) -> str:
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(all_lines[-lines:])
    except Exception:
        return "(无法读取日志)"


# ---------------------------------------------------------------------------
# batch extraction
# ---------------------------------------------------------------------------

def batch_extract(
    ckpt_dir: str,
    cfg_path: str,
    data_list_file: str,
    output_root: str,
    extractor_script: str,
    bg_sample_num: int | None = None,
    fv_sample_num: int | None = None,
    boundary_sample_num: int = 0,
    num_workers: int = 4,
    device: str = "cuda",
    recursive: bool = True,
    skip_existing: bool = True,
    max_models: int = 0,
    use_relative_path: bool = True,
    timeout_sec: int = 36000,
) -> dict:
    """批量提取所有模型的特征，返回汇总 dict。"""
    print("=" * 80)
    print("批量特征提取")
    print("=" * 80)
    print(f"  Checkpoint目录  : {ckpt_dir}")
    print(f"  输出根目录      : {output_root}")
    print(f"  配置文件        : {cfg_path}")
    print(f"  数据列表        : {data_list_file}")
    print(f"  提取脚本        : {extractor_script}")
    print(f"  设备            : {device}")
    print(f"  背景采样数      : {bg_sample_num}")
    print(f"  边界采样数      : {boundary_sample_num}")
    print(f"  全局采样数      : {fv_sample_num}")
    print(f"  num_workers     : {num_workers}")
    print(f"  递归搜索        : {recursive}")
    print(f"  跳过已存在      : {skip_existing}")
    print("=" * 80)

    # 查找 checkpoints
    print("\n查找 checkpoints...")
    ckpts = find_all_checkpoints(ckpt_dir, recursive=recursive)
    if not ckpts:
        print(f"❌ 未找到任何 checkpoint 文件 {CKPT_EXTS}")
        return {}

    print(f"找到 {len(ckpts)} 个 checkpoint(s)")
    if max_models > 0:
        ckpts = ckpts[:max_models]
        print(f"限制处理前 {max_models} 个")

    output_root_p = Path(output_root)
    output_root_p.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    start_time = datetime.now()

    for i, ckpt in enumerate(ckpts, 1):
        model_id = make_model_id(ckpt, use_relative_path=use_relative_path, base_dir=ckpt_dir)
        print(f"\n[{i}/{len(ckpts)}] {model_id}")
        print(f"  路径: {ckpt}")

        success, out_dir, error = extract_features_for_one_model(
            ckpt_path=ckpt,
            cfg_path=cfg_path,
            data_list_file=data_list_file,
            output_dir=output_root_p / model_id,
            extractor_script=extractor_script,
            bg_sample_num=bg_sample_num,
            fv_sample_num=fv_sample_num,
            boundary_sample_num=boundary_sample_num,
            num_workers=num_workers,
            device=device,
            skip_existing=skip_existing,
            timeout_sec=timeout_sec,
        )

        results.append({
            "model_id":   model_id,
            "ckpt_path":  str(ckpt),
            "output_dir": str(out_dir),
            "success":    success,
            "error":      error,
        })

    # 汇总
    elapsed = datetime.now() - start_time
    success_count = sum(1 for r in results if r["success"])
    fail_count    = len(results) - success_count

    print("\n" + "=" * 80)
    print("批量提取完成")
    print("=" * 80)
    print(f"  总数  : {len(results)}")
    print(f"  成功  : {success_count}")
    print(f"  失败  : {fail_count}")
    print(f"  耗时  : {elapsed}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ckpt_dir":           str(ckpt_dir),
            "cfg_path":           str(cfg_path),
            "data_list_file":     str(data_list_file),
            "output_root":        str(output_root),
            "device":             device,
            "bg_sample_num":      bg_sample_num,
            "fv_sample_num":      fv_sample_num,
            "boundary_sample_num": boundary_sample_num,
            "num_workers":        num_workers,
        },
        "stats": {
            "total":           len(results),
            "success":         success_count,
            "failed":          fail_count,
            "elapsed_seconds": elapsed.total_seconds(),
        },
        "results": results,
    }

    summary_path = output_root_p / "batch_extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")

    if fail_count > 0:
        print("\n失败的模型:")
        for r in results:
            if not r["success"]:
                # 只显示首行
                first_line = (r["error"] or "").splitlines()[0][:120]
                print(f"  - {r['model_id']}: {first_line}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量递归提取目录下所有模型的特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分类任务，encoder-only config
  python extract_features.py \\
      --ckpt_dir ../models_ResEncL \\
      --cfg Configs/ABI_ResEncL_T1_encoder_only.json \\
      --data_list_file DataLists/ABIDE_CMU_T1_all.json \\
      --output_root ./features/ABI_T1 \\
      --device cuda

  # 分割任务
  python extract_features.py \\
      --ckpt_dir ../models \\
      --cfg Configs/MS_FLAIR_ResEncL.json \\
      --data_list_file DataLists/MS_FLAIR.json \\
      --output_root ./features/MSF \\
      --bg_sample_num 2560
        """,
    )

    # ---- 必需参数 ----
    parser.add_argument("--ckpt_dir",        required=True,  help="Checkpoint 根目录")
    parser.add_argument("--cfg",             required=True,  help="配置 JSON 文件")
    parser.add_argument("--data_list_file",  required=True,  help="数据列表 JSON 文件")
    parser.add_argument("--output_root",     required=True,  help="特征保存根目录")

    # ---- 采样参数 ----
    parser.add_argument("--bg_sample_num",       type=int, default=None,
                        help="背景采样数量（分割任务建议 2560）")
    parser.add_argument("--fv_sample_num",       type=int, default=None,
                        help="全局采样数量（默认使用 config 中的值）")
    parser.add_argument("--boundary_sample_num", type=int, default=0,
                        help="GT mask 形态学边界 voxel 采样数（分割任务建议 256，0=不采样）")

    # ---- 运行参数 ----
    parser.add_argument("--device",          default="cuda",
                        help="计算设备: cuda / cpu / cuda:1 等（默认 cuda）")
    parser.add_argument("--num_workers",     type=int, default=4,
                        help="DataLoader num_workers（默认 4，不稳定时改为 0）")
    parser.add_argument("--timeout",         type=int, default=36000,
                        help="单模型超时秒数（默认 36000 = 10 小时）")

    # ---- 控制参数 ----
    parser.add_argument("--no_recursive",    action="store_true",
                        help="不递归搜索子目录")
    parser.add_argument("--force",           action="store_true",
                        help="强制重新提取（忽略已存在的 feature_metadata.json）")
    parser.add_argument("--max_models",      type=int, default=0,
                        help="限制处理模型数量（0 = 全部）")
    parser.add_argument("--no_relative_path",action="store_true",
                        help="只用文件名作为模型 ID（不保留子目录层级）")

    # ---- 脚本路径（一般不需要改） ----
    parser.add_argument("--extractor",       default=None,
                        help="extract_features_only.py 的路径（默认自动查找）")

    args = parser.parse_args()

    # ---- 验证输入 ----
    errors = []
    if not Path(args.ckpt_dir).exists():
        errors.append(f"Checkpoint 目录不存在: {args.ckpt_dir}")
    if not Path(args.cfg).exists():
        errors.append(f"配置文件不存在: {args.cfg}")
    if not Path(args.data_list_file).exists():
        errors.append(f"数据列表文件不存在: {args.data_list_file}")
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    # ---- 定位提取脚本 ----
    extractor = args.extractor or _find_extractor_script()
    print(f"✓ 提取脚本: {extractor}")

    # ---- 执行批量提取 ----
    summary = batch_extract(
        ckpt_dir=args.ckpt_dir,
        cfg_path=args.cfg,
        data_list_file=args.data_list_file,
        output_root=args.output_root,
        extractor_script=extractor,
        bg_sample_num=args.bg_sample_num,
        fv_sample_num=args.fv_sample_num,
        boundary_sample_num=args.boundary_sample_num,
        num_workers=args.num_workers,
        device=args.device,
        recursive=not args.no_recursive,
        skip_existing=not args.force,
        max_models=args.max_models,
        use_relative_path=not args.no_relative_path,
        timeout_sec=args.timeout,
    )

    if not summary:
        sys.exit(1)

    failed = summary.get("stats", {}).get("failed", 0)
    if failed == 0:
        print("\n✓ 全部成功！")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} 个模型失败，请检查对应的 extraction.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
