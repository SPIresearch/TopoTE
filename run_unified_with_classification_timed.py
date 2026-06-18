import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from utils.metric_registry import METRIC_REGISTRY
from utils.openmind_match import infer_openmind_method_key
from utils.openmind_reference import get_openmind_value, get_openmind_cls_value
from utils.ranking_metrics import kendall_tau_b
from utils.precomputed_features import load_precomputed_store, describe, layer_sort_key

CKPT_EXTS = (".pth", ".ckpt", ".pt")


def _nat_key(s: str):
    """Natural-sort key so that 'stages.10' sorts after 'stages.2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def find_ckpts(root: str, recursive: bool = True) -> list[str]:
    """Glob for checkpoint files under *root*."""
    files: list[str] = []
    for ext in CKPT_EXTS:
        pattern = os.path.join(root, "**", f"*{ext}") if recursive else os.path.join(root, f"*{ext}")
        files += glob.glob(pattern, recursive=recursive)
    return sorted(set(files))


def find_feature_targets(feature_root: str) -> list[str]:
    """Enumerate model feature targets under a feature root.

    Expected layouts:
      (1) <feature_root>/<model_name>/case_0000/*.pkl ...
      (2) <feature_root>/<model_name>.pkl
    """
    p = Path(feature_root)
    if not p.exists():
        raise FileNotFoundError(f"feature_dir not found: {feature_root}")
    if p.is_file() and p.suffix.lower() == ".pkl":
        return [str(p)]

    if p.is_dir():
        kids = list(p.iterdir())
        # If this directory itself looks like a model directory (contains case_* with *.pkl)
        has_case_layer = any((d.is_dir() and list(d.glob("*.pkl"))) for d in kids)
        if has_case_layer:
            return [str(p)]

        targets = []
        for child in kids:
            if child.is_dir():
                targets.append(str(child))
            elif child.is_file() and child.suffix.lower() == ".pkl":
                targets.append(str(child))
        return sorted(targets)

    return []


def detect_task_type(cfg_path: str) -> str:
    """
    从配置文件中检测任务类型
    
    Returns:
        "classification" or "segmentation"
    """
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        task_type = cfg.get("task_type", "segmentation").lower().strip()
        return task_type
    except Exception as e:
        print(f"[WARN] 无法读取配置文件检测任务类型: {e}")
        return "segmentation"


# ---------------------------------------------------------------------------
# single-checkpoint evaluation  (subprocess wrapper)
# ---------------------------------------------------------------------------


def run_one(
    ckpt: str,
    cfg: str,
    data_list_file: str,
    metric: str,
    output_dir: str,
    metric_params: list[str] | None = None,
    main_script: str = "main_unified_with_classification.py",
    pkl_path: str = "",
    decoder_loading_policy: str = "",
    reconstruction_methods: str = "",
    contrastive_methods: str = "",
    unknown_decoder_policy: str = "",
) -> tuple[str | None, str | None]:
    """
    Invoke main_unified_with_classification.py for one checkpoint.

    Returns (result_json_path, error_snippet).
    Exactly one of the two is None.
    """
    if ckpt:
        model_name = Path(ckpt).stem
        display_name = os.path.basename(ckpt)
    elif pkl_path:
        pp = Path(pkl_path)
        model_name = pp.name if pp.is_dir() else pp.stem
        display_name = model_name
    else:
        model_name = "unknown"
        display_name = "unknown"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["python", main_script, "--cfg", cfg, "--metric", metric, "--output_dir", output_dir, "--model_name", model_name]
    if data_list_file:
        cmd += ["--data_list_file", data_list_file]
    if ckpt:
        cmd += ["--model_path", ckpt]
    if pkl_path:
        cmd += ["--pkl_path", pkl_path]
    if metric_params:
        cmd += ["--metric_params"] + metric_params
    if decoder_loading_policy:
        cmd += ["--decoder_loading_policy", decoder_loading_policy]
    if reconstruction_methods:
        cmd += ["--reconstruction_methods", reconstruction_methods]
    if contrastive_methods:
        cmd += ["--contrastive_methods", contrastive_methods]
    if unknown_decoder_policy:
        cmd += ["--unknown_decoder_policy", unknown_decoder_policy]

    try:
        print(f"  ==> {display_name}")
        # stdout 直接透传终端（子进程打印的拓扑分析过程可见），
        # stderr 单独捕获用于错误报告
        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.returncode != 0:
            snippet = result.stderr.decode("utf-8", errors="ignore")[-2000:]
            return None, snippet

        res_json = Path(output_dir) / f"{model_name}_results.json"
        if res_json.exists():
            return str(res_json), None
        return None, "result JSON was not created"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# OpenMind alignment  +  Kendall τ-b
# ---------------------------------------------------------------------------


def attach_openmind_and_kendall(
    df: pd.DataFrame,
    score_col: str,
    out_csv: str,
    arch: str,
    col: str,
) -> pd.DataFrame:
    """
    1. Map each row's checkpoint filename  →  OpenMind method key  →  reference score.
    2. Compute Kendall τ-b between *score_col* and the reference scores.
    3. Write a short summary text file alongside the CSV.
    """
    method_keys, ref_vals = [], []
    for _, row in df.iterrows():
        key = infer_openmind_method_key(str(row.get("file", "")))
        method_keys.append(key)
        ref_vals.append(get_openmind_value(key, col=col, arch=arch) if key else None)

    df["openmind_method"] = method_keys
    df[f"openmind_{col}"] = ref_vals

    # only pairs where both values are present
    valid = df[(df[score_col].notna()) & (df[f"openmind_{col}"].notna())]
    n = len(valid)

    tau = (
        kendall_tau_b(
            valid[score_col].astype(float).tolist(),
            valid[f"openmind_{col}"].astype(float).tolist(),
        )
        if n >= 2
        else float("nan")
    )

    print(f"  kendall_tau_b ({score_col} vs OpenMind-{col}, arch={arch}, n={n}): {tau}")

    # persist summary
    summary_base = Path(str(out_csv)).with_suffix("")
    summary = summary_base.parent / (summary_base.name + f"_kendall_{score_col}_{arch}_{col}.txt")
    summary.write_text(
        f"kendall_tau_b ({score_col} vs OpenMind-{col}, arch={arch})\n"
        f"n={n}\n"
        f"tau_b={tau}\n",
        encoding="utf-8",
    )
    print(f"  saved: {summary}")
    return df


def attach_openmind_cls_and_kendall(
    df: pd.DataFrame,
    score_col: str,
    out_csv: str,
    arch: str,
    col: str,
    cls_metric: str = "ap",
) -> pd.DataFrame:
    """
    Classification 版 OpenMind 对齐 + Kendall τ-b。

    使用 get_openmind_cls_value() 查 Table 3（AP / BAcc）。
    col 取值: ABI / MRN / RSN / Mean
    cls_metric 取值: ap / bacc
    """
    method_keys, ref_vals = [], []
    for _, row in df.iterrows():
        key = infer_openmind_method_key(str(row.get("file", "")))
        method_keys.append(key)
        ref_vals.append(
            get_openmind_cls_value(key, col=col, arch=arch, metric=cls_metric)
            if key else None
        )

    df["openmind_method"]       = method_keys
    df[f"openmind_{col}_{cls_metric}"] = ref_vals

    ref_col = f"openmind_{col}_{cls_metric}"
    valid = df[(df[score_col].notna()) & (df[ref_col].notna())]
    n = len(valid)

    tau = (
        kendall_tau_b(
            valid[score_col].astype(float).tolist(),
            valid[ref_col].astype(float).tolist(),
        )
        if n >= 2
        else float("nan")
    )

    print(f"  kendall_tau_b ({score_col} vs OpenMind-{col}-{cls_metric}, arch={arch}, n={n}): {tau}")

    summary_base = Path(str(out_csv)).with_suffix("")
    summary = summary_base.parent / (
        summary_base.name + f"_kendall_{score_col}_{arch}_{col}_{cls_metric}.txt"
    )
    summary.write_text(
        f"kendall_tau_b ({score_col} vs OpenMind-{col}-{cls_metric}, arch={arch})\n"
        f"n={n}\n"
        f"tau_b={tau}\n",
        encoding="utf-8",
    )
    print(f"  saved: {summary}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Batch metric evaluation + OpenMind ranking (Segmentation + Classification)")

    # required
    parser.add_argument("--cfg", type=str, required=True, help="Config JSON (包含task_type)")
    # NOTE: In PKL feature mode, data_list_file is not needed.
    parser.add_argument("--data_list_file", type=str, default="", help="Data-list JSON (ignored in PKL feature mode)")
    # Two modes (pick one):
    #   - ckpt mode: provide --ckpt_dir (will search *.pth/*.ckpt/*.pt)
    #   - feature mode: provide --feature_dir (will enumerate model subdirs or *.pkl)
    parser.add_argument("--ckpt_dir", type=str, default="", help="Root directory containing checkpoints")
    parser.add_argument("--feature_dir", type=str, default="", help="Root directory containing precomputed features (PKL)")
    parser.add_argument("--metric", type=str, required=True, choices=list(METRIC_REGISTRY.keys()),
                        help="Metric to evaluate")

    # metric tunables  (forwarded verbatim to main_unified.py)
    parser.add_argument("--metric_params", nargs="*", default=[],
                        help="Metric hyper-params as  key=value  tokens")
    parser.add_argument("--decoder_loading_policy", type=str, default="",
                        help="decoder loading strategy forwarded to main script")
    parser.add_argument("--reconstruction_methods", type=str, default="",
                        help="comma-separated reconstruction methods for openmind_hybrid")
    parser.add_argument("--contrastive_methods", type=str, default="",
                        help="comma-separated contrastive methods for openmind_hybrid")
    parser.add_argument("--unknown_decoder_policy", type=str, default="",
                        help="fallback policy for unknown methods: encoder_only/full")

    # search & output
    parser.add_argument("--recursive", action="store_true", help="Recursively search ckpt_dir")
    parser.add_argument("--max_models", type=int, default=0, help="Limit number of checkpoints (0 = all)")
    parser.add_argument("--out_csv", type=str, default="rank.csv", help="Ranked output CSV")
    parser.add_argument("--output_dir", type=str, default="./results1", help="Per-model result JSONs")
    parser.add_argument(
        "--layerwise_csv",
        type=str,
        default="",
        help=(
            "(Optional) Write per-model per-layer scores as a long-form CSV. "
            "Default: <out_csv_stem>_layerwise_long.csv"
        ),
    )

    # OpenMind
    parser.add_argument("--openmind_arch", type=str, default="resenc_l",
                        help="Architecture key: resenc_l | primus_m")
    parser.add_argument("--openmind_col", type=str, default="All",
                        help="OpenMind column: All | ID | OOD | <site-code>")
    parser.add_argument("--no_openmind", action="store_true", help="Skip OpenMind Kendall computation")
    parser.add_argument("--openmind_cls_metric", type=str, default="ap", choices=["ap", "bacc"],
                        help="Classification reference metric: ap (Average Precision) or bacc (Balanced Accuracy). Default: ap")

    # main script
    parser.add_argument("--main_script", type=str, default="main_unified_with_classification.py",
                        help="Main evaluation script to use")

    args = parser.parse_args()

    # ---- 检测任务类型 ----
    task_type = detect_task_type(args.cfg)
    print(f"\n{'=' * 70}")
    print(f"  🎯 检测到任务类型: {task_type.upper()}")
    print(f"{'=' * 70}\n")

    # ---- discover targets (ckpt mode OR feature mode) ----
    mode = "feature" if args.feature_dir else "ckpt"
    if mode == "feature":
        targets = find_feature_targets(args.feature_dir)
        if not targets:
            raise RuntimeError(
                f"No feature targets found in {args.feature_dir}. Expected subdirs per model or *.pkl files."
            )
        if args.max_models > 0:
            targets = targets[: args.max_models]
        print(f"Found {len(targets)} feature target(s)  |  metric = {args.metric}\n")
    else:
        if not args.ckpt_dir:
            raise RuntimeError("You must provide --ckpt_dir (ckpt mode) or --feature_dir (PKL feature mode)")
        targets = find_ckpts(args.ckpt_dir, recursive=args.recursive)
        if not targets:
            raise RuntimeError(f"No checkpoints ({CKPT_EXTS}) found in {args.ckpt_dir}")
        if args.max_models > 0:
            targets = targets[: args.max_models]
        print(f"Found {len(targets)} checkpoint(s)  |  metric = {args.metric}\n")

    # ---- evaluate each target ----
    rows: list[dict] = []
    layer_rows: list[dict] = []
    for t in targets:
        ckpt = t if mode == "ckpt" else ""
        pkl_path = t if mode == "feature" else ""

        # unify display columns (compute early so we can log timing)
        file_label = (
            os.path.basename(ckpt)
            if mode == "ckpt"
            else (Path(pkl_path).name if Path(pkl_path).is_dir() else Path(pkl_path).stem)
        )

        t0 = time.perf_counter()
        res_json, err = run_one(
            ckpt=ckpt,
            cfg=args.cfg,
            data_list_file=args.data_list_file,
            metric=args.metric,
            output_dir=args.output_dir,
            metric_params=args.metric_params or None,
            main_script=args.main_script,
            pkl_path=pkl_path,
            decoder_loading_policy=args.decoder_loading_policy,
            reconstruction_methods=args.reconstruction_methods,
            contrastive_methods=args.contrastive_methods,
            unknown_decoder_policy=args.unknown_decoder_policy,
        )
        dt = time.perf_counter() - t0
        print(f"    ⏱️  wall-time: {dt:.2f}s ({dt/60.0:.2f} min)  |  {file_label}")

        row: dict = {"file": file_label, "path": (ckpt if mode == "ckpt" else pkl_path), "wall_time_sec": float(dt)}

        if res_json is None:
            row.update(overall_score=None, encoder_last=None, decoder_last=None,
                       status="error", error=err)
            print(f"    [ERROR] {row['file']}")
        else:
            try:
                data = json.loads(Path(res_json).read_text(encoding="utf-8"))
                row.update(
                    overall_score=data.get("overall_score"),
                    encoder_last=data.get("encoder_last"),
                    decoder_last=data.get("decoder_last"),
                    task_type=data.get("task_type", task_type),
                    status="ok" if data.get("overall_score") is not None else "no_score",
                    result_json=res_json,
                )
                print(f"    [OK]  overall={data.get('overall_score')}")

                # ---- collect per-layer records (one row per model-layer) ----
                ls = data.get("layer_scores") or {}
                lay = data.get("layers") or list(ls.keys())
                for layer in lay:
                    layer_rows.append(
                        {
                            "file": file_label,
                            "model_name": data.get("model_name", file_label),
                            "task_type": data.get("task_type", task_type),
                            "metric": data.get("metric", args.metric),
                            "layer": layer,
                            "layer_score": ls.get(layer, None),
                            "result_json": res_json,
                        }
                    )
            except Exception as exc:
                row.update(overall_score=None, encoder_last=None, decoder_last=None,
                           status="parse_error", error=str(exc))

        rows.append(row)

    # ---- rank (higher is better) ----
    df = pd.DataFrame(rows)
    df["_sort"] = df["overall_score"].astype(float).where(df["overall_score"].notna(), -1e18)
    df = df.sort_values(["_sort", "file"], ascending=[False, True]).drop(columns=["_sort"])


    # ---- OpenMind Kendall τ-b (仅对分割任务) ----
    if not args.no_openmind and task_type == "segmentation":
        print("\n📊 计算OpenMind相关性 (仅适用于分割任务)...")
        
        score_cols = ["overall_score", "encoder_last", "decoder_last"]
        for score_col in score_cols:
            if score_col in df.columns and df[score_col].notna().any():
                df = attach_openmind_and_kendall(
                    df, score_col, args.out_csv, args.openmind_arch, args.openmind_col
                )
    elif not args.no_openmind and task_type == "classification":
        print("\n📊 计算OpenMind相关性 (分类任务)...")
        print(f"  arch={args.openmind_arch}  col={args.openmind_col}  cls_metric={args.openmind_cls_metric}")
        score_cols = ["overall_score", "encoder_last"]
        for score_col in score_cols:
            if score_col in df.columns and df[score_col].notna().any():
                df = attach_openmind_cls_and_kendall(
                    df, score_col, args.out_csv,
                    arch=args.openmind_arch,
                    col=args.openmind_col,
                    cls_metric=args.openmind_cls_metric,
                )

    # ---- write CSV ----
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # ---- write per-layer long-form CSV (requested: 每个模型每一层的分数记录) ----
    if layer_rows:
        layer_df = pd.DataFrame(layer_rows)
        if not layer_df.empty:
            layer_df["_layer_sort"] = layer_df["layer"].map(layer_sort_key)
            layer_df = layer_df.sort_values(["model_name", "_layer_sort", "layer"], ascending=[True, True, True])
            layer_df = layer_df.drop(columns=["_layer_sort"])

            out_base = Path(args.out_csv)
            layerwise_csv = Path(args.layerwise_csv) if args.layerwise_csv else out_base.with_suffix("")
            if not args.layerwise_csv:
                layerwise_csv = layerwise_csv.parent / (layerwise_csv.name + "_layerwise_long.csv")

            layerwise_csv.parent.mkdir(parents=True, exist_ok=True)
            layer_df.to_csv(layerwise_csv, index=False)
            print(f"Saved layerwise CSV: {layerwise_csv}")

    # ---- summary table ----
    display_cols = [c for c in (
        "file", "wall_time_sec", "overall_score",
        "encoder_last", "decoder_last",
        "task_type", "status",
    ) if c in df.columns]
    print(f"\n{'=' * 70}")
    print(df[display_cols].head(20).to_string(index=False))
    print(f"{'=' * 70}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
