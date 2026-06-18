#!/usr/bin/env python3
"""
TDA Adaptive 融合脚本 - 带逐模型特征目录支持（严格对齐论文公式）

与 simple 版的区别仅在于：支持从每个模型的独立特征目录加载特征做额外分析。
但 α 门控仍然是 task-level 共享的（由 --num_classes 决定）。

论文核心公式:
  - 任务复杂度:  κ = log(|C|)                        (Section 2.4)
  - 门控因子:    α = tanh(κ/2) = 2·sigmoid(κ)-1
  - 融合分数:    S_ϕ = α · N(T^GRTD) + (1-α) · N(T^LBTC)  (Eq. 6)
  - 层分配:      GRTD 取 decoder 层平均, LBTC 取 encoder 层平均 (Section 6)
  - 评估指标:    weighted Kendall τ*,w                  (Eq. 7)

用法示例：
    python adaptive_fusion_per_model.py \
        --dataset MSF \
        --results_dir results1 \
        --num_classes 1 \
        --openmind_col MSF \
        --out_csv rank_MSF.csv
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.fusion_weights import build_fusion_layer_weights


# =====================================================================
# 论文 Eq. 6: Task-Adaptive Gating
# =====================================================================

def compute_task_complexity(num_classes: int) -> float:
    """
    论文 Section 2.4: κ = log(|C|)
    |C| 是前景语义类别数（不含背景）
    
    当 |C|=1 时 κ=0，α=0。
    """
    assert num_classes >= 1, f"num_classes must be >= 1, got {num_classes}"
    return math.log(num_classes)


def sigmoid_gate(kappa: float) -> float:
    """
    α = tanh(κ/2) = 2·sigmoid(κ)-1
    This maps κ ∈ [0, +∞) to α ∈ [0, 1).

    α → 1: 偏重全局 GRTD（多器官等复杂任务）
    α → 0: 偏重局部 LBTC（局灶病变等简单任务）
    """
    return math.tanh(kappa / 2.0)


# =====================================================================
# 论文 Eq. 7: Weighted Kendall τ*,w
# =====================================================================

def weighted_kendall_tau(pred_scores: List[float], gt_scores: List[float]) -> float:
    """
    论文 Eq. 7:
        τ*,w = 1 - 2 * Σ_{i<j} w(i)w(j) 1[discordant(i,j)]
                       / Σ_{i<j} w(i)w(j)

    其中模型按 ground-truth 性能降序排列，位置 i 的权重 w(i) = 1/(i+1)。
    """
    n = len(pred_scores)
    if n < 2:
        return 0.0

    # 按 gt_scores 降序排列
    order = np.argsort(-np.array(gt_scores))
    pred_ordered = np.array(pred_scores)[order]

    weights = np.array([1.0 / (i + 1) for i in range(n)])

    discordant_sum = 0.0
    total_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            w_ij = weights[i] * weights[j]
            total_sum += w_ij
            if pred_ordered[i] < pred_ordered[j]:
                discordant_sum += w_ij

    if total_sum == 0:
        return 0.0

    tau = 1.0 - 2.0 * discordant_sum / total_sum
    return float(tau)


# =====================================================================
# JSON 加载与层分数提取
# =====================================================================

def _find_metric_dir(dataset_dir: Path, metric: str) -> Path:
    aliases = {
        "TDA": {"tda"},
        "TDA_lbtc": {"tda_lbtc", "lbtc"},
    }
    wanted = aliases.get(metric, {metric.lower()})
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return child
    raise FileNotFoundError(
        f"Metric directory not found under {dataset_dir}: aliases={sorted(wanted)}"
    )


def load_model_results(dataset_dir: Path, metric: str) -> Dict[str, dict]:
    """加载 TDA(GRTD) 或 TDA_lbtc/LBTC 结果 JSON，递归兼容 results_* 子目录。"""
    metric_dir = _find_metric_dir(dataset_dir, metric)
    if not metric_dir.exists():
        raise FileNotFoundError(f"Metric directory not found: {metric_dir}")

    results: Dict[str, dict] = {}
    for json_path in sorted(metric_dir.rglob("*_results.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            model_name = data.get("model_name", json_path.stem.replace("_results", ""))
            if model_name not in results:
                results[model_name] = data
        except Exception as e:
            print(f"  [WARN] 跳过 {json_path.name}: {e}")

    return results


def _to_float(x: Any) -> Optional[float]:
    """尽可能把各种形态的值转成 float"""
    if x is None:
        return None
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, dict):
        for k in ("value", "mean", "avg", "score"):
            if k in x:
                return _to_float(x[k])
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_float(x[0])
    try:
        return float(x)
    except Exception:
        return None


def _canonical_layer_name(name: Any) -> str:
    """Normalize layer keys so JSON keys and fusion-weight keys can match.

    Result JSONs may use module names such as ``encoder.stages.3`` while
    ``utils.fusion_weights`` uses keys such as ``encoder_stages_3``.  We map
    dots, slashes and hyphens to underscores and collapse repeated underscores.
    """
    s = str(name).strip()
    for ch in (".", "/", "-"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.lower().strip("_")


def _weighted_layer_average(layer_scores: dict, layer_weights: Dict[str, float]) -> Optional[float]:
    # Build an alias map to support both dot-style keys
    # (encoder.stages.3) and underscore-style keys (encoder_stages_3).
    score_map = {_canonical_layer_name(k): v for k, v in layer_scores.items()}

    weighted_sum = 0.0
    weight_sum = 0.0
    missing_layers = []
    for layer, weight in layer_weights.items():
        key = _canonical_layer_name(layer)
        val = _to_float(score_map.get(key))
        if val is None:
            missing_layers.append(layer)
            continue
        weighted_sum += weight * val
        weight_sum += weight
    if weight_sum <= 0.0:
        if missing_layers:
            print(f"  [WARN] no fusion layers matched. Missing examples: {missing_layers[:5]}")
        return None
    return float(weighted_sum / weight_sum)


def extract_encoder_avg(results_dict: dict, layer_weights: Dict[str, float]) -> Optional[float]:
    """
    从 layer_scores 中提取 encoder 层的平均分数。
    层名示例: encoder_stem, encoder_stages_0, encoder_stages_1, ...
    """
    layer_scores = results_dict.get("layer_scores", {})
    if not layer_scores:
        return None

    return _weighted_layer_average(layer_scores, layer_weights)


def extract_decoder_avg(results_dict: dict, layer_weights: Dict[str, float]) -> Optional[float]:
    """
    从 layer_scores 中提取 decoder 层的平均分数。
    层名示例: decoder_stages_0, decoder_stages_1, ...
    """
    layer_scores = results_dict.get("layer_scores", {})
    if not layer_scores:
        return None

    return _weighted_layer_average(layer_scores, layer_weights)


# =====================================================================
# Kendall τ (带 OpenMind 参考表)
# =====================================================================

def compute_weighted_kendall_with_openmind(
    model_scores: Dict[str, float],
    openmind_arch: str,
    openmind_col: str,
) -> Tuple[float, int]:
    """用 weighted Kendall τ*,w 对比 OpenMind ground-truth"""
    try:
        from utils.openmind_reference import get_openmind_segmentation_table
        from utils.openmind_match import infer_openmind_method_key
    except ImportError:
        print("  [WARN] 无法导入 utils.openmind_*，跳过 Kendall τ 计算")
        return 0.0, 0

    openmind_table = get_openmind_segmentation_table(arch=openmind_arch)

    pred_list = []
    gt_list = []
    for model_name, score in model_scores.items():
        method_key = infer_openmind_method_key(model_name)
        if method_key is None or method_key not in openmind_table:
            continue

        gt_score = openmind_table[method_key].get(openmind_col.upper())
        if gt_score is None:
            continue

        pred_list.append(score)
        gt_list.append(gt_score)

    if len(pred_list) < 2:
        return 0.0, 0

    tau = weighted_kendall_tau(pred_list, gt_list)
    return tau, len(pred_list)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Topology-Driven Adaptive Fusion - per-model 目录版（论文公式严格版）"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称（如 MSF, ACD, TPC, ISL, HNT, KIT）")
    parser.add_argument("--results_dir", type=str, default="results1",
                        help="结果根目录")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="目标任务的前景语义类别数 |C|（不含背景，>=1）")
    parser.add_argument("--openmind_arch", type=str, default="resenc_l")
    parser.add_argument("--openmind_col", type=str, required=True,
                        help="OpenMind 表中的列名（如 MSF, ACD）")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--weight_profile", type=str, default="paper",
                        choices=["paper", "sharp", "robust", "wide", "flat", "custom"],
                        help="层融合权重 profile，默认 paper（你的原始高斯权重）")
    parser.add_argument("--decoder_sigma", type=float, default=None,
                        help="覆盖 decoder 高斯 sigma（sharp/robust/wide 可用）")
    parser.add_argument("--encoder_sigma", type=float, default=None,
                        help="覆盖 encoder 高斯 sigma（sharp/robust/wide 可用）")
    parser.add_argument("--decoder_weights", type=str, default=None,
                        help="custom profile 时使用，逗号分隔 5 个数")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="custom profile 时使用，逗号分隔 6 个数")

    args = parser.parse_args()

    dataset_dir = Path(args.results_dir) / args.dataset

    print("=" * 70)
    print(f"Topology-Driven Adaptive Fusion (per-model 版): {args.dataset}")
    print("=" * 70)

    fusion_weights = build_fusion_layer_weights(
        profile=args.weight_profile,
        decoder_sigma=args.decoder_sigma,
        encoder_sigma=args.encoder_sigma,
        decoder_custom=args.decoder_weights,
        encoder_custom=args.encoder_weights,
    )
    decoder_layer_weights = fusion_weights["decoder"]
    encoder_layer_weights = fusion_weights["encoder"]

    print(f"\n📊 层融合权重 profile: {args.weight_profile}")
    print(f"  decoder: {decoder_layer_weights}")
    print(f"  encoder: {encoder_layer_weights}")

    # ========== 1. 计算 Task-Adaptive 门控 α ==========
    # 论文 Eq. 6: α = tanh(κ/2) = 2·sigmoid(κ)-1, κ = log(|C|)
    # α 由任务类别数决定，所有模型共享

    kappa = compute_task_complexity(args.num_classes)
    alpha = sigmoid_gate(kappa)

    print(f"\n📊 步骤 1: 计算 Task-Adaptive 门控因子 α")
    print(f"  |C| = {args.num_classes}")
    print(f"  κ = log({args.num_classes}) = {kappa:.4f}")
    print(f"  α = tanh({kappa:.4f} / 2) = {alpha:.4f}")
    print(f"  → GRTD 权重: {alpha:.4f}, LBTC 权重: {1 - alpha:.4f}")

    # ========== 2. 加载 GRTD / LBTC 结果 ==========
    print(f"\n📊 步骤 2: 加载 GRTD(TDA) / LBTC 结果...")

    grtd_results = load_model_results(dataset_dir, "TDA")
    lbtc_results = load_model_results(dataset_dir, "TDA_lbtc")

    print(f"  ✓ GRTD (TDA):  {len(grtd_results)} 模型")
    print(f"  ✓ LBTC:        {len(lbtc_results)} 模型")

    # ========== 3. 提取层平均分数 ==========
    # 论文 Section 6: GRTD → decoder 层平均, LBTC → encoder 层平均
    print(f"\n📊 步骤 3: 逐模型提取层平均分数 (GRTD←decoder, LBTC←encoder)...")

    model_data: Dict[str, dict] = {}

    for model_name in grtd_results.keys():
        if model_name not in lbtc_results:
            print(f"  [SKIP] {model_name}: LBTC 缺失")
            continue

        print(f"\n  ─── {model_name} ───")

        # GRTD: 取 decoder 层平均
        grtd_score = extract_decoder_avg(grtd_results[model_name], decoder_layer_weights)
        # LBTC: 取 encoder 层平均
        lbtc_score = extract_encoder_avg(lbtc_results[model_name], encoder_layer_weights)

        if grtd_score is None or lbtc_score is None:
            print(f"    [MISS] GRTD(dec)={grtd_score}, LBTC(enc)={lbtc_score} -> 跳过")
            continue

        print(f"    GRTD (decoder avg): {grtd_score:.4f}")
        print(f"    LBTC (encoder avg): {lbtc_score:.4f}")

        model_data[model_name] = {
            "grtd_raw": float(grtd_score),
            "lbtc_raw": float(lbtc_score),
        }

    print(f"\n  ✓ 有效模型: {len(model_data)}")
    if len(model_data) < 2:
        print("  ❌ 有效模型 < 2，无法归一化")
        sys.exit(1)

    # ========== 4. Min-Max 归一化 N(·) ==========
    print(f"\n📊 步骤 4: 跨模型 Min-Max 归一化 N(·)...")

    grtd_scores = [d["grtd_raw"] for d in model_data.values()]
    lbtc_scores = [d["lbtc_raw"] for d in model_data.values()]

    grtd_min, grtd_max = min(grtd_scores), max(grtd_scores)
    lbtc_min, lbtc_max = min(lbtc_scores), max(lbtc_scores)

    grtd_range = (grtd_max - grtd_min) if abs(grtd_max - grtd_min) > 1e-9 else 1.0
    lbtc_range = (lbtc_max - lbtc_min) if abs(lbtc_max - lbtc_min) > 1e-9 else 1.0

    print(f"  GRTD (decoder) 原始范围: [{grtd_min:.4f}, {grtd_max:.4f}]")
    print(f"  LBTC (encoder) 原始范围: [{lbtc_min:.4f}, {lbtc_max:.4f}]")

    print(f"\n  {'模型':30s}  {'GRTD_raw':>10s}  {'GRTD_norm':>10s}  "
          f"{'LBTC_raw':>10s}  {'LBTC_norm':>10s}")
    print(f"  {'-' * 82}")

    for model_name, data in model_data.items():
        data["grtd_norm"] = (data["grtd_raw"] - grtd_min) / grtd_range
        data["lbtc_norm"] = (data["lbtc_raw"] - lbtc_min) / lbtc_range

        print(f"  {model_name:30s}  {data['grtd_raw']:10.4f}  {data['grtd_norm']:10.4f}  "
              f"{data['lbtc_raw']:10.4f}  {data['lbtc_norm']:10.4f}")

    # ========== 5. Adaptive Fusion ==========
    # 论文 Eq. 6: S_ϕ = α · N(T^GRTD) + (1-α) · N(T^LBTC)
    print(f"\n📊 步骤 5: Adaptive Fusion")
    print(f"  S_ϕ = {alpha:.4f} × N(GRTD) + {1 - alpha:.4f} × N(LBTC)")
    print(f"\n  {'模型':30s}  {'α':>8s}  {'GRTD_norm':>10s}  {'LBTC_norm':>10s}  {'S_ϕ':>10s}")
    print(f"  {'-' * 78}")

    for model_name, data in model_data.items():
        fused = alpha * data["grtd_norm"] + (1.0 - alpha) * data["lbtc_norm"]
        data["fused_score"] = fused

        print(f"  {model_name:30s}  {alpha:8.4f}  {data['grtd_norm']:10.4f}  "
              f"{data['lbtc_norm']:10.4f}  {fused:10.4f}")

    # ========== 6. 排序 ==========
    print(f"\n📊 步骤 6: 按 S_ϕ 排序（越大越好）...")
    print("=" * 70)

    sorted_models = sorted(model_data.items(),
                           key=lambda x: x[1]["fused_score"], reverse=True)

    for rank, (name, data) in enumerate(sorted_models, 1):
        print(f"  {rank:2d}. {name:30s}  S_ϕ = {data['fused_score']:.4f}")

    # ========== 7. Weighted Kendall τ*,w ==========
    print(f"\n📊 步骤 7: Weighted Kendall τ*,w (论文 Eq. 7)...")

    model_scores = {name: data["fused_score"] for name, data in model_data.items()}

    try:
        tau, n = compute_weighted_kendall_with_openmind(
            model_scores, args.openmind_arch, args.openmind_col
        )
        print(f"  Weighted Kendall τ*,w = {tau:.4f} (n={n})")
    except Exception as e:
        print(f"  [WARN] Kendall τ 计算失败: {e}")

    # ========== 8. 保存 ==========
    if args.out_csv:
        print(f"\n📊 保存: {args.out_csv}")
        rows = []
        for name, data in sorted_models:
            rows.append({
                "model_name": name,
                "grtd_raw": data["grtd_raw"],
                "lbtc_raw": data["lbtc_raw"],
                "grtd_norm": data["grtd_norm"],
                "lbtc_norm": data["lbtc_norm"],
                "fused_score": data["fused_score"],
                "alpha": alpha,
                "kappa": kappa,
                "weight_profile": args.weight_profile,
            })
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
