import argparse
import json
import re
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.metric_registry import get_metric, METRIC_REGISTRY
from utils.sliding_window_sampling import ms_sliding_window_sampling, GLOBAL_KEY
from utils.layer_resolver import resolve_layers_and_sample_num
from utils.get_model import get_model, reinit_decoder_modules
from utils.utility import get_loader, setup_seed, load_pretrained_model
from utils.precomputed_features import load_precomputed_store, describe, layer_sort_key  # PKL mode

# 导入分类任务专用模块
try:
    from utils.sampling_classification import batch_classification_sampling
    from utils.classification import get_classification_loader
    CLASSIFICATION_SUPPORT = True
except ImportError:
    CLASSIFICATION_SUPPORT = False
    print("[WARN] Classification support not available. Install sampling_classification.py and utils_classification.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nat_key(s: str):
    """Natural-sort key so that 'stages.10' sorts after 'stages.2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _avg_last_k(items: list, k: int = 5) -> float | None:
    """Average the scores of the last *k* layers (by natural order)."""
    if not items:
        return None
    items = sorted(items, key=lambda pair: layer_sort_key(pair[0]))
    vals = [v for _, v in items[-k:]]
    return float(np.mean(vals)) if vals else None


def compute_encoder_decoder_last(layer_scores: dict, layers: list, last_k: int = 5, task_type: str = "segmentation"):
    """Split layers into encoder / decoder(head) groups and compute averages.

    Returns two values:
      - encoder_avg: mean score across **all** encoder layers
      - decoder_avg: mean score across **all** decoder (or cls_head) layers

    This ensures the kendall correlation is computed on the full encoder/decoder
    averages, matching the paper's description.
    """
    task_type = (task_type or "segmentation").lower().strip()

    enc, dec = [], []
    for layer in layers:
        score = layer_scores.get(layer)
        if score is None:
            continue
        if "encoder" in layer or "eva" in layer or "down_projection" in layer:
            enc.append((layer, score))
        if task_type == "classification":
            if "cls_head" in layer:
                dec.append((layer, score))
        else:
            if "decoder" in layer or "up_projection" in layer:
                dec.append((layer, score))

    # Average ALL encoder / decoder layers (not just last K)
    enc_avg = float(np.mean([v for _, v in enc])) if enc else None
    dec_avg = float(np.mean([v for _, v in dec])) if dec else None
    return enc_avg, dec_avg


def _parse_metric_params(tokens: list[str]) -> dict:
    """Parse 'key=value' CLI tokens into a dict, auto-casting numbers and bools."""
    out = {}
    for token in tokens:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        # auto-cast
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        out[k] = v
    return out


def _decoder_loading_options_from_cfg_and_args(configs: dict, args):
    policy = args.decoder_loading_policy or configs.get("decoder_loading_policy", None)
    reconstruction_methods = args.reconstruction_methods or configs.get("reconstruction_methods", None)
    contrastive_methods = args.contrastive_methods or configs.get("contrastive_methods", None)
    unknown_policy = args.unknown_decoder_policy or configs.get("unknown_decoder_policy", "encoder_only")
    return {
        "decoder_loading_policy": policy,
        "reconstruction_methods": reconstruction_methods,
        "contrastive_methods": contrastive_methods,
        "unknown_policy": unknown_policy,
    }


# ---------------------------------------------------------------------------
# core evaluation loop for SEGMENTATION
# ---------------------------------------------------------------------------


def evaluate_segmentation(
    configs: dict,
    test_loader,
    model,
    metric,
    model_name: str,
    ckpt_path: str = "",
    layers: list = None,
    sample_num: dict = None,
) -> tuple[dict, float | None, list[str]]:
    """
    分割任务评估
    """
    metric.reset()
    model.eval()
    
    with torch.no_grad():
        for case_idx, val_data in enumerate(test_loader):
            print(f"  case {case_idx} ...", end=" ", flush=True)

            try:
                data = val_data["data"].permute(0, 1, 4, 3, 2).cuda()
                seg = val_data["seg"].permute(0, 1, 4, 3, 2).cuda()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[OOM – skipped]")
                    torch.cuda.empty_cache()
                    continue
                raise

            seg = F.interpolate(seg.float(), size=data.shape[2:], mode="nearest").long()

            feature_dict = ms_sliding_window_sampling(
                layers,
                sample_num,
                data,
                seg,
                [configs["roi_z"], configs["roi_y"], configs["roi_x"]],
                configs["sw_batch_size"],
                model,
                overlap=configs.get("infer_overlap", 0.0),
                mode=configs.get("window_mode", "constant"),
                fv_sample_num=configs.get("fv_sample_num", None),
                bg_sample_num=configs.get("bg_sample_num", 0),
                boundary_sample_num=configs.get("boundary_sample_num", 0),
                seed=int(configs.get("sampling_seed", 0)),
            )

            metric.begin_case(case_idx)
            for layer in layers:
                metric.accumulate(layer, feature_dict[layer])

            print("done")

    # --- compute per-layer scores ---
    layer_scores: dict[str, float | None] = {}
    for layer in layers:
        score = metric.compute(layer)
        layer_scores[layer] = score
        tag = f"{score:.6f}" if score is not None else "None"
        print(f"  [{layer}]  {metric.name} = {tag}")

    overall_score = metric.aggregate_overall(layer_scores, layers)
    print(f"  [overall]  {metric.name} = {overall_score}")

    return layer_scores, overall_score, layers


# ---------------------------------------------------------------------------
# core evaluation loop for CLASSIFICATION
# ---------------------------------------------------------------------------


def evaluate_classification(
    configs: dict,
    test_loader,
    model,
    metric,
    model_name: str,
    ckpt_path: str = "",
    layers: list = None,
    sample_num: dict = None,
) -> tuple[dict, float | None, list[str]]:
    """
    分类任务评估
    
    关键差异:
    - 使用 batch_classification_sampling 而非 ms_sliding_window_sampling
    - 不需要seg标签,使用图像级label
    - 按图像类别分组特征
    """
    if not CLASSIFICATION_SUPPORT:
        raise RuntimeError(
            "Classification support not available. "
            "Please ensure sampling_classification.py and utils_classification.py are in the path."
        )
    
    metric.reset()
    model.eval()
    
    # 使用分类任务专用的批量采样函数
    print("📊 开始分类任务特征采样...")
    feature_dict = batch_classification_sampling(
        layers=layers,
        sample_num=sample_num,
        data_loader=test_loader,
        model=model,
        roi_size=[configs["roi_z"], configs["roi_y"], configs["roi_x"]],
        sw_batch_size=configs["sw_batch_size"],
        overlap=configs.get("infer_overlap", 0.0),
        mode=configs.get("window_mode", "constant"),
        fv_sample_num=configs.get("fv_sample_num", None),
        seed=int(configs.get("sampling_seed", 0)),
    )
    
    print("✅ 特征采样完成")
    
    # 统计每个类别的样本数
    print("\n📊 类别特征统计:")
    for layer in layers:
        print(f"  [{layer}]")
        for key, feats in feature_dict[layer].items():
            if key == GLOBAL_KEY:
                print(f"    {key}: {feats.shape[0]} 个全局特征点")
            else:
                print(f"    类别 {key}: {feats.shape[0]} 个特征点")
    
    # 计算每层的度量分数
    # 注意：对于分类任务，所有图像已经在batch_classification_sampling中处理完毕
    # 这里不需要按case循环，直接对整个feature_dict计算
    metric.begin_case(0)  # 整个数据集作为一个case
    for layer in layers:
        metric.accumulate(layer, feature_dict[layer])

    # --- compute per-layer scores ---
    layer_scores: dict[str, float | None] = {}
    for layer in layers:
        score = metric.compute(layer)
        layer_scores[layer] = score
        tag = f"{score:.6f}" if score is not None else "None"
        print(f"  [{layer}]  {metric.name} = {tag}")

    overall_score = metric.aggregate_overall(layer_scores, layers)
    print(f"  [overall]  {metric.name} = {overall_score}")

    return layer_scores, overall_score, layers


# ---------------------------------------------------------------------------
# PKL mode evaluation (works for both tasks)
# ---------------------------------------------------------------------------


def evaluate_pkl(
    configs: dict,
    model,
    metric,
    model_name: str,
    ckpt_path: str = "",
) -> tuple[dict, float | None, list[str]]:
    """
    PKL模式评估 (适用于分割和分类任务)

    - segmentation: case-by-case accumulate（与分割采样一致）
    - classification (recommended): 先聚合全数据集，再对每层只 accumulate 一次
      这样才能对齐 OpenMind 的分类相关性评估（每个模型在一个目标数据集对应一个分数）。
    """
    metric.reset()

    store = load_precomputed_store(configs, model_name=model_name, ckpt_path=ckpt_path)
    print("[PKL mode] loaded precomputed features: ", describe(store.store))

    if isinstance(configs.get("layers_override", None), list):
        layers = list(configs["layers_override"])
    else:
        layers = store.layers()

    case_ids = store.case_ids()
    if not case_ids:
        raise RuntimeError("No cases found in PKL store")

    task_type = str(configs.get("task_type", "segmentation")).lower().strip()

    # Decide whether to aggregate classification PKLs into a single pseudo-case.
    # For TDA-family this is needed; for CCFV/LogME/GBC/LEEP we keep case-by-case to preserve per-image structure.
    metric_name = str(getattr(metric, "name", "")).lower()
    default_agg = metric_name.startswith("tda")
    pkl_cls_aggregate = bool(configs.get("pkl_cls_aggregate", default_agg))

    # Heuristic: TDA family treats label==0 specially in some implementations; shift class labels by +1 for safety
    shift_cls_labels = bool(configs.get("cls_label_shift", metric_name.startswith("tda")))

    def _normalize_cls_fd(fd: dict) -> dict:
        # drop background keys for classification; shift class labels if requested
        out = {}
        for k, v in fd.items():
            if k == GLOBAL_KEY:
                out[GLOBAL_KEY] = v
                continue
            if isinstance(k, int):
                if k == -1:
                    # background: ignore for classification
                    continue
                kk = k + 1 if shift_cls_labels else k
                out[kk] = v
        # if no GLOBAL_KEY, create from concatenated class vectors
        if GLOBAL_KEY not in out:
            arrs = [v for k, v in out.items() if isinstance(k, int) and v is not None and len(v) > 0]
            if arrs:
                out[GLOBAL_KEY] = np.concatenate(arrs, axis=0)
        return out

    if task_type == "classification" and pkl_cls_aggregate:
        print("[PKL cls] aggregate all cases into one dataset-level pseudo-case ...")
        # For each layer, aggregate per-class arrays across all cases
        metric.begin_case(0)
        for layer in layers:
            per_cls = {}   # int label -> list of (n_i, C)
            globals_ = []  # list of (n_i, C)
            missing = 0
            for case_id in case_ids:
                fd = store.get(case_id, layer)
                if fd is None:
                    missing += 1
                    continue
                fd = _normalize_cls_fd(fd)
                if GLOBAL_KEY in fd and fd[GLOBAL_KEY] is not None and len(fd[GLOBAL_KEY]) > 0:
                    globals_.append(fd[GLOBAL_KEY])
                for k, v in fd.items():
                    if k == GLOBAL_KEY:
                        continue
                    if isinstance(k, int) and v is not None and len(v) > 0:
                        per_cls.setdefault(int(k), []).append(v)

            if not per_cls and not globals_:
                # nothing to accumulate
                continue

            agg_fd = {}
            for k, vs in per_cls.items():
                agg_fd[int(k)] = np.concatenate(vs, axis=0)
            if globals_:
                agg_fd[GLOBAL_KEY] = np.concatenate(globals_, axis=0)
            else:
                # fallback global
                all_vs = [agg_fd[k] for k in agg_fd if isinstance(k, int)]
                if all_vs:
                    agg_fd[GLOBAL_KEY] = np.concatenate(all_vs, axis=0)

            metric.accumulate(layer, agg_fd)

            if missing:
                print(f"  layer {layer}: aggregated (missing_cases={missing})")
        # compute scores
        layer_scores: dict[str, float | None] = {}
        for layer in layers:
            layer_scores[layer] = metric.compute(layer)

        overall_score = metric.aggregate_overall(layer_scores, layers)
        print(f"  [overall]  {metric.name} = {overall_score}")
        return layer_scores, overall_score, layers

    # ---- default: segmentation / legacy cls case-by-case ----
    for case_idx, case_id in enumerate(case_ids):
        print(f"  case {case_idx} ({case_id}) ...", end=" ", flush=True)
        metric.begin_case(case_idx)
        missing = 0
        for layer in layers:
            fd = store.get(case_id, layer)
            if fd is None:
                missing += 1
                continue
            if task_type == "classification":
                fd = _normalize_cls_fd(fd)
            metric.accumulate(layer, fd)
        if missing:
            print(f"done  (missing_layers={missing})")
        else:
            print("done")

    layer_scores: dict[str, float | None] = {}
    for layer in layers:
        layer_scores[layer] = metric.compute(layer)

    overall_score = metric.aggregate_overall(layer_scores, layers)
    print(f"  [overall]  {metric.name} = {overall_score}")
    return layer_scores, overall_score, layers


def evaluate(
    configs: dict,
    test_loader,
    model,
    metric,
    model_name: str,
    ckpt_path: str = "",
) -> tuple[dict, float | None, list[str]]:
    """
    统一评估函数，根据配置自动选择评估模式
    
    支持三种模式:
    1. feature_source="pkl": 从预计算的PKL文件加载特征
    2. task_type="classification": 分类任务 (从模型提取特征)
    3. task_type="segmentation" (默认): 分割任务 (从模型提取特征)
    """
    feature_source = str(configs.get("feature_source", "model")).lower().strip()
    task_type = str(configs.get("task_type", "segmentation")).lower().strip()
    
    # 解析layers和sample_num
    if feature_source != "pkl":
        layers, sample_num = resolve_layers_and_sample_num(configs, model)
    else:
        # PKL模式下,layers从store中读取
        layers = None
        sample_num = None
    
    # 1. PKL模式 (适用于两种任务)
    if feature_source == "pkl":
        print(f"🔄 使用 PKL 模式")
        return evaluate_pkl(configs, model, metric, model_name, ckpt_path)
    
    # 2. 分类任务模式
    elif task_type == "classification":
        print(f"📊 使用分类任务模式")
        return evaluate_classification(
            configs, test_loader, model, metric, model_name, ckpt_path, layers, sample_num
        )
    
    # 3. 分割任务模式 (默认)
    else:
        print(f"🔬 使用分割任务模式")
        return evaluate_segmentation(
            configs, test_loader, model, metric, model_name, ckpt_path, layers, sample_num
        )


def main():
    parser = argparse.ArgumentParser(description="Unified transferability metric evaluation (Segmentation + Classification)")

    # required
    parser.add_argument("--cfg", type=str, required=True, help="Config JSON (layers, model, ROI, task_type, …)")
    # NOTE: In PKL mode we do not need any dataset forward pass.
    # Keep this arg for backward compatibility (segmentation/classification modes).
    parser.add_argument("--data_list_file", type=str, default="", help="Data-list JSON with 'val' split (ignored in PKL mode)")
    parser.add_argument("--metric", type=str, required=True, choices=list(METRIC_REGISTRY.keys()),
                        help="Which metric to compute")

    # model
    parser.add_argument("--model_path", type=str, default="",
                        help="Checkpoint path.  Empty string  →  random init (for baselines)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name used in output filenames")

    parser.add_argument(
        "--decoder_loading_policy",
        type=str,
        default="",
        help="decoder loading strategy: encoder_only | full | openmind_hybrid",
    )
    parser.add_argument(
        "--reconstruction_methods",
        type=str,
        default="",
        help="comma-separated methods treated as reconstruction in openmind_hybrid",
    )
    parser.add_argument(
        "--contrastive_methods",
        type=str,
        default="",
        help="comma-separated methods treated as contrastive in openmind_hybrid",
    )
    parser.add_argument(
        "--unknown_decoder_policy",
        type=str,
        default="",
        help="fallback when method cannot be inferred in openmind_hybrid: encoder_only/full",
    )

    # metric tunables  (also readable from cfg JSON key 'metric_params')
    parser.add_argument("--metric_params", nargs="*", default=[],
                        help="Override metric hyper-params as  key=value  tokens")

    # output
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to write result JSON")

    # PKL overrides (useful when running from pre-saved features)
    parser.add_argument("--pkl_path", type=str, default="",
                        help="Override PKL feature path (file or directory). If set, forces feature_source=pkl")
    parser.add_argument("--pkl_root", type=str, default="",
                        help="Override PKL root directory. If set, forces feature_source=pkl")
    parser.add_argument("--pkl_dataset", type=str, default="",
                        help="Optional dataset subdir under pkl_root (e.g., HNT).")

    args = parser.parse_args()
    setup_seed(0)

    # ---- load config ----
    configs = json.load(open(args.cfg, "r"))
    load_opts = _decoder_loading_options_from_cfg_and_args(configs, args)

    # ---- PKL CLI overrides (force PKL-only path resolution) ----
    if args.pkl_path:
        configs["feature_source"] = "pkl"
        configs["pkl_path"] = args.pkl_path
    if args.pkl_root:
        configs["feature_source"] = "pkl"
        configs["pkl_root"] = args.pkl_root
    if args.pkl_dataset:
        configs["pkl_dataset"] = args.pkl_dataset
    
    # 检测任务类型
    task_type = str(configs.get("task_type", "segmentation")).lower().strip()
    feature_source = str(configs.get("feature_source", "model")).lower().strip()
    
    print(f"\n{'=' * 70}")
    print(f"  🎯 任务类型: {task_type.upper()}")
    print(f"  📦 特征来源: {feature_source.upper()}")
    print(f"{'=' * 70}\n")

    # ---- merge metric params: config < CLI ----
    cfg_params = configs.get("metric_params", {})
    cli_params = _parse_metric_params(args.metric_params)
    merged_params = {**cfg_params, **cli_params}

    print(f"metric       : {args.metric}")
    print(f"metric_params: {merged_params}")
    if load_opts["decoder_loading_policy"]:
        print(
            "decoder_load : "
            f"policy={load_opts['decoder_loading_policy']} | "
            f"recon={load_opts['reconstruction_methods'] or 'default'} | "
            f"contrastive={load_opts['contrastive_methods'] or 'default'} | "
            f"unknown={load_opts['unknown_policy']}"
        )

    # ---- instantiate metric ----
    metric = get_metric(args.metric, **merged_params)

    # ---- LBTC paper-faithful sampling defaults ----
    # LBTC needs spatial GT-mask boundary anchors and enough background/context
    # samples. If the config forgets these knobs, enable the paper defaults for
    # tda_lbtc in segmentation mode. Explicit positive config values
    # are preserved.
    def _has_positive_budget(v) -> bool:
        if isinstance(v, dict):
            for vv in v.values():
                try:
                    if int(vv) > 0:
                        return True
                except Exception:
                    continue
            return False
        try:
            return int(v) > 0
        except Exception:
            return False

    metric_key = str(args.metric).lower().strip()
    if task_type == "segmentation":
        # Paper-faithful background sampling defaults (Appendix B):
        #   - GRTD/TDA, GBC and LogME keep background voxels as class 0
        #     with a 2560-voxel budget per case/layer.
        #   - LEEP excludes background voxels.
        #   - LBTC also needs background/context around GT boundary anchors.
        if metric_key in {"tda", "grtd", "gbc", "logme"}:
            if not _has_positive_budget(configs.get("bg_sample_num", 0)):
                configs["bg_sample_num"] = 2560
                print(f"[{metric_key.upper()}] bg_sample_num not set; using 2560 to retain background as class 0")
        elif metric_key == "leep":
            if configs.get("bg_sample_num", 0) != 0:
                print("[LEEP] forcing bg_sample_num=0 (background excluded for LEEP)")
            configs["bg_sample_num"] = 0

    if task_type == "segmentation" and metric_key == "tda_lbtc":
        default_boundary_num = int(
            merged_params.get(
                "num_boundary_patches",
                merged_params.get("lbtc_num_boundary_patches", 50),
            )
        )
        if not _has_positive_budget(configs.get("boundary_sample_num", 0)):
            configs["boundary_sample_num"] = default_boundary_num
            print(f"[LBTC] boundary_sample_num not set; using {default_boundary_num} for spatial GT-boundary anchors")
        if not _has_positive_budget(configs.get("bg_sample_num", 0)):
            configs["bg_sample_num"] = 2560
            print("[LBTC] bg_sample_num not set; using 2560 for local boundary context")

    # ---- data + model ----
    model_name = args.model_name or (Path(args.model_path).stem if args.model_path else "random_init")
    
    if feature_source == "pkl":
        test_loader = None
        model = get_model(args, configs).cuda()
        if args.model_path:
            model = load_pretrained_model(
                args.model_path,
                model,
                model_name=model_name,
                decoder_loading_policy=load_opts["decoder_loading_policy"] or None,
                reconstruction_methods=load_opts["reconstruction_methods"] or None,
                contrastive_methods=load_opts["contrastive_methods"] or None,
                unknown_policy=load_opts["unknown_policy"],
            )
        model = reinit_decoder_modules(model, configs)
    else:
        if not args.data_list_file:
            raise RuntimeError("--data_list_file is required when feature_source != 'model'")
        # 根据任务类型选择数据加载器
        if task_type == "classification":
            if not CLASSIFICATION_SUPPORT:
                raise RuntimeError(
                    "分类任务支持不可用。请确保以下文件在路径中:\n"
                    "  - sampling_classification.py\n"
                    "  - utils_classification.py"
                )
            print("📊 使用分类任务数据加载器")
            test_loader = get_classification_loader(args, configs)
        else:
            print("🔬 使用分割任务数据加载器")
            test_loader = get_loader(args, configs)

        model = get_model(args, configs).cuda()
        if args.model_path:
            model = load_pretrained_model(
                args.model_path,
                model,
                model_name=model_name,
                decoder_loading_policy=load_opts["decoder_loading_policy"] or None,
                reconstruction_methods=load_opts["reconstruction_methods"] or None,
                contrastive_methods=load_opts["contrastive_methods"] or None,
                unknown_policy=load_opts["unknown_policy"],
            )
        else:
            print("[WARN] no checkpoint – using random initialisation")

        model = reinit_decoder_modules(model, configs)

    # ---- evaluate ----
    print(f"\n{'=' * 60}")
    print(f"  {model_name}  |  metric = {args.metric}")
    print(f"{'=' * 60}")

    layer_scores, overall_score, layers = evaluate(
        configs, test_loader, model, metric, model_name=model_name, ckpt_path=args.model_path
    )
    
    encdec_last_k = int(configs.get("encdec_last_k", 5 if task_type=="classification" else (1 if feature_source=="pkl" else 5)))
    enc_last, dec_last = compute_encoder_decoder_last(layer_scores, layers, last_k=encdec_last_k, task_type=task_type)

    # ---- persist ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": model_name,
        "task_type": task_type,
        "metric": args.metric,
        "decoder_loading_policy": load_opts["decoder_loading_policy"] or "encoder_only(legacy)",
        "overall_score": overall_score,
        "layer_scores": layer_scores,
        "layers": layers,
        "encoder_last": enc_last,
        "decoder_last": dec_last,
        "metric_params": merged_params,
    }

    json_path = output_dir / f"{model_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- persist per-layer scores (one row per layer) ----
    # 你要的“每个模型每一层的记录”：这里为每个模型单独落盘一个 layer_scores CSV
    layer_csv = output_dir / f"{model_name}_layer_scores.csv"
    with open(layer_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", f"{args.metric}", "is_encoder", "is_decoder"])
        for layer in layers:
            s = layer_scores.get(layer, None)
            layer_str = str(layer)
            # encoder: "encoder.*", "eva.*", "down_projection.*"
            is_enc = int(
                "encoder" in layer_str
                or "eva" in layer_str
                or "down_projection" in layer_str
            )
            # decoder: "decoder.*", "up_projection.*", "cls_head.*"
            if task_type == "classification":
                is_dec = int("cls_head" in layer_str)
            else:
                is_dec = int(
                    "decoder" in layer_str
                    or "up_projection" in layer_str
                )
            w.writerow([
                layer,
                "" if s is None else float(s),
                is_enc,
                is_dec,
            ])

    print(f"\nresult JSON : {json_path}")
    print(f"layerwise CSV: {layer_csv}")
    print(f"overall {args.metric}: {overall_score}")


if __name__ == "__main__":
    main()
