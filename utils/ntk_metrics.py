"""
ntk_metrics.py
==============
Strict Jacobian-based NTK utilities for the setting:

    Frozen pre-trained encoder + randomly initialized decoder

Important
---------
The old detached-feature implementation based on K = X X^T is **not** a strict
NTK. This file now provides utilities for computing a true empirical Jacobian
kernel with respect to decoder parameters only.

Practical definition
--------------------
For decoder stage l, let h_l(x, p) ∈ R^{C_l} be the feature vector at sampled
spatial point p. We define a fixed random scalar readout

    g_l(x, p) = w_l^T h_l(x, p),

where w_l is sampled once and then frozen. The strict empirical NTK is

    Θ_l[i, j] = < ∂g_i / ∂θ_dec , ∂g_j / ∂θ_dec >

where θ_dec contains decoder parameters only.

This is a *true Jacobian NTK* for the chosen readout. It is considerably more
expensive than feature-Gram proxies, so the intended regime is:
    - a small number of cases,
    - a small number of sampled points per case,
    - decoder stages only.

Why the MetricComputer path is disabled
---------------------------------------
The project's MetricComputer API only sees detached numpy feature dictionaries.
That interface cannot recover Jacobians with respect to decoder parameters.
Therefore the MetricComputer-like NTK classes below are now explicit stubs that
raise a helpful error, and strict NTK should be run through the dedicated
runner script.

Exports used by the strict runner
---------------------------------
- run_strict_ntk_analysis(...)
- canonical_layer_name(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class _StrictNTKNotAvailableInDetachedPipeline:
    """Compatibility stub so old registry imports do not break."""

    name = "strict_ntk_requires_runner"

    def __init__(self, *args, **kwargs):
        self._msg = (
            "Strict Jacobian NTK is no longer available through the detached "
            "feature MetricComputer pipeline. Please use the dedicated strict "
            "runner (run_strict_ntk_experiments.py / run_ntk_experiments.sh), "
            "which has access to the model, raw inputs, and decoder parameters."
        )

    def reset(self):
        raise RuntimeError(self._msg)

    def begin_case(self, case_idx: int):
        raise RuntimeError(self._msg)

    def accumulate(self, layer: str, feature_dict: dict):
        raise RuntimeError(self._msg)

    def compute(self, layer: str):
        raise RuntimeError(self._msg)

    def aggregate_overall(self, layer_scores, layers):
        raise RuntimeError(self._msg)


class NTKCondMetric(_StrictNTKNotAvailableInDetachedPipeline):
    name = "ntk_cond"


class NTKTraceMetric(_StrictNTKNotAvailableInDetachedPipeline):
    name = "ntk_trace"


class NTKLayerwiseMetric(_StrictNTKNotAvailableInDetachedPipeline):
    name = "ntk_layerwise"


class NTKSpectrumMetric(_StrictNTKNotAvailableInDetachedPipeline):
    name = "ntk_spectrum"


@dataclass
class StrictNTKSettings:
    max_cases: int = 3
    fg_points_per_case: int = 4
    bg_points_per_case: int = 4
    ntk_reg: float = 1e-6
    spectrum_topk: int = 8
    random_readout_seed: int = 0
    crop_mode: str = "center"
    include_background_in_sampling: bool = True
    ignore_background_for_spectrum: bool = True


def canonical_layer_name(layer_name: str) -> str:
    return str(layer_name).replace(".", "_")


def _center_crop_or_pad_3d(x: torch.Tensor, roi: Sequence[int]) -> torch.Tensor:
    rz, ry, rx = [int(v) for v in roi]
    _, _, Z, Y, X = x.shape

    pad_z = max(0, rz - Z)
    pad_y = max(0, ry - Y)
    pad_x = max(0, rx - X)

    if pad_z or pad_y or pad_x:
        x = F.pad(
            x,
            [
                pad_x // 2, pad_x - pad_x // 2,
                pad_y // 2, pad_y - pad_y // 2,
                pad_z // 2, pad_z - pad_z // 2,
            ],
            value=0,
        )
        _, _, Z, Y, X = x.shape

    sz = max(0, (Z - rz) // 2)
    sy = max(0, (Y - ry) // 2)
    sx = max(0, (X - rx) // 2)

    return x[:, :, sz:sz + rz, sy:sy + ry, sx:sx + rx]


def prepare_case_patch(
    data: torch.Tensor,
    seg: torch.Tensor,
    roi: Sequence[int],
    crop_mode: str = "center",
) -> Tuple[torch.Tensor, torch.Tensor]:
    crop_mode = str(crop_mode).lower().strip()
    if crop_mode != "center":
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}; only 'center' is supported")

    data_p = _center_crop_or_pad_3d(data, roi)
    seg_p = _center_crop_or_pad_3d(seg, roi)
    return data_p, seg_p


def select_decoder_param_items(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    items: List[Tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if "decoder" in name and p.requires_grad:
            items.append((name, p))
    if not items:
        raise RuntimeError("No decoder parameters found. Check the model structure.")
    return items


def freeze_encoder_unfreeze_decoder(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        if "decoder" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def _capture_stage_output(model: torch.nn.Module, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
    holder = {}

    def _hook(_m, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        holder["feat"] = t

    handle = model.decoder.stages[stage_idx].register_forward_hook(_hook)
    try:
        _ = model(x)
    finally:
        handle.remove()

    if "feat" not in holder:
        raise RuntimeError(f"Failed to capture decoder stage {stage_idx}")
    feat = holder["feat"]
    if not isinstance(feat, torch.Tensor) or feat.ndim != 5:
        raise RuntimeError(f"Captured output of stage {stage_idx} is invalid")
    return feat


def _sample_stage_points(
    seg_case: np.ndarray,
    feat_shape_zyx: Sequence[int],
    fg_points_per_case: int,
    bg_points_per_case: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    fZ, fY, fX = [int(v) for v in feat_shape_zyx]

    seg_t = torch.from_numpy(seg_case.astype(np.float32))[None, None]
    seg_r = F.interpolate(seg_t, size=(fZ, fY, fX), mode="nearest")[0, 0].numpy().astype(np.int32)

    coords_list = []
    labels_list = []

    fg_classes = [int(c) for c in np.unique(seg_r) if int(c) != 0]
    if fg_classes and fg_points_per_case > 0:
        per_fg = max(1, fg_points_per_case // len(fg_classes))
        for c in fg_classes:
            idx = np.argwhere(seg_r == c)
            if idx.size == 0:
                continue
            k = min(per_fg, len(idx))
            sel = rng.choice(len(idx), size=k, replace=False)
            coords_list.append(idx[sel])
            labels_list.append(np.full(k, c, dtype=np.int32))

    if bg_points_per_case > 0:
        bg_idx = np.argwhere(seg_r == 0)
        if len(bg_idx) > 0:
            k = min(bg_points_per_case, len(bg_idx))
            sel = rng.choice(len(bg_idx), size=k, replace=False)
            coords_list.append(bg_idx[sel])
            labels_list.append(np.zeros(k, dtype=np.int32))

    if not coords_list:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    coords = np.concatenate(coords_list, axis=0).astype(np.int64)
    labels = np.concatenate(labels_list, axis=0).astype(np.int32)
    return coords, labels


def _flatten_grads_to_cpu(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32, device="cpu"))
        else:
            parts.append(g.detach().reshape(-1).to(device="cpu", dtype=torch.float32))
    return torch.cat(parts, dim=0)


def _build_fixed_readout(
    channels: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(channels).astype(np.float32)
    w /= max(np.linalg.norm(w), 1e-12)
    return torch.tensor(w, device=device, dtype=dtype)


def _compute_stage_grad_rows(
    model: torch.nn.Module,
    stage_feat: torch.Tensor,
    seg_case: np.ndarray,
    decoder_params: Sequence[torch.nn.Parameter],
    stage_idx: int,
    case_idx: int,
    settings: StrictNTKSettings,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], dict]:
    _, C, fZ, fY, fX = stage_feat.shape
    rng = np.random.default_rng(settings.random_readout_seed + 1009 * case_idx + 97 * stage_idx)

    coords, labels = _sample_stage_points(
        seg_case=seg_case,
        feat_shape_zyx=(fZ, fY, fX),
        fg_points_per_case=settings.fg_points_per_case,
        bg_points_per_case=settings.bg_points_per_case,
        rng=rng,
    )
    if len(coords) == 0:
        return None, None, {"n_points": 0}

    z = torch.as_tensor(coords[:, 0], device=stage_feat.device, dtype=torch.long)
    y = torch.as_tensor(coords[:, 1], device=stage_feat.device, dtype=torch.long)
    x = torch.as_tensor(coords[:, 2], device=stage_feat.device, dtype=torch.long)

    H = stage_feat[0, :, z, y, x].permute(1, 0).contiguous()
    w = _build_fixed_readout(
        channels=H.shape[1],
        seed=settings.random_readout_seed + 7919 * stage_idx,
        device=H.device,
        dtype=H.dtype,
    )
    s = H @ w

    rows: List[torch.Tensor] = []
    model.zero_grad(set_to_none=True)

    for i in range(s.shape[0]):
        grads = torch.autograd.grad(
            s[i],
            decoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        row = _flatten_grads_to_cpu(grads, decoder_params)
        rows.append(row)

    G = torch.stack(rows, dim=0)
    meta = {
        "n_points": int(G.shape[0]),
        "n_params": int(G.shape[1]),
        "n_fg_points": int((labels != 0).sum()),
        "n_bg_points": int((labels == 0).sum()),
    }
    return G, labels, meta


def _kernel_from_grad_rows(G: torch.Tensor) -> np.ndarray:
    return (G @ G.T).numpy().astype(np.float64)


def _kernel_condition_stats(K: np.ndarray, reg: float = 1e-6) -> Tuple[float, float, float, float]:
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)
    lam_max = float(np.max(eigvals)) if eigvals.size else 0.0
    lam_min = float(np.min(eigvals[eigvals > 0])) if np.any(eigvals > 0) else float(reg)
    lam_min = max(lam_min, reg)
    kappa = min(lam_max / lam_min if lam_min > 0 else 1e8, 1e8)
    inv_kappa = 1.0 / kappa
    return float(kappa), float(inv_kappa), float(lam_max), float(lam_min)


def _kernel_trace(K: np.ndarray) -> float:
    N = max(1, K.shape[0])
    return float(np.trace(K) / N)


def _kernel_spectrum_alignment(
    K: np.ndarray,
    labels: np.ndarray,
    topk: int = 8,
    ignore_background: bool = True,
) -> Optional[float]:
    y = np.asarray(labels, dtype=np.int32)
    if ignore_background:
        keep = y != 0
        if keep.sum() < 2:
            return None
        y = y[keep]
        K = K[np.ix_(keep, keep)]

    classes = np.unique(y)
    if len(classes) < 2:
        return None

    eigvals, eigvecs = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k = min(int(topk), len(eigvals))
    if k < 1:
        return None

    Y = np.zeros((len(y), len(classes)), dtype=np.float64)
    for ci, c in enumerate(classes):
        Y[y == c, ci] = 1.0
    Y -= Y.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Y, axis=0, keepdims=True)
    Y = Y / np.maximum(norms, 1e-12)

    U = eigvecs[:, :k]
    lam = np.maximum(eigvals[:k], 0.0)
    proj = (U.T @ Y) ** 2
    weighted = (lam[:, None] * proj).sum(axis=0)
    align = float(np.mean(weighted))
    return float(np.log(align + 1e-10))


def run_strict_ntk_analysis(
    model: torch.nn.Module,
    loader,
    roi: Sequence[int],
    settings: StrictNTKSettings,
    device: str = "cuda",
) -> dict:
    model.eval()
    freeze_encoder_unfreeze_decoder(model)

    decoder_param_items = select_decoder_param_items(model)
    decoder_params = [p for _, p in decoder_param_items]

    n_stages = len(model.decoder.stages)
    layer_names = [f"decoder.stages.{i}" for i in range(n_stages)]

    per_layer_grad_rows: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    per_layer_labels: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}
    per_layer_meta: Dict[str, List[dict]] = {ln: [] for ln in layer_names}

    processed_cases = 0
    for case_idx, val_data in enumerate(loader):
        if settings.max_cases > 0 and processed_cases >= settings.max_cases:
            break

        data = val_data["data"].permute(0, 1, 4, 3, 2).to(device)
        seg = val_data["seg"].permute(0, 1, 4, 3, 2).to(device)
        seg = F.interpolate(seg.float(), size=data.shape[2:], mode="nearest").long()

        data_p, seg_p = prepare_case_patch(data, seg, roi=roi, crop_mode=settings.crop_mode)
        seg_np = seg_p[0, 0].detach().cpu().numpy().astype(np.int32)

        print(f"  strict-ntk case {case_idx} ...", flush=True)

        for stage_idx, layer_name in enumerate(layer_names):
            model.zero_grad(set_to_none=True)
            stage_feat = _capture_stage_output(model, stage_idx, data_p)
            G, labels, meta = _compute_stage_grad_rows(
                model=model,
                stage_feat=stage_feat,
                seg_case=seg_np,
                decoder_params=decoder_params,
                stage_idx=stage_idx,
                case_idx=case_idx,
                settings=settings,
            )
            if G is not None and labels is not None and G.shape[0] >= 2:
                per_layer_grad_rows[layer_name].append(G)
                per_layer_labels[layer_name].append(labels)
                per_layer_meta[layer_name].append(meta)

            del stage_feat
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        processed_cases += 1

    layer_scores_cond: Dict[str, Optional[float]] = {}
    layer_scores_trace: Dict[str, Optional[float]] = {}
    layer_scores_spectrum: Dict[str, Optional[float]] = {}
    layerwise_kappa_json: Dict[str, dict] = {}

    for layer_name in layer_names:
        canon = canonical_layer_name(layer_name)
        if not per_layer_grad_rows[layer_name]:
            layer_scores_cond[layer_name] = None
            layer_scores_trace[layer_name] = None
            layer_scores_spectrum[layer_name] = None
            continue

        G = torch.cat(per_layer_grad_rows[layer_name], dim=0)
        labels = np.concatenate(per_layer_labels[layer_name], axis=0)

        K = _kernel_from_grad_rows(G)
        kappa, inv_kappa, lam_max, lam_min = _kernel_condition_stats(K, reg=settings.ntk_reg)
        trace_val = _kernel_trace(K)
        spectrum_val = _kernel_spectrum_alignment(
            K,
            labels,
            topk=settings.spectrum_topk,
            ignore_background=settings.ignore_background_for_spectrum,
        )

        layer_scores_cond[layer_name] = inv_kappa
        layer_scores_trace[layer_name] = trace_val
        layer_scores_spectrum[layer_name] = spectrum_val

        total_points = int(sum(m["n_points"] for m in per_layer_meta[layer_name]))
        total_fg = int(sum(m["n_fg_points"] for m in per_layer_meta[layer_name]))
        total_bg = int(sum(m["n_bg_points"] for m in per_layer_meta[layer_name]))

        layerwise_kappa_json[canon] = {
            "original_layer_name": layer_name,
            "kappa": kappa,
            "lambda_max": lam_max,
            "lambda_min": lam_min,
            "inv_kappa": inv_kappa,
            "trace": trace_val,
            "spectrum_alignment": spectrum_val,
            "is_decoder": True,
            "is_encoder": False,
            "n_points": total_points,
            "n_fg_points": total_fg,
            "n_bg_points": total_bg,
        }

        print(
            f"    [{layer_name}]  inv_kappa={inv_kappa:.6e}  "
            f"trace={trace_val:.6e}  spectrum={spectrum_val}"
        )

    def _decoder_avg(layer_scores: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [layer_scores[l] for l in layer_names if layer_scores.get(l) is not None]
        return float(np.mean(vals)) if vals else None

    results = {
        "processed_cases": processed_cases,
        "settings": {
            "max_cases": settings.max_cases,
            "fg_points_per_case": settings.fg_points_per_case,
            "bg_points_per_case": settings.bg_points_per_case,
            "ntk_reg": settings.ntk_reg,
            "spectrum_topk": settings.spectrum_topk,
            "random_readout_seed": settings.random_readout_seed,
            "crop_mode": settings.crop_mode,
            "ignore_background_for_spectrum": settings.ignore_background_for_spectrum,
        },
        "layers": layer_names,
        "metrics": {
            "ntk_cond": {
                "layer_scores": layer_scores_cond,
                "overall_score": _decoder_avg(layer_scores_cond),
                "decoder_last": _decoder_avg(layer_scores_cond),
            },
            "ntk_trace": {
                "layer_scores": layer_scores_trace,
                "overall_score": _decoder_avg(layer_scores_trace),
                "decoder_last": _decoder_avg(layer_scores_trace),
            },
            "ntk_spectrum": {
                "layer_scores": layer_scores_spectrum,
                "overall_score": _decoder_avg(layer_scores_spectrum),
                "decoder_last": _decoder_avg(layer_scores_spectrum),
            },
        },
        "layerwise_kappa_json": layerwise_kappa_json,
    }
    return results
