import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    fall_back_tuple,
    look_up_option
)

GLOBAL_KEY = "__global__"         # global random samples for Fv
BOUNDARY_KEY = "__boundary__"     # spatially-detected boundary voxel features
BOUNDARY_LABEL_KEY = "__boundary_labels__"  # class labels for boundary voxels


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


class FeatureExtractor(nn.Module):
    """
    Forward-hook based feature extractor.
    支持 CNN (5D: B,C,Z,Y,X) 和 Transformer (3D: B,seq_len,embed_dim) 两种输出格式。
    Transformer 输出会自动 reshape 为 (B, embed_dim, Z, Y, X) 以兼容下游采样。

    Args:
        model: 要提取特征的模型
        layers: 层名称列表
        token_grid_size: (Z,Y,X) token 网格大小。对于 PrimusM (input=160³, patch=8³) 为 (20,20,20)。
                         如果为 None 则自动推断。
    """

    def __init__(self, model: nn.Module, layers: list, token_grid_size: tuple = None):
        super().__init__()
        self.model = model
        self.layers = layers
        self.token_grid_size = token_grid_size
        self._features = {layer: torch.empty(0) for layer in layers}
        self.handlers = {layer: None for layer in layers}

        named = dict([*self.model.named_modules()])
        for layer_id in layers:
            if layer_id not in named:
                keys = list(named.keys())
                hint = (str(layer_id).split('.', 1)[0] or '').lower()
                if hint:
                    cand = [k for k in keys if hint in k]
                else:
                    cand = []
                if not cand:
                    cand = [k for k in keys if ('encoder' in k or 'decoder' in k or 'eva' in k)]
                if not cand:
                    cand = keys
                raise KeyError(
                    f"Layer '{layer_id}' not found in model.named_modules(). "
                    f"Example candidates (first 120):\n" + "\n".join(cand[:120])
                )
            layer = named[layer_id]
            self.handlers[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def _reshape_tokens_to_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 (B, seq_len, embed_dim) reshape 为 (B, embed_dim, Z, Y, X)"""
        B, S, C = tensor.shape
        if self.token_grid_size is not None:
            gz, gy, gx = self.token_grid_size
        else:
            side = round(S ** (1.0 / 3.0))
            gz = gy = gx = side
        expected_s = gz * gy * gx
        if S > expected_s:
            tensor = tensor[:, :expected_s, :]  # 截断 register tokens
        elif S < expected_s:
            # 不应发生，但做防御
            pad = torch.zeros(B, expected_s - S, C, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor.permute(0, 2, 1).reshape(B, C, gz, gy, gx)

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            # Transformer token 输出: (B, seq_len, embed_dim) -> (B, C, Z, Y, X)
            if output.ndim == 3:
                output = self._reshape_tokens_to_volume(output)
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features

    def remove_handler(self):
        for handler in self.handlers.values():
            handler.remove()


def _allocate_proportional_budget(
    counts: Dict[int, int],
    total_budget: int,
    rng: np.random.Generator,
) -> Dict[int, int]:
    """
    Allocate integer sample budgets proportional to voxel counts.
    - counts: {label: voxel_count}, typically for foreground labels (exclude 0).
    - total_budget: total points to sample among these labels.
    Returns: {label: k_label} with sum <= total_budget (after caps).
    """
    total_budget = int(max(total_budget, 0))
    if total_budget == 0 or not counts:
        return {lb: 0 for lb in counts}

    labels = [int(k) for k in counts.keys()]
    w = np.asarray([max(int(counts[lb]), 0) for lb in labels], dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        # fallback: uniform
        base = total_budget // len(labels)
        rem = total_budget % len(labels)
        budgets = {lb: base for lb in labels}
        for i, lb in enumerate(labels[:rem]):
            budgets[lb] += 1
    else:
        raw = w / s * total_budget
        floors = np.floor(raw).astype(np.int64)
        budgets = {lb: int(floors[i]) for i, lb in enumerate(labels)}
        used = int(floors.sum())
        remaining = total_budget - used
        # distribute remaining by largest fractional parts (Hamilton method)
        if remaining > 0:
            frac = raw - floors
            order = np.argsort(-frac)
            for t in range(remaining):
                lb = labels[int(order[t % len(labels)])]
                budgets[lb] += 1

    # cap by available voxels; if capping frees budget, redistribute
    def capacity(lb: int) -> int:
        return int(max(counts.get(lb, 0), 0))

    freed = 0
    for lb in list(budgets.keys()):
        cap = capacity(lb)
        if budgets[lb] > cap:
            freed += budgets[lb] - cap
            budgets[lb] = cap

    if freed > 0:
        # redistribute freed samples to labels with remaining capacity, proportionally to their remaining capacity
        caps_left = {lb: capacity(lb) - budgets[lb] for lb in budgets if capacity(lb) - budgets[lb] > 0}
        if caps_left:
            # sample labels with probability proportional to remaining capacity
            lb_list = list(caps_left.keys())
            cap_arr = np.asarray([caps_left[lb] for lb in lb_list], dtype=np.float64)
            cap_sum = float(cap_arr.sum())
            if cap_sum > 0:
                probs = cap_arr / cap_sum
                extra = rng.choice(len(lb_list), size=freed, replace=True, p=probs)
                for j in extra:
                    lb = lb_list[int(j)]
                    budgets[lb] += 1
                    # enforce cap
                    if budgets[lb] > capacity(lb):
                        budgets[lb] = capacity(lb)

    return budgets


def _sample_coords_from_mask(
    mask_idx: torch.Tensor,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    mask_idx: (M,5) int64 indices for labels == lb (b, c, z, y, x)
    return: (k,5) numpy int64 subset.
    """
    if k <= 0 or mask_idx.numel() == 0:
        return np.empty((0, 5), dtype=np.int64)
    mask_np = mask_idx.detach().cpu().numpy()
    M = mask_np.shape[0]
    k = int(min(k, M))
    if k == M:
        return mask_np.astype(np.int64, copy=False)
    sel = rng.choice(M, size=k, replace=False)
    return mask_np[sel].astype(np.int64, copy=False)


def _sample_global_coords(
    labels_t: torch.Tensor,
    k_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Global random sampling over valid voxels of the (B,1,Z,Y,X) label tensor.

    We sample uniformly over foreground + real background for Fv, but exclude
    padded sentinel voxels (label == -1). This matches the paper setting where
    padding is assigned -1 and removed from all downstream computations.
    """
    if k_total is None or int(k_total) <= 0:
        return np.empty((0, 5), dtype=np.int64)

    assert labels_t.ndim == 5 and labels_t.shape[1] == 1, "labels expected shape (B,1,Z,Y,X)"
    valid_idx = torch.nonzero(labels_t >= 0, as_tuple=False)
    if valid_idx.numel() == 0:
        return np.empty((0, 5), dtype=np.int64)

    valid_np = valid_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    k_total = int(min(int(k_total), valid_np.shape[0]))
    if k_total <= 0:
        return np.empty((0, 5), dtype=np.int64)
    if k_total == valid_np.shape[0]:
        return valid_np

    sel = rng.choice(valid_np.shape[0], size=k_total, replace=False)
    return valid_np[sel].astype(np.int64, copy=False)


def _gather_vectors_from_feature(
    one_feat: torch.Tensor,
    z_local: np.ndarray,
    y_local: np.ndarray,
    x_local: np.ndarray,
    rz: int,
    ry: int,
    rx: int,
) -> np.ndarray:
    """
    Gather vectors from one feature map given local ROI coords (z_local,y_local,x_local)
    using nearest mapping ROI->feature.
    one_feat: (C, fz, fy, fx)
    return: (N, C) numpy
    """
    if one_feat.ndim != 4:
        return np.empty((0, 0), dtype=np.float32)

    C, fz, fy, fx = one_feat.shape
    if z_local.size == 0:
        return np.empty((0, C), dtype=np.float32)

    # ROI coord -> feature coord (nearest mapping)
    zf = np.zeros_like(z_local, dtype=np.int64) if rz <= 1 else np.rint(z_local * (fz - 1) / (rz - 1)).astype(np.int64)
    yf = np.zeros_like(y_local, dtype=np.int64) if ry <= 1 else np.rint(y_local * (fy - 1) / (ry - 1)).astype(np.int64)
    xf = np.zeros_like(x_local, dtype=np.int64) if rx <= 1 else np.rint(x_local * (fx - 1) / (rx - 1)).astype(np.int64)

    zf = np.clip(zf, 0, fz - 1)
    yf = np.clip(yf, 0, fy - 1)
    xf = np.clip(xf, 0, fx - 1)

    zf_t = torch.as_tensor(zf, device=one_feat.device, dtype=torch.long)
    yf_t = torch.as_tensor(yf, device=one_feat.device, dtype=torch.long)
    xf_t = torch.as_tensor(xf, device=one_feat.device, dtype=torch.long)

    # (C,N) -> (N,C)
    vecs = one_feat[:, zf_t, yf_t, xf_t].permute(1, 0).detach().cpu().numpy()
    return vecs


def ms_sliding_window_sampling(
    layers: list,
    sample_num: dict,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.0,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    # --- new options (paper-faithful sampling) ---
    fv_sample_num: Optional[Union[int, Dict[str, int]]] = None,
    bg_sample_num: Optional[Union[int, Dict[str, int]]] = 0,
    boundary_sample_num: Optional[int] = 0,
    seed: int = 0,
    token_grid_size: tuple = None,
) -> Dict[str, Dict[Union[int, str], np.ndarray]]:
    """
    Paper-faithful multi-scale sampling:
    - Ccons: sample foreground voxels only, with per-class sample sizes proportional to voxel counts (class imbalance aware).
    - Fv: sample *globally* and uniformly over the entire feature map (all voxels), independent of classes.

    Return format:
        res[layer][label_int] = (N, C) sampled vectors for that class (foreground by default, background optional)
        res[layer][GLOBAL_KEY] = (Ng, C) globally sampled vectors for Fv

    Notes:
    - `sample_num[layer]` is interpreted as the *total foreground* sample budget per case for that layer.
    - `fv_sample_num` controls global sampling budget. If None, it defaults to `sample_num[layer]`.
    - To include background class sampling (mainly for GBC), set bg_sample_num>0.
    """
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    rz, ry, rx = roi_size  # ROI spatial (in padded image coords)

    # --- pad inputs to at least roi ---
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    pad_before = []  # (z,y,x) before padding
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
        pad_before.append(half)
    # pad_before built in reverse order (x,y,z)
    pad_before = pad_before[::-1]  # now (z,y,x)

    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    # --- labels tensor (MetaTensor or Tensor) and pad the same way (avoid coord shift bugs) ---
    if hasattr(labels, "as_tensor"):
        labels_t = labels.as_tensor()
    else:
        labels_t = torch.as_tensor(labels)
    if labels_t.dtype != torch.long:
        labels_t = labels_t.long()
    labels_t = F.pad(labels_t, pad=pad_size, mode="constant", value=-1).long()

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)
    total_slices = num_win * batch_size

    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError("OOM in importance_map. Try smaller roi or mode='constant'.") from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # --- rng ---
    rng = np.random.default_rng(int(seed))

    # --- prepare sampling coordinates (in padded image coord system) ---
    uniq = torch.unique(labels_t).detach().cpu().tolist()
    uniq_int = [int(u) for u in uniq]

    # counts for foreground labels (>0)
    fg_labels = [u for u in uniq_int if u > 0]
    fg_counts = {}
    for lb in fg_labels:
        fg_counts[lb] = int((labels_t == lb).sum().item())

    # background counts if needed
    bg_count = int((labels_t == 0).sum().item())

    # helper to fetch layer-wise int option
    def _layer_opt(opt, layer: str, default: int) -> int:
        if opt is None:
            return default
        if isinstance(opt, dict):
            return int(opt.get(layer, default))
        return int(opt)

    # res_dict: layer -> {label -> list[vectors], GLOBAL_KEY -> list[vectors]}
    res_dict: Dict[str, Dict[Union[int, str], list]] = {layer: {GLOBAL_KEY: []} for layer in layers}
    # also keep per-class keys, but only for labels we actually sample
    for layer in layers:
        for lb in fg_labels:
            res_dict[layer][lb] = []
        if _layer_opt(bg_sample_num, layer, 0) > 0:
            res_dict[layer][0] = []
        if boundary_sample_num and int(boundary_sample_num) > 0:
            res_dict[layer][BOUNDARY_KEY] = []
            res_dict[layer][BOUNDARY_LABEL_KEY] = []

    # remaining coords to be collected per layer
    remain_class_coords: Dict[str, Dict[int, np.ndarray]] = {layer: {} for layer in layers}
    remain_global_coords: Dict[str, np.ndarray] = {}

    # --- spatial boundary voxel detection (paper §3.2: ∂Y via morphological gradient) ---
    # A voxel belongs to ∂Y if it is a foreground voxel AND has at least one
    # spatially adjacent voxel from a different class (6-connectivity, 3-D).
    # We pre-compute a single boundary-anchor set from the GT mask, then give
    # every layer its own copy below. This avoids the previous bug where the
    # first processed layer consumed/deleted boundary anchors and later layers
    # could not use the same spatial anchors.
    base_boundary_coords: Optional[np.ndarray] = None   # (K_bnd, 5)  [b,c,z,y,x]
    base_boundary_labels: Optional[np.ndarray] = None   # (K_bnd,)    class id

    if boundary_sample_num and int(boundary_sample_num) > 0 and len(fg_labels) > 0:
        try:
            from scipy.ndimage import binary_erosion
            B_dim = labels_t.shape[0]
            struct6 = np.zeros((3, 3, 3), dtype=bool)
            # face-connected neighbours only (6-connectivity)
            struct6[1, 1, 0] = struct6[1, 1, 2] = True
            struct6[1, 0, 1] = struct6[1, 2, 1] = True
            struct6[0, 1, 1] = struct6[2, 1, 1] = True

            bnd_coords_list, bnd_label_list = [], []
            for b_idx in range(B_dim):
                lab_np = labels_t[b_idx, 0].cpu().numpy().astype(np.int32)
                for lb in fg_labels:
                    fg_mask = (lab_np == lb)
                    if not fg_mask.any():
                        continue
                    # Interior voxels survive erosion; boundary = fg & NOT interior
                    interior = binary_erosion(fg_mask, structure=struct6, border_value=0)
                    bnd_mask = fg_mask & ~interior
                    bnd_vox = np.argwhere(bnd_mask)          # (M, 3) → (z, y, x)
                    if bnd_vox.shape[0] == 0:
                        continue
                    n = bnd_vox.shape[0]
                    # Build (n, 5) coord array: [batch, channel=0, z, y, x]
                    bc = np.empty((n, 5), dtype=np.int64)
                    bc[:, 0] = b_idx
                    bc[:, 1] = 0
                    bc[:, 2:] = bnd_vox
                    bnd_coords_list.append(bc)
                    bnd_label_list.append(np.full(n, lb, dtype=np.int64))

            if bnd_coords_list:
                all_bnd_coords = np.concatenate(bnd_coords_list, axis=0)
                all_bnd_labels = np.concatenate(bnd_label_list, axis=0)
                k_bnd = min(int(boundary_sample_num), all_bnd_coords.shape[0])
                sel = rng.choice(all_bnd_coords.shape[0], k_bnd, replace=False)
                base_boundary_coords = all_bnd_coords[sel]
                base_boundary_labels = all_bnd_labels[sel]
        except Exception as _bnd_err:
            # scipy not available or other error → silently fall back (LBTC will use feature-space proxy)
            base_boundary_coords = None
            base_boundary_labels = None

    # Each layer receives an independent copy of the same GT-mask boundary anchors.
    # This keeps LBTC comparable across encoder layers and prevents anchor leakage
    # caused by deleting a shared remain_boundary_coords array.
    remain_boundary_coords_by_layer: Dict[str, Optional[np.ndarray]] = {}
    remain_boundary_labels_by_layer: Dict[str, Optional[np.ndarray]] = {}
    for layer in layers:
        if base_boundary_coords is not None and base_boundary_labels is not None:
            remain_boundary_coords_by_layer[layer] = base_boundary_coords.copy()
            remain_boundary_labels_by_layer[layer] = base_boundary_labels.copy()
        else:
            remain_boundary_coords_by_layer[layer] = None
            remain_boundary_labels_by_layer[layer] = None

    for layer in layers:
        # foreground proportional budgets
        total_fg_budget = int(sample_num.get(layer, 0))
        budgets = _allocate_proportional_budget(fg_counts, total_fg_budget, rng=rng)

        for lb, k in budgets.items():
            if k <= 0:
                remain_class_coords[layer][lb] = np.empty((0, 5), dtype=np.int64)
                continue
            idx = torch.nonzero(labels_t == int(lb), as_tuple=False)
            remain_class_coords[layer][lb] = _sample_coords_from_mask(idx, int(k), rng=rng)

        # optional background sampling
        k_bg = _layer_opt(bg_sample_num, layer, 0)
        if k_bg and int(k_bg) > 0:
            idx0 = torch.nonzero(labels_t == 0, as_tuple=False)
            remain_class_coords[layer][0] = _sample_coords_from_mask(idx0, int(k_bg), rng=rng)

        # global Fv coords
        k_g = _layer_opt(fv_sample_num, layer, total_fg_budget)
        remain_global_coords[layer] = _sample_global_coords(labels_t, int(k_g), rng=rng)

    # --- hook ---
    feat_extractor = FeatureExtractor(predictor, layers, token_grid_size=token_grid_size)

    # --- sliding window loop ---
    it = tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size)
    for slice_g in it:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]

        window_data = torch.cat(
            [convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]
        ).to(sw_device)

        features = feat_extractor(window_data)

        for layer in layers:
            feat_map = features[layer]
            if isinstance(feat_map, (tuple, list)):
                feat_map = feat_map[0]

            # zip over batch dimension
            for win_slice, one_feat in zip(unravel_slice, feat_map):
                if one_feat.ndim != 4:
                    continue  # skip non-3D feature maps

                # which batch is this window from?
                b0 = int(win_slice[0].start)
                z0, z1 = int(win_slice[2].start), int(win_slice[2].stop)
                y0, y1 = int(win_slice[3].start), int(win_slice[3].stop)
                x0, x1 = int(win_slice[4].start), int(win_slice[4].stop)

                # ---- global sampling (Fv) ----
                gcoords = remain_global_coords.get(layer, None)
                if gcoords is not None and gcoords.size > 0:
                    # filter points inside this ROI and this batch
                    mask = (
                        (gcoords[:, 0] == b0) &
                        (gcoords[:, 2] >= z0) & (gcoords[:, 2] < z1) &
                        (gcoords[:, 3] >= y0) & (gcoords[:, 3] < y1) &
                        (gcoords[:, 4] >= x0) & (gcoords[:, 4] < x1)
                    )
                    if np.any(mask):
                        sel_idx = np.where(mask)[0]
                        pts = gcoords[sel_idx]
                        z_local = (pts[:, 2] - z0).astype(np.int64, copy=False)
                        y_local = (pts[:, 3] - y0).astype(np.int64, copy=False)
                        x_local = (pts[:, 4] - x0).astype(np.int64, copy=False)

                        vecs = _gather_vectors_from_feature(one_feat, z_local, y_local, x_local, rz, ry, rx)
                        if vecs.size > 0:
                            res_dict[layer][GLOBAL_KEY].append(vecs)
                        # remove collected points
                        remain_global_coords[layer] = np.delete(gcoords, sel_idx, axis=0)

                # ---- per-class sampling (Ccons / GBC) ----
                for lb_int, coords in list(remain_class_coords[layer].items()):
                    if coords is None or coords.size == 0:
                        continue
                    mask = (
                        (coords[:, 0] == b0) &
                        (coords[:, 2] >= z0) & (coords[:, 2] < z1) &
                        (coords[:, 3] >= y0) & (coords[:, 3] < y1) &
                        (coords[:, 4] >= x0) & (coords[:, 4] < x1)
                    )
                    if not np.any(mask):
                        continue
                    sel_idx = np.where(mask)[0]
                    pts = coords[sel_idx]
                    z_local = (pts[:, 2] - z0).astype(np.int64, copy=False)
                    y_local = (pts[:, 3] - y0).astype(np.int64, copy=False)
                    x_local = (pts[:, 4] - x0).astype(np.int64, copy=False)

                    vecs = _gather_vectors_from_feature(one_feat, z_local, y_local, x_local, rz, ry, rx)
                    if vecs.size > 0:
                        res_dict[layer][lb_int].append(vecs)
                    remain_class_coords[layer][lb_int] = np.delete(coords, sel_idx, axis=0)

                # ---- boundary voxel sampling (LBTC: spatial ∂Y anchors) ----
                bnd_coords = remain_boundary_coords_by_layer.get(layer, None)
                bnd_labels = remain_boundary_labels_by_layer.get(layer, None)
                if bnd_coords is not None and bnd_labels is not None and bnd_coords.size > 0:
                    bnd_mask = (
                        (bnd_coords[:, 0] == b0) &
                        (bnd_coords[:, 2] >= z0) & (bnd_coords[:, 2] < z1) &
                        (bnd_coords[:, 3] >= y0) & (bnd_coords[:, 3] < y1) &
                        (bnd_coords[:, 4] >= x0) & (bnd_coords[:, 4] < x1)
                    )
                    if np.any(bnd_mask):
                        bnd_sel = np.where(bnd_mask)[0]
                        pts = bnd_coords[bnd_sel]
                        z_local = (pts[:, 2] - z0).astype(np.int64, copy=False)
                        y_local = (pts[:, 3] - y0).astype(np.int64, copy=False)
                        x_local = (pts[:, 4] - x0).astype(np.int64, copy=False)
                        vecs = _gather_vectors_from_feature(one_feat, z_local, y_local, x_local, rz, ry, rx)
                        if vecs.size > 0:
                            res_dict[layer][BOUNDARY_KEY].append(vecs)
                            res_dict[layer][BOUNDARY_LABEL_KEY].append(
                                bnd_labels[bnd_sel].astype(np.int64)
                            )
                        remain_boundary_coords_by_layer[layer] = np.delete(bnd_coords, bnd_sel, axis=0)
                        remain_boundary_labels_by_layer[layer] = np.delete(bnd_labels, bnd_sel, axis=0)

    feat_extractor.remove_handler()

    # list -> ndarray (concat chunks)
    out: Dict[str, Dict[Union[int, str], np.ndarray]] = {layer: {} for layer in layers}
    for layer in layers:
        for k, vlist in res_dict[layer].items():
            if len(vlist) == 0:
                if k == BOUNDARY_LABEL_KEY:
                    out[layer][k] = np.empty((0,), dtype=np.int64)
                else:
                    out[layer][k] = np.empty((0, 0), dtype=np.float32)
                continue
            if k == BOUNDARY_LABEL_KEY:
                # 1-D integer label arrays
                out[layer][k] = np.concatenate(vlist, axis=0).astype(np.int64)
            elif isinstance(vlist[0], np.ndarray) and vlist[0].ndim == 2:
                out[layer][k] = np.concatenate(vlist, axis=0)
            else:
                out[layer][k] = np.asarray(vlist)

    return out
