import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Dict, Sequence, Union, Optional
from tqdm import tqdm

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, convert_data_type, fall_back_tuple, look_up_option

# 复用原有的 FeatureExtractor 和辅助函数
from .sliding_window_sampling import (
    FeatureExtractor,
    GLOBAL_KEY,
    _get_scan_interval,
)


# =============================================================================
# NOTE
# =============================================================================
# OpenMind 的“分类数据集 transferability 表”（LogME / GBC / LEEP / Kendall 等）
# 通常是“每个样本（case）对应一个特征向量”。
# 因此这里额外提供 pooling 版：对每层 activation 做空间均值池化得到 (1, C)。
# 旧的 point-cloud 采样版仍保留（classification_sliding_window_sampling）。


def _as_tensor(x):
    """MONAI/Hook 可能返回 tuple/list；这里统一取第一个张量并转 torch.Tensor。"""
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x


def classification_sliding_window_pooling(
    layers: list,
    inputs: torch.Tensor,
    image_label: int,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.0,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
) -> Dict[str, Dict[Union[int, str], np.ndarray]]:
    """分类任务：对每层 activation 做空间均值池化，得到每个 case 一条向量。

    返回结构与原采样接口保持一致：
        res[layer][image_label] = (1, C)
        res[layer][GLOBAL_KEY] = (1, C)  # 方便复用现有 Fv / overall 逻辑

    说明：
      - **强烈建议 overlap=0.0**（默认就是 0），否则 sliding-window 有重叠会导致平均被重复计数。
      - 对于 linear / MLP（分类头）输出 (B, D)，会按 batch 做平均（B通常为 sw_batch_size）。
    """

    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2

    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    batch_size, _, *image_size_ = inputs.shape
    if batch_size != 1:
        raise ValueError("分类 pooling 仅支持 batch_size=1（逐 case 处理）")

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)

    # Pad inputs 到至少 roi_size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])

    inputs = F.pad(
        inputs,
        pad=pad_size,
        mode=look_up_option(padding_mode, PytorchPadMode),
        value=cval,
    )

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)

    # importance_map：这里主要用于一致性；pooling 默认不做重建 blend。
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        importance_map = compute_importance_map(
            valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device
        )

    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    model = predictor
    if isinstance(model, FeatureExtractor):
        feature_extractor = model
        owns_extractor = False
    else:
        feature_extractor = FeatureExtractor(model, layers)
        owns_extractor = True

    # running sums per layer
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    iterator = tqdm(range(0, num_win, sw_batch_size), disable=not progress)
    for start in iterator:
        end = min(start + sw_batch_size, num_win)
        slices_batch = slices[start:end]

        sw_inputs = torch.cat(
            [inputs[(slice(None), slice(None)) + tuple(slc)] for slc in slices_batch],
            dim=0,
        ).to(sw_device)

        with torch.no_grad():
            layer_features = feature_extractor(sw_inputs)

        for layer in layers:
            feat = _as_tensor(layer_features[layer])  # (B, C, ...)
            feat = feat.to(device)

            if feat.ndim >= 3:
                # sum over spatial dims -> (B, C)
                spatial_dims = tuple(range(2, feat.ndim))
                feat_sum = feat.sum(dim=spatial_dims)  # (B, C)
                vox = int(np.prod(feat.shape[2:]))
                batch_sum = feat_sum.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                sums[layer] = sums.get(layer, 0.0) + batch_sum
                counts[layer] = counts.get(layer, 0) + int(vox * feat.shape[0])
            elif feat.ndim == 2:
                # (B, D)
                batch_sum = feat.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                sums[layer] = sums.get(layer, 0.0) + batch_sum
                counts[layer] = counts.get(layer, 0) + int(feat.shape[0])
            elif feat.ndim == 1:
                vec = feat.detach().cpu().numpy().astype(np.float64)
                sums[layer] = sums.get(layer, 0.0) + vec
                counts[layer] = counts.get(layer, 0) + 1
            else:
                raise ValueError(f"Unexpected feature dim for layer={layer}: {tuple(feat.shape)}")

    if owns_extractor:
        feature_extractor.remove_handler()

    result: Dict[str, Dict[Union[int, str], np.ndarray]] = {}
    for layer in layers:
        if layer not in sums or counts.get(layer, 0) <= 0:
            continue
        emb = (sums[layer] / float(counts[layer])).astype(np.float32, copy=False)
        emb = emb.reshape(1, -1)
        result[layer] = {int(image_label): emb, GLOBAL_KEY: emb}

    return result


# =============================================================================
# 原 point-cloud 采样（保留）
# =============================================================================


def classification_sliding_window_sampling(
    layers: list,
    sample_num: dict,
    inputs: torch.Tensor,
    image_label: int,  # 图像级标签（单个整数）
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.0,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    fv_sample_num: Optional[Union[int, Dict[str, int]]] = None,
    seed: int = 0,
) -> Dict[str, Dict[Union[int, str], np.ndarray]]:
    """
    分类任务的特征采样函数（point-cloud）

    关键差异:
    - 输入: image_label（图像级标签，如0/1）而非 labels（像素级标签张量）
    - 采样: 从整个图像中随机采样，不按像素类别分配
    - 输出: 按图像标签分组的特征

    Returns:
        res[layer][image_label] = (N, C) 该图像该层的采样特征
        res[layer][GLOBAL_KEY] = (Ng, C) 全局采样特征（用于Fv）
    """

    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2

    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    batch_size, _, *image_size_ = inputs.shape
    if batch_size != 1:
        raise ValueError("分类任务采样仅支持batch_size=1")

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)

    # Pad inputs
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])

    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)

    # Importance map for blending
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(
                valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device
            )
        except BaseException as e:
            raise RuntimeError("OOM in importance_map. Try smaller roi or mode='constant'.") from e

    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # 初始化特征提取器
    model = predictor
    if isinstance(model, FeatureExtractor):
        feature_extractor = model
    else:
        feature_extractor = FeatureExtractor(model, layers)

    layer_all_features = {layer: [] for layer in layers}
    rng = np.random.default_rng(int(seed))

    # Sliding window inference
    iterator = tqdm(range(0, num_win, sw_batch_size), disable=not progress)
    for start in iterator:
        end = min(start + sw_batch_size, num_win)
        slices_batch = slices[start:end]

        sw_inputs = torch.cat(
            [inputs[(slice(None), slice(None)) + tuple(slc)] for slc in slices_batch],
            dim=0,
        ).to(sw_device)

        with torch.no_grad():
            layer_features = feature_extractor(sw_inputs)

        for layer in layers:
            feat = _as_tensor(layer_features[layer])
            layer_all_features[layer].append(feat)

    if not isinstance(predictor, FeatureExtractor):
        feature_extractor.remove_handler()

    result = {layer: {} for layer in layers}

    for layer in layers:
        n_samples = int(sample_num.get(layer, 200)) if isinstance(sample_num, dict) else int(sample_num)

        all_feats = torch.cat(layer_all_features[layer], dim=0)
        C = all_feats.shape[1] if all_feats.ndim >= 2 else 1

        if all_feats.ndim == 5:  # 3D
            flat_feats = all_feats.permute(1, 0, 2, 3, 4).reshape(C, -1).T
        elif all_feats.ndim == 4:  # 2D
            flat_feats = all_feats.permute(1, 0, 2, 3).reshape(C, -1).T
        elif all_feats.ndim == 2:
            flat_feats = all_feats
        else:
            raise ValueError(f"Unexpected feature dimension: {all_feats.shape}")

        flat_feats_np = flat_feats.detach().cpu().numpy()

        if flat_feats_np.shape[0] > n_samples:
            indices = rng.choice(flat_feats_np.shape[0], size=n_samples, replace=False)
            sampled = flat_feats_np[indices]
        else:
            sampled = flat_feats_np

        result[layer][int(image_label)] = sampled

        if fv_sample_num is not None:
            fv_n = int(fv_sample_num.get(layer, n_samples)) if isinstance(fv_sample_num, dict) else int(fv_sample_num)
        else:
            fv_n = n_samples

        if flat_feats_np.shape[0] > fv_n:
            fv_indices = rng.choice(flat_feats_np.shape[0], size=fv_n, replace=False)
            fv_sampled = flat_feats_np[fv_indices]
        else:
            fv_sampled = flat_feats_np

        result[layer][GLOBAL_KEY] = fv_sampled

    return result


def batch_classification_sampling(
    layers: list,
    sample_num: dict,
    data_loader,
    model,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    **kwargs
) -> Dict[str, Dict[int, np.ndarray]]:
    """批量处理分类任务（point-cloud）。

    返回：
        result[layer][label] = (N_total, C)
        result[layer][GLOBAL_KEY] = (Ng_total, C)
    """
    result = {layer: {} for layer in layers}

    print("开始批量采样...")
    for idx, batch_data in enumerate(tqdm(data_loader)):
        img = batch_data["data"].cuda()
        label = int(batch_data["label"].item())

        single_result = classification_sliding_window_sampling(
            layers=layers,
            sample_num=sample_num,
            inputs=img,
            image_label=label,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            **kwargs
        )

        for layer in layers:
            if label not in result[layer]:
                result[layer][label] = []
            result[layer][label].append(single_result[layer][label])

            if GLOBAL_KEY not in result[layer]:
                result[layer][GLOBAL_KEY] = []
            result[layer][GLOBAL_KEY].append(single_result[layer][GLOBAL_KEY])

    for layer in layers:
        for label in list(result[layer].keys()):
            if label == GLOBAL_KEY:
                result[layer][GLOBAL_KEY] = np.concatenate(result[layer][GLOBAL_KEY], axis=0)
            else:
                result[layer][label] = np.concatenate(result[layer][label], axis=0)

    return result


__all__ = [
    "classification_sliding_window_pooling",
    "classification_sliding_window_sampling",
    "batch_classification_sampling",
    "GLOBAL_KEY",
]
