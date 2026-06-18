from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import math
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree

from utils.ccfv import cal_w_distance, feature_variety, gaussian_stats, cal_w_distance_stats
from utils.ccfv_slow import cal_w_distance as cal_w_distance_slow, feature_variety as feature_variety_slow, gaussian_stats as gaussian_stats_slow, cal_w_distance_stats as cal_w_distance_stats_slow
from utils.gbc_metric import gbc_score
from utils.leep_metric import leep_score
from utils.logme_metric import logme_score
from utils.sliding_window_sampling import GLOBAL_KEY
from utils.tda import TDARTDLiteMetric, TDALBTCMetric
from utils.tlc_metric import TLCMetric
from utils.ntk_metrics import (
    NTKCondMetric,
    NTKTraceMetric,
    NTKLayerwiseMetric,
    NTKSpectrumMetric,
)

# ===========================================================================
# 层名识别辅助函数（兼容 ResEncL 和 PrimusM）
# ===========================================================================

def is_encoder_layer(layer_name: str) -> bool:
    """判断层名是否属于 encoder（兼容 ResEncL 和 PrimusM）"""
    l = layer_name.lower()
    if "encoder" in l or "enc" in l:
        return True
    # PrimusM: eva.blocks.X, down_projection
    if l.startswith("eva.") or l == "eva":
        return True
    if "down_projection" in l:
        return True
    return False


def is_decoder_layer(layer_name: str) -> bool:
    """判断层名是否属于 decoder（兼容 ResEncL 和 PrimusM）"""
    l = layer_name.lower()
    if "decoder" in l or "dec" in l:
        return True
    if "up_projection" in l:
        return True
    return False


# ===========================================================================
# 论文对齐的层选择工具函数
# ===========================================================================

def _nat_sort_key(s: str) -> list:
    """Natural-sort key so 'stages.10' sorts after 'stages.2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _deepest_encoder_score(
    layer_scores: Dict[str, Optional[float]],
    layers: List[str],
) -> Optional[float]:
    """Return the score of the deepest encoder layer.

    "Deepest" means last in the *config list order* — the user lists layers
    from shallow to deep, so the last encoder entry is the deepest.
    Natural-sort is intentionally NOT used here because names like
    'encoder.stem' would sort after 'encoder.stages.4' alphabetically
    (「ste」>「sta」), which is wrong.

    Used by GBC and LogME: both papers measure transferability in the
    backbone / feature-extractor output space (encoder last stage).
    """
    enc = [
        l for l in layers          # preserve config order
        if is_encoder_layer(l) and layer_scores.get(l) is not None
    ]
    if not enc:
        return None
    return layer_scores[enc[-1]]   # last = deepest in config order


def _deepest_output_score(
    layer_scores: Dict[str, Optional[float]],
    layers: List[str],
) -> Optional[float]:
    """Return the score of the model output layer, for LEEP (use_logits=True).

    Search priority (preserving config list order in each group):
      1. cls_head layers  (classification task)
      2. decoder layers   (segmentation task — deepest = final output)
      3. Any layer with a valid score (fallback)
    """
    # 1. classification head
    cls = [
        l for l in layers
        if "cls_head" in l.lower() and layer_scores.get(l) is not None
    ]
    if cls:
        return layer_scores[cls[-1]]

    # 2. decoder / segmentation head (last in list = deepest output)
    dec = [
        l for l in layers
        if is_decoder_layer(l) and layer_scores.get(l) is not None
    ]
    if dec:
        return layer_scores[dec[-1]]

    # 3. fallback: last valid layer in config order
    valid = [l for l in layers if layer_scores.get(l) is not None]
    if valid:
        return layer_scores[valid[-1]]
    return None


# ===========================================================================
# Abstract base
# ===========================================================================


class MetricComputer(ABC):
    """Abstract base class for all transferability metrics."""

    # --- lifecycle -----------------------------------------------------------

    @abstractmethod
    def reset(self):
        """Clear internal state.  Called once before each (dataset, model) pair."""
        ...

    def begin_case(self, case_idx: int):
        """Optional hook before accumulating layers for a new case.
        Override when you need deterministic-but-case-specific RNG."""
        pass

    # --- accumulator ---------------------------------------------------------

    @abstractmethod
    def accumulate(self, layer: str, feature_dict: dict):
        """Feed one case's sampled features for one layer.

        feature_dict:
            int  keys  → (N, C) ndarray   per-class sampled feature vectors
            GLOBAL_KEY → (Ng, C) ndarray  globally sampled vectors (for Fv)
        """
        ...

    # --- scoring -------------------------------------------------------------

    @abstractmethod
    def compute(self, layer: str) -> Optional[float]:
        """Return the metric score for *layer* after all cases are accumulated.
        Returns None when data is insufficient.
        **Convention: higher is better for every metric.**"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in file names and logs (e.g. 'ccfv')."""
        ...

    def aggregate_overall(
        self,
        layer_scores: Dict[str, Optional[float]],
        layers: List[str],
    ) -> Optional[float]:
        valid = [layer_scores[l] for l in layers if layer_scores.get(l) is not None]
        return float(np.mean(valid)) if valid else None


# ===========================================================================
# CCFV
# ===========================================================================


class CCFVMetric(MetricComputer):

    """CC-FV (Class Consistency + Feature Variety).

    原论文定义：
      - Class Consistency: 同一语义类在不同 case 的特征分布应一致 (W2 距离)；
      - Feature Variety: 特征在超球面上应“分散/均匀”(hyperspherical energy 的倒数)；
      - Score = log((Fv+eps)/(Ccons+eps)).

    我们这里兼容 *分割* 与 *分类* 两类数据：
      - 分割：每个 case 有多个语义类 key（含背景 0）；按论文做法：逐 case 计算 Fv，再跨 case 计算 Ccons。
      - 分类：每个 case 只有一个图像级 label key（0/1/...），不存在“背景类 0”。
              此时逐 case 的 GLOBAL_KEY 通常只有 1 个向量，导致 Fv=0。
              因此我们切换为 *数据集级* Fv：把所有 case 的 GLOBAL_KEY(或回退特征)拼起来算一次 Fv。

    关键点：
      - 背景处理：
          * 分割模式：跳过 label=0 (背景) 与 label<0；
          * 分类模式：仅跳过 label<0（如旧采样里用 -1 表示 background）。
      - 模式判定：根据单个 case 的“非 global class key 数量”最大值：>1 视为分割，否则视为分类。
    """

    def __init__(
        self,
        energy_eps: float = 1e-3,
        inv_eps: float = 1e-8,
        reduce: str = 'mean',
        # Fv 计算最多使用多少点（无论 per-case 或 dataset-level）
        max_points: int | None = 2048,
        # Fv 计算方法：auto/full/block（默认 auto：小样本 full，大样本 block，均为精确）
        fv_method: str = 'auto',
        fv_block_size: int = 2048,
        fv_use_float32: bool = True,
        # W2 计算时的协方差 jitter
        w2_eps: float = 1e-6,
        # Ccons 计算：all_pairs(论文式) 或 to_proto(近似，加速)
        ccons_mode: str = 'all_pairs',
        # all_pairs 时 pair 数过大就随机采样（近似）
        ccons_max_pairs: int = 20000,
        # 是否对每个 case/类别先预计算 (mean,cov) 再做 pairwise（大幅加速）
        ccons_precompute_stats: bool = True,
        seed: int = 0,
    ):
        self._energy_eps = float(energy_eps)
        self._inv_eps = float(inv_eps)
        self._reduce = str(reduce)
        self._max_points = None if max_points is None else int(max_points)
        # --- FV speed knobs (exact unless you choose sample in utils.ccfv) ---
        self._fv_method = str(fv_method)
        self._fv_block_size = int(fv_block_size)
        self._fv_use_float32 = bool(fv_use_float32)

        # --- W2 / Ccons knobs ---
        self._w2_eps = float(w2_eps)
        self._ccons_precompute_stats = bool(ccons_precompute_stats)
        self._ccons_mode = str(ccons_mode)
        self._ccons_max_pairs = int(ccons_max_pairs)
        self._seed = int(seed)

        # --- internal state (reset clears these) ---
        self._fv_cases: dict[str, list[float]] = {}
        self._global_points: dict[str, list[object]] = {}
        self._class_feats: dict[str, dict[int, list[object]]] = {}
        self._max_non_global_keys: dict[str, int] = {}
        self._rng = None
        self.reset()

    @property
    def name(self) -> str:
        return 'ccfv'

    # --- lifecycle -----------------------------------------------------------

    def reset(self):
        self._fv_cases = {}
        self._global_points = {}
        self._class_feats = {}
        self._max_non_global_keys = {}
        import numpy as _np
        self._rng = _np.random.default_rng(self._seed)

    # --- accumulator ---------------------------------------------------------

    def accumulate(self, layer: str, feature_dict: dict):
        import numpy as _np

        self._fv_cases.setdefault(layer, [])
        self._global_points.setdefault(layer, [])
        self._class_feats.setdefault(layer, {})
        self._max_non_global_keys.setdefault(layer, 0)

        # --- track how many class keys exist per case (for auto seg/cls detection)
        non_global_int_keys = [k for k in feature_dict.keys() if k != GLOBAL_KEY and isinstance(k, (int, _np.integer))]
        self._max_non_global_keys[layer] = max(self._max_non_global_keys[layer], len(non_global_int_keys))

        # ---- choose a feature set for Fv ----
        case_feats = None
        if GLOBAL_KEY in feature_dict:
            g = _np.asarray(feature_dict[GLOBAL_KEY])
            if g.size > 0:
                case_feats = g if g.ndim == 2 else g.reshape(g.shape[0], -1)

        if case_feats is None:
            parts = []
            for lb, f in feature_dict.items():
                if lb == GLOBAL_KEY or not isinstance(lb, (int, _np.integer)):
                    continue
                f = _np.asarray(f)
                if f.size == 0:
                    continue
                parts.append(f if f.ndim == 2 else f.reshape(f.shape[0], -1))
            if parts:
                case_feats = _np.concatenate(parts, axis=0)

        # store for dataset-level Fv fallback
        if case_feats is not None and case_feats.size > 0:
            # optional downsampling
            if self._max_points and case_feats.shape[0] > self._max_points:
                sel = self._rng.choice(case_feats.shape[0], size=self._max_points, replace=False)
                case_feats = case_feats[sel]
            self._global_points[layer].append(case_feats)

            # per-case Fv only makes sense when >=2 points
            if case_feats.shape[0] >= 2:
                fv = feature_variety(
                    case_feats,
                    energy_eps=self._energy_eps,
                    inv_eps=self._inv_eps,
                    l2_normalize=True,
                    reduce=self._reduce,
                    method=self._fv_method,
                    block_size=self._fv_block_size,
                    use_float32=self._fv_use_float32,
                )
                if fv > 0.0:
                    self._fv_cases[layer].append(float(fv))

        # ---- Ccons: store per-class features for cross-case Wasserstein ----
        for lb, feats in feature_dict.items():
            if lb == GLOBAL_KEY or not isinstance(lb, (int, _np.integer)):
                continue
            feats = _np.asarray(feats)
            if feats.size == 0:
                continue
            if feats.ndim == 1:
                feats = feats.reshape(feats.shape[0], -1)
            self._class_feats[layer].setdefault(int(lb), []).append(feats)

    # --- scoring -------------------------------------------------------------

    def _is_segmentation_like(self, layer: str) -> bool:
        # >1 class-key in a single case => segmentation-like
        return int(self._max_non_global_keys.get(layer, 0)) > 1

    def _compute_fv(self, layer: str) -> float | None:
        import numpy as _np
        fv_list = self._fv_cases.get(layer, [])
        # segmentation: prefer per-case
        if fv_list:
            return float(_np.mean(fv_list))

        # classification / fallback: dataset-level variety
        chunks = self._global_points.get(layer, [])
        if not chunks:
            # last fallback: concat all per-class feats
            parts = []
            for lb, lst in self._class_feats.get(layer, {}).items():
                if not lst:
                    continue
                parts.extend(lst)
            chunks = parts

        if not chunks:
            return None

        X = _np.concatenate([_np.asarray(x) for x in chunks if _np.asarray(x).size > 0], axis=0)
        if X.size == 0 or X.shape[0] < 2:
            return None

        if self._max_points and X.shape[0] > self._max_points:
            sel = self._rng.choice(X.shape[0], size=self._max_points, replace=False)
            X = X[sel]

        fv = feature_variety(
            X,
            energy_eps=self._energy_eps,
            inv_eps=self._inv_eps,
            l2_normalize=True,
            reduce=self._reduce,
            method=self._fv_method,
            block_size=self._fv_block_size,
            use_float32=self._fv_use_float32,
        )
        return float(fv) if fv > 0.0 else None

    def _iter_pairs(self, n: int):
        """Yield (i,j) pairs; if too many, sample random pairs."""
        if n < 2:
            return
        total = n * (n - 1) // 2
        if self._ccons_max_pairs and total > self._ccons_max_pairs:
            # random sampling with replacement (good enough + deterministic via rng)
            for _ in range(self._ccons_max_pairs):
                i = int(self._rng.integers(0, n))
                j = int(self._rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                if i < j:
                    yield i, j
                else:
                    yield j, i
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    yield i, j

    def compute(self, layer: str) -> float | None:
        import numpy as _np
        EPS = 1e-8

        Fv = self._compute_fv(layer)
        if Fv is None:
            return None

        seg_like = self._is_segmentation_like(layer)

        # Ccons: mean pairwise W2 over classes
        dist_sum, dist_cnt = 0.0, 0
        for lb, feats_list in self._class_feats.get(layer, {}).items():
            lb = int(lb)
            if lb < 0:
                continue
            if seg_like and lb == 0:
                # segmentation background
                continue
            n = len(feats_list)
            if n < 2:
                continue

            if self._ccons_precompute_stats:
                stats_list = [gaussian_stats(f, eps=self._w2_eps) for f in feats_list]

                if self._ccons_mode.lower() == 'to_proto':
                    proto = _np.concatenate(feats_list, axis=0)
                    proto_stats = gaussian_stats(proto, eps=self._w2_eps)
                    for s in stats_list:
                        dist_sum += cal_w_distance_stats(s, proto_stats)
                        dist_cnt += 1
                else:
                    for i, j in self._iter_pairs(n):
                        dist_sum += cal_w_distance_stats(stats_list[i], stats_list[j])
                        dist_cnt += 1
            else:
                if self._ccons_mode.lower() == 'to_proto':
                    proto = _np.concatenate(feats_list, axis=0)
                    for f in feats_list:
                        dist_sum += cal_w_distance(f, proto, eps=self._w2_eps)
                        dist_cnt += 1
                else:
                    for i, j in self._iter_pairs(n):
                        dist_sum += cal_w_distance(feats_list[i], feats_list[j], eps=self._w2_eps)
                        dist_cnt += 1

        if dist_cnt == 0:
            return None

        Ccons = dist_sum / dist_cnt
        return float(_np.log((Fv + EPS) / (Ccons + EPS)))

# ===========================================================================
# CCFV (Slow backend: SciPy sqrtm + full hyperspherical energy)
# ===========================================================================

class CCFVSlowMetric(CCFVMetric):
    """CC-FV with the *original* (non-accelerated) math backend.

    Differences vs CCFVMetric:
      - W2 uses SciPy sqrtm (same as the original ccfv.py).
      - Feature variety uses the full LxL hyperspherical energy (no blockwise path).

    Note: This keeps the *outer* logic (seg/cls handling, optional max_points,
    optional max_pairs sampling) identical to CCFVMetric, so your timing comparison
    isolates the backend speedup as much as possible.
    """

    @property
    def name(self) -> str:
        return "ccfv_slow"

    # --- accumulator ---------------------------------------------------------

    def accumulate(self, layer: str, feature_dict: dict):
        import numpy as _np

        self._fv_cases.setdefault(layer, [])
        self._global_points.setdefault(layer, [])
        self._class_feats.setdefault(layer, {})
        self._max_non_global_keys.setdefault(layer, 0)

        non_global_int_keys = [k for k in feature_dict.keys() if k != GLOBAL_KEY and isinstance(k, (int, _np.integer))]
        self._max_non_global_keys[layer] = max(self._max_non_global_keys[layer], len(non_global_int_keys))

        # ---- choose a feature set for Fv ----
        case_feats = None
        if GLOBAL_KEY in feature_dict:
            g = _np.asarray(feature_dict[GLOBAL_KEY])
            if g.size > 0:
                case_feats = g if g.ndim == 2 else g.reshape(g.shape[0], -1)

        if case_feats is None:
            parts = []
            for lb, f in feature_dict.items():
                if lb == GLOBAL_KEY or not isinstance(lb, (int, _np.integer)):
                    continue
                f = _np.asarray(f)
                if f.size == 0:
                    continue
                parts.append(f if f.ndim == 2 else f.reshape(f.shape[0], -1))
            if parts:
                case_feats = _np.concatenate(parts, axis=0)

        # store for dataset-level Fv fallback
        if case_feats is not None and case_feats.size > 0:
            # optional downsampling (keep identical outer behavior to the fast metric)
            if self._max_points and case_feats.shape[0] > self._max_points:
                sel = self._rng.choice(case_feats.shape[0], size=self._max_points, replace=False)
                case_feats = case_feats[sel]
            self._global_points[layer].append(case_feats)

            # per-case Fv only makes sense when >=2 points
            if case_feats.shape[0] >= 2:
                fv = feature_variety_slow(
                    case_feats,
                    energy_eps=self._energy_eps,
                    inv_eps=self._inv_eps,
                    l2_normalize=True,
                    reduce=self._reduce,
                )
                if fv > 0.0:
                    self._fv_cases[layer].append(float(fv))

        # ---- Ccons: store per-class features for cross-case Wasserstein ----
        for lb, feats in feature_dict.items():
            if lb == GLOBAL_KEY or not isinstance(lb, (int, _np.integer)):
                continue
            feats = _np.asarray(feats)
            if feats.size == 0:
                continue
            if feats.ndim == 1:
                feats = feats.reshape(feats.shape[0], -1)
            self._class_feats[layer].setdefault(int(lb), []).append(feats)

    # --- scoring -------------------------------------------------------------

    def _compute_fv(self, layer: str) -> float | None:
        import numpy as _np
        fv_list = self._fv_cases.get(layer, [])
        if fv_list:
            return float(_np.mean(fv_list))

        chunks = self._global_points.get(layer, [])
        if not chunks:
            parts = []
            for _, lst in self._class_feats.get(layer, {}).items():
                if lst:
                    parts.extend(lst)
            chunks = parts

        if not chunks:
            return None

        X = _np.concatenate([_np.asarray(x) for x in chunks if _np.asarray(x).size > 0], axis=0)
        if X.size == 0 or X.shape[0] < 2:
            return None

        if self._max_points and X.shape[0] > self._max_points:
            sel = self._rng.choice(X.shape[0], size=self._max_points, replace=False)
            X = X[sel]

        fv = feature_variety_slow(
            X,
            energy_eps=self._energy_eps,
            inv_eps=self._inv_eps,
            l2_normalize=True,
            reduce=self._reduce,
        )
        return float(fv) if fv > 0.0 else None

    def compute(self, layer: str) -> float | None:
        import numpy as _np
        EPS = 1e-8

        Fv = self._compute_fv(layer)
        if Fv is None:
            return None

        seg_like = self._is_segmentation_like(layer)

        dist_sum, dist_cnt = 0.0, 0
        for lb, feats_list in self._class_feats.get(layer, {}).items():
            lb = int(lb)
            if lb < 0:
                continue
            if seg_like and lb == 0:
                continue
            n = len(feats_list)
            if n < 2:
                continue

            if self._ccons_precompute_stats:
                stats_list = [gaussian_stats_slow(f, eps=self._w2_eps) for f in feats_list]

                if self._ccons_mode.lower() == "to_proto":
                    proto = _np.concatenate(feats_list, axis=0)
                    proto_stats = gaussian_stats_slow(proto, eps=self._w2_eps)
                    for s in stats_list:
                        dist_sum += cal_w_distance_stats_slow(s, proto_stats)
                        dist_cnt += 1
                else:
                    for i, j in self._iter_pairs(n):
                        dist_sum += cal_w_distance_stats_slow(stats_list[i], stats_list[j])
                        dist_cnt += 1
            else:
                if self._ccons_mode.lower() == "to_proto":
                    proto = _np.concatenate(feats_list, axis=0)
                    for f in feats_list:
                        dist_sum += cal_w_distance_slow(f, proto, eps=self._w2_eps)
                        dist_cnt += 1
                else:
                    for i, j in self._iter_pairs(n):
                        dist_sum += cal_w_distance_slow(feats_list[i], feats_list[j], eps=self._w2_eps)
                        dist_cnt += 1

        if dist_cnt == 0:
            return None

        Ccons = dist_sum / dist_cnt
        return float(_np.log((Fv + EPS) / (Ccons + EPS)))


class GBCMetric(MetricComputer):

    def __init__(
        self,
        gaussian_type: str = "spherical",
        eps: float = 1e-4,
        pca_dim: Optional[int] = 64,
        pca_random_state: int = 0,
        ignore_background: bool = False,
        subsample_per_case: Optional[int] = 1000,
        max_per_class: int = 20000,
    ):
        self._gaussian_type = gaussian_type
        self._eps = eps
        self._pca_dim = pca_dim
        self._pca_random_state = pca_random_state
        self._ignore_background = ignore_background
        self._subsample_per_case = subsample_per_case
        self._max_per_class = max_per_class
        # --- internal state ---
        self._feat_all: Dict[str, List[np.ndarray]] = {}
        self._lab_all: Dict[str, List[np.ndarray]] = {}
        self._case_idx: int = 0

    @property
    def name(self) -> str:
        return "gbc"

    # --- lifecycle -----------------------------------------------------------

    def reset(self):
        self._feat_all = {}
        self._lab_all = {}
        self._case_idx = 0

    def begin_case(self, case_idx: int):
        self._case_idx = int(case_idx)

    # --- accumulator ---------------------------------------------------------

    def accumulate(self, layer: str, feature_dict: dict):
        self._feat_all.setdefault(layer, [])
        self._lab_all.setdefault(layer, [])

        parts = []
        labs = []
        for lb, feats in feature_dict.items():
            if lb == GLOBAL_KEY or not isinstance(lb, (int, np.integer)):
                continue
            lb_int = int(lb)
            if self._ignore_background and lb_int == 0:
                continue
            feats = np.asarray(feats)
            if feats.size == 0:
                continue
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            parts.append(feats)
            labs.append(np.full(feats.shape[0], lb_int, dtype=np.int64))

        if not parts:
            return

        X = np.concatenate(parts, axis=0)
        y = np.concatenate(labs, axis=0)

        # optional balanced subsampling per case
        if self._subsample_per_case and X.shape[0] > self._subsample_per_case:
            rng = np.random.default_rng(self._case_idx)
            budget = int(min(self._subsample_per_case, X.shape[0]))
            keep = self._balanced_subsample(y, budget, rng)
            X, y = X[keep], y[keep]

        self._feat_all[layer].append(X)
        self._lab_all[layer].append(y)

    def _balanced_subsample(self, y, budget, rng):
        """Allocate ~equal quota per class; fill leftovers from remaining pool."""
        ignore = {0} if self._ignore_background else set()
        labels = [int(lb) for lb in np.unique(y) if int(lb) not in ignore]
        if not labels:
            return np.arange(y.shape[0], dtype=np.int64)

        budget = int(min(budget, y.shape[0]))
        per_class = budget // len(labels)
        remainder = budget % len(labels)

        chosen = []
        for i, lb in enumerate(labels):
            idx = np.where(y == lb)[0]
            k = min(per_class + (1 if i < remainder else 0), idx.size)
            if k > 0:
                chosen.append(rng.choice(idx, size=k, replace=False))

        chosen = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)

        # fill remaining budget
        if chosen.size < budget:
            mask = np.ones(y.shape[0], dtype=bool)
            mask[chosen] = False
            pool = np.where(mask & ~np.isin(y, list(ignore)))[0]
            need = budget - chosen.size
            if pool.size > 0 and need > 0:
                extra = rng.choice(pool, size=min(need, pool.size), replace=False)
                chosen = np.concatenate([chosen, extra])

        return chosen.astype(np.int64)

    # --- scoring -------------------------------------------------------------

    def compute(self, layer: str) -> Optional[float]:
        if not self._feat_all.get(layer):
            return None

        X = np.concatenate(self._feat_all[layer], axis=0)
        y = np.concatenate(self._lab_all[layer], axis=0)

        # per-class hard cap
        if self._max_per_class:
            rng = np.random.default_rng(0)
            parts = []
            for lb in np.unique(y):
                idx = np.where(y == lb)[0]
                if idx.size > self._max_per_class:
                    idx = rng.choice(idx, size=self._max_per_class, replace=False)
                parts.append(idx)
            keep = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
            X, y = X[keep], y[keep]

        ignore = [0] if self._ignore_background else None
        return gbc_score(
            X, y,
            gaussian_type=self._gaussian_type,
            eps=self._eps,
            pca_dim=self._pca_dim,
            pca_random_state=self._pca_random_state,
            ignore_labels=ignore,
        )

    def aggregate_overall(self, layer_scores, layers):
        """Return score of the deepest encoder layer.

        Paper (GBC, CVPR 2022): transferability is measured in the source
        model's *feature space*, i.e. the backbone output before any task-
        specific head.  For ResEncL this is encoder.stages.4.

        Falls back to the last valid layer if no encoder layer is found.
        """
        score = _deepest_encoder_score(layer_scores, layers)
        if score is not None:
            return score
        # Fallback: last layer in natural sort order
        valid = [l for l in layers if layer_scores.get(l) is not None]
        if valid:
            return layer_scores[sorted(valid, key=_nat_sort_key)[-1]]
        return None


# ===========================================================================
# LEEP
# ===========================================================================


class LEEPMetric(MetricComputer):
    """Log Expected Empirical Prediction (Nguyen et al., ICML 2020).

    Two operating modes controlled by ``use_logits``:

    ``use_logits=True``  (default, paper-faithful)
        The features stored in feature_dict are the source model's **output
        logits** (shape N×S, where S = number of source-task classes).
        Softmax is applied to obtain θ(x_i) ∈ Δ^S, which is then used in
        the paper's EEP formula:
            LEEP = 1/n · Σ_i log Σ_{z∈Z} P̂(y_i|z) · θ(x_i)_z
        To use this mode, configure the **model output layer** (e.g. the
        segmentation head's final conv, or the classification head) as the
        feature extraction layer.  Do NOT use an intermediate encoder layer.

    ``use_logits=False``  (N-LEEP / centroid proxy, kept for backward compat)
        Intermediate features (any dimension) are used.  L2-normalised class
        centroids stand in for the source model head, and cosine-softmax gives
        a proxy θ(x_i).  This is closer to N-LEEP (Li et al., CVPR 2021) than
        to the original LEEP and is provided only as a fallback when you cannot
        hook the model's output layer.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        use_logits: bool = True,
        temperature: float = 1.0,       # only used in use_logits=False mode
    ):
        self._ignore_background = ignore_background
        self._use_logits = bool(use_logits)
        self._temperature = float(temperature)
        # --- internal state (cleared by reset) ---
        self._features: Dict[str, List[np.ndarray]] = {}
        self._labels:   Dict[str, List[np.ndarray]] = {}

    @property
    def name(self) -> str:
        return "leep"

    # --- lifecycle -----------------------------------------------------------

    def reset(self):
        self._features = {}
        self._labels   = {}

    # --- accumulator ---------------------------------------------------------

    def accumulate(self, layer: str, feature_dict: dict):
        self._features.setdefault(layer, [])
        self._labels.setdefault(layer, [])

        for lb, feats in feature_dict.items():
            if lb == GLOBAL_KEY or not isinstance(lb, (int, np.integer)):
                continue
            lb_int = int(lb)
            if self._ignore_background and lb_int == 0:
                continue
            feats = np.asarray(feats)
            if feats.size == 0:
                continue
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
            self._features[layer].append(feats)
            self._labels[layer].append(
                np.full(feats.shape[0], lb_int, dtype=np.int64)
            )

    # --- scoring -------------------------------------------------------------

    def compute(self, layer: str) -> Optional[float]:
        if not self._features.get(layer):
            return None

        X = np.concatenate(self._features[layer], axis=0)  # (N, D)
        y = np.concatenate(self._labels[layer],   axis=0)  # (N,)
        N = X.shape[0]
        if N < 2:
            return None

        # Remap target labels to contiguous [0, K)
        classes = np.unique(y)
        K = len(classes)
        if K < 2:
            return None
        label_map = {int(c): i for i, c in enumerate(classes)}
        y_mapped = np.array([label_map[int(v)] for v in y], dtype=np.int64)

        if self._use_logits:
            return self._leep_from_logits(X, y_mapped)
        else:
            return self._leep_centroid_proxy(X, y_mapped, K)

    # ------------------------------------------------------------------
    # Paper-faithful LEEP (use_logits=True)
    # ------------------------------------------------------------------

    def _leep_from_logits(
        self,
        logits: np.ndarray,    # (N, S) — source model output logits
        y_target: np.ndarray,  # (N,)   — target labels in [0, K)
    ) -> Optional[float]:
        """True LEEP as in Nguyen et al. (2020).

        logits[i] = θ(x_i) raw scores over source-task label space Z
        (size S = number of source classes).  We apply softmax to get
        proper probabilities, then follow the paper's three-step recipe.
        """
        S = logits.shape[1]
        if S < 1:
            return None

        # --- Step 1: source predictions θ(x_i) via softmax ---
        # Numerically stable softmax
        lgt = logits - logits.max(axis=1, keepdims=True)
        exp_lgt = np.exp(lgt)
        theta = exp_lgt / exp_lgt.sum(axis=1, keepdims=True)  # (N, S)

        # --- Step 2 & 3: delegate to paper formula ---
        return leep_score(theta, y_target)

    # ------------------------------------------------------------------
    # Centroid-proxy fallback (use_logits=False)
    # ------------------------------------------------------------------

    def _leep_centroid_proxy(
        self,
        X: np.ndarray,         # (N, D) intermediate features
        y_mapped: np.ndarray,  # (N,)   target labels in [0, K)
        K: int,
    ) -> Optional[float]:
        """N-LEEP-style approximation: L2-normalised centroids → cosine softmax.

        This is NOT the paper's LEEP.  Use it only when the model output
        layer is unavailable (use_logits=False).  In this mode the 'source
        label space' Z is the set of target classes, making the measure
        circular.  For a proper transferability score prefer use_logits=True.
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_n = X / np.maximum(norms, 1e-12)                      # (N, D)

        # L2-normalised class centroids
        centroids = np.zeros((K, X.shape[1]), dtype=np.float64)
        for j in range(K):
            centroids[j] = X_n[y_mapped == j].mean(axis=0)
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(c_norms, 1e-12)      # (K, D)

        sim = X_n @ centroids.T / self._temperature              # (N, K)
        sim -= sim.max(axis=1, keepdims=True)
        exp_sim = np.exp(sim)
        predictions = exp_sim / exp_sim.sum(axis=1, keepdims=True)  # (N, K)

        return leep_score(predictions, y_mapped)

    # --- overall aggregation -------------------------------------------------

    def aggregate_overall(self, layer_scores, layers):
        """Return the paper-faithful layer score.

        use_logits=True  (default):
            Uses the model's output logits as θ(x_i).  The relevant layer is
            the model's *final output* (segmentation/classification head), not
            an intermediate encoder layer.  We look for (in priority order):
              1. cls_head layers         (classification task)
              2. deepest decoder layer   (segmentation task)
              3. last valid layer overall (fallback)

        use_logits=False  (N-LEEP proxy):
            Uses intermediate features; the deepest encoder layer is selected
            (same reasoning as GBC / LogME).
        """
        if self._use_logits:
            score = _deepest_output_score(layer_scores, layers)
        else:
            score = _deepest_encoder_score(layer_scores, layers)

        if score is not None:
            return score
        # Fallback: mean of all valid layers
        valid = [layer_scores[l] for l in layers if layer_scores.get(l) is not None]
        return float(np.mean(valid)) if valid else None


class LogMEMetric(MetricComputer):

    def __init__(
        self,
        ignore_background: bool = False,
        subsample_per_case: Optional[int] = None,
        max_per_class: int = 20000,
        seed: int = 0,
    ):
        self._ignore_background = ignore_background
        self._subsample_per_case = subsample_per_case
        self._max_per_class = max_per_class
        self._seed = seed
        # --- internal state ---
        self._feat_all: Dict[str, List[np.ndarray]] = {}
        self._lab_all: Dict[str, List[np.ndarray]] = {}
        self._case_idx: int = 0

    @property
    def name(self) -> str:
        return "logme"

    # --- lifecycle -----------------------------------------------------------

    def reset(self):
        self._feat_all = {}
        self._lab_all = {}
        self._case_idx = 0

    def begin_case(self, case_idx: int):
        self._case_idx = int(case_idx)

    # --- accumulator ---------------------------------------------------------

    def accumulate(self, layer: str, feature_dict: dict):
        self._feat_all.setdefault(layer, [])
        self._lab_all.setdefault(layer, [])

        parts = []
        labs = []
        for lb, feats in feature_dict.items():
            if lb == GLOBAL_KEY or not isinstance(lb, (int, np.integer)):
                continue
            lb_int = int(lb)
            if self._ignore_background and lb_int == 0:
                continue
            feats = np.asarray(feats)
            if feats.size == 0:
                continue
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            parts.append(feats)
            labs.append(np.full(feats.shape[0], lb_int, dtype=np.int64))

        if not parts:
            return

        X = np.concatenate(parts, axis=0)
        y = np.concatenate(labs, axis=0)

        # optional subsampling per case
        if self._subsample_per_case and X.shape[0] > self._subsample_per_case:
            rng = np.random.default_rng(self._case_idx)
            budget = int(min(self._subsample_per_case, X.shape[0]))
            idx = rng.choice(X.shape[0], size=budget, replace=False)
            X, y = X[idx], y[idx]

        self._feat_all[layer].append(X)
        self._lab_all[layer].append(y)

    # --- scoring -------------------------------------------------------------

    def compute(self, layer: str) -> Optional[float]:
        if not self._feat_all.get(layer):
            return None

        X = np.concatenate(self._feat_all[layer], axis=0)
        y = np.concatenate(self._lab_all[layer], axis=0)

        # per-class hard cap
        if self._max_per_class:
            rng = np.random.default_rng(0)
            parts = []
            for lb in np.unique(y):
                idx = np.where(y == lb)[0]
                if idx.size > self._max_per_class:
                    idx = rng.choice(idx, size=self._max_per_class, replace=False)
                parts.append(idx)
            keep = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
            X, y = X[keep], y[keep]

        if X.shape[0] < 2:
            return None

        return logme_score(X, y)

    def aggregate_overall(self, layer_scores, layers):
        """Return score of the deepest encoder layer.

        Paper (LogME, ICML 2021): uses features {f_i = φ(x_i)} from the
        pre-trained feature extractor φ, i.e. the encoder output.
        For ResEncL this is encoder.stages.4.

        Falls back to mean of all valid layers if no encoder layer is found.
        """
        score = _deepest_encoder_score(layer_scores, layers)
        if score is not None:
            return score
        # Fallback: mean across all valid layers
        valid = [layer_scores[l] for l in layers if layer_scores.get(l) is not None]
        return float(np.mean(valid)) if valid else None


# ===========================================================================
# Registry
# ===========================================================================


METRIC_REGISTRY: Dict[str, type] = {
    "ccfv": CCFVMetric,
    "ccfv_slow": CCFVSlowMetric,
    "gbc": GBCMetric,
    "leep": LEEPMetric,
    "logme": LogMEMetric,
    "tda": TDARTDLiteMetric,
    "tda_lbtc": TDALBTCMetric,
    # ---- NTK-based metrics (Section 9 theoretical analysis) ----
    "ntk_cond":       NTKCondMetric,       # Exp.1: condition number proxy
    "ntk_trace":      NTKTraceMetric,      # Exp.2: NTK-Trace baseline
    "ntk_layerwise":  NTKLayerwiseMetric,  # Exp.1: per-layer κ JSON dump
    "ntk_spectrum":   NTKSpectrumMetric,   # Exp.3: eigenvector-label alignment
    # ---- TLC: Topological Label Coherence (Section 3.6, classification/regression) ----
    "tlc":            TLCMetric,            # Section 3.6: MST leakage rate on encoder features
}


def get_metric(name: str, **kwargs) -> MetricComputer:

    key = name.lower().strip()
    if key not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    cls = METRIC_REGISTRY[key]
    valid_params = {
        p for p in inspect.signature(cls.__init__).parameters if p != "self"
    }
    return cls(**{k: v for k, v in kwargs.items() if k in valid_params})