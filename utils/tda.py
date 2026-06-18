from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np

try:
    from scipy.spatial.distance import cdist
    from scipy.ndimage import binary_erosion
except Exception:
    cdist = None
    binary_erosion = None

try:
    from utils.sliding_window_sampling import GLOBAL_KEY, BOUNDARY_KEY, BOUNDARY_LABEL_KEY
except Exception:
    GLOBAL_KEY = "__global__"
    BOUNDARY_KEY = "__boundary__"
    BOUNDARY_LABEL_KEY = "__boundary_labels__"

GLOBAL_ALIASES = {"__global__", "__GLOBAL__", "global", "GLOBAL", "GLOBAL_KEY"}


def _is_global_key(k: Any) -> bool:
    if k == GLOBAL_KEY:
        return True
    if isinstance(k, str) and k.lower() in {x.lower() for x in GLOBAL_ALIASES}:
        return True
    return False


def _to_numpy(x: Any) -> np.ndarray:
    # torch tensor -> numpy
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_2d(x: Any) -> np.ndarray:
    a = _to_numpy(x)
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        return a.reshape(-1, 1)
    # 兜底：把第0维当 N
    return a.reshape((a.shape[0], -1))


def _label_distance(y: np.ndarray, label_far: float) -> np.ndarray:
    """Binary label distance: intra-class=0, inter-class=label_far."""
    y = np.asarray(y).reshape(-1, 1)
    eq = (y == y.T)
    D = np.where(eq, 0.0, float(label_far)).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D


def _mst_weight_prim(W: np.ndarray) -> float:
    """Dense Prim MST O(N^2)."""
    W = np.asarray(W, dtype=np.float32)
    N = W.shape[0]
    if N <= 1:
        return 0.0

    in_tree = np.zeros(N, dtype=bool)
    min_cost = W[0].copy()
    min_cost[0] = np.inf
    in_tree[0] = True

    total = 0.0
    for _ in range(N - 1):
        j = int(np.argmin(min_cost))
        c = float(min_cost[j])
        if not np.isfinite(c):
            break
        total += c
        in_tree[j] = True
        min_cost[j] = np.inf
        min_cost = np.minimum(min_cost, W[j])
        min_cost[in_tree] = np.inf
    return float(total)


def _mst_edges_prim(W: np.ndarray) -> List[Tuple[int, int]]:
    """
    Prim's MST algorithm that returns edges instead of total weight.
    Returns list of (u, v) tuples representing MST edges.
    """
    W = np.asarray(W, dtype=np.float32)
    N = W.shape[0]
    if N <= 1:
        return []
    
    in_tree = np.zeros(N, dtype=bool)
    min_cost = W[0].copy()
    min_cost[0] = np.inf
    parent = np.full(N, -1, dtype=np.int32)
    in_tree[0] = True
    
    edges = []
    for _ in range(N - 1):
        j = int(np.argmin(min_cost))
        c = float(min_cost[j])
        if not np.isfinite(c):
            break
        
        # Add edge
        if parent[j] >= 0:
            edges.append((parent[j], j))
        
        in_tree[j] = True
        
        # Update costs and parents
        for k in range(N):
            if not in_tree[k] and W[j, k] < min_cost[k]:
                min_cost[k] = W[j, k]
                parent[k] = j
        
        min_cost[j] = np.inf
    
    return edges


def _parse_int_set(spec: Union[str, List[int], Tuple[int, ...], set, None]) -> set:
    if spec is None:
        return set()
    if isinstance(spec, set):
        return set(int(x) for x in spec)
    if isinstance(spec, (list, tuple)):
        return set(int(x) for x in spec)
    s = str(spec).strip()
    if s == "":
        return set()
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = set()
    for p in parts:
        try:
            out.add(int(p))
        except Exception:
            pass
    return out


def _safe_quantile_bins(scores: np.ndarray, num_bins: int) -> np.ndarray:
    """
    分桶：按分位数切箱，使每个桶尽量均匀。
    返回 bin_id: [0..num_bins-1]
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0 or num_bins <= 1:
        return np.zeros((scores.size,), dtype=np.int32)

    # 计算阈值：1/num_bins, 2/num_bins, ...
    qs = [i / num_bins for i in range(1, num_bins)]
    thr = np.quantile(scores, qs).astype(np.float32)

    # digitize: 输出 0..len(thr)
    bin_id = np.digitize(scores, thr, right=False).astype(np.int32)
    # clip 保守
    bin_id = np.clip(bin_id, 0, num_bins - 1)
    return bin_id


def _safe_uniform_bins(scores: np.ndarray, num_bins: int) -> np.ndarray:
    """
    分桶：按 min~max 等宽切箱。
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0 or num_bins <= 1:
        return np.zeros((scores.size,), dtype=np.int32)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros((scores.size,), dtype=np.int32)
    edges = np.linspace(lo, hi, num_bins + 1, dtype=np.float32)[1:-1]  # 去掉两端
    bin_id = np.digitize(scores, edges, right=False).astype(np.int32)
    bin_id = np.clip(bin_id, 0, num_bins - 1)
    return bin_id


class TDARTDLiteMetric:

    def __init__(
        self,
        feature_metric: str = "euclidean",
        aux_op: str = "min",
        ignore_labels: str = "",
        per_class_max: int = 800,
        label_far: float = -1.0,
        seed: int = 0,
        debug_save_dir: str = "",
        n_clusters_per_class: int = 0,
    ):
        self.feature_metric = feature_metric
        self.aux_op = aux_op
        self.ignore_labels = [int(x) for x in str(ignore_labels).split(",") if str(x).strip() != ""]
        self.per_class_max = int(per_class_max)
        self.label_far = float(label_far)
        self.seed = int(seed)
        self._debug_save_dir = str(debug_save_dir).strip()
        self.n_clusters_per_class = int(n_clusters_per_class)

        self._case_idx = 0
        self._rng = np.random.default_rng(self.seed)
        self._scores: Dict[str, List[float]] = {}
        self._debug_case_counter: int = 0

    @property
    def name(self) -> str:
        return "tda"

    def reset(self):
        self._case_idx = 0
        self._rng = np.random.default_rng(self.seed)
        self._scores = {}

    def begin_case(self, case_idx: int):
        self._case_idx = int(case_idx)
        self._rng = np.random.default_rng(self.seed + self._case_idx)

    def _build_xy(self, feature_dict: Dict[Any, Any]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        parts = []
        ys = []
        for k, v in feature_dict.items():
            if _is_global_key(k):
                continue
            try:
                lb = int(k)
            except Exception:
                continue
            if lb in self.ignore_labels:
                continue
            arr = _to_2d(v).astype(np.float32, copy=False)
            if arr.shape[0] < 1:
                continue
            if self.per_class_max > 0 and arr.shape[0] > self.per_class_max:
                idx = self._rng.choice(arr.shape[0], size=self.per_class_max, replace=False)
                arr = arr[idx]
            parts.append(arr)
            ys.append(np.full((arr.shape[0],), lb, dtype=np.int32))

        if not parts:
            return None
        X = np.concatenate(parts, axis=0)
        y = np.concatenate(ys, axis=0)
        if X.shape[0] < 2:
            return None
        return X, y

    def _cluster_to_prototypes(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        对每个类单独做 MiniBatchKMeans，取簇中心作为 prototype。
        每类最多 self.n_clusters_per_class 个 prototype。
        只在 self.n_clusters_per_class > 0 时调用。
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise RuntimeError(
                "scikit-learn is required for prototype clustering. "
                "Install: pip install scikit-learn"
            )

        proto_feats, proto_labels = [], []
        for c in np.unique(y):
            mask = (y == c)
            Xc = X[mask]
            k = min(self.n_clusters_per_class, Xc.shape[0])
            if k <= 1:
                # 点数不足，直接保留原始点
                proto_feats.append(Xc)
                proto_labels.append(np.full(Xc.shape[0], c, dtype=np.int32))
                continue
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.seed,
                n_init=3,
                batch_size=max(256, k * 10),
            )
            km.fit(Xc)
            proto_feats.append(km.cluster_centers_.astype(np.float32))
            proto_labels.append(np.full(k, c, dtype=np.int32))

        X_proto = np.vstack(proto_feats)
        y_proto = np.concatenate(proto_labels)
        return X_proto, y_proto

    def accumulate(self, layer: str, feature_dict: Dict[Any, Any]):
        self._scores.setdefault(layer, [])
        if cdist is None:
            raise RuntimeError("scipy is required for RTD-Lite (scipy.spatial.distance.cdist). Please install scipy.")

        xy = self._build_xy(feature_dict)
        if xy is None:
            return
        X, y = xy

        # ── Prototype clustering（可选）──────────────────────────────────────
        # 当 n_clusters_per_class > 0 时，对每个类单独做 K-Means 聚类，
        # 用簇中心（prototype）代替原始点建 MST，降低计算量同时保留拓扑结构。
        if self.n_clusters_per_class > 0 and X.shape[0] > self.n_clusters_per_class * len(np.unique(y)):
            X, y = self._cluster_to_prototypes(X, y)
            if X.shape[0] < 2:
                return
        # ────────────────────────────────────────────────────────────────────

        # feature distances
        Df = cdist(X, X, metric=self.feature_metric).astype(np.float32, copy=False)

        # auto label_far: paper-version lambda = median pairwise feature distance
        # on the current sampled set (off-diagonal entries).
        if self.label_far < 0:
            N = Df.shape[0]
            if N > 1:
                tri_u = np.triu_indices(N, k=1)
                vals = Df[tri_u]
            else:
                vals = np.asarray([], dtype=np.float32)
            lf = float(np.median(vals)) if vals.size else 1.0
            if not np.isfinite(lf) or lf <= 0:
                lf = 1.0
            label_far_use = lf
        else:
            label_far_use = self.label_far

        Dl = _label_distance(y, label_far=label_far_use)

        if self.aux_op == "min":
            Da = np.minimum(Df, Dl)
        elif self.aux_op == "max":
            Da = np.maximum(Df, Dl)
        else:
            raise ValueError("aux_op must be min or max")

        # Fix: intra-class edges in Da are all 0, causing Prim's argmin to
        # break ties by array index (sampling order dependent).  Add a tiny
        # feature-distance-proportional perturbation to intra-class edges so
        # Prim always selects the same (nearest) intra-class neighbour,
        # making MST(Da) deterministic for a given PKL regardless of how
        # many times the metric is computed.
        #
        # Perturbation scale: eps = label_far * 1e-5
        # → max intra-class edge weight ≈ 1e-5 * label_far  (negligible)
        # → inter-class edges unaffected (min(Df, label_far), unchanged)
        max_df = float(np.max(Df)) if Df.size > 0 else 1.0
        if max_df > 0:
            eps = float(label_far_use) * 1e-5
            same_class = (y.reshape(-1, 1) == y.reshape(1, -1))
            Da = Da.astype(np.float64)
            Da[same_class] = eps * (Df[same_class].astype(np.float64) / max_df)
            np.fill_diagonal(Da, 0.0)
            Da = Da.astype(np.float32)

        # ── Debug: save MST(Da) topology to JSON ──────────────────────────
        if self._debug_save_dir:
            import json, os
            os.makedirs(self._debug_save_dir, exist_ok=True)
            da_edges = _mst_edges_prim(Da)
            Wa_dbg = _mst_weight_prim(Da)
            Wf_dbg = _mst_weight_prim(Df)
            record = {
                "case_idx": self._debug_case_counter,
                "layer": layer,
                "n_points": int(len(y)),
                "classes": [int(c) for c in np.unique(y).tolist()],
                "label_far": float(label_far_use),
                "Wf": float(Wf_dbg),
                "Wa": float(Wa_dbg),
                "delta": float(abs(Wf_dbg - Wa_dbg)),
                "n_edges": len(da_edges),
                "n_cross_edges": int(sum(1 for u, v in da_edges if y[u] != y[v])),
                "edges": [
                    {
                        "u": int(u), "v": int(v),
                        "label_u": int(y[u]), "label_v": int(y[v]),
                        "is_cross": int(y[u] != y[v]),
                        "da_weight": float(Da[u, v]),
                        "df_weight": float(Df[u, v]),
                    }
                    for u, v in da_edges
                ],
            }
            fname = f"case{self._debug_case_counter:04d}_{layer.replace('.', '_')}.json"
            with open(os.path.join(self._debug_save_dir, fname), "w") as _f:
                json.dump(record, _f, indent=2)
            self._debug_case_counter += 1
        # ── End debug ──────────────────────────────────────────────────────

        # Paper-version GRTD: macro-averaged, per-class normalized
        # tree-weight discrepancy on feature-MST edges.
        #
        # For each class c:
        #   E_c = MST(Df) edges with at least one endpoint of class c
        #   delta_c = (sum_{e in E_c} w_feat(e) - sum_{e in E_c} w_sem(e))
        #             / (sum_{e in E_c} w_feat(e) + eps)
        #   T_GRTD = - mean_c(delta_c)
        #
        # Important: w_sem(e) is evaluated on the same feature-MST edge e,
        # with same-class cost 0 and cross-class cost min(w_feat(e), lambda).
        # Do not use the perturbed Da values here, because the paper formula
        # requires exact same-class semantic cost 0.
        classes = np.unique(y)
        edge_list = _mst_edges_prim(Df)
        if not edge_list:
            return

        edge_arr = np.array(edge_list, dtype=np.int32)  # (E, 2)
        eu, ev = edge_arr[:, 0], edge_arr[:, 1]
        wf_edges = Df[eu, ev].astype(np.float64)
        same_edge = (y[eu] == y[ev])
        wsem_edges = np.where(
            same_edge,
            0.0,
            np.minimum(wf_edges, float(label_far_use)),
        ).astype(np.float64)

        eps_norm = 1e-8
        per_class_delta = []
        for c in classes:
            # E_c: feature-MST edges where at least one endpoint belongs to class c
            in_Ec = (y[eu] == c) | (y[ev] == c)
            if not np.any(in_Ec):
                continue

            sum_wf = float(np.sum(wf_edges[in_Ec]))
            sum_wsem = float(np.sum(wsem_edges[in_Ec]))
            if sum_wf <= 0.0 or not np.isfinite(sum_wf):
                continue

            delta_c = (sum_wf - sum_wsem) / (sum_wf + eps_norm)
            per_class_delta.append(float(delta_c))

        if not per_class_delta:
            return

        delta = float(np.mean(per_class_delta))
        self._scores[layer].append(delta)

    def compute(self, layer: str) -> Optional[float]:
        vals = self._scores.get(layer)
        if not vals:
            return None
        return -float(np.mean(vals))  # Negative: higher is better

    def aggregate_overall(self, layer_scores: Dict[str, Optional[float]], layers: List[str]) -> Optional[float]:
        vals = [layer_scores.get(l) for l in layers if layer_scores.get(l) is not None]
        if not vals:
            return None
        return float(np.mean(vals))


class TDALBTCMetric(TDARTDLiteMetric):
    def __init__(
        self,
        feature_metric: str = "euclidean",
        ignore_labels: str = "",
        
        # LBTC 特定参数
        num_boundary_patches: int = 50,      # 采样边界补丁数量
        patch_size: int = 16,                # 补丁大小 w×w
        foreground_labels: Union[str, List[int], Tuple[int, ...], set] = "nonzero",
        
        # 采样控制
        per_patch_max_points: int = 200,     # 每个补丁最多采样点数

        # 边界锚点来源
        #   spatial_gt:   论文主版本；使用 sampler 中由 GT mask morphological gradient 得到的 __boundary__ 锚点
        #   feature_proxy:旧近似版本；使用前景/背景在 feature space 中最近的 30% 点作为边界锚点
        #   auto:         优先 spatial_gt，若旧缓存/PKL 中没有 __boundary__ 则回退 feature_proxy
        boundary_anchor_mode: str = "spatial_gt",

        seed: int = 0,
    ):
        super().__init__(
            feature_metric=feature_metric,
            aux_op="min",  # LBTC不使用这个参数
            ignore_labels=ignore_labels,
            per_class_max=0,  # 禁用全局采样
            label_far=-1.0,   # LBTC不使用这个参数
            seed=seed,
        )
        
        self.num_boundary_patches = int(num_boundary_patches)
        self.patch_size = int(patch_size)
        self.foreground_labels_spec = foreground_labels
        self.per_patch_max_points = int(per_patch_max_points)
        self.boundary_anchor_mode = str(boundary_anchor_mode).strip().lower()
        valid_modes = {"auto", "spatial_gt", "spatial", "gt", "morphological", "feature_proxy", "feature", "legacy", "approx"}
        if self.boundary_anchor_mode not in valid_modes:
            raise ValueError(
                f"Unknown boundary_anchor_mode={boundary_anchor_mode!r}. "
                "Use 'auto', 'spatial_gt', or 'feature_proxy'."
            )

    @property
    def name(self) -> str:
        return "tda_lbtc"

    def _resolve_foreground_set(self, feature_dict: Dict[Any, Any]) -> set:
        """解析前景标签集合"""
        spec = self.foreground_labels_spec
        if isinstance(spec, str):
            s = spec.strip().lower()
            if s in {"nonzero", "all_nonzero", "non_background", "nonbackground"}:
                fg = set()
                for k in feature_dict.keys():
                    if _is_global_key(k):
                        continue
                    try:
                        lb = int(k)
                    except Exception:
                        continue
                    if lb in self.ignore_labels:
                        continue
                    if lb != 0:
                        fg.add(lb)
                return fg
        return _parse_int_set(spec)

    def _identify_boundary_samples(
        self, 
        features_by_label: Dict[int, np.ndarray],
        fg_set: set
    ) -> Tuple[np.ndarray, np.ndarray]:

        fg_feats = []
        bg_feats = []
        
        for lb, feats in features_by_label.items():
            if lb in fg_set:
                fg_feats.append(feats)
            elif lb == 0:  # 背景
                bg_feats.append(feats)
        
        if not fg_feats or not bg_feats:
            return (
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )
        
        fg_all = np.concatenate(fg_feats, axis=0)
        bg_all = np.concatenate(bg_feats, axis=0)
        
        # 计算前景到背景的距离矩阵（采样以避免计算爆炸）
        max_fg = min(2000, fg_all.shape[0])
        max_bg = min(2000, bg_all.shape[0])

        fg_sample_idx = (
            self._rng.choice(fg_all.shape[0], size=max_fg, replace=False)
            if fg_all.shape[0] > max_fg
            else np.arange(fg_all.shape[0])
        )
        bg_sample_idx = (
            self._rng.choice(bg_all.shape[0], size=max_bg, replace=False)
            if bg_all.shape[0] > max_bg
            else np.arange(bg_all.shape[0])
        )
        
        fg_sample = fg_all[fg_sample_idx]
        bg_sample = bg_all[bg_sample_idx]
        
        # 计算距离
        dist_matrix = cdist(fg_sample, bg_sample, metric=self.feature_metric)
        
        # 前景边界：到背景最近的点
        min_dist_to_bg = np.min(dist_matrix, axis=1)
        fg_boundary_threshold = np.percentile(min_dist_to_bg, 30)  # 最近的30%
        fg_boundary_mask = min_dist_to_bg <= fg_boundary_threshold
        fg_boundary = fg_sample[fg_boundary_mask]
        
        # 背景边界：到前景最近的点
        min_dist_to_fg = np.min(dist_matrix, axis=0)
        bg_boundary_threshold = np.percentile(min_dist_to_fg, 30)
        bg_boundary_mask = min_dist_to_fg <= bg_boundary_threshold
        bg_boundary = bg_sample[bg_boundary_mask]
        
        return fg_boundary, bg_boundary

    def _sample_boundary_patches_feature_proxy(
        self,
        features_by_label: Dict[int, np.ndarray],
        fg_set: set,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        旧近似实现：在 feature space 里找前景/背景最近的 30% 点作为边界锚点。

        这个模式不依赖 GT mask 的空间形态学边界，适合旧缓存/没有 __boundary__ 的特征文件，
        但严格性弱于论文主文的 morphological-gradient boundary anchors。

        返回：List of (patch_features, patch_labels)
        """
        # 识别边界点
        fg_boundary, bg_boundary = self._identify_boundary_samples(features_by_label, fg_set)

        if fg_boundary.shape[0] == 0 or bg_boundary.shape[0] == 0:
            return []
        
        # 收集所有特征和标签
        all_feats = []
        all_labels = []
        
        for lb, feats in features_by_label.items():
            all_feats.append(feats)
            all_labels.append(np.full(feats.shape[0], lb, dtype=np.int32))
        
        all_feats = np.concatenate(all_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 合并边界点作为锚点候选
        boundary_points = np.concatenate([fg_boundary, bg_boundary], axis=0) if fg_boundary.shape[0] > 0 else bg_boundary
        
        if boundary_points.shape[0] == 0:
            return []
        
        # 采样N个锚点
        num_anchors = min(self.num_boundary_patches, boundary_points.shape[0])
        anchor_indices = self._rng.choice(boundary_points.shape[0], size=num_anchors, replace=False)
        anchors = boundary_points[anchor_indices]
        
        patches = []
        
        # 对每个锚点，提取"局部补丁"（特征空间中最近的点）
        for anchor in anchors:
            # 计算到锚点的距离
            distances = np.linalg.norm(all_feats - anchor, axis=1)
            
            # 选择最近的patch_size个点
            patch_size_actual = min(self.patch_size, all_feats.shape[0])
            nearest_indices = np.argpartition(distances, patch_size_actual-1)[:patch_size_actual]
            
            patch_feats = all_feats[nearest_indices]
            patch_labels = all_labels[nearest_indices]
            
            # 确保补丁中既有前景又有背景（否则没有意义）
            has_fg = np.any(patch_labels > 0)
            has_bg = np.any(patch_labels == 0)
            
            if has_fg and has_bg:
                # 如果点数过多，采样
                if self.per_patch_max_points > 0 and patch_feats.shape[0] > self.per_patch_max_points:
                    sample_idx = self._rng.choice(
                        patch_feats.shape[0], 
                        size=self.per_patch_max_points, 
                        replace=False
                    )
                    patch_feats = patch_feats[sample_idx]
                    patch_labels = patch_labels[sample_idx]
                
                patches.append((patch_feats, patch_labels))
        
        return patches

    def _extract_spatial_boundary_anchors(
        self,
        feature_dict: Dict[Any, Any],
        fg_set: set,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        读取 sampler 提供的 GT 空间边界锚点。

        sliding_window_sampling.ms_sliding_window_sampling 在 boundary_sample_num>0 时会根据
        GT mask 的 morphological gradient / 6-neighbour erosion 生成：
          - BOUNDARY_KEY:       边界 voxel 的特征向量，shape=(K, C)
          - BOUNDARY_LABEL_KEY: 对应 GT 标签，shape=(K,)
        这才是论文主文 LBTC 的边界锚点来源。
        """
        if BOUNDARY_KEY not in feature_dict or BOUNDARY_LABEL_KEY not in feature_dict:
            return None

        anchors = _to_2d(feature_dict.get(BOUNDARY_KEY)).astype(np.float32, copy=False)
        labels = _to_numpy(feature_dict.get(BOUNDARY_LABEL_KEY)).reshape(-1).astype(np.int32, copy=False)

        if anchors.shape[0] == 0 or labels.shape[0] == 0:
            return None

        # 防御：如果缓存文件长度不一致，截断到共同长度。
        n = min(anchors.shape[0], labels.shape[0])
        anchors = anchors[:n]
        labels = labels[:n]

        keep = np.ones((n,), dtype=bool)
        if self.ignore_labels:
            keep &= ~np.isin(labels, np.asarray(self.ignore_labels, dtype=np.int32))
        # 通常 morphological boundary 只会给前景边界；这里仍按 foreground_labels 做一次过滤，
        # 避免旧缓存把背景边界也混进来。若用户显式传空集合则不过滤。
        if fg_set:
            keep &= np.isin(labels, np.asarray(sorted(fg_set), dtype=np.int32))

        anchors = anchors[keep]
        labels = labels[keep]

        if anchors.shape[0] == 0:
            return None
        return anchors, labels

    def _collect_pool_arrays(
        self,
        features_by_label: Dict[int, np.ndarray],
        extra_features: Optional[np.ndarray] = None,
        extra_labels: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """拼接局部 patch 检索用的候选点池。"""
        all_feats: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for lb, feats in features_by_label.items():
            arr = _to_2d(feats).astype(np.float32, copy=False)
            if arr.shape[0] == 0:
                continue
            all_feats.append(arr)
            all_labels.append(np.full(arr.shape[0], int(lb), dtype=np.int32))

        # 把边界锚点也加入候选池，确保每个 anchor 至少可以被自身/相近边界点覆盖到。
        if extra_features is not None and extra_labels is not None and extra_features.shape[0] > 0:
            ef = _to_2d(extra_features).astype(np.float32, copy=False)
            el = np.asarray(extra_labels, dtype=np.int32).reshape(-1)
            n = min(ef.shape[0], el.shape[0])
            if n > 0:
                all_feats.append(ef[:n])
                all_labels.append(el[:n])

        if not all_feats:
            return None

        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)
        if X.shape[0] < 2:
            return None
        return X, y

    def _sample_boundary_patches_spatial_gt(
        self,
        feature_dict: Dict[Any, Any],
        features_by_label: Dict[int, np.ndarray],
        fg_set: set,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        论文主版本：使用 GT mask morphological gradient 得到的空间边界 voxel 作为 anchors。

        注意：当前 feature_dict 中不保存完整空间坐标，因此这里采用“空间 GT 边界 anchor +
        特征空间近邻 patch”的折中实现：anchor 的来源严格来自 GT 空间边界；local patch
        通过该 anchor 在当前层特征空间中的最近邻构建。
        """
        anchor_pack = self._extract_spatial_boundary_anchors(feature_dict, fg_set)
        if anchor_pack is None:
            return []
        boundary_features, boundary_labels = anchor_pack

        pool = self._collect_pool_arrays(features_by_label, boundary_features, boundary_labels)
        if pool is None:
            return []
        all_feats, all_labels = pool

        num_anchors = min(self.num_boundary_patches, boundary_features.shape[0])
        if num_anchors <= 0:
            return []
        anchor_indices = self._rng.choice(boundary_features.shape[0], size=num_anchors, replace=False)
        anchors = boundary_features[anchor_indices]

        patches: List[Tuple[np.ndarray, np.ndarray]] = []
        for anchor in anchors:
            # 以 GT 空间边界点对应的 feature 为中心，在当前层 feature space 中取 w 个近邻。
            distances = np.linalg.norm(all_feats - anchor.reshape(1, -1), axis=1)
            k = min(self.patch_size, all_feats.shape[0])
            if k <= 1:
                continue
            if k >= all_feats.shape[0]:
                nearest_indices = np.arange(all_feats.shape[0])
            else:
                nearest_indices = np.argpartition(distances, k - 1)[:k]

            patch_feats = all_feats[nearest_indices]
            patch_labels = all_labels[nearest_indices]

            # 论文的 leakage 是跨“不同语义类”的边，不应只限制为 fg-vs-bg；
            # 因此只要求 patch 内至少有两个语义标签。
            if np.unique(patch_labels).size < 2:
                continue

            if self.per_patch_max_points > 0 and patch_feats.shape[0] > self.per_patch_max_points:
                sample_idx = self._rng.choice(
                    patch_feats.shape[0],
                    size=self.per_patch_max_points,
                    replace=False,
                )
                patch_feats = patch_feats[sample_idx]
                patch_labels = patch_labels[sample_idx]

            patches.append((patch_feats, patch_labels))

        return patches

    def _sample_boundary_patches(
        self,
        feature_dict: Dict[Any, Any],
        features_by_label: Dict[int, np.ndarray],
        fg_set: set,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """按 boundary_anchor_mode 选择论文版或旧近似版 LBTC 采样。"""
        mode = self.boundary_anchor_mode

        if mode in {"feature_proxy", "feature", "legacy", "approx"}:
            return self._sample_boundary_patches_feature_proxy(features_by_label, fg_set)

        if mode in {"spatial_gt", "spatial", "gt", "morphological"}:
            return self._sample_boundary_patches_spatial_gt(feature_dict, features_by_label, fg_set)

        # auto: 优先使用论文版 GT 空间边界；如果旧 PKL / 旧 sampler 没有边界锚点，则回退旧近似实现。
        patches = self._sample_boundary_patches_spatial_gt(feature_dict, features_by_label, fg_set)
        if patches:
            return patches
        return self._sample_boundary_patches_feature_proxy(features_by_label, fg_set)

    def _compute_local_divergence(
        self,
        patch_features: np.ndarray,
        patch_labels: np.ndarray
    ) -> float:
        """
        计算单个补丁的局部拓扑散度
        
        ∆k = (1/|Vk|-1) * Σ I(Y(u) ≠ Y(v))
        
        其中边(u,v)在特征MST中
        """
        N = patch_features.shape[0]
        if N < 2:
            return 0.0
        
        # 构建特征距离矩阵
        dist_matrix = cdist(patch_features, patch_features, metric=self.feature_metric)
        
        # 计算MST（返回边）
        mst_edges = _mst_edges_prim(dist_matrix)
        
        if not mst_edges:
            return 0.0
        
        # 计算违反边的数量
        num_violations = 0
        for u, v in mst_edges:
            if patch_labels[u] != patch_labels[v]:
                num_violations += 1
        
        # 局部散度 = 违反边比例
        divergence = num_violations / len(mst_edges) if len(mst_edges) > 0 else 0.0
        
        return divergence

    def accumulate(self, layer: str, feature_dict: Dict[Any, Any]):

        self._scores.setdefault(layer, [])
        
        if cdist is None:
            raise RuntimeError("scipy is required. Please install scipy.")
        
        # 解析前景集合
        fg_set = self._resolve_foreground_set(feature_dict)
        
        # 收集特征
        features_by_label: Dict[int, np.ndarray] = {}
        for k, v in feature_dict.items():
            if _is_global_key(k):
                continue
            try:
                lb = int(k)
            except Exception:
                continue
            if lb in self.ignore_labels:
                continue
            
            arr = _to_2d(v).astype(np.float32, copy=False)
            if arr.shape[0] < 1:
                continue
            
            features_by_label[lb] = arr
        
        if not features_by_label:
            return
        
        # 采样边界补丁。默认 spatial_gt：论文版 spatial-GT boundary anchors；可用 feature_proxy 跑旧近似实现。
        patches = self._sample_boundary_patches(feature_dict, features_by_label, fg_set)
        
        if not patches:
            return
        
        # 计算每个补丁的局部散度
        divergences = []
        for patch_feats, patch_labels in patches:
            div = self._compute_local_divergence(patch_feats, patch_labels)
            divergences.append(div)
        
        # 聚合：平均散度
        avg_divergence = float(np.mean(divergences)) if divergences else 0.0
        
        # 存储 leakage rate。论文版 LBTC 在 compute() 中返回 1 - mean(leakage)。
        self._scores[layer].append(avg_divergence)

    def compute(self, layer: str) -> Optional[float]:
        """
        计算论文版 LBTC 层级分数。

        LBTC = 1 - mean(leakage_rate)，范围约为 [0, 1]，越大越好。
        """
        vals = self._scores.get(layer)
        if not vals:
            return None
        score = 1.0 - float(np.mean(vals))
        return float(np.clip(score, 0.0, 1.0))


class TDARTDLiteBinnedMetric(TDARTDLiteMetric):

    def __init__(
        self,
        feature_metric: str = "euclidean",
        aux_op: str = "min",
        ignore_labels: str = "",
        foreground_labels: Union[str, List[int], Tuple[int, ...], set] = "nonzero",
        outside_mode: str = "background_only",
        background_labels: Union[str, List[int], Tuple[int, ...], set] = "0",
        num_bins: int = 3,
        bin_strategy: str = "quantile",
        pseudo_label_base: int = 100000,
        per_foreground_class_max: int = 15000,
        outside_max_total: int = 0,          # 0 = 不限制
        per_bin_max: int = 0,                # 0 = 不限制
        distance_to_fg_mode: str = "mean",
        fg_sample_for_distance: int = 0,     # 0 = 不限制
        label_far: float = -1.0,
        seed: int = 0,
    ):
        super().__init__(
            feature_metric=feature_metric,
            aux_op=aux_op,
            ignore_labels=ignore_labels,
            per_class_max=0,
            label_far=label_far,
            seed=seed,
        )

        self.foreground_labels_spec = foreground_labels
        self.outside_mode = str(outside_mode)
        self.background_labels = _parse_int_set(background_labels)

        self.num_bins = int(num_bins)
        self.bin_strategy = str(bin_strategy).lower()
        self.pseudo_label_base = int(pseudo_label_base)

        self.per_foreground_class_max = int(per_foreground_class_max)
        self.outside_max_total = int(outside_max_total)
        self.per_bin_max = int(per_bin_max)

        self.distance_to_fg_mode = str(distance_to_fg_mode).lower()
        self.fg_sample_for_distance = int(fg_sample_for_distance)

    @property
    def name(self) -> str:
        return "tda"

    def _resolve_foreground_set(self, feature_dict: Dict[Any, Any]) -> set:
        spec = self.foreground_labels_spec
        if isinstance(spec, str):
            s = spec.strip().lower()
            if s in {"nonzero", "all_nonzero", "non_background", "nonbackground"}:
                fg = set()
                for k in feature_dict.keys():
                    if _is_global_key(k):
                        continue
                    try:
                        lb = int(k)
                    except Exception:
                        continue
                    if lb in self.ignore_labels:
                        continue
                    if lb != 0:
                        fg.add(lb)
                return fg
        return _parse_int_set(spec)

    def _collect_points(
        self, feature_dict: Dict[Any, Any]
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        fg_set = self._resolve_foreground_set(feature_dict)

        fg_map: Dict[int, np.ndarray] = {}
        outside_parts: List[np.ndarray] = []

        for k, v in feature_dict.items():
            if _is_global_key(k):
                continue
            try:
                lb = int(k)
            except Exception:
                continue
            if lb in self.ignore_labels:
                continue

            arr = _to_2d(v).astype(np.float32, copy=False)
            if arr.shape[0] < 1:
                continue

            is_fg = (lb in fg_set) if fg_set else False

            if is_fg:
                if self.per_foreground_class_max > 0 and arr.shape[0] > self.per_foreground_class_max:
                    idx = self._rng.choice(arr.shape[0], size=self.per_foreground_class_max, replace=False)
                    arr = arr[idx]
                fg_map[lb] = arr
            else:
                if self.outside_mode == "background_only":
                    if lb in self.background_labels:
                        outside_parts.append(arr)
                else:
                    outside_parts.append(arr)

        outside = np.concatenate(outside_parts, axis=0) if outside_parts else np.zeros((0, 1), dtype=np.float32)
        if outside.ndim != 2:
            outside = outside.reshape((outside.shape[0], -1))

        if self.outside_max_total > 0 and outside.shape[0] > self.outside_max_total:
            idx = self._rng.choice(outside.shape[0], size=self.outside_max_total, replace=False)
            outside = outside[idx]

        return fg_map, outside

    def _get_fg_points_for_distance(self, fg_map: Dict[int, np.ndarray]) -> np.ndarray:
        if not fg_map:
            return np.zeros((0, 1), dtype=np.float32)
        
        all_fg = np.concatenate(list(fg_map.values()), axis=0)
        
        if self.fg_sample_for_distance > 0 and all_fg.shape[0] > self.fg_sample_for_distance:
            idx = self._rng.choice(all_fg.shape[0], size=self.fg_sample_for_distance, replace=False)
            all_fg = all_fg[idx]
        
        return all_fg

    def _compute_distance_scores(self, outside: np.ndarray, fg_points: np.ndarray) -> np.ndarray:
        if outside.size == 0 or fg_points.size == 0:
            return np.zeros((outside.shape[0],), dtype=np.float32)
        
        dist_matrix = cdist(outside, fg_points, metric=self.feature_metric).astype(np.float32, copy=False)
        
        mode = self.distance_to_fg_mode
        
        if mode == "centroid":
            centroid = np.mean(fg_points, axis=0, keepdims=True)
            scores = cdist(outside, centroid, metric=self.feature_metric).reshape(-1)
        elif mode == "min":
            scores = np.min(dist_matrix, axis=1)
        elif mode == "mean":
            scores = np.mean(dist_matrix, axis=1)
        elif mode == "median":
            scores = np.median(dist_matrix, axis=1)
        elif mode.startswith("percentile_"):
            try:
                percentile = float(mode.split("_")[1])
                scores = np.percentile(dist_matrix, percentile, axis=1)
            except Exception:
                scores = np.mean(dist_matrix, axis=1)
        else:
            scores = np.mean(dist_matrix, axis=1)
        
        return scores.astype(np.float32)

    def _bin_outside(self, outside: np.ndarray, fg_map: Dict[int, np.ndarray]) -> List[np.ndarray]:
        if outside.size == 0:
            return [np.zeros((0, outside.shape[1] if outside.ndim > 1 else 1), dtype=np.float32) 
                    for _ in range(self.num_bins)]

        fg_points = self._get_fg_points_for_distance(fg_map)
        
        if fg_points.size == 0:
            return [np.zeros((0, outside.shape[1]), dtype=np.float32) for _ in range(self.num_bins)]

        scores = self._compute_distance_scores(outside, fg_points)

        if self.bin_strategy == "uniform":
            bin_id = _safe_uniform_bins(scores, self.num_bins)
        else:
            bin_id = _safe_quantile_bins(scores, self.num_bins)

        bins: List[np.ndarray] = []
        for b in range(self.num_bins):
            pts = outside[bin_id == b]
            if self.per_bin_max > 0 and pts.shape[0] > self.per_bin_max:
                idx = self._rng.choice(pts.shape[0], size=self.per_bin_max, replace=False)
                pts = pts[idx]
            bins.append(pts)
        return bins

    def _relabel_feature_dict(self, feature_dict: Dict[Any, Any]) -> Dict[int, np.ndarray]:
        fg_map, outside = self._collect_points(feature_dict)
        bins = self._bin_outside(outside, fg_map)

        out: Dict[int, np.ndarray] = {}
        for lb, pts in fg_map.items():
            if pts.shape[0] > 0:
                out[int(lb)] = pts

        for b, pts in enumerate(bins):
            if pts.shape[0] > 0:
                out[int(self.pseudo_label_base + b)] = pts

        return out

    def accumulate(self, layer: str, feature_dict: Dict[Any, Any]):
        if cdist is None:
            raise RuntimeError("scipy is required. Please install scipy.")
        relabeled = self._relabel_feature_dict(feature_dict)
        super().accumulate(layer, relabeled)
