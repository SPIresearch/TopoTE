"""
Topological Label Coherence (TLC) Metric — Section 3.6

For classification:  dY(a, b) = 1[a ≠ b]
For regression:      ρ_TLC = Σ_{(i,j)∈E(MST)} (y_i-y_j)^2 / ((n-1)·σ_y²)

Score:  S_TLC(ϕ) = 1 − ρ_TLC

Key design choices vs. TDA/LBTC:
  - Operates on ENCODER features only (no decoder forward pass needed).
  - Aggregates the full dataset into a single feature pool before computing
    the MST (set `pkl_cls_aggregate=True` in config, which is the default
    for TDA-family and is also applied here via the `name` prefix check in
    evaluate_pkl).
  - No ignore_labels: all provided class keys contribute to the MST.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Union

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

try:
    from utils.sliding_window_sampling import GLOBAL_KEY
except ImportError:
    GLOBAL_KEY = "__global__"

GLOBAL_ALIASES = {"__global__", "__GLOBAL__", "global", "GLOBAL"}


# ---------------------------------------------------------------------------
# helpers (mirrors tda.py to stay self-contained)
# ---------------------------------------------------------------------------

def _is_global_key(k: Any) -> bool:
    if k == GLOBAL_KEY:
        return True
    if isinstance(k, str) and k.lower() in {x.lower() for x in GLOBAL_ALIASES}:
        return True
    return False


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_2d(x: Any) -> np.ndarray:
    a = _to_numpy(x)
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape((a.shape[0], -1))


def _mst_edges_prim(W: np.ndarray) -> List[tuple]:
    """Dense Prim MST — returns list of (u, v) edge tuples."""
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
        if parent[j] >= 0:
            edges.append((int(parent[j]), int(j)))
        in_tree[j] = True
        # vectorised update
        improve = (~in_tree) & (W[j] < min_cost)
        min_cost[improve] = W[j][improve]
        parent[improve] = j
        min_cost[j] = np.inf
    return edges


# ---------------------------------------------------------------------------
# label-distance functions (Eq. 1 in the paper)
# ---------------------------------------------------------------------------

def _dY_classification(yi: int, yj: int) -> float:
    """Binary 0/1 indicator for cross-class pairs."""
    return float(yi != yj)


def _dY_regression_sq(yi: float, yj: float) -> float:
    """Squared label variation for regression TLC, matching Eq. (6)."""
    diff = float(yi) - float(yj)
    return diff * diff


# ---------------------------------------------------------------------------
# TLCMetric
# ---------------------------------------------------------------------------

class TLCMetric:
    """
    Topological Label Coherence for global prediction tasks (classification
    and regression).

    Parameters
    ----------
    task          : "classification" (default) or "regression"
    per_class_max : max features per class to subsample before building the MST.
                    Helps keep O(N²) Prim tractable for large pools.
                    Set to -1 to disable subsampling.
    feature_metric: distance metric passed to scipy cdist (default "euclidean").
    seed          : RNG seed for subsampling reproducibility.
    """

    def __init__(
        self,
        task: str = "classification",
        per_class_max: int = 800,
        feature_metric: str = "euclidean",
        seed: int = 0,
    ):
        self.task = task.lower().strip()
        self.per_class_max = int(per_class_max)
        self.feature_metric = feature_metric
        self.seed = int(seed)

        self._case_idx: int = 0
        self._rng = np.random.default_rng(self.seed)
        # layer -> list of ρ_TLC values (one per accumulate call)
        self._scores: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ API

    @property
    def name(self) -> str:
        return "tlc"

    def reset(self):
        self._case_idx = 0
        self._rng = np.random.default_rng(self.seed)
        self._scores = {}

    def begin_case(self, case_idx: int):
        self._case_idx = int(case_idx)
        self._rng = np.random.default_rng(self.seed + self._case_idx)

    # ------------------------------------------------------------------ core

    def _build_xy(
        self, feature_dict: Dict[Any, Any]
    ) -> Optional[tuple]:
        """
        Collect (X, y) from feature_dict.

        feature_dict keys:
          - integer class label → np.ndarray (n_i, C)  [class features]
          - GLOBAL_KEY          → ignored here (TLC uses class-labelled pts)
        """
        parts: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for k, v in feature_dict.items():
            if _is_global_key(k):
                continue
            try:
                lb = int(k)
            except (TypeError, ValueError):
                continue
            arr = _to_2d(v).astype(np.float32, copy=False)
            if arr.shape[0] < 1:
                continue
            # optional subsampling
            if self.per_class_max > 0 and arr.shape[0] > self.per_class_max:
                idx = self._rng.choice(arr.shape[0], size=self.per_class_max, replace=False)
                arr = arr[idx]
            parts.append(arr)
            ys.append(np.full(arr.shape[0], lb, dtype=np.int64))

        if not parts:
            return None
        X = np.concatenate(parts, axis=0)
        y = np.concatenate(ys, axis=0)
        if X.shape[0] < 2:
            return None
        return X, y

    def accumulate(self, layer: str, feature_dict: Dict[Any, Any]):
        """
        Compute ρ_TLC for this layer and store it.

        Called once per case (or once for the full dataset when
        pkl_cls_aggregate=True).
        """
        if cdist is None:
            raise RuntimeError(
                "scipy is required for TLC (scipy.spatial.distance.cdist). "
                "Install with: pip install scipy"
            )

        self._scores.setdefault(layer, [])

        xy = self._build_xy(feature_dict)
        if xy is None:
            return
        X, y = xy

        n = X.shape[0]
        if n < 2:
            return

        # 1. Build MST on feature distances
        Df = cdist(X, X, metric=self.feature_metric).astype(np.float32)
        edges = _mst_edges_prim(Df)

        if not edges:
            return

        # 2. Compute label variation along MST edges
        if self.task == "regression":
            labels_float = y.astype(np.float64)
            sigma2 = float(np.var(labels_float))
            if sigma2 <= 0.0:
                rho = 0.0
            else:
                total = sum(
                    _dY_regression_sq(labels_float[u], labels_float[v])
                    for u, v in edges
                )
                # Regression TLC Eq. (6): normalized continuous variation
                # along the feature MST, using empirical label variance for
                # scale invariance.
                rho = total / ((n - 1) * sigma2)

        else:
            # classification — Macro-Averaged TLC (Definition 3.x)
            #
            # For each class c, E_c(T_n) = MST edges with at least one
            # endpoint of class c.
            #   ρ_{n,c} = |cross-class edges in E_c| / |E_c|
            #   ρ_n^TLC = (1/C) * Σ_c ρ_{n,c}
            #
            # This gives each class equal weight regardless of how many
            # points it has, handling severe class imbalance in medical data.
            edge_arr = np.array(edges, dtype=np.int32)  # (E, 2)
            yu = y[edge_arr[:, 0]]
            yv = y[edge_arr[:, 1]]
            is_cross = (yu != yv)                       # bool (E,)

            classes = np.unique(y)
            C = len(classes)

            per_class_rho = []
            for c in classes:
                # E_c: edges where at least one endpoint belongs to class c
                in_Ec = (yu == c) | (yv == c)
                n_Ec = int(np.sum(in_Ec))
                if n_Ec == 0:
                    # class c has no MST edges — skip (shouldn't happen)
                    continue
                n_cross_c = int(np.sum(is_cross & in_Ec))
                per_class_rho.append(n_cross_c / n_Ec)

            if not per_class_rho:
                return
            rho = float(np.mean(per_class_rho))   # macro-average over classes
        score = 1.0 - rho   # S_TLC = 1 − ρ_TLC

        if np.isfinite(score):
            self._scores[layer].append(float(score))

    def compute(self, layer: str) -> Optional[float]:
        """Mean S_TLC across all accumulated cases for this layer."""
        vals = self._scores.get(layer, [])
        if not vals:
            return None
        return float(np.mean(vals))

    def aggregate_overall(
        self,
        layer_scores: Dict[str, Optional[float]],
        layers: List[str],
    ) -> Optional[float]:
        """
        Overall score = mean across all encoder layers.

        For TLC we only use encoder features; no decoder layers should be
        present in the config (ABI_encoder_only.json only lists encoder.*).
        """
        vals = [
            layer_scores[l]
            for l in layers
            if layer_scores.get(l) is not None
        ]
        if not vals:
            return None
        return float(np.mean(vals))
