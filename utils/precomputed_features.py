from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# project global key
try:
    from utils.sliding_window_sampling import GLOBAL_KEY
except Exception:
    GLOBAL_KEY = "__global__"

# accept common aliases in saved PKLs
GLOBAL_ALIASES = {"__global__", "__GLOBAL__", "global", "GLOBAL", "GLOBAL_KEY"}

# common wrapper keys
WRAP_KEYS = ["features", "feats", "feat", "x", "emb", "embedding", "data"]


def describe(obj: Any) -> str:
    """Small summary string for debugging prints."""
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    except Exception:
        pass
    if hasattr(obj, "shape") and not isinstance(obj, (dict, list, tuple)):
        try:
            return f"{type(obj).__name__}(shape={obj.shape})"
        except Exception:
            pass
    if isinstance(obj, dict):
        ks = list(obj.keys())
        return f"dict(len={len(ks)}) keys_sample={ks[:5]}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)})"
    return str(type(obj))


def _is_global_key(k: Any) -> bool:
    if k == GLOBAL_KEY:
        return True
    if isinstance(k, str) and k.lower() in {x.lower() for x in GLOBAL_ALIASES}:
        return True
    return False


def _unwrap_array(v: Any) -> Optional[np.ndarray]:
    """Extract numpy array from common wrappers (torch tensor, dict wrapper, tuple/list wrapper)."""
    if v is None:
        return None

    # torch tensor -> numpy
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        try:
            v = v.detach().cpu().numpy()
        except Exception:
            pass

    if isinstance(v, np.ndarray):
        return v

    # tuple/list: (feats, idx) or [feats, ...]
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        # list of arrays -> concat
        if all(isinstance(x, np.ndarray) for x in v):
            try:
                return np.concatenate(v, axis=0)
            except Exception:
                return np.asarray(v[0])
        return _unwrap_array(v[0])

    # dict wrapper like {"feats":..., "indices":...}
    if isinstance(v, dict):
        for kk in WRAP_KEYS:
            if kk in v:
                return _unwrap_array(v[kk])

    return None


def _to_2d(x: Any) -> np.ndarray:
    """
    Convert array-like to 2D (N,C).
    - point cloud: (N,C) stays
    - (C,H,W) -> (H*W, C)
    - (C,D,H,W) -> (D*H*W, C)
    - otherwise: reshape (N, -1)
    """
    a = np.asarray(x)

    if a.ndim == 2:
        return a
    if a.ndim == 1:
        return a.reshape(-1, 1)

    # feature map channels-first
    if a.ndim == 3:
        c, h, w = a.shape
        out = a.reshape(c, -1).T  # (h*w, c)
        if out.shape[0] > 2_000_000:
            raise ValueError(
                f"Feature-map too large after flatten: {out.shape}. "
                f"Your PKL may be raw feature-map instead of sampled point features."
            )
        return out

    if a.ndim == 4:
        c = a.shape[0]
        out = a.reshape(c, -1).T  # (d*h*w, c)
        if out.shape[0] > 2_000_000:
            raise ValueError(
                f"Feature-map too large after flatten: {out.shape}. "
                f"Your PKL may be raw feature-map instead of sampled point features."
            )
        return out

    # generic: keep first dim as N
    return a.reshape((a.shape[0], -1))


def _is_feature_dict(obj: Any) -> bool:
    """
    Heuristic: dict with class keys (int / str-digit) OR global key (incl aliases).
    Values should be array-like / wrapper.
    """
    if not isinstance(obj, dict):
        return False

    has_global = any(_is_global_key(k) for k in obj.keys())
    has_cls = any(
        isinstance(k, (int, np.integer)) or (isinstance(k, str) and k.isdigit())
        for k in obj.keys()
    )
    has_payload = any(_unwrap_array(v) is not None for v in obj.values())
    return has_payload and (has_cls or has_global)


def _standardize_feature_dict(fd: Dict[Any, Any]) -> Dict[Any, np.ndarray]:
    """
    Normalize:
      - global aliases -> GLOBAL_KEY
      - class keys -> int
      - unwrap torch/dict wrappers
      - ensure 2D float32
    """
    out: Dict[Any, np.ndarray] = {}

    for k, v in fd.items():
        arr = _unwrap_array(v)
        if arr is None:
            continue

        if _is_global_key(k):
            out[GLOBAL_KEY] = _to_2d(arr).astype(np.float32, copy=False)
            continue

        lb: Optional[int] = None
        if isinstance(k, (int, np.integer)):
            lb = int(k)
        elif isinstance(k, str) and k.isdigit():
            lb = int(k)

        if lb is None:
            continue

        out[lb] = _to_2d(arr).astype(np.float32, copy=False)

    return out


def layer_sort_key(layer: str) -> Tuple[int, int, str]:
    """
    Stable ordering for your filename style:
      encoder_stem < encoder_stages_0 < ... < encoder_stages_N < decoder_stages_0 < ... < decoder_stages_N
    """
    s = str(layer)
    t = s.replace(".", "_")

    if "encoder" in t:
        group = 0
    elif "decoder" in t:
        group = 1
    else:
        group = 2

    if "stem" in t:
        idx = -1
    else:
        nums = re.findall(r"(\d+)", t)
        idx = int(nums[-1]) if nums else 999

    return (group, idx, t)


def _iter_case_layer_payload(obj: Any) -> Iterator[Tuple[str, str, Any]]:

    if isinstance(obj, dict):
        # case -> layer -> fd
        for case_id, v1 in obj.items():
            if isinstance(v1, dict):
                for layer, payload in v1.items():
                    if _is_feature_dict(payload):
                        yield str(case_id), str(layer), payload

        # layer -> case -> fd
        for layer, v1 in obj.items():
            if isinstance(v1, dict):
                for case_id, payload in v1.items():
                    if _is_feature_dict(payload):
                        yield str(case_id), str(layer), payload

        # layer -> fd (single case)
        for layer, payload in obj.items():
            if _is_feature_dict(payload):
                yield "case_0", str(layer), payload

    if isinstance(obj, (list, tuple)):
        for i, rec in enumerate(obj):
            if not isinstance(rec, dict):
                continue
            case_id = str(rec.get("case_id", rec.get("id", f"case_{i}")))
            if "layers" in rec and isinstance(rec["layers"], dict):
                for layer, payload in rec["layers"].items():
                    if _is_feature_dict(payload):
                        yield case_id, str(layer), payload
            else:
                for layer, payload in rec.items():
                    if _is_feature_dict(payload):
                        yield case_id, str(layer), payload


@dataclass
class PrecomputedStore:
    """Standardized store: store[case_id][layer] -> standardized feature_dict."""

    store: Dict[str, Dict[str, Dict[Any, np.ndarray]]]

    def case_ids(self) -> List[str]:
        return sorted(self.store.keys())

    def layers(self) -> List[str]:
        s = set()
        for c in self.store.values():
            s.update(c.keys())
        return sorted(s, key=layer_sort_key)

    def get(self, case_id: str, layer: str) -> Optional[Dict[Any, np.ndarray]]:
        return self.store.get(case_id, {}).get(layer)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_store_from_obj(obj: Any) -> PrecomputedStore:
    store: Dict[str, Dict[str, Dict[Any, np.ndarray]]] = {}
    for case_id, layer, payload in _iter_case_layer_payload(obj):
        fd = _standardize_feature_dict(payload)
        if not fd:
            continue
        store.setdefault(case_id, {})[layer] = fd
    return PrecomputedStore(store=store)


def _auto_find_model_dir(root: Path, model_name: str) -> Optional[Path]:
    """Search root/*/model_name when root/model_name doesn't exist."""
    matches = []
    if not root.exists() or not root.is_dir():
        return None
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / model_name
        if cand.exists() and cand.is_dir():
            matches.append(cand)
    return matches[0] if matches else None


def resolve_pkl_path(configs: Dict[str, Any], model_name: str, ckpt_path: str = "") -> str:

    ckpt_stem = Path(ckpt_path).stem if ckpt_path else model_name

    tpl = configs.get("pkl_path_template", "")
    if isinstance(tpl, str) and tpl:
        return tpl.format(model_name=model_name, ckpt_stem=ckpt_stem, ckpt_path=ckpt_path)

    pkl_path = configs.get("pkl_path", "")
    if isinstance(pkl_path, str) and pkl_path:
        if "{model_name}" in pkl_path or "{ckpt_stem}" in pkl_path:
            return pkl_path.format(model_name=model_name, ckpt_stem=ckpt_stem, ckpt_path=ckpt_path)
        return pkl_path

    root = configs.get("pkl_root", "")
    if not root:
        raise ValueError("PKL mode requires cfg key: pkl_path or pkl_root or pkl_path_template")

    rootp = Path(root)

    ds = configs.get("pkl_dataset", configs.get("dataset_tag", configs.get("feature_dataset", "")))
    if isinstance(ds, str) and ds:
        rootp = rootp / ds

    # single-file option
    cand = rootp / f"{model_name}.pkl"
    if cand.exists():
        return str(cand)

    # model directory option
    model_dir = rootp / model_name
    if model_dir.exists() and model_dir.is_dir():
        return str(model_dir)

    auto = _auto_find_model_dir(rootp, model_name)
    if auto is not None:
        return str(auto)

    if rootp.exists():
        return str(rootp)

    raise FileNotFoundError(f"Cannot resolve precomputed feature path from pkl_root={root}")


def _load_feature_dict_from_file(fp: Path) -> Optional[Dict[Any, np.ndarray]]:
    obj = load_pickle(str(fp))

    # if the top-level itself is feature_dict
    if _is_feature_dict(obj):
        fd = _standardize_feature_dict(obj)
        return fd if fd else None

    # sometimes user saved as {layer: feature_dict} even per-file
    if isinstance(obj, dict):
        for _, payload in obj.items():
            if _is_feature_dict(payload):
                fd = _standardize_feature_dict(payload)
                return fd if fd else None

    return None


def load_precomputed_store(configs: Dict[str, Any], model_name: str, ckpt_path: str = "") -> PrecomputedStore:

    path = resolve_pkl_path(configs, model_name=model_name, ckpt_path=ckpt_path)
    p = Path(path)

    if p.is_file():
        obj = load_pickle(str(p))
        return build_store_from_obj(obj)

    if not p.is_dir():
        raise FileNotFoundError(f"pkl path not found: {path}")

    # (L3) case directories
    case_dirs = sorted([d for d in p.iterdir() if d.is_dir()])
    has_case_layer = any(list(d.glob("*.pkl")) for d in case_dirs)

    if has_case_layer:
        store: Dict[str, Dict[str, Dict[Any, np.ndarray]]] = {}
        for case_dir in case_dirs:
            layer_files = sorted(case_dir.glob("*.pkl"), key=lambda x: layer_sort_key(x.stem))
            if not layer_files:
                continue
            case_id = case_dir.name
            for lf in layer_files:
                layer = lf.stem
                fd = _load_feature_dict_from_file(lf)
                if fd is None:
                    continue
                store.setdefault(case_id, {})[layer] = fd
        return PrecomputedStore(store=store)

    # (L2) per-case files under this directory
    files = sorted(list(p.glob("*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files found under directory: {p}")

    store: Dict[str, Dict[str, Dict[Any, np.ndarray]]] = {}
    for fp in files:
        case_id = fp.stem
        obj = load_pickle(str(fp))
        tmp = build_store_from_obj(obj).store
        if tmp:
            # may contain multiple cases; merge
            for cid, layers in tmp.items():
                store.setdefault(cid, {}).update(layers)
        else:
            # simplest: {layer: feature_dict}
            if isinstance(obj, dict):
                for layer, payload in obj.items():
                    if _is_feature_dict(payload):
                        store.setdefault(case_id, {})[str(layer)] = _standardize_feature_dict(payload)

    return PrecomputedStore(store=store)
