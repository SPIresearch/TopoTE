from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

DECODER_LAYERS: List[str] = [
    "decoder_stages_0",
    "decoder_stages_1",
    "decoder_stages_2",
    "decoder_stages_3",
    "decoder_stages_4",
]

ENCODER_LAYERS: List[str] = [
    "encoder_stages_0",
    "encoder_stages_1",
    "encoder_stages_2",
    "encoder_stages_3",
    "encoder_stages_4",
    "encoder_stages_5",
]

PAPER_DECODER_WEIGHTS: Dict[str, float] = {
    "decoder_stages_0": 0.022,
    "decoder_stages_1": 0.229,
    "decoder_stages_2": 0.499,
    "decoder_stages_3": 0.229,
    "decoder_stages_4": 0.022,
}

PAPER_ENCODER_WEIGHTS: Dict[str, float] = {
    "encoder_stages_0": 0.002,
    "encoder_stages_1": 0.038,
    "encoder_stages_2": 0.240,
    "encoder_stages_3": 0.444,
    "encoder_stages_4": 0.240,
    "encoder_stages_5": 0.038,
}


def _normalize(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / s


def _gaussian_vector(n: int, center: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    idx = np.arange(n, dtype=np.float64)
    raw = np.exp(-0.5 * ((idx - float(center)) / sigma) ** 2)
    return _normalize(raw)


def _to_weight_dict(layer_names: List[str], values: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(layer_names, values)}


def _parse_custom_weights(text: Optional[str], expected_len: int) -> Optional[np.ndarray]:
    if text is None:
        return None
    items = [t.strip() for t in text.split(",") if t.strip() != ""]
    if len(items) != expected_len:
        raise ValueError(
            f"Expected {expected_len} weights, got {len(items)}: {text}"
        )
    arr = np.array([float(x) for x in items], dtype=np.float64)
    return _normalize(arr)


def build_fusion_layer_weights(
    profile: str = "paper",
    decoder_sigma: Optional[float] = None,
    encoder_sigma: Optional[float] = None,
    decoder_custom: Optional[str] = None,
    encoder_custom: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return {"decoder": {...}, "encoder": {...}}.

    profile:
      - paper  : exact paper weights (your current baseline)
      - sharp  : narrower Gaussian (more center-sensitive)
      - robust : slightly wider Gaussian (less sensitive to one layer)
      - wide   : much wider Gaussian (strong robustness, weaker locality)
      - flat   : uniform weights
      - custom : parse from --decoder_custom/--encoder_custom
    """
    p = profile.lower()

    if p == "paper":
        dec = PAPER_DECODER_WEIGHTS.copy()
        enc = PAPER_ENCODER_WEIGHTS.copy()
        return {"decoder": dec, "encoder": enc}

    if p == "flat":
        dec = _to_weight_dict(
            DECODER_LAYERS, np.ones(len(DECODER_LAYERS), dtype=np.float64)
        )
        enc = _to_weight_dict(
            ENCODER_LAYERS, np.ones(len(ENCODER_LAYERS), dtype=np.float64)
        )
        return {"decoder": dec, "encoder": enc}

    if p == "custom":
        dec_vec = _parse_custom_weights(decoder_custom, len(DECODER_LAYERS))
        enc_vec = _parse_custom_weights(encoder_custom, len(ENCODER_LAYERS))
        if dec_vec is None or enc_vec is None:
            raise ValueError(
                "profile=custom requires both decoder_custom and encoder_custom."
            )
        return {
            "decoder": _to_weight_dict(DECODER_LAYERS, dec_vec),
            "encoder": _to_weight_dict(ENCODER_LAYERS, enc_vec),
        }

    sigma_map = {
        "sharp": (0.80, 1.00),
        "robust": (1.20, 1.50),
        "wide": (1.60, 2.00),
    }
    if p not in sigma_map:
        raise ValueError(
            f"Unknown profile: {profile}. "
            "Use one of: paper, sharp, robust, wide, flat, custom."
        )

    dec_sigma_default, enc_sigma_default = sigma_map[p]
    dec_sigma = float(decoder_sigma) if decoder_sigma is not None else dec_sigma_default
    enc_sigma = float(encoder_sigma) if encoder_sigma is not None else enc_sigma_default

    dec_center = (len(DECODER_LAYERS) - 1) / 2.0
    enc_center = len(ENCODER_LAYERS) / 2.0

    dec_vec = _gaussian_vector(len(DECODER_LAYERS), dec_center, dec_sigma)
    enc_vec = _gaussian_vector(len(ENCODER_LAYERS), enc_center, enc_sigma)

    return {
        "decoder": _to_weight_dict(DECODER_LAYERS, dec_vec),
        "encoder": _to_weight_dict(ENCODER_LAYERS, enc_vec),
    }
