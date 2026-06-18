from __future__ import annotations

"""Hard-coded reference numbers from the OpenMind paper.

We keep these numbers *in-code* for two reasons:
1) Quick sanity-checks: are we loading the right checkpoint / metric?
2) Plot overlays: show paper baselines alongside our runs.

Sources:
- Table 2: segmentation (DSC)
- Table 3: classification (Average Precision / Balanced Accuracy)
"""

from typing import Dict, Optional

# -----------------------------
# Table 2: Segmentation results
# -----------------------------

# ResEnc-L (CNN)
OPENMIND_SEG_RESENC_L: Dict[str, Dict[str, float]] = {
    "nnunet_def_1k": {"ATL": 58.7, "SBM": 59.98, "ISL": 78.4, "HNT": 62.98, "HAN": 53.37, "MSF": 52.19, "TPC": 79.5, "YBM": 58.43, "COS": 46.19, "ACD": 91.1, "AMO": 88.0, "KIT": 87.21, "ID": 61.08, "OOD": 88.77, "All": 68.0},
    "nnunet_def": {"ATL": 56.08, "SBM": 60.41, "ISL": 78.22, "HNT": 59.37, "HAN": 32.27, "MSF": 55.84, "TPC": 76.78, "YBM": 56.73, "COS": 49.31, "ACD": 90.72, "AMO": 83.88, "KIT": 77.61, "ID": 58.33, "OOD": 84.07, "All": 64.77},
    "scratch_1k": {"ATL": 58.21, "SBM": 53.43, "ISL": 79.14, "HNT": 65.75, "HAN": 58.24, "MSF": 54.9, "TPC": 79.94, "YBM": 56.12, "COS": 71.57, "ACD": 92.09, "AMO": 88.73, "KIT": 87.48, "ID": 64.15, "OOD": 89.43, "All": 70.47},
    "scratch": {"ATL": 57.02, "SBM": 54.29, "ISL": 78.09, "HNT": 63.3, "HAN": 56.11, "MSF": 55.47, "TPC": 76.18, "YBM": 54.42, "COS": 65.2, "ACD": 91.97, "AMO": 85.24, "KIT": 84.03, "ID": 62.23, "OOD": 87.08, "All": 68.44},
    "voco": {"ATL": 57.14, "SBM": 59.62, "ISL": 77.5, "HNT": 63.48, "HAN": 51.12, "MSF": 54.9, "TPC": 75.12, "YBM": 56.92, "COS": 63.49, "ACD": 91.44, "AMO": 85.6, "KIT": 85.71, "ID": 62.14, "OOD": 87.58, "All": 68.5},
    "swinunetr": {"ATL": 56.07, "SBM": 57.25, "ISL": 77.45, "HNT": 61.64, "HAN": 49.42, "MSF": 54.82, "TPC": 74.69, "YBM": 57.05, "COS": 65.68, "ACD": 90.53, "AMO": 84.95, "KIT": 85.54, "ID": 61.56, "OOD": 87.01, "All": 67.92},
    "simclr": {"ATL": 57.15, "SBM": 59.72, "ISL": 78.01, "HNT": 63.32, "HAN": 51.56, "MSF": 55.68, "TPC": 77.77, "YBM": 59.14, "COS": 68.2, "ACD": 91.76, "AMO": 86.06, "KIT": 84.85, "ID": 63.4, "OOD": 87.56, "All": 69.44},
    "vf": {"ATL": 57.42, "SBM": 59.88, "ISL": 78.18, "HNT": 64.32, "HAN": 51.67, "MSF": 57.42, "TPC": 76.11, "YBM": 59.31, "COS": 63.98, "ACD": 91.57, "AMO": 85.38, "KIT": 86.21, "ID": 63.14, "OOD": 87.72, "All": 69.29},
    "mg": {"ATL": 58.03, "SBM": 61.57, "ISL": 77.58, "HNT": 65.11, "HAN": 54.69, "MSF": 55.25, "TPC": 77.14, "YBM": 58.67, "COS": 71.27, "ACD": 91.74, "AMO": 86.35, "KIT": 86.17, "ID": 64.37, "OOD": 88.09, "All": 70.3},
    "mae": {"ATL": 58.25, "SBM": 62.41, "ISL": 77.89, "HNT": 66.58, "HAN": 55.14, "MSF": 56.84, "TPC": 77.96, "YBM": 60.07, "COS": 70.85, "ACD": 91.98, "AMO": 86.78, "KIT": 86.12, "ID": 65.11, "OOD": 88.3, "All": 70.91},
    "s3d": {"ATL": 58.76, "SBM": 64.09, "ISL": 78.05, "HNT": 65.74, "HAN": 52.81, "MSF": 56.08, "TPC": 78.81, "YBM": 59.18, "COS": 66.66, "ACD": 92.01, "AMO": 86.16, "KIT": 86.01, "ID": 64.46, "OOD": 88.06, "All": 70.36},
}

# Primus-M (Transformer)
OPENMIND_SEG_PRIMUS_M: Dict[str, Dict[str, float]] = {
    "scratch_1k": {"ATL": 56.77, "SBM": 48.5, "ISL": 76.59, "HNT": 58.4, "HAN": 53.4, "MSF": 53.27, "TPC": 76.32, "YBM": 52.53, "COS": 64.68, "ACD": 90.89, "AMO": 87.24, "KIT": 85.57, "ID": 60.05, "OOD": 87.9, "All": 67.01},
    "scratch": {"ATL": 51.51, "SBM": 43.26, "ISL": 75.23, "HNT": 55.3, "HAN": 50.6, "MSF": 54.0, "TPC": 73.31, "YBM": 50.3, "COS": 62.11, "ACD": 90.93, "AMO": 80.17, "KIT": 76.73, "ID": 57.29, "OOD": 82.61, "All": 63.62},
    "voco": {"ATL": 46.8, "SBM": 34.15, "ISL": 73.29, "HNT": 51.06, "HAN": 47.64, "MSF": 52.64, "TPC": 65.52, "YBM": 44.75, "COS": 52.16, "ACD": 87.26, "AMO": 65.81, "KIT": 70.21, "ID": 52.0, "OOD": 74.43, "All": 57.61},
    "swinunetr": {"ATL": 47.23, "SBM": 36.31, "ISL": 73.84, "HNT": 50.15, "HAN": 46.49, "MSF": 52.8, "TPC": 66.23, "YBM": 44.49, "COS": 54.82, "ACD": 87.92, "AMO": 66.17, "KIT": 70.39, "ID": 52.49, "OOD": 74.82, "All": 58.07},
    "simclr": {"ATL": 54.61, "SBM": 42.62, "ISL": 75.43, "HNT": 56.75, "HAN": 50.8, "MSF": 53.59, "TPC": 70.08, "YBM": 48.36, "COS": 58.2, "ACD": 89.97, "AMO": 75.75, "KIT": 81.27, "ID": 56.72, "OOD": 82.33, "All": 63.12},
    "vf": {"ATL": 58.62, "SBM": 47.37, "ISL": 77.56, "HNT": 62.37, "HAN": 56.18, "MSF": 55.0, "TPC": 74.98, "YBM": 53.96, "COS": 69.74, "ACD": 91.41, "AMO": 84.95, "KIT": 86.17, "ID": 61.75, "OOD": 87.51, "All": 68.19},
    "mg": {"ATL": 56.5, "SBM": 47.34, "ISL": 76.76, "HNT": 58.42, "HAN": 54.02, "MSF": 54.67, "TPC": 73.65, "YBM": 49.77, "COS": 60.74, "ACD": 90.89, "AMO": 82.15, "KIT": 84.45, "ID": 59.1, "OOD": 85.83, "All": 65.78},
    "mae": {"ATL": 61.16, "SBM": 56.67, "ISL": 77.12, "HNT": 66.12, "HAN": 57.24, "MSF": 56.02, "TPC": 78.31, "YBM": 54.35, "COS": 72.02, "ACD": 92.16, "AMO": 87.16, "KIT": 86.74, "ID": 64.34, "OOD": 88.69, "All": 70.42},
    "simmim": {"ATL": 60.28, "SBM": 51.68, "ISL": 77.53, "HNT": 62.76, "HAN": 56.74, "MSF": 55.91, "TPC": 77.0, "YBM": 52.9, "COS": 70.87, "ACD": 91.98, "AMO": 86.57, "KIT": 85.92, "ID": 62.85, "OOD": 88.16, "All": 69.18},
}


# ------------------------------
# Table 3: Classification results
# ------------------------------
# Datasets: MRN / RSN / ABI (+ Mean)
# Metrics: Average Precision (AP) and Balanced Accuracy (BAcc)

# ResEnc-L (CNN)
OPENMIND_CLS_RESENC_L_AP: Dict[str, Dict[str, float]] = {
    "scratch": {"MRN": 66.33, "RSN": 65.39, "ABI": 57.18, "Mean": 62.97},
    "voco": {"MRN": 71.99, "RSN": 70.33, "ABI": 62.23, "Mean": 68.18},
    "swinunetr": {"MRN": 70.72, "RSN": 70.37, "ABI": 62.32, "Mean": 67.80},
    "simclr": {"MRN": 68.74, "RSN": 73.02, "ABI": 59.17, "Mean": 66.98},
    "vf": {"MRN": 68.46, "RSN": 66.22, "ABI": 58.01, "Mean": 64.23},
    "mg": {"MRN": 68.85, "RSN": 71.21, "ABI": 59.69, "Mean": 66.59},
    "mae": {"MRN": 64.53, "RSN": 67.86, "ABI": 58.31, "Mean": 63.57},
    "s3d": {"MRN": 70.08, "RSN": 69.25, "ABI": 60.65, "Mean": 66.66},
}

OPENMIND_CLS_RESENC_L_BACC: Dict[str, Dict[str, float]] = {
    "scratch": {"MRN": 76.61, "RSN": 62.25, "ABI": 53.29, "Mean": 64.05},
    "voco": {"MRN": 80.35, "RSN": 65.57, "ABI": 60.18, "Mean": 68.70},
    "swinunetr": {"MRN": 79.25, "RSN": 66.60, "ABI": 59.88, "Mean": 68.58},
    "simclr": {"MRN": 78.27, "RSN": 67.96, "ABI": 57.66, "Mean": 67.96},
    "vf": {"MRN": 78.43, "RSN": 63.28, "ABI": 55.60, "Mean": 65.77},
    "mg": {"MRN": 79.09, "RSN": 66.14, "ABI": 57.24, "Mean": 67.49},
    "mae": {"MRN": 76.72, "RSN": 64.21, "ABI": 56.23, "Mean": 65.72},
    "s3d": {"MRN": 78.56, "RSN": 64.20, "ABI": 56.41, "Mean": 66.39},
}

# Primus-M (Transformer)
OPENMIND_CLS_PRIMUS_M_AP: Dict[str, Dict[str, float]] = {
    "scratch": {"MRN": 60.58, "RSN": 58.91, "ABI": 52.87, "Mean": 57.45},
    "voco": {"MRN": 67.92, "RSN": 62.09, "ABI": 53.22, "Mean": 61.08},
    "swinunetr": {"MRN": 64.98, "RSN": 60.79, "ABI": 56.84, "Mean": 60.87},
    "simclr": {"MRN": 67.67, "RSN": 60.26, "ABI": 55.59, "Mean": 61.18},
    "vf": {"MRN": 66.41, "RSN": 61.66, "ABI": 58.37, "Mean": 62.14},
    "mg": {"MRN": 68.86, "RSN": 61.24, "ABI": 52.41, "Mean": 60.84},
    "mae": {"MRN": 67.00, "RSN": 60.28, "ABI": 53.09, "Mean": 60.13},
    "simmim": {"MRN": 65.27, "RSN": 59.62, "ABI": 53.94, "Mean": 59.61},
}

OPENMIND_CLS_PRIMUS_M_BACC: Dict[str, Dict[str, float]] = {
    "scratch": {"MRN": 75.49, "RSN": 57.10, "ABI": 52.05, "Mean": 61.55},
    "voco": {"MRN": 77.68, "RSN": 61.64, "ABI": 51.28, "Mean": 63.53},
    "swinunetr": {"MRN": 76.61, "RSN": 59.22, "ABI": 55.60, "Mean": 63.81},
    "simclr": {"MRN": 76.69, "RSN": 59.52, "ABI": 54.15, "Mean": 63.45},
    "vf": {"MRN": 76.67, "RSN": 60.14, "ABI": 56.06, "Mean": 64.29},
    "mg": {"MRN": 77.73, "RSN": 59.97, "ABI": 54.13, "Mean": 63.94},
    "mae": {"MRN": 77.26, "RSN": 58.83, "ABI": 51.84, "Mean": 62.64},
    "simmim": {"MRN": 77.15, "RSN": 58.49, "ABI": 53.93, "Mean": 63.19},
}


def get_openmind_segmentation_table(arch: str = "resenc_l") -> Dict[str, Dict[str, float]]:
    a = (arch or "").lower().strip()
    if a in ("resenc-l", "resenc_l", "resencl", "cnn", "resenc_l_(cnn)"):
        return OPENMIND_SEG_RESENC_L
    if a in ("primus-m", "primus_m", "primus", "transformer", "primus_m_(transformer)"):
        return OPENMIND_SEG_PRIMUS_M
    raise KeyError(f"Unknown arch={arch}. Expected resenc_l or primus_m.")


def get_openmind_value(method_key: str, col: str = "All", arch: str = "resenc_l") -> Optional[float]:
    """Return OpenMind *segmentation* value (Table 2).

    Returns None if method_key not present.
    """
    table = get_openmind_segmentation_table(arch)
    mk = (method_key or "").lower().strip()
    if mk not in table:
        return None

    c = (col or "All").strip()
    c_lower = c.lower()
    if c_lower == "all":
        c_key = "All"
    elif c_lower == "id":
        c_key = "ID"
    elif c_lower == "ood":
        c_key = "OOD"
    else:
        c_key = c.upper()

    return table[mk].get(c_key)


def get_openmind_classification_table(arch: str = "resenc_l", metric: str = "ap") -> Dict[str, Dict[str, float]]:
    """Return OpenMind classification table (Table 3).

    Args:
        arch: "resenc_l" (CNN) or "primus_m" (Transformer)
        metric: "ap" (Average Precision) or "bacc"/"balanced_accuracy"
    """
    a = (arch or "").lower().strip()
    m = (metric or "ap").lower().strip()
    if m in ("ap", "average_precision", "avg_precision"):
        if a in ("resenc-l", "resenc_l", "resencl", "cnn", "resenc_l_(cnn)"):
            return OPENMIND_CLS_RESENC_L_AP
        if a in ("primus-m", "primus_m", "primus", "transformer", "primus_m_(transformer)"):
            return OPENMIND_CLS_PRIMUS_M_AP
    if m in ("bacc", "balanced_accuracy", "bal_acc", "balancedacc", "balanced-accuracy"):
        if a in ("resenc-l", "resenc_l", "resencl", "cnn", "resenc_l_(cnn)"):
            return OPENMIND_CLS_RESENC_L_BACC
        if a in ("primus-m", "primus_m", "primus", "transformer", "primus_m_(transformer)"):
            return OPENMIND_CLS_PRIMUS_M_BACC
    raise KeyError(f"Unknown arch={arch} or metric={metric}. Expected arch in (resenc_l, primus_m) and metric in (ap, bacc).")


def get_openmind_cls_value(method_key: str, col: str = "Mean", arch: str = "resenc_l", metric: str = "ap") -> Optional[float]:
    """Return OpenMind classification value (Table 3).

    Returns None if method_key not present.
    """
    table = get_openmind_classification_table(arch=arch, metric=metric)
    mk = (method_key or "").lower().strip()
    if mk not in table:
        return None

    c = (col or "Mean").strip()
    c_lower = c.lower()
    if c_lower in ("mean", "avg", "average"):
        c_key = "Mean"
    else:
        c_key = c.upper()  # MRN/RSN/ABI
    return table[mk].get(c_key)
