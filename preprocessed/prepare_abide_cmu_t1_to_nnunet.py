#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare ABIDE-CMU (ABI) classification dataset as T1w-only 3D volumes.

What it does:
1) Scan ABIDE-cmu folder (CMU_a/CMU_a/<SUB_ID>/..., CMU_b/CMU_b/<SUB_ID>/...)
2) Read phenotype file (xlsx/csv) to get DX_GROUP labels (1=autism, 2=control)
   and REMAP to 0-based (0=autism, 1=control) immediately.
3) Find T1w NIfTI per subject (handles per-subject archives: .zip/.tar/.tar.gz/.tgz)
4) Export to nnUNet_raw/DatasetXXX_ABIDE_CMU_T1/imagesTr/ABIDE_<ID>_0000.nii.gz
5) Write DataLists/ABIDE_CMU_T1_all_val.json  (labels are 0-based)
6) Write Configs/ABI_ResEncL_T1_full.json and Configs/ABI_ResEncL_T1_encoder_only.json
   (based on ABI_ResEncL.json template, with task_type and TLC fields injected)

Notes:
- We do NOT use fMRI here (align with OpenMind ABI: T1w only).
- For "encoder+cls head" variant, the safest is: sample encoder layers only
  (no network surgery needed — TLC operates on encoder features exclusively).
- DX_GROUP remapping (1->0, 2->1) is performed once in main() immediately
  after _read_phenotype(); every downstream file sees 0-based labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------

def _copy_or_link(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        try:
            os.symlink(str(src), str(dst))
            return
        except Exception:
            pass  # Windows without admin rights — fall through to copy
    shutil.copy2(str(src), str(dst))


def _is_nifti(p: Path) -> bool:
    return p.name.endswith(".nii") or p.name.endswith(".nii.gz")


def _is_archive(p: Path) -> bool:
    name = p.name.lower()
    if name.endswith(".nii.gz"):   # .nii.gz is NOT a generic archive
        return False
    return name.endswith((".zip", ".tar", ".tgz", ".tar.gz"))


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_archives_if_any(
    subject_dir: Path,
    extract_dirname: str = "__extracted__",
) -> Tuple[Optional[Path], bool]:
    """Extract archives under subject_dir into subject_dir/__extracted__/ (idempotent).

    Returns
    -------
    out_dir : Path or None
        Directory where archives were extracted (None if no archives exist).
    all_ok : bool
        False if at least one archive failed to extract.  The caller should
        skip the subject in that case to avoid using stale / partial data.
    """
    archives = [p for p in subject_dir.rglob("*") if p.is_file() and _is_archive(p)]
    if not archives:
        return None, True

    out_dir = subject_dir / extract_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for arc in archives:
        marker = out_dir / (arc.stem.replace(".", "_") + "_DONE")
        if marker.exists():
            continue  # already extracted in a previous run

        print(f"[EXTRACT] {arc} -> {out_dir}")
        try:
            if zipfile.is_zipfile(arc):
                with zipfile.ZipFile(arc, "r") as zf:
                    zf.extractall(out_dir)
            elif tarfile.is_tarfile(arc):
                with tarfile.open(arc, "r:*") as tf:
                    tf.extractall(out_dir)
            else:
                print(f"[WARN] Unknown archive format (skipped): {arc}")
                all_ok = False
                continue
            marker.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[WARN] Failed to extract {arc}: {exc}")
            all_ok = False

    return out_dir, all_ok


# ---------------------------------------------------------------------------
# T1w NIfTI selection heuristic
# ---------------------------------------------------------------------------

def _score_t1_candidate(p: Path) -> Tuple[int, int]:
    """
    Return (priority, size_MB_bucket).  Higher is a better T1w candidate.
    """
    name = p.name.lower()
    pr = 0

    # Positive keywords — ordered from most specific to least
    if "mprage" in name:
        pr += 50
    if "t1" in name or "t1w" in name:
        pr += 40
    if "anat" in name or "smri" in name:
        pr += 20
    # ABIDE preprocessed outputs often carry these suffixes
    if "brain" in name:
        pr += 10   # skull-stripped T1 — still the right file
    if "preproc" in name or "preprocessed" in name:
        pr += 5

    # Negative keywords: fMRI and other non-T1 modalities
    neg = ["func", "bold", "rest", "fmri", "rsfmri", "task",
           "epi", "fieldmap", "dwi", "adc", "dti"]
    if any(k in name for k in neg):
        pr -= 100

    try:
        size_mb = int(p.stat().st_size // (1024 * 1024))
    except Exception:
        size_mb = 0
    return pr, size_mb


def _find_t1_nifti(subject_dir: Path) -> Optional[Path]:
    """
    Search for the best T1w NIfTI file within subject_dir (including any
    extracted sub-directories).

    Strategy:
      1) Rank all NIfTI files by keyword score.
      2) If the top-ranked file has a negative score (looks non-T1), fall
         back to the largest NIfTI file as a last resort.
    """
    candidates = [p for p in subject_dir.rglob("*") if p.is_file() and _is_nifti(p)]
    if not candidates:
        return None

    ranked = sorted(candidates, key=_score_t1_candidate, reverse=True)
    best = ranked[0]

    if _score_t1_candidate(best)[0] < 0:
        # Every candidate looks like a non-T1 file — use the largest as fallback
        ranked_by_size = sorted(
            candidates,
            key=lambda p: p.stat().st_size if p.exists() else 0,
            reverse=True,
        )
        best = ranked_by_size[0] if ranked_by_size else best

    return best


# ---------------------------------------------------------------------------
# Phenotype parsing
# ---------------------------------------------------------------------------

def _detect_columns(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Auto-detect subject-ID and diagnosis columns from phenotype headers."""
    lower = [c.lower() for c in cols]

    sid_col = None
    for cand in ["sub_id", "subject", "subject_id", "subid", "participant_id", "id"]:
        if cand in lower:
            sid_col = cols[lower.index(cand)]
            break

    dx_col = None
    for cand in ["dx_group", "dx", "diagnosis", "group"]:
        if cand in lower:
            dx_col = cols[lower.index(cand)]
            break

    return sid_col, dx_col


def _read_phenotype(phenotype_path: Path) -> Dict[str, int]:
    """
    Return mapping: subject_id (7-digit zero-padded string) -> raw DX_GROUP int.

    ABIDE convention: DX_GROUP 1 = autism, 2 = control.

    NOTE: the 1/2 -> 0/1 remapping is done in main() right after this call,
    so every downstream object (datalist, dataset.json, configs) sees 0-based
    integer labels {0, 1}.
    """
    import pandas as pd  # deferred so module is importable without pandas installed

    if phenotype_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(phenotype_path)
    else:
        df = pd.read_csv(phenotype_path)

    sid_col, dx_col = _detect_columns(list(df.columns))
    if sid_col is None or dx_col is None:
        raise RuntimeError(
            f"Cannot detect required columns in phenotype file.\n"
            f"  Got columns : {list(df.columns)}\n"
            f"  Need one of : sub_id / subject / participant_id  (for subject ID)\n"
            f"            and: dx_group / dx / diagnosis / group  (for diagnosis)"
        )

    mapping: Dict[str, int] = {}
    for _, row in df.iterrows():
        sid = row[sid_col]
        dx  = row[dx_col]

        # Skip NaN entries
        if sid is None or (isinstance(sid, float) and sid != sid):
            continue
        if dx  is None or (isinstance(dx,  float) and dx  != dx):
            continue

        # Normalize subject ID to 7-digit zero-padded string
        try:
            sid_str = str(int(str(sid).strip())).zfill(7)
        except Exception:
            sid_str = re.sub(r"\D", "", str(sid).strip())
            sid_str = sid_str.zfill(7) if sid_str else ""

        if not sid_str:
            continue

        try:
            dx_int = int(dx)
        except Exception:
            continue

        mapping[sid_str] = dx_int

    return mapping


# ---------------------------------------------------------------------------
# Subject-directory discovery
# ---------------------------------------------------------------------------

def _extract_site_archives(abide_root: Path, sites: List[str]) -> None:
    """
    If a site folder does not exist but a same-named archive does
    (e.g. CMU_a.tgz / CMU_a.zip), extract it first — idempotent.

    Supported patterns:
      <abide_root>/CMU_a.tgz   -> extracts to <abide_root>/  (archive should contain CMU_a/)
      <abide_root>/CMU_a.zip   -> same
    """
    for site in sites:
        site_dir = abide_root / site
        if site_dir.exists():
            continue  # already a directory — nothing to do

        # Look for <site>.tgz / <site>.tar.gz / <site>.zip etc.
        candidates = [
            abide_root / f"{site}.tgz",
            abide_root / f"{site}.tar.gz",
            abide_root / f"{site}.tar",
            abide_root / f"{site}.zip",
        ]
        arc = next((p for p in candidates if p.exists()), None)
        if arc is None:
            print(f"[WARN] Neither folder nor archive found for site '{site}' under {abide_root}")
            continue

        print(f"[EXTRACT SITE] {arc} -> {abide_root}")
        try:
            if arc.name.endswith(".zip"):
                with zipfile.ZipFile(arc, "r") as zf:
                    zf.extractall(abide_root)
            else:
                with tarfile.open(arc, "r:*") as tf:
                    tf.extractall(abide_root)
            print(f"[EXTRACT SITE] Done: {site_dir}")
        except Exception as exc:
            print(f"[ERROR] Failed to extract site archive {arc}: {exc}")
            raise


def _collect_subject_dirs(abide_root: Path, sites: List[str]) -> Dict[str, Path]:
    """
    Return mapping: subject_id -> subject_dir.

    Handles two common ABIDE layouts:
      <abide_root>/CMU_a/CMU_a/<ID>/   (nested site folder)
      <abide_root>/CMU_a/<ID>/          (flat)
    """
    out: Dict[str, Path] = {}

    for site in sites:
        nested = abide_root / site / site
        flat   = abide_root / site
        base   = nested if nested.exists() else flat

        if not base.exists():
            print(f"[WARN] Site folder not found: {site}  (tried {nested} and {flat})")
            continue

        for d in base.iterdir():
            if not d.is_dir():
                continue
            name = d.name.strip()
            if re.fullmatch(r"\d{6,8}", name):
                out[name.zfill(7)] = d

    return out


# ---------------------------------------------------------------------------
# nnU-Net raw export
# ---------------------------------------------------------------------------

def export_nnunet_raw_classification(
    items: List[Dict],
    out_root: Path,
    dataset_id: int,
    dataset_name: str,
    prefix: str = "ABIDE",
    mode: str = "copy",
    channel_name: str = "T1w",
) -> Tuple[Path, List[Dict]]:
    """
    Export images to nnU-Net raw layout (imagesTr, classification task).

    Returns (dataset_dir, ccfv_items).
    Each ccfv_item: {"data": <posix_path>, "label": int, "case_id": str}.
    Labels are already 0-based (remapping was done in main()).
    """
    dataset_dir = out_root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"
    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)

    ccfv_items: List[Dict] = []
    for it in items:
        sid   = it["case_id"]
        label = int(it["label"])   # 0-based
        dst   = dataset_dir / "imagesTr" / f"{prefix}_{sid}_0000.nii.gz"
        _copy_or_link(Path(it["img"]), dst, mode=mode)
        ccfv_items.append({"data": str(dst.as_posix()), "label": label, "case_id": sid})

    # dataset.json: labels are 0-based after DX_GROUP remapping
    dataset_json = {
        "channel_names": {"0": channel_name},
        "labels": {"autism": 0, "control": 1},   # 0-based; DX_GROUP 1->0, 2->1
        "numTraining": len(ccfv_items),
        "file_ending": ".nii.gz",
        "dataset_name": f"Dataset{dataset_id:03d}_{dataset_name}",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dataset_dir, ccfv_items


# ---------------------------------------------------------------------------
# Datalist writer
# ---------------------------------------------------------------------------

def write_datalist(ccfv_items: List[Dict], ccfv_root: Path, name: str) -> Path:
    datalist_dir = ccfv_root / "DataLists"
    datalist_dir.mkdir(parents=True, exist_ok=True)
    path = datalist_dir / name
    path.write_text(
        json.dumps({"val": ccfv_items, "fold_id": 0}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

# Fields injected into every generated classification config.
# These align with ABI_classification.json and the TLC/metric pipeline.
_CLS_CFG_FIELDS: dict = {
    # Triggers the classification evaluation path in main_unified_with_classification.py
    "task_type": "classification",
    # PKL mode: aggregate all cases into one dataset-level pseudo-case before
    # building the MST. Required for TLC and TDA-family metrics.
    "pkl_cls_aggregate": True,
    # DX_GROUP has been remapped to 0/1 at read time — no further shift needed.
    "cls_label_shift": False,
    # Default metric hyper-params forwarded to TLCMetric.__init__()
    "metric_params": {
        "task": "classification",
        "per_class_max": 800,
        "feature_metric": "euclidean",
        "seed": 0,
    },
}


def write_cfgs_from_template(
    ccfv_root: Path,
    template_cfg: Path,
    out_full_name: str,
    out_enc_only_name: str,
    num_input_channels: int = 1,
) -> Tuple[Path, Path]:
    """
    Generate two config JSON files from a template:

    full (encoder + decoder layers)
        Useful when you want to compare decoder-level features too.

    encoder-only (encoder.* layers only)
        The recommended config for TLC: the metric operates exclusively on
        encoder features and needs no decoder forward pass.

    Both configs receive all keys from _CLS_CFG_FIELDS so the framework
    automatically routes to the classification evaluation path.
    """
    cfg_dir = ccfv_root / "Configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # ---- full config ----
    cfg = json.loads(template_cfg.read_text(encoding="utf-8"))
    cfg["num_input_channels"] = int(num_input_channels)
    cfg.update(_CLS_CFG_FIELDS)
    full_path = cfg_dir / out_full_name
    full_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- encoder-only config ----
    enc_cfg = json.loads(template_cfg.read_text(encoding="utf-8"))
    enc_cfg["num_input_channels"] = int(num_input_channels)
    enc_cfg.update(_CLS_CFG_FIELDS)

    # Prefer an explicit "encoder_layers" key in the template; otherwise derive
    # from any layer whose name starts with "encoder."
    enc_layers: List[str] = enc_cfg.get("encoder_layers") or [
        lyr for lyr in enc_cfg.get("layers", []) if str(lyr).startswith("encoder.")
    ]
    enc_cfg["layers"] = enc_layers

    if isinstance(enc_cfg.get("sample_num"), dict):
        enc_cfg["sample_num"] = {
            k: v for k, v in enc_cfg["sample_num"].items() if k in set(enc_layers)
        }

    enc_only_path = cfg_dir / out_enc_only_name
    enc_only_path.write_text(json.dumps(enc_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    return full_path, enc_only_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare ABIDE-CMU T1w data for classification transferability evaluation."
    )
    ap.add_argument("--abide_root",       required=True,
                    help="ABIDE-cmu root directory (contains CMU_a/, CMU_b/, phenotypic file)")
    ap.add_argument("--phenotype",        required=True,
                    help="Phenotypic file: phenotypic_CMU.xlsx or phenotypic_CMU.csv")
    ap.add_argument("--ccfv_root",        required=True,
                    help="Project repo root (DataLists/ and Configs/ will be written here)")
    ap.add_argument("--sites",            nargs="+", default=["CMU_a", "CMU_b"],
                    help="Site sub-folders to scan (default: CMU_a CMU_b)")
    ap.add_argument("--dataset_id",       type=int,  default=7)
    ap.add_argument("--dataset_name",     default="ABIDE_CMU_T1")
    ap.add_argument("--prefix",           default="ABIDE")
    ap.add_argument("--mode",             default="copy", choices=["copy", "symlink"])
    ap.add_argument("--no_nnunet",        action="store_true",
                    help="Skip nnUNet export and write a datalist that points to "
                         "original T1 paths (WARNING: those absolute paths are not "
                         "portable to other machines)")
    ap.add_argument("--datalist_name",    default="ABIDE_CMU_T1_all_val.json")
    ap.add_argument("--template_cfg",     default="ABI_ResEncL.json",
                    help="Template config JSON to patch (must contain 'layers' key)")
    ap.add_argument("--cfg_full_name",    default="ABI_ResEncL_T1_full.json")
    ap.add_argument("--cfg_enc_only_name",default="ABI_ResEncL_T1_encoder_only.json")
    args = ap.parse_args()

    abide_root     = Path(args.abide_root)
    phenotype_path = Path(args.phenotype)
    ccfv_root      = Path(args.ccfv_root)
    template_cfg   = Path(args.template_cfg)
    if not template_cfg.is_absolute():
        template_cfg = ccfv_root / template_cfg

    for p, tag in [(abide_root, "abide_root"), (phenotype_path, "phenotype"),
                   (ccfv_root, "ccfv_root"), (template_cfg, "template_cfg")]:
        if not p.exists():
            raise RuntimeError(f"Path not found [{tag}]: {p}")

    # ------------------------------------------------------------------
    # Step 1 — Read phenotype and remap DX_GROUP to 0-based labels
    # ------------------------------------------------------------------
    print("[1/5] Reading phenotype...")
    sid2dx = _read_phenotype(phenotype_path)
    print(f"  subjects in phenotype file : {len(sid2dx)}")

    dx_raw = set(sid2dx.values())
    if dx_raw <= {1, 2}:
        # Standard ABIDE encoding: 1=autism, 2=control → remap to 0=autism, 1=control
        sid2dx = {sid: dx - 1 for sid, dx in sid2dx.items()}
        print("  DX_GROUP remapped: 1->0 (autism), 2->1 (control)")
    elif dx_raw <= {0, 1}:
        print("  DX_GROUP already 0-based {0, 1}; no remapping applied")
    else:
        print(
            f"  [WARN] Unexpected DX_GROUP values {dx_raw}. "
            "No automatic remapping — verify labels manually before use."
        )

    # ------------------------------------------------------------------
    # Step 2 — Collect subject directories
    # ------------------------------------------------------------------
    print("[2/5] Scanning subject folders (extracting site-level archives if needed)...")
    _extract_site_archives(abide_root, args.sites)
    sid2dir = _collect_subject_dirs(abide_root, args.sites)
    print(f"  subject folders found : {len(sid2dir)}")

    # ------------------------------------------------------------------
    # Step 3 — Locate T1w NIfTI for each subject
    # ------------------------------------------------------------------
    print("[3/5] Finding T1w NIfTI (extracting archives where needed)...")
    raw_items: List[Dict] = []
    miss_label = miss_extract = miss_t1 = 0

    for sid, sdir in sorted(sid2dir.items()):
        if sid not in sid2dx:
            miss_label += 1
            continue

        # Extract any per-subject archives; skip subject on failure to avoid
        # picking up stale / partial files from a previously interrupted run.
        _, extract_ok = _extract_archives_if_any(sdir)
        if not extract_ok:
            miss_extract += 1
            print(f"  [SKIP_EXTRACT_FAIL] {sid}  ({sdir})")
            continue

        t1 = _find_t1_nifti(sdir)
        if t1 is None:
            miss_t1 += 1
            print(f"  [MISS_T1] {sid}  ({sdir})")
            continue

        raw_items.append({"case_id": sid, "img": str(t1), "label": int(sid2dx[sid])})

    if not raw_items:
        raise RuntimeError(
            "No valid subjects found. "
            "Check --sites names and phenotype column names."
        )

    print(
        f"  usable subjects   : {len(raw_items)}\n"
        f"  missing label     : {miss_label}\n"
        f"  extract failures  : {miss_extract}\n"
        f"  missing T1        : {miss_t1}"
    )
    dist = dict(sorted(Counter(it["label"] for it in raw_items).items()))
    print(f"  label distribution (0=autism, 1=control) : {dist}")

    # ------------------------------------------------------------------
    # Step 4 — Export nnUNet raw layout and write datalist
    # ------------------------------------------------------------------
    if args.no_nnunet:
        print("[4/5] Writing datalist only (--no_nnunet; paths are NOT portable)...")
        ccfv_items = [
            {"data": it["img"], "label": int(it["label"]), "case_id": it["case_id"]}
            for it in raw_items
        ]
        datalist_path = write_datalist(ccfv_items, ccfv_root, args.datalist_name)
        dataset_dir = None
    else:
        print("[4/5] Exporting to nnUNet_raw...")
        dataset_dir, ccfv_items = export_nnunet_raw_classification(
            raw_items,
            out_root=ccfv_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            prefix=args.prefix,
            mode=args.mode,
        )
        datalist_path = write_datalist(ccfv_items, ccfv_root, args.datalist_name)

    # ------------------------------------------------------------------
    # Step 5 — Generate config files from template
    # ------------------------------------------------------------------
    print("[5/5] Writing configs (full + encoder-only) from template...")
    cfg_full_path, cfg_enc_only_path = write_cfgs_from_template(
        ccfv_root=ccfv_root,
        template_cfg=template_cfg,
        out_full_name=args.cfg_full_name,
        out_enc_only_name=args.cfg_enc_only_name,
        num_input_channels=1,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=================================================")
    print(f"[DONE] usable cases   : {len(ccfv_items)}")
    if dataset_dir is not None:
        print(f"[DONE] nnUNet raw dir  : {dataset_dir}")
        print(f"[DONE] dataset.json   : {dataset_dir / 'dataset.json'}")
    else:
        print("[SKIP] nnUNet raw export (--no_nnunet)")
    print(f"[DONE] datalist        : {datalist_path}")
    print(f"[DONE] cfg (full)      : {cfg_full_path}")
    print(f"[DONE] cfg (enc-only)  : {cfg_enc_only_path}")
    print("=================================================")
    print("Example — run TLC metric directly (encoder-only config):")
    print(
        f"  python main_unified_with_classification.py \\\n"
        f"    --cfg {cfg_enc_only_path} \\\n"
        f"    --data_list_file {datalist_path} \\\n"
        f"    --metric tlc \\\n"
        f"    --model_path <ckpt> \\\n"
        f"    --output_dir output/results/ABI_T1"
    )
    print("\nExample — extract features for PKL mode (encoder-only):")
    print(
        f"  python extract_features_only.py \\\n"
        f"    --cfg {cfg_enc_only_path} \\\n"
        f"    --data_list_file {datalist_path} \\\n"
        f"    --model_path <ckpt> \\\n"
        f"    --feature_save_dir output/features/ABI_T1/enc_only"
    )
    print("=================================================")


if __name__ == "__main__":
    main()
