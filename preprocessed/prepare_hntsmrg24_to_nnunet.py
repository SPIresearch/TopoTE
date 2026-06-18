import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _find_one(folder: Path, patterns: List[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    for p in patterns:
        hits = sorted(folder.glob(p))
        if hits:
            return hits[0]
    return None


def _case_id_to_name(case_id: str, prefix: str = "HNTSMRG24") -> str:
    try:
        n = int(re.sub(r"\D", "", case_id) or case_id)
        return f"{prefix}_{n:04d}"
    except Exception:
        safe = re.sub(r"[^0-9A-Za-z]+", "_", case_id).strip("_")
        return f"{prefix}_{safe}"


def _copy_or_link(src: Path, dst: Path, mode: str = "symlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be 'symlink' or 'copy'")


def build_items(
    hnt_root: Path,
    input_time: str = "preRT",
    modality: str = "T2",
    use_registered: bool = False,
    label_time: str = "preRT",
) -> List[Dict]:
    """Build a list of dicts: {data: <path>, seg: <path>} for ALL cases."""
    items: List[Dict] = []
    case_dirs = sorted([p for p in hnt_root.iterdir() if p.is_dir()])

    for case_dir in case_dirs:
        case_id = case_dir.name
        pre_dir = case_dir / "preRT"
        mid_dir = case_dir / "midRT"

        # decide image path
        if input_time.lower() == "midrt":
            img = _find_one(mid_dir, [f"{case_id}_midRT_{modality}.nii.gz", f"*midRT*{modality}*.nii.gz"])
        else:
            # preRT
            if use_registered:
                img = _find_one(mid_dir, [f"{case_id}_preRT_{modality}_registered.nii.gz", f"*preRT*{modality}*registered*.nii.gz"])
                if img is None:
                    # fallback to raw preRT
                    img = _find_one(pre_dir, [f"{case_id}_preRT_{modality}.nii.gz", f"*preRT*{modality}*.nii.gz"])
            else:
                img = _find_one(pre_dir, [f"{case_id}_preRT_{modality}.nii.gz", f"*preRT*{modality}*.nii.gz"])

        # decide label path
        if label_time.lower() == "midrt":
            seg = _find_one(mid_dir, [f"{case_id}_midRT_mask.nii.gz", "*midRT*mask*.nii.gz", "*midRT*Mask*.nii.gz"])
        else:
            # preRT
            if use_registered:
                seg = _find_one(mid_dir, [f"{case_id}_preRT_mask_registered.nii.gz", "*preRT*mask*registered*.nii.gz"])
                if seg is None:
                    seg = _find_one(pre_dir, [f"{case_id}_preRT_mask.nii.gz", "*preRT*mask*.nii.gz"])
            else:
                seg = _find_one(pre_dir, [f"{case_id}_preRT_mask.nii.gz", "*preRT*mask*.nii.gz"])

        if img is None or seg is None:
            print(f"[SKIP] case={case_id}: img={img} seg={seg}")
            continue

        items.append({"case_id": case_id, "img": str(img), "seg": str(seg)})

    return items


def export_nnunet_raw(
    items: List[Dict],
    out_root: Path,
    dataset_id: int,
    dataset_name: str,
    prefix: str = "HNTSMRG24",
    mode: str = "symlink",
) -> Tuple[Path, List[Dict]]:
    """Export as nnU-Net raw (single modality) and return ccfv items with new paths."""
    dataset_dir = out_root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"
    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    ccfv_items: List[Dict] = []

    for it in items:
        case_name = _case_id_to_name(it["case_id"], prefix=prefix)
        src_img = Path(it["img"])
        src_seg = Path(it["seg"])

        dst_img = imagesTr / f"{case_name}_0000.nii.gz"
        dst_seg = labelsTr / f"{case_name}.nii.gz"

        _copy_or_link(src_img, dst_img, mode=mode)
        _copy_or_link(src_seg, dst_seg, mode=mode)

        ccfv_items.append({"data": str(dst_img), "seg": str(dst_seg)})

    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(ccfv_items),
        "file_ending": ".nii.gz",
        "dataset_name": f"Dataset{dataset_id:03d}_{dataset_name}",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False), encoding="utf-8")

    return dataset_dir, ccfv_items


def write_ccfv_datalist(ccfv_items: List[Dict], ccfv_root: Path, name: str) -> Path:
    datalist_dir = ccfv_root / "DataLists"
    datalist_dir.mkdir(parents=True, exist_ok=True)
    path = datalist_dir / name
    payload = {"val": ccfv_items, "fold_id": 0}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_encoder_cfg(ccfv_root: Path, name: str, roi_xyz: Tuple[int,int,int], layers: List[str], sample_num: List[int]) -> Path:
    cfg_dir = ccfv_root / "Configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / name
    payload = {
        "dataset": "HNTSMRG24",
        "num_input_channels": 1,
        "num_classes": 1,
        "roi_z": int(roi_xyz[0]),
        "roi_y": int(roi_xyz[1]),
        "roi_x": int(roi_xyz[2]),
        "infer_overlap": 0.25,
        "window_mode": "constant",
        "sw_batch_size": 2,
        "layers": layers,
        "sample_num": sample_num,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hnt_root", type=str, default="../HNTSMRG24_train", help="HNTSMRG24_train root")
    ap.add_argument("--ccfv_root", type=str, default="../CCFV-main", help="CCFV-main repo root (where Configs/ and DataLists/ exist)")
    ap.add_argument("--out_root", type=str, default="", help="Where to create nnUNet_raw. If empty, uses <ccfv_root>.")
    ap.add_argument("--dataset_id", type=int, default=100)
    ap.add_argument("--dataset_name", type=str, default="HNTSMRG24_preRT")
    ap.add_argument("--prefix", type=str, default="HNTSMRG24")
    ap.add_argument("--mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--input_time", type=str, default="preRT", choices=["preRT", "midRT"])
    ap.add_argument("--label_time", type=str, default="preRT", choices=["preRT", "midRT"])
    ap.add_argument("--modality", type=str, default="T2")
    ap.add_argument("--use_registered", action="store_true", help="Use *_registered under midRT/ for preRT image/label if present")
    ap.add_argument("--no_nnunet", action="store_true", help="Only write datalist (paths point to original files), skip nnU-Net export")
    ap.add_argument("--datalist_name", type=str, default="HNTSMRG24_preRT_all_val.json")
    ap.add_argument("--cfg_name", type=str, default="HNTSMRG24_preRT_T2_ResEncL_Encoder.json")
    ap.add_argument("--roi_z", type=int, default=96)
    ap.add_argument("--roi_y", type=int, default=128)
    ap.add_argument("--roi_x", type=int, default=128)
    args = ap.parse_args()

    hnt_root = Path(args.hnt_root)
    ccfv_root = Path(args.ccfv_root)
    out_root = Path(args.out_root) if args.out_root else ccfv_root

    raw_items = build_items(
        hnt_root=hnt_root,
        input_time=args.input_time,
        modality=args.modality,
        use_registered=args.use_registered,
        label_time=args.label_time,
    )

    if len(raw_items) == 0:
        raise RuntimeError("No valid cases found. Check folder names and file patterns.")

    if args.no_nnunet:
        # datalist points to original files
        ccfv_items = [{"data": it["img"], "seg": it["seg"]} for it in raw_items]
        datalist_path = write_ccfv_datalist(ccfv_items, ccfv_root, args.datalist_name)
        dataset_dir = None
    else:
        dataset_dir, ccfv_items = export_nnunet_raw(
            raw_items, out_root, args.dataset_id, args.dataset_name, prefix=args.prefix, mode=args.mode
        )
        datalist_path = write_ccfv_datalist(ccfv_items, ccfv_root, args.datalist_name)

    # Encoder cfg template (you can edit layers/sample_num to match your model)
    default_layers = [
        "encoder.stem",
        "encoder.stages.0",
        "encoder.stages.1",
        "encoder.stages.2",
        "encoder.stages.3",
    ]
    default_sample_num = [200, 200, 150, 100, 50]
    cfg_path = write_encoder_cfg(
        ccfv_root,
        args.cfg_name,
        roi_xyz=(args.roi_z, args.roi_y, args.roi_x),
        layers=default_layers,
        sample_num=default_sample_num,
    )

    print("=================================================")
    print("[DONE] cases:", len(ccfv_items))
    if dataset_dir is not None:
        print("[DONE] nnU-Net raw:", dataset_dir)
        print("[DONE] dataset.json:", Path(dataset_dir) / "dataset.json")
    else:
        print("[SKIP] nnU-Net raw export (--no_nnunet enabled).")
    print("[DONE] datalist:", datalist_path)
    print("[DONE] encoder cfg:", cfg_path)
    print("=================================================")
    print("Run examples:")
    print(f"  python main.py --cfg {cfg_path} --data_list_file {datalist_path} --model_path <ckpt>")
    print(f"  python main_gbc.py --cfg {cfg_path} --data_list_file {datalist_path} --model_path <ckpt>")
    print("=================================================")

if __name__ == "__main__":
    main()
