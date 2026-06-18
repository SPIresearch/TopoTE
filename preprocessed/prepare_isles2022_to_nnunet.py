import argparse
import json
import re
import shutil
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import nibabel as nib


def sanitize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def first_match(glob_iter):
    for p in glob_iter:
        if p is not None and p.exists():
            return p
    return None


def find_isles_files(root: Path, sub: str, ses: str):
    sub_dir = root / sub / ses
    dwi = first_match((sub_dir / "dwi").glob("*_dwi.nii*"))
    adc = first_match((sub_dir / "dwi").glob("*_adc.nii*"))
    msk = first_match((root / "derivatives" / sub / ses).rglob("*_msk.nii*"))
    return dwi, adc, msk


def sitk_same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize() == b.GetSize()
            and a.GetSpacing() == b.GetSpacing()
            and a.GetOrigin() == b.GetOrigin()
            and a.GetDirection() == b.GetDirection())


def resample_to_ref(moving: sitk.Image, ref: sitk.Image, is_label: bool):
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32
    tx = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(moving, ref, tx, interp, 0, out_type)


def binarize_label_to_uint8(in_path: Path, out_path: Path):
    img = nib.load(str(in_path))
    data = img.get_fdata()
    out = (data > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(out, img.affine, img.header), str(out_path))


def write_dataset_json(dataset_dir: Path, num_training: int):
    ds = {
        "channel_names": {"0": "DWI", "1": "ADC"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": int(num_training),
        "file_ending": ".nii.gz"
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", required=True, help="ISLES-2022 根目录（含 derivatives/ 和 sub-*/）")
    ap.add_argument("--nnunet_raw", required=True, help="nnUNet_raw 根目录")
    ap.add_argument("--dataset_id", type=int, default=2)
    ap.add_argument("--dataset_name", type=str, default="ISLES2022_DWIADC")
    ap.add_argument("--force_resample_to_dwi", action="store_true",
                    help="若 ADC/label 与 DWI 网格不一致，则重采样到 DWI")
    ap.add_argument("--binarize_label", action="store_true", help="label>0 二值化保存为 uint8")
    ap.add_argument("--overwrite", action="store_true", help="若 Dataset 已存在则删除重建")
    args = ap.parse_args()

    bids_root = Path(args.bids_root).expanduser().resolve()
    nnunet_raw = Path(args.nnunet_raw).expanduser().resolve()

    dataset_dir = nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)

    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    subs = sorted([p.name for p in bids_root.iterdir() if p.is_dir() and p.name.startswith("sub-") and p.name != "derivatives"])
    if not subs:
        raise RuntimeError("未找到 sub-* 目录，请确认 bids_root 是否正确")

    n_train, skipped = 0, 0

    for sub in subs:
        ses_list = sorted([p.name for p in (bids_root / sub).iterdir() if p.is_dir() and p.name.startswith("ses-")])
        if not ses_list:
            ses_list = ["ses-0001"]

        for ses in ses_list:
            dwi_p, adc_p, msk_p = find_isles_files(bids_root, sub, ses)
            if dwi_p is None or adc_p is None or msk_p is None:
                skipped += 1
                continue

            cid = sanitize(f"{sub}_{ses}")

            out_dwi = imagesTr / f"{cid}_0000.nii.gz"
            out_adc = imagesTr / f"{cid}_0001.nii.gz"
            out_msk = labelsTr / f"{cid}.nii.gz"

            if not args.force_resample_to_dwi:
                shutil.copy2(dwi_p, out_dwi)
                shutil.copy2(adc_p, out_adc)
                if args.binarize_label:
                    binarize_label_to_uint8(msk_p, out_msk)
                else:
                    shutil.copy2(msk_p, out_msk)
            else:
                # 用 DWI 作为 reference，保证 ADC/mask 同网格
                dwi = sitk.ReadImage(str(dwi_p))
                adc = sitk.ReadImage(str(adc_p))
                msk = sitk.ReadImage(str(msk_p))

                # 写 DWI（保持原样，转 float32）
                dwi_f = sitk.Cast(dwi, sitk.sitkFloat32)
                sitk.WriteImage(dwi_f, str(out_dwi))

                # ADC -> DWI
                if not sitk_same_grid(adc, dwi):
                    adc_r = resample_to_ref(adc, dwi, is_label=False)
                else:
                    adc_r = sitk.Cast(adc, sitk.sitkFloat32)
                sitk.WriteImage(adc_r, str(out_adc))

                # mask -> DWI (NN)
                if not sitk_same_grid(msk, dwi):
                    msk_r = resample_to_ref(msk, dwi, is_label=True)
                else:
                    msk_r = sitk.Cast(msk, sitk.sitkUInt8)
                sitk.WriteImage(msk_r, str(out_msk))

                # 二值化（可选再做一次，确保 0/1）
                if args.binarize_label:
                    binarize_label_to_uint8(out_msk, out_msk)

            n_train += 1
            print(f"[OK] {cid} | DWI+ADC+MSK")

    write_dataset_json(dataset_dir, n_train)
    print("\n====== DONE ======")
    print("nnU-Net dataset:", dataset_dir)
    print("Training cases:", n_train)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()
