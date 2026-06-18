#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_tpc_to_nnunet.py
=======================
输入结构（你现在的）：
TPC/
  imagesTr/             # 里面是每个case一个压缩包(或也可能直接是 .nii/.nii.gz)
  cow_seg_labelsTr/     # 里面是每个case一个压缩包(或也可能直接是 .nii/.nii.gz)

输出结构（nnU-Net raw 标准）：
nnUNet_raw/
  DatasetXXX_<dataset_dir_name>/
    imagesTr/<case>_0000.nii.gz
    labelsTr/<case>.nii.gz
    dataset.json
    prepare_report.csv

关键：
- imagesTr/labelsTr 内的“每个文件可能是 zip/tar/tgz，也可能已是 .nii/.nii.gz”
- 解压在临时目录，但最终一定 COPY 成真实文件（不symlink，避免“无法显示/断链”）
"""

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

ARCHIVE_EXTS = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2")


def is_archive(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(ext) for ext in ARCHIVE_EXTS)


def is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = archive.name.lower()
    if n.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
        return
    if any(n.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(out_dir)
        return
    raise ValueError(f"Unknown archive: {archive}")


def list_nii_files(root: Path) -> List[Path]:
    nii_gz = list(root.rglob("*.nii.gz"))
    nii = [p for p in root.rglob("*.nii") if not p.name.endswith(".nii.gz")]
    return sorted(nii_gz + nii)


def safe_stem_nii(p: Path) -> str:
    n = p.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return p.stem


def stem_without_archive_suffix(p: Path) -> str:
    """
    foo.tar.gz -> foo
    foo.zip    -> foo
    foo.nii.gz -> foo
    """
    name = p.name
    low = name.lower()
    for ext in (".tar.gz", ".tar.bz2", ".nii.gz", ".zip", ".tgz", ".tbz2", ".tar", ".nii"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return p.stem


def normalize_case_name(s: str) -> str:
    """
    topcow_ct_001_0000 -> topcow_ct_001
    """
    if s.endswith("_0000"):
        return s[:-5]
    return s


def gzip_copy(src_nii: Path, dst_nii_gz: Path) -> None:
    dst_nii_gz.parent.mkdir(parents=True, exist_ok=True)
    with open(src_nii, "rb") as f_in, gzip.open(dst_nii_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def copy_nii_to_nii_gz(src: Path, dst: Path) -> None:
    """
    统一输出 .nii.gz
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if src.name.endswith(".nii.gz"):
        shutil.copy2(src, dst)
    elif src.name.endswith(".nii"):
        gzip_copy(src, dst)
    else:
        raise ValueError(f"Not a NIfTI: {src}")


def score_as_label(p: Path) -> int:
    s = p.as_posix().lower()
    score = 0
    for kw, w in [("seg", 6), ("label", 6), ("mask", 6), ("gt", 5),
                 ("labels", 3), ("annotation", 3)]:
        if kw in s:
            score += w
    return score


def score_as_image(p: Path) -> int:
    s = p.as_posix().lower()
    score = 0
    for kw, w in [("ct", 5), ("image", 4), ("img", 3), ("volume", 2), ("data", 1)]:
        if kw in s:
            score += w
    for kw in ["seg", "label", "mask", "gt"]:
        if kw in s:
            score -= 10
    return score


def pick_one_nifti(nii_files: List[Path], kind: str) -> Optional[Path]:
    """
    kind: 'image' or 'label'
    若压缩包内多个nii，做个鲁棒选择
    """
    if not nii_files:
        return None
    if len(nii_files) == 1:
        return nii_files[0]
    if kind == "image":
        return sorted(nii_files, key=score_as_image, reverse=True)[0]
    else:
        return sorted(nii_files, key=score_as_label, reverse=True)[0]


def find_next_dataset_id(nnunet_raw: Path) -> int:
    pat = re.compile(r"^Dataset(\d{3})_")
    mx = 0
    if nnunet_raw.exists():
        for p in nnunet_raw.iterdir():
            if p.is_dir():
                m = pat.match(p.name)
                if m:
                    mx = max(mx, int(m.group(1)))
    return mx + 1 if mx > 0 else 1


def write_dataset_json(dataset_dir: Path, modality_name: str, labels_map: Dict[str, int], num_training: int) -> None:
    d = {
        "channel_names": {"0": modality_name},
        "labels": labels_map,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_case_nifti(src_file: Path, kind: str) -> Tuple[Optional[Path], List[str]]:
    """
    给一个文件（可能是 .nii/.nii.gz 或 archive），返回 (nifti_path, warnings)
    nifti_path 是临时解压目录里的实际nii(调用者负责copy)，或直接是原文件（若已是nii）
    """
    warns = []
    if is_nifti(src_file):
        return src_file, warns
    if is_archive(src_file):
        with tempfile.TemporaryDirectory(prefix="tpc_unpack_") as td:
            td_path = Path(td)
            extract_archive(src_file, td_path)
            niis = list_nii_files(td_path)
            if not niis:
                return None, ["no_nii_in_archive"]
            chosen = pick_one_nifti(niis, kind=kind)
            if chosen is None:
                return None, ["no_nii_after_pick"]
            # 注意：chosen 在临时目录里，不能直接返回（临时目录会被删）
            # 这里做法：把 chosen 复制到一个“新的临时文件”再返回其路径（由调用者再copy到最终输出）
            tmp_keep_dir = Path(tempfile.mkdtemp(prefix="tpc_keep_"))
            kept = tmp_keep_dir / chosen.name
            shutil.copy2(chosen, kept)
            warns.append("extracted_from_archive")
            return kept, warns
    return None, ["unsupported_file_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True, help="输入TPC根目录（里面有 imagesTr 和 labels目录）")
    ap.add_argument("--out_root", type=str, required=True, help="nnUNet_raw 根目录")
    ap.add_argument("--dataset_dir_name", type=str, required=True, help="数据集名，如 TPC -> DatasetXXX_TPC")
    ap.add_argument("--labels_dir_name", type=str, required=True, help="TPC下标签目录名，如 cow_seg_labelsTr")
    ap.add_argument("--images_dir_name", type=str, default="imagesTr", help="TPC下影像目录名，默认 imagesTr")
    ap.add_argument("--modality_name", type=str, default="CT", help="dataset.json 里 channel_names['0']，如 CT")
    ap.add_argument("--dataset_id", type=int, default=-1, help="可选：指定 Dataset ID；不填则自动取下一个")
    ap.add_argument("--labels_json", type=str, default='{"background":0,"cow":1}', help="labels映射JSON")
    ap.add_argument("--dry_run", action="store_true", help="只扫描不写文件")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    img_dir = in_root / args.images_dir_name
    seg_dir = in_root / args.labels_dir_name
    if not img_dir.exists():
        raise FileNotFoundError(f"images dir not found: {img_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {seg_dir}")

    nnunet_raw = Path(args.out_root)
    dsid = args.dataset_id if args.dataset_id > 0 else find_next_dataset_id(nnunet_raw)
    dataset_dir = nnunet_raw / f"Dataset{dsid:03d}_{args.dataset_dir_name}"
    out_imagesTr = dataset_dir / "imagesTr"
    out_labelsTr = dataset_dir / "labelsTr"
    out_imagesTr.mkdir(parents=True, exist_ok=True)
    out_labelsTr.mkdir(parents=True, exist_ok=True)

    labels_map = json.loads(args.labels_json)

    # 读取两个目录下所有文件（每个case一个文件：可能是 archive 也可能是 nii/nii.gz）
    img_files = sorted([p for p in img_dir.iterdir() if p.is_file()])
    seg_files = sorted([p for p in seg_dir.iterdir() if p.is_file()])

    # 用“文件基名”作为case键（去掉压缩后缀/nii后缀），然后 normalize(_0000)
    def key_of(p: Path) -> str:
        return normalize_case_name(stem_without_archive_suffix(p))

    img_map: Dict[str, Path] = {key_of(p): p for p in img_files}
    seg_map: Dict[str, Path] = {key_of(p): p for p in seg_files}

    all_cases = sorted(set(img_map.keys()) | set(seg_map.keys()))

    report_rows = []
    ok = 0
    skip = 0

    def record(status: str, case: str, img_src: str, seg_src: str, img_out: str, seg_out: str, warns: List[str]):
        report_rows.append({
            "status": status,
            "case": case,
            "image_src": img_src,
            "label_src": seg_src,
            "image_out": img_out,
            "label_out": seg_out,
            "warnings": "|".join(warns) if warns else ""
        })

    for case in all_cases:
        if case not in img_map:
            record("SKIP_NO_IMAGE_FILE", case, "", str(seg_map.get(case, "")), "", "", ["missing_image_file"])
            skip += 1
            continue
        if case not in seg_map:
            record("SKIP_NO_LABEL_FILE", case, str(img_map.get(case, "")), "", "", "", ["missing_label_file"])
            skip += 1
            continue

        img_src_file = img_map[case]
        seg_src_file = seg_map[case]

        img_nifti, w1 = resolve_case_nifti(img_src_file, kind="image")
        seg_nifti, w2 = resolve_case_nifti(seg_src_file, kind="label")
        warns = w1 + w2

        if img_nifti is None or seg_nifti is None:
            record("SKIP_CANNOT_RESOLVE_NIFTI", case, str(img_src_file), str(seg_src_file), "", "", warns + ["resolve_failed"])
            skip += 1
            continue

        out_img = out_imagesTr / f"{case}_0000.nii.gz"
        out_seg = out_labelsTr / f"{case}.nii.gz"

        try:
            if not args.dry_run:
                copy_nii_to_nii_gz(img_nifti, out_img)
                copy_nii_to_nii_gz(seg_nifti, out_seg)
            record("OK", case, str(img_src_file), str(seg_src_file), str(out_img), str(out_seg), warns)
            ok += 1
        except Exception as e:
            record("ERROR_WRITE", case, str(img_src_file), str(seg_src_file), str(out_img), str(out_seg),
                   warns + [f"{type(e).__name__}:{e}"])
            skip += 1

        # 如果 resolve_case_nifti 为 archive 生成了 tpc_keep_* 临时目录文件，这里可以清理其父目录
        # （不强制，保持简单；如果你case很多，想清理我再给你加自动清理）

    if not args.dry_run:
        write_dataset_json(dataset_dir, args.modality_name, labels_map, ok)
        with open(dataset_dir / "prepare_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["status","case","image_src","label_src","image_out","label_out","warnings"])
            w.writeheader()
            for r in report_rows:
                w.writerow(r)

    print("=================================================")
    print(f"[DONE] {dataset_dir}")
    print(f"  OK={ok}  SKIP/ERR={skip}")
    if not args.dry_run:
        print(f"  dataset.json       : {dataset_dir/'dataset.json'}")
        print(f"  prepare_report.csv : {dataset_dir/'prepare_report.csv'}")
    else:
        print("  DRY_RUN: no files written")
    print("=================================================")


if __name__ == "__main__":
    main()
