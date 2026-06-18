import os
import json
import glob
import argparse
import numpy as np

try:
    import nibabel as nib
except Exception as e:
    raise RuntimeError("需要 nibabel：pip install nibabel") from e


def find_cases(ms_root: str):
    images_tr = os.path.join(ms_root, "imagesTr")
    labels_tr = os.path.join(ms_root, "labelsTr")
    imgs = sorted(glob.glob(os.path.join(images_tr, "*_0000.nii.gz")))
    if not imgs:
        raise RuntimeError(f"没找到 imagesTr/*_0000.nii.gz：{images_tr}")

    cases = []
    for img in imgs:
        base = os.path.basename(img)
        case_id = base.replace("_0000.nii.gz", "")
        lab = os.path.join(labels_tr, f"{case_id}.nii.gz")
        if not os.path.exists(lab):
            raise RuntimeError(f"缺少 label：{lab}（对应 image：{img}）")
        cases.append((case_id, img, lab))
    return cases


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def link_or_copy(src, dst, mode="symlink"):
    if os.path.exists(dst):
        return
    ensure_dir(os.path.dirname(dst))
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        import shutil
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError("mode must be symlink/copy/hardlink")


def welford_update(mean, m2, count, x):
    # x: 1D array
    for v in x:
        count += 1
        delta = v - mean
        mean += delta / count
        delta2 = v - mean
        m2 += delta * delta2
    return mean, m2, count


def compute_mean_std(cases, nonzero=True, max_vox_per_case=200000, clip_pct=(0.5, 99.5), seed=0):
    """
    用 Welford 在线统计全数据 mean/std，避免一次性载入巨量数据。
    MRI 背景 0 多，默认只统计非零体素（更合理）。
    """
    rng = np.random.default_rng(seed)
    mean = 0.0
    m2 = 0.0
    count = 0

    minX = minY = minZ = 10**9

    for _, img_path, _ in cases:
        img = nib.load(img_path).get_fdata(dtype=np.float32)  # shape (X,Y,Z)
        X, Y, Z = img.shape
        minX = min(minX, X); minY = min(minY, Y); minZ = min(minZ, Z)

        arr = img.reshape(-1)
        if nonzero:
            arr = arr[arr != 0]
        if arr.size == 0:
            continue

        # 可选：轻度 clip 去掉极端值
        if clip_pct is not None:
            lo, hi = np.percentile(arr, clip_pct)
            arr = np.clip(arr, lo, hi)

        # 下采样，避免太慢
        if arr.size > max_vox_per_case:
            idx = rng.choice(arr.size, size=max_vox_per_case, replace=False)
            arr = arr[idx]

        mean, m2, count = welford_update(mean, m2, count, arr.astype(np.float64))

    if count < 2:
        raise RuntimeError("统计 mean/std 失败：有效体素过少（可能全是0？）")

    var = m2 / (count - 1)
    std = float(np.sqrt(max(var, 1e-12)))

    # 注意：nibabel shape 是 (X,Y,Z)，而你的评估里会 permute 成 (Z,Y,X)
    min_dims_zyx = (int(minZ), int(minY), int(minX))
    return float(mean), float(std), min_dims_zyx


def make_nnunet_raw(ms_root, nnunet_raw_root, dataset_id, dataset_name, cases, mode="symlink"):
    ds = f"Dataset{dataset_id:03d}_{dataset_name}"
    out_root = os.path.join(nnunet_raw_root, ds)
    out_images = os.path.join(out_root, "imagesTr")
    out_labels = os.path.join(out_root, "labelsTr")
    ensure_dir(out_images)
    ensure_dir(out_labels)

    for case_id, img, lab in cases:
        dst_img = os.path.join(out_images, f"{case_id}_0000.nii.gz")
        dst_lab = os.path.join(out_labels, f"{case_id}.nii.gz")
        link_or_copy(img, dst_img, mode=mode)
        link_or_copy(lab, dst_lab, mode=mode)

    # nnU-Net dataset.json（最基础字段即可）
    dataset_json = {
        "name": dataset_name,
        "description": f"{dataset_name} (single-modality FLAIR) prepared for nnU-Net / CCFV-GBC",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(out_root, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    return out_root


def make_ccfv_datalist(out_json_path, cases, use_paths_from="nnunet_or_src", nnunet_ds_root=None):
    val = []
    for case_id, img, lab in cases:
        if use_paths_from == "nnunet_or_src" and nnunet_ds_root is not None:
            data_path = os.path.join(nnunet_ds_root, "imagesTr", f"{case_id}_0000.nii.gz")
            seg_path  = os.path.join(nnunet_ds_root, "labelsTr", f"{case_id}.nii.gz")
        else:
            data_path = img
            seg_path  = lab

        val.append({"data": os.path.abspath(data_path), "seg": os.path.abspath(seg_path)})

    obj = {"val": val, "fold_id": 0}
    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_encoder_cfg(out_cfg_path, mean, std, roi_zyx, input_channels=1):
    """
    生成 ResEncL 的 encoder 多尺度评估 config
    - layers: encoder.stem + encoder.stages.0..3
    - sample_num: 可按需要再调大/调小
    """
    roi_z, roi_y, roi_x = roi_zyx

    cfg = {
        "layers": ["encoder.stem", "encoder.stages.0", "encoder.stages.1", "encoder.stages.2", "encoder.stages.3"],
        "sample_num": {
            "encoder.stem": 100,
            "encoder.stages.0": 150,
            "encoder.stages.1": 200,
            "encoder.stages.2": 300,
            "encoder.stages.3": 400
        },
        "model": {
            "name": "ResEncL",
            "num_input_channels": int(input_channels),
            "deep_supervision": True
        },
        "num_classes": 1,

        # sliding window ROI（顺序是 z,y,x；会在 evaluate 里用）
        "roi_x": int(roi_x),
        "roi_y": int(roi_y),
        "roi_z": int(roi_z),

        # 归一化（CCFV-main/utils/utility.py 里 NormalizeIntensityd 用这两个）
        "mean": float(mean),
        "std": float(std),

        "sw_batch_size": 2,
        "window_mode": "gaussian",
        "infer_overlap": 0.0
    }

    ensure_dir(os.path.dirname(out_cfg_path))
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms_root", type=str, required=True, help="MS_FLAIR 根目录（包含 imagesTr/ labelsTr/）")
    ap.add_argument("--ccfv_root", type=str, required=True, help="你的 CCFV-main 根目录")
    ap.add_argument("--dataset_id", type=int, default=11, help="nnU-Net Dataset id（避免和现有冲突）")
    ap.add_argument("--dataset_name", type=str, default="MS_FLAIR", help="nnU-Net Dataset name")
    ap.add_argument("--nnunet_raw_root", type=str, default=None, help="nnUNet_raw 根目录（不填则优先读环境变量 nnUNet_raw，否则用 /data/nnUNet/nnUNet_raw）")
    ap.add_argument("--mode", type=str, default="symlink", choices=["symlink", "copy", "hardlink"])
    ap.add_argument("--no_nnunet", action="store_true", help="不创建 nnUNet_raw/DatasetXXX，仅生成 datalist+cfg（直接用原始路径）")
    ap.add_argument("--max_vox_per_case", type=int, default=200000)
    ap.add_argument("--nonzero", action="store_true", help="只用非零体素统计 mean/std（推荐 MRI 用）")
    ap.add_argument("--roi_cap_z", type=int, default=96)
    ap.add_argument("--roi_cap_y", type=int, default=160)
    ap.add_argument("--roi_cap_x", type=int, default=160)
    args = ap.parse_args()

    cases = find_cases(args.ms_root)
    print(f"[OK] found cases: {len(cases)}")

    mean, std, min_zyx = compute_mean_std(
        cases,
        nonzero=args.nonzero,
        max_vox_per_case=args.max_vox_per_case
    )
    # ROI 不能超过最小尺寸；再做一个上限 cap（默认 96/160/160）
    roi_z = min(args.roi_cap_z, min_zyx[0])
    roi_y = min(args.roi_cap_y, min_zyx[1])
    roi_x = min(args.roi_cap_x, min_zyx[2])
    roi_zyx = (roi_z, roi_y, roi_x)

    print(f"[STATS] mean={mean:.6f}, std={std:.6f}, min(Z,Y,X)={min_zyx}, roi(Z,Y,X)={roi_zyx}")

    nnunet_ds_root = None
    if not args.no_nnunet:
        nnunet_raw_root = args.nnunet_raw_root or os.environ.get("nnUNet_raw") or "/data/nnUNet/nnUNet_raw"
        nnunet_ds_root = make_nnunet_raw(
            args.ms_root, nnunet_raw_root, args.dataset_id, args.dataset_name, cases, mode=args.mode
        )
        print(f"[nnU-Net RAW] created: {nnunet_ds_root}")

    # datalist 输出到 CCFV-main/DataLists/
    datalist_path = os.path.join(args.ccfv_root, "DataLists", "MS_FLAIR_val.json")
    make_ccfv_datalist(datalist_path, cases, use_paths_from="nnunet_or_src", nnunet_ds_root=nnunet_ds_root)
    print(f"[DATALIST] saved: {datalist_path}")

    # config 输出到 CCFV-main/Configs/
    cfg_path = os.path.join(args.ccfv_root, "Configs", "MS_FLAIR_FLAIR_ResEncL_Encoder.json")
    make_encoder_cfg(cfg_path, mean, std, roi_zyx, input_channels=1)
    print(f"[CFG] saved: {cfg_path}")

    print("\n====== DONE ======")
    print("You can now run:")
    print(f"  python main.py --cfg {cfg_path} --data_list_file {datalist_path} --model_path <your_model>")
    print(f"  python main_gbc.py --cfg {cfg_path} --data_list_file {datalist_path} --model_path <your_model>")


if __name__ == "__main__":
    main()
