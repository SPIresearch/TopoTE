import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------- robust imports (兼容脚本 / 包模式) ----------
try:
    from .sliding_window_sampling import ms_sliding_window_sampling, GLOBAL_KEY
    from .layer_resolver import resolve_layers_and_sample_num
    from .get_model import get_model, reinit_decoder_modules
    from .utility import setup_seed, load_pretrained_model
except Exception:
    import sys as _sys
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from utils.sliding_window_sampling import ms_sliding_window_sampling, GLOBAL_KEY
    from utils.layer_resolver import resolve_layers_and_sample_num
    from utils.get_model import get_model, reinit_decoder_modules
    from utils.utility import setup_seed, load_pretrained_model

# ---------- MONAI（做版本兼容，避免 RenameKeysd 等） ----------
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd
try:
    from monai.transforms import EnsureChannelFirstd
except Exception:
    from monai.transforms import AddChanneld as EnsureChannelFirstd

from monai.data import DataLoader, Dataset

# ---------- nibabel for affine-aware resampling ----------
try:
    import nibabel as nib
    from nibabel.processing import resample_from_to
except Exception:
    nib = None
    resample_from_to = None


def _project_root() -> Path:
    return Path.cwd()  # 改为返回当前工作目录


def _norm_path(p: str, root: Path) -> str:
    if p is None or not isinstance(p, str):
        return p
    pp = p.replace("\\", "/")
    path = Path(pp)
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)


def _infer_case_id_from_path(p: str) -> str:
    name = Path(p.replace("\\", "/")).name
    for suf in [".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".npz"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    if len(name) >= 5 and name[-5] == "_" and name[-4:].isdigit():
        name = name[:-5]
    return name


def _guess_modal_keys(first_item: Dict[str, Any]) -> List[str]:
    if "data_t1" in first_item and "data_fmri" in first_item:
        return ["data_t1", "data_fmri"]
    if "data0" in first_item and "data1" in first_item:
        return ["data0", "data1"]
    if "data" in first_item:
        return ["data"]
    keys = [k for k in first_item.keys() if k.startswith("data")]
    return sorted(keys)


def _get_affine_from(d: Dict[str, Any], key: str, x: Any) -> np.ndarray:
    mk = f"{key}_meta_dict"
    if mk in d and isinstance(d[mk], dict) and "affine" in d[mk]:
        aff = d[mk]["affine"]
        return np.asarray(aff, dtype=np.float64)
    if hasattr(x, "affine") and x.affine is not None:
        try:
            return np.asarray(x.affine, dtype=np.float64)
        except Exception:
            pass
    return np.eye(4, dtype=np.float64)


def _set_affine_meta(d: Dict[str, Any], key: str, affine: np.ndarray, spatial_shape: Tuple[int, int, int]):
    mk = f"{key}_meta_dict"
    if mk not in d or not isinstance(d[mk], dict):
        d[mk] = {}
    d[mk]["affine"] = np.asarray(affine, dtype=np.float64)
    d[mk]["spatial_shape"] = tuple(int(v) for v in spatial_shape)


def _torch_resize_to(x_cxyz: torch.Tensor, target_xyz: Tuple[int, int, int]) -> torch.Tensor:
    xt = torch.as_tensor(x_cxyz).float().unsqueeze(0)  # (1,C,X,Y,Z)
    yt = F.interpolate(xt, size=target_xyz, mode="trilinear", align_corners=False)
    return yt.squeeze(0)


def _resample_channelwise_to_ref(
    x_cxyz: torch.Tensor,
    affine_x: np.ndarray,
    ref_shape_xyz: Tuple[int, int, int],
    affine_ref: np.ndarray,
    order: int = 1,
) -> torch.Tensor:
    if resample_from_to is None or nib is None:
        return _torch_resize_to(x_cxyz, ref_shape_xyz)

    x_np = torch.as_tensor(x_cxyz).cpu().numpy()
    out = []
    for c in range(x_np.shape[0]):
        img = nib.Nifti1Image(x_np[c], affine_x)
        tgt = (ref_shape_xyz, affine_ref)
        rimg = resample_from_to(img, tgt, order=order)
        out.append(rimg.get_fdata(dtype=np.float32))
    out_np = np.stack(out, axis=0).astype(np.float32)
    return torch.from_numpy(out_np)


class ResampleModalitiesToFirstd:
    def __init__(self, keys: List[str], ref_key: Optional[str] = None, policy: str = "first", order: int = 1):
        self.keys = list(keys)
        self.ref_key = ref_key
        self.policy = policy
        self.order = int(order)

    def __call__(self, d: Dict[str, Any]):
        if self.ref_key and self.ref_key in self.keys:
            rk = self.ref_key
        else:
            if self.policy == "min_voxels":
                best_k = self.keys[0]
                best_v = None
                for k in self.keys:
                    x = torch.as_tensor(d[k])
                    if x.ndim == 3:
                        shp = tuple(int(v) for v in x.shape)
                    else:
                        shp = tuple(int(v) for v in x.shape[-3:])
                    vox = int(shp[0] * shp[1] * shp[2])
                    if best_v is None or vox < best_v:
                        best_v = vox
                        best_k = k
                rk = best_k
            else:
                rk = self.keys[0]

        ref = torch.as_tensor(d[rk])
        if ref.ndim == 3:
            ref = ref.unsqueeze(0)
        ref_shape_xyz = tuple(int(v) for v in ref.shape[1:])
        aff_ref = _get_affine_from(d, rk, d[rk])

        for k in self.keys:
            if k == rk:
                x = torch.as_tensor(d[k])
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                d[k] = x.float()
                _set_affine_meta(d, k, aff_ref, ref_shape_xyz)
                continue

            x = torch.as_tensor(d[k])
            if x.ndim == 3:
                x = x.unsqueeze(0)
            shp = tuple(int(v) for v in x.shape[1:])
            aff_x = _get_affine_from(d, k, d[k])

            same_shape = (shp == ref_shape_xyz)
            same_aff = np.allclose(aff_x, aff_ref, atol=1e-5)
            if same_shape and same_aff:
                d[k] = x.float()
                _set_affine_meta(d, k, aff_ref, ref_shape_xyz)
                continue

            y = _resample_channelwise_to_ref(x.float(), aff_x, ref_shape_xyz, aff_ref, order=self.order)
            d[k] = y
            _set_affine_meta(d, k, aff_ref, ref_shape_xyz)

        d["_resample_ref_key"] = rk
        return d


class PackModalitiesToDatad:
    def __init__(self, modal_keys: List[str], out_key: str = "data"):
        self.modal_keys = list(modal_keys)
        self.out_key = out_key

    def __call__(self, d: Dict[str, Any]):
        arrs = [torch.as_tensor(d[k]) for k in self.modal_keys]
        fixed = []
        for a in arrs:
            a = a.float()
            if a.ndim == 3:
                a = a.unsqueeze(0)
            fixed.append(a)
        d[self.out_key] = torch.cat(fixed, dim=0)  # (C_total,X,Y,Z)
        for k in self.modal_keys:
            if k != self.out_key:
                d.pop(k, None)
        return d


def _load_val_list_and_task(args, configs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, int]:
    data_list = json.load(open(args.data_list_file, "r", encoding="utf-8"))
    val_list = data_list.get("val", [])
    if not val_list:
        raise ValueError(f"data_list_file 里没有 'val' 或为空: {args.data_list_file}")

    first = val_list[0]
    if "seg" in first:
        task = "seg"
    elif "label" in first:
        task = "cls"
    else:
        raise ValueError("无法判断任务类型：每条数据需要包含 'seg' 或 'label'。")

    root = _project_root()
    for item in val_list:
        for k, v in list(item.items()):
            if isinstance(v, str) and (k.startswith("data") or k in ("seg",)):
                item[k] = _norm_path(v, root)
        if "case_id" not in item:
            cand = None
            if "data" in item and isinstance(item["data"], str):
                cand = item["data"]
            else:
                ks = [kk for kk in item.keys() if kk.startswith("data") and isinstance(item[kk], str)]
                if ks:
                    cand = item[sorted(ks)[0]]
            if cand:
                item["case_id"] = _infer_case_id_from_path(cand)

    label_offset = int(configs.get("label_offset", 0))
    if task == "cls" and ("label_offset" not in configs):
        labels = []
        for it in val_list:
            lb = it.get("label")
            if isinstance(lb, (list, tuple)):
                lb = lb[0]
            labels.append(int(lb))
        uniq = sorted(set(labels))
        if (0 not in uniq) and (len(uniq) > 0) and (min(uniq) == 1):
            label_offset = 1
        else:
            label_offset = 0

    return val_list, task, int(label_offset)


def _build_seg_loader(val_list: List[Dict[str, Any]], num_workers: int) -> DataLoader:
    val_transforms = Compose([
        LoadImaged(keys=["data", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["data", "seg"]),
        NormalizeIntensityd(keys=["data"], nonzero=True, channel_wise=True),
    ])
    val_ds = Dataset(data=val_list, transform=val_transforms)
    return DataLoader(val_ds, batch_size=1, num_workers=int(num_workers), shuffle=False, drop_last=False)


def _build_cls_loader(val_list: List[Dict[str, Any]], configs: Dict[str, Any], num_workers: int) -> DataLoader:
    first = val_list[0]
    modal_keys = _guess_modal_keys(first)
    if not modal_keys:
        raise ValueError(f"分类 DataList 未找到 data 字段: keys={list(first.keys())}")

    ref_key = configs.get("resample_ref_key", None)  # e.g., "data_t1" or "data_fmri"
    policy = configs.get("resample_ref_policy", "first")  # "first" | "min_voxels"

    val_transforms = Compose([
        LoadImaged(keys=modal_keys, reader="NibabelReader", image_only=False),
        EnsureChannelFirstd(keys=modal_keys),
        ResampleModalitiesToFirstd(keys=modal_keys, ref_key=ref_key, policy=policy, order=1),
        NormalizeIntensityd(keys=modal_keys, nonzero=True, channel_wise=True),
        PackModalitiesToDatad(modal_keys, out_key="data"),
    ])
    val_ds = Dataset(data=val_list, transform=val_transforms)
    return DataLoader(val_ds, batch_size=1, num_workers=int(num_workers), shuffle=False, drop_last=False)


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_features_per_case(feature_dict: Dict[str, Dict[Any, Any]], save_dir: str, case_idx: int, case_id: Optional[str] = None):
    case_name = f"case_{case_id}" if case_id else f"case_{case_idx:04d}"
    case_dir = Path(save_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for layer_name, class_features in feature_dict.items():
        layer_filename = layer_name.replace(".", "_") + ".pkl"
        layer_path = case_dir / layer_filename
        with open(layer_path, "wb") as f:
            pickle.dump(class_features, f)
        saved_files.append(str(layer_path))

        stats = {}
        for class_id, feats in class_features.items():
            if class_id == GLOBAL_KEY:
                stats["global"] = feats.shape if isinstance(feats, np.ndarray) else len(feats)
            else:
                stats[f"class_{class_id}"] = feats.shape if isinstance(feats, np.ndarray) else len(feats)
        print(f"    保存: {layer_path.name} | {stats}")
    return saved_files


def _make_brain_mask(data_bczyx: torch.Tensor) -> torch.Tensor:
    s = data_bczyx.abs().sum(dim=1, keepdim=True)
    return s > 1e-6


def _remap_cls_feature_dict(feature_dict: Dict[str, Dict[Any, Any]]) -> Dict[str, Dict[Any, Any]]:
    out: Dict[str, Dict[Any, Any]] = {}
    for layer, cls_map in feature_dict.items():
        out[layer] = {}
        for k, v in cls_map.items():
            if k == GLOBAL_KEY:
                out[layer][GLOBAL_KEY] = v
            else:
                kk = int(k)
                out[layer][-1 if kk == 0 else kk - 1] = v
    return out


def _pool_cls_feature_dict(feature_dict: Dict[str, Dict[Any, np.ndarray]],
                           keep_bg: bool = False,
                           bg_label: int = -1) -> Dict[str, Dict[Any, np.ndarray]]:
    """Pool per-layer sampled points to a single embedding per key (mean over points).

    For classification alignment (LogME/GBC/LEEP/OpenMind Kendall), it's often better to
    represent each case with ONE embedding per layer. This converts:
        key -> (N_points, C)  ==>  key -> (1, C)

    - Always keeps GLOBAL_KEY ("__global__") if present (pooled).
    - Drops background (bg_label) by default unless keep_bg=True.
    """
    out: Dict[str, Dict[Any, np.ndarray]] = {}
    for layer, fd in feature_dict.items():
        if not isinstance(fd, dict):
            out[layer] = fd
            continue
        fd2: Dict[Any, np.ndarray] = {}
        for k, v in fd.items():
            if v is None:
                continue
            if k == bg_label and not keep_bg:
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] > 0:
                fd2[k] = v.mean(axis=0, keepdims=True)
            else:
                fd2[k] = v
        out[layer] = fd2
    return out


def extract_and_save_features(configs: Dict[str, Any], test_loader, model, feature_save_dir: str, model_name: str = "model", task: str = "seg", device: torch.device = torch.device("cuda")):
    model.eval()
    feature_save_dir = Path(feature_save_dir)
    feature_save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"特征提取: {model_name}")
    print(f"任务类型: {task}")
    print(f"保存目录: {feature_save_dir}")
    print("=" * 80)

    layers, sample_num = resolve_layers_and_sample_num(configs, model)

    metadata = {
        "model_name": model_name,
        "task": task,
        "layers": layers,
        "sample_budgets": sample_num,
        "label_offset": int(configs.get("label_offset", 0)),
        "resample_ref_key": configs.get("resample_ref_key", None),
        "resample_ref_policy": configs.get("resample_ref_policy", "first"),
        "fv_sample_num": configs.get("fv_sample_num", None),
        "bg_sample_num": configs.get("bg_sample_num", 0),
        "cases": [],
    }

    with torch.no_grad():
        for case_idx, val_data in enumerate(test_loader):
            print(f"\n{'='*60}")
            print(f"Case {case_idx}")
            print(f"{'='*60}")
            if "_resample_ref_key" in val_data:
                rk = val_data["_resample_ref_key"]
                if isinstance(rk, (list, tuple)):
                    rk = rk[0]
                print(f"[RESAMPLE] ref_key={rk} policy={configs.get('resample_ref_policy','first')}")

            case_id = None
            if "case_id" in val_data:
                cid = val_data["case_id"]
                case_id = cid[0] if isinstance(cid, (list, tuple)) else cid

            data = val_data["data"].permute(0, 1, 4, 3, 2).to(device)

            if task == "seg":
                seg = val_data["seg"].permute(0, 1, 4, 3, 2).to(device)
                seg = F.interpolate(seg.float(), size=data.shape[2:], mode="nearest").long()
                feature_dict = ms_sliding_window_sampling(
                    layers, sample_num, data, seg,
                    [configs["roi_z"], configs["roi_y"], configs["roi_x"]],
                    configs["sw_batch_size"], model,
                    overlap=configs.get("infer_overlap", 0.0),
                    mode=configs.get("window_mode", "constant"),
                    fv_sample_num=configs.get("fv_sample_num", None),
                    bg_sample_num=configs.get("bg_sample_num", 0),
                    boundary_sample_num=configs.get("boundary_sample_num", 0),
                    seed=int(configs.get("sampling_seed", 0)),
                    progress=True,
                    token_grid_size=tuple(configs["token_grid_size"]) if "token_grid_size" in configs else None,
                )
                lb = val_data["label"]
                if torch.is_tensor(lb):
                    raw_label = int(lb.item()) if lb.numel() == 1 else int(lb[0].item())
                else:
                    raw_label = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

                offset = int(configs.get("label_offset", 0))
                image_label = raw_label - offset
                if image_label < 0:
                    raise ValueError(f"label_offset={offset} 导致 label<0: raw_label={raw_label}")

                pseudo = _make_brain_mask(data).long() * int(image_label + 1)
                feature_dict = ms_sliding_window_sampling(
                    layers, sample_num, data, pseudo,
                    [configs["roi_z"], configs["roi_y"], configs["roi_x"]],
                    configs["sw_batch_size"], model,
                    overlap=configs.get("infer_overlap", 0.0),
                    mode=configs.get("window_mode", "constant"),
                    fv_sample_num=configs.get("fv_sample_num", None),
                    bg_sample_num=configs.get("bg_sample_num", 0),
                    boundary_sample_num=0,
                    seed=int(configs.get("sampling_seed", 0)),
                    progress=True,
                    token_grid_size=tuple(configs["token_grid_size"]) if "token_grid_size" in configs else None,
                )
                feature_dict = _remap_cls_feature_dict(feature_dict)

                # optional: pool points -> 1 embedding per key (classification-friendly)
                cls_pool = str(configs.get("cls_pool", "none")).lower()
                cls_sampling = str(configs.get("cls_sampling", "point")).lower()
                if cls_pool in ("mean", "avg", "pool") or cls_sampling in ("embed", "embedding"):
                    keep_bg = bool(configs.get("cls_keep_bg", False))
                    feature_dict = _pool_cls_feature_dict(feature_dict, keep_bg=keep_bg, bg_label=-1)

            saved_files = save_features_per_case(feature_dict, feature_save_dir, case_idx, case_id)
            metadata["cases"].append({"case_idx": case_idx, "case_id": str(case_id) if case_id else None, "files": saved_files})

    metadata_path = feature_save_dir / "feature_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n元数据已保存: {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="仅提取和保存特征（支持分割/分类 DataList，多模态支持重采样对齐）")
    parser.add_argument("--cfg", type=str, required=True, help="配置JSON文件")
    parser.add_argument("--data_list_file", type=str, required=True, help="数据列表JSON")
    parser.add_argument("--model_path", type=str, default="", help="模型checkpoint路径")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称")
    parser.add_argument("--feature_save_dir", type=str, required=True, help="特征保存目录")
    parser.add_argument("--bg_sample_num", type=int, default=None, help="背景采样数量（分类会 remap 到 -1）")
    parser.add_argument("--fv_sample_num", type=int, default=None, help="全局采样数量")
    parser.add_argument("--boundary_sample_num", type=int, default=0,
                        help="GT mask 形态学边界 voxel 采样数（分割任务建议 256，0=不采样）")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers（0最稳，>0更快）")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:0")

    args = parser.parse_args()
    setup_seed(0)

    configs = json.load(open(args.cfg, "r", encoding="utf-8"))
    if args.bg_sample_num is not None:
        configs["bg_sample_num"] = int(args.bg_sample_num)
    if args.fv_sample_num is not None:
        configs["fv_sample_num"] = int(args.fv_sample_num)
    if args.boundary_sample_num:
        configs["boundary_sample_num"] = int(args.boundary_sample_num)

    val_list, task, label_offset = _load_val_list_and_task(args, configs)
    configs.setdefault("label_offset", int(label_offset))
    if task == "cls":
        print(f"[CLS] label_offset={configs['label_offset']} (raw_label -> normalized_label = raw - offset)")
        print(f"[CLS] resample_ref_key={configs.get('resample_ref_key', None)} policy={configs.get('resample_ref_policy','first')}")
        if resample_from_to is None:
            print("[WARN] nibabel.resample_from_to 不可用，将退化为按 shape 的 trilinear resize（不看 affine）")

    test_loader = _build_seg_loader(val_list, args.num_workers) if task == "seg" else _build_cls_loader(val_list, configs, args.num_workers)

    device = _device_from_arg(args.device)

    model = get_model(args, configs).to(device)


    model_name = args.model_name or (Path(args.model_path).stem if args.model_path else "random_init")
    if args.model_path:
        model = load_pretrained_model(args.model_path, model)
        print(f"✓ 加载checkpoint: {args.model_path}")
    else:
        print("⚠️  使用随机初始化模型")

    # 关键新增：在只加载 encoder checkpoint 之后，按 configs["decoder_init"] 重新初始化 decoder
    model = reinit_decoder_modules(model, configs)

    metadata = extract_and_save_features(
        configs=configs,
        test_loader=test_loader,
        model=model,
        feature_save_dir=args.feature_save_dir,
        model_name=model_name,
        task=task,
        device=device,
    )

    print("\n" + "=" * 80)
    print("✓ 特征提取完成！")
    print(f"  保存位置: {args.feature_save_dir}")
    print(f"  处理cases: {len(metadata['cases'])}")
    print(f"  提取层数: {len(metadata['layers'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()