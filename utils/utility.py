import json
import os
import random
import re
from collections import OrderedDict

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
)

DEFAULT_RECONSTRUCTION_METHODS = {
    "mae",
    "mg",
    "s3d",
    "simmim",
    "swinunetr",
    "vf",
}
DEFAULT_CONTRASTIVE_METHODS = {
    "voco",
    "simclr",
}


def get_loader(args, configs):
    data_list = json.load(open(args.data_list_file))
    val_list = data_list["val"]

    val_transforms = Compose([
        LoadImaged(keys=["data", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["data", "seg"]),
        NormalizeIntensityd(keys=["data"], nonzero=True, channel_wise=True),
    ])

    val_ds = Dataset(data=val_list, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    return val_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "network_weights",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "student",
            "teacher",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix(key: str):
    for p in ["module.", "model.", "network.", "net.", "student.", "teacher."]:
        if key.startswith(p):
            return key[len(p):]
    return key


def _is_encoder_key(k: str) -> bool:
    """判断一个 key 是否属于 encoder（而非 decoder / seg_layers）。

    ResEncL 结构:
        encoder.stem.*          -> encoder ✅
        encoder.stages.*        -> encoder ✅
        decoder.stages.*        -> decoder ❌
        decoder.seg_layers.*    -> seg head ❌

    PrimusM 结构:
        down_projection.*       -> encoder (stem) ✅
        eva.*                   -> encoder (transformer blocks) ✅
        up_projection.*         -> decoder (PatchDecode) ❌
        decoder.*               -> decoder (EvaMAE 的 decoder eva) ❌
        seg_layers.*            -> seg head ❌
    """
    decoder_prefixes = (
        "decoder.",
        "up_projection.",
        "seg_layers.",
    )
    for pfx in decoder_prefixes:
        if k.startswith(pfx):
            return False

    encoder_prefixes = (
        "encoder.",
        "eva.",
        "down_projection.",
    )
    for pfx in encoder_prefixes:
        if k.startswith(pfx):
            return True

    return False


def _normalize_policy_name(policy):
    key = str(policy).strip().lower()
    aliases = {
        "encoder_only": "encoder_only",
        "encoder-only": "encoder_only",
        "random_decoder": "encoder_only",
        "random-decoder": "encoder_only",
        "full": "full",
        "full_model": "full",
        "full-model": "full",
        "with_decoder": "full",
        "with-decoder": "full",
        "openmind_hybrid": "openmind_hybrid",
        "openmind-hybrid": "openmind_hybrid",
        "auto_openmind": "openmind_hybrid",
        "auto-openmind": "openmind_hybrid",
        "hybrid": "openmind_hybrid",
    }
    return aliases.get(key, key)


def _parse_method_set(methods, defaults):
    if methods is None:
        return {str(x).strip().lower() for x in defaults if str(x).strip()}

    if isinstance(methods, str):
        parts = methods.replace(";", ",").split(",")
        return {p.strip().lower() for p in parts if p.strip()}

    if isinstance(methods, (list, tuple, set)):
        return {str(x).strip().lower() for x in methods if str(x).strip()}

    return {str(x).strip().lower() for x in defaults if str(x).strip()}


def _infer_openmind_method_tag(path: str, model_name: str = ""):
    candidates = [str(model_name or ""), os.path.basename(str(path or ""))]
    candidates = [c for c in candidates if c]

    known = sorted(
        set(DEFAULT_RECONSTRUCTION_METHODS) | set(DEFAULT_CONTRASTIVE_METHODS),
        key=len,
        reverse=True,
    )

    for cand in candidates:
        low = cand.lower()

        m = re.search(r"openmind[-_]?([a-z0-9]+)", low)
        if m:
            raw = m.group(1).strip().lower()
            if raw in known:
                return raw

        for tag in known:
            if re.search(rf"(^|[^a-z0-9]){re.escape(tag)}([^a-z0-9]|$)", low):
                return tag

    return None


def _resolve_encoder_only(
    path: str,
    encoder_only: bool = True,
    decoder_loading_policy: str | None = None,
    model_name: str = "",
    reconstruction_methods=None,
    contrastive_methods=None,
    unknown_policy: str = "encoder_only",
):
    if decoder_loading_policy is None:
        return bool(encoder_only), "legacy_encoder_only" if bool(encoder_only) else "legacy_full"

    policy = _normalize_policy_name(decoder_loading_policy)
    if policy == "encoder_only":
        return True, "encoder_only"
    if policy == "full":
        return False, "full"
    if policy != "openmind_hybrid":
        raise ValueError(
            f"Unknown decoder loading policy: '{decoder_loading_policy}'. "
            "Supported: encoder_only | full | openmind_hybrid"
        )

    rec_methods = _parse_method_set(reconstruction_methods, DEFAULT_RECONSTRUCTION_METHODS)
    con_methods = _parse_method_set(contrastive_methods, DEFAULT_CONTRASTIVE_METHODS)
    method = _infer_openmind_method_tag(path=path, model_name=model_name)

    if method in rec_methods:
        return False, f"openmind_hybrid:reconstruction:{method}"
    if method in con_methods:
        return True, f"openmind_hybrid:contrastive:{method}"

    unknown = _normalize_policy_name(unknown_policy)
    if unknown == "full":
        return False, f"openmind_hybrid:unknown->{unknown}"
    return True, f"openmind_hybrid:unknown->{unknown}"


def load_pretrained_model(
    path,
    model,
    min_match_ratio=0.01,
    encoder_only=True,
    decoder_loading_policy=None,
    model_name="",
    reconstruction_methods=None,
    contrastive_methods=None,
    unknown_policy="encoder_only",
):
    """加载预训练权重。

    Args:
        path: checkpoint 文件路径
        model: 目标模型
        min_match_ratio: 最低匹配率阈值
        encoder_only: 兼容旧逻辑。True 时默认只加载 encoder，False 时加载 encoder+decoder。
        decoder_loading_policy:
            - encoder_only  : 只加载 encoder（decoder 随机）
            - full          : 加载 encoder+decoder
            - openmind_hybrid:
                reconstruction-based 方法加载 decoder，
                contrastive-based 方法保持随机 decoder。
        model_name: 可选，辅助 openmind_hybrid 识别方法名
        reconstruction_methods: openmind_hybrid 下视为 reconstruction 的方法集合
        contrastive_methods: openmind_hybrid 下视为 contrastive 的方法集合
        unknown_policy: openmind_hybrid 下未识别方法时回退策略（encoder_only/full）
    """
    effective_encoder_only, policy_trace = _resolve_encoder_only(
        path=path,
        encoder_only=encoder_only,
        decoder_loading_policy=decoder_loading_policy,
        model_name=model_name,
        reconstruction_methods=reconstruction_methods,
        contrastive_methods=contrastive_methods,
        unknown_policy=unknown_policy,
    )

    ckpt = torch.load(path, map_location="cpu")
    sd_raw = _extract_state_dict(ckpt)

    if not isinstance(sd_raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(sd_raw)} in {path}")

    sd = OrderedDict((_strip_prefix(k), v) for k, v in sd_raw.items())
    model_sd = model.state_dict()

    matched = {}
    mismatched = []
    skipped_decoder = []

    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            if effective_encoder_only and not _is_encoder_key(k):
                skipped_decoder.append(k)
                continue
            matched[k] = v
        elif k in model_sd and hasattr(v, "shape"):
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    ratio_direct = len(matched) / max(len(model_sd), 1)
    if ratio_direct < 0.1:
        print(f"[LOAD] Direct match ratio too low ({ratio_direct:.2%}), trying EvaMAE -> PrimusM remapping...")

        remap_strategies = [
            lambda k: f"encoder.{k}" if (k.startswith("eva.") or k.startswith("down_projection.")) else k,
            lambda k: k,
            lambda k: k.replace("encoder.", "", 1) if k.startswith("encoder.") else k,
        ]

        best_matched = matched
        best_ratio = ratio_direct
        best_skipped = skipped_decoder

        for i, remap_fn in enumerate(remap_strategies):
            trial_matched = {}
            trial_skipped = []
            for k, v in sd.items():
                new_k = remap_fn(k)
                if new_k in model_sd and hasattr(v, "shape") and v.shape == model_sd[new_k].shape:
                    if effective_encoder_only and not _is_encoder_key(new_k):
                        trial_skipped.append(new_k)
                        continue
                    trial_matched[new_k] = v

            trial_ratio = len(trial_matched) / max(len(model_sd), 1)
            print(
                f"  Strategy {i}: mapped {len(trial_matched)}/{len(model_sd)} ({trial_ratio:.2%})"
                f" (skipped {len(trial_skipped)} decoder keys)"
            )

            if trial_ratio > best_ratio:
                best_matched = trial_matched
                best_ratio = trial_ratio
                best_skipped = trial_skipped

        matched = best_matched
        skipped_decoder = best_skipped
        print(f"[LOAD] Best remap: {len(matched)}/{len(model_sd)} ({best_ratio:.2%})")

    first_key = "encoder.stem.convs.0.conv.weight"
    if first_key in sd and first_key in model_sd:
        w_ckpt = sd[first_key]
        w_model = model_sd[first_key]
        if hasattr(w_ckpt, "shape") and hasattr(w_model, "shape") and w_ckpt.ndim == 5 and w_model.ndim == 5:
            oc1, ic1, kz1, ky1, kx1 = w_ckpt.shape
            oc2, ic2, kz2, ky2, kx2 = w_model.shape
            if oc1 == oc2 and (kz1, ky1, kx1) == (kz2, ky2, kx2) and ic1 != ic2:
                if ic1 == 1 and ic2 > 1:
                    w_new = w_ckpt.repeat(1, ic2, 1, 1, 1) / float(ic2)
                    matched[first_key] = w_new
                    print(f"[LOAD] adapt first conv: {first_key} {w_ckpt.shape} -> {w_new.shape}")
                elif ic1 > 1 and ic2 == 1:
                    w_new = w_ckpt.mean(dim=1, keepdim=True)
                    matched[first_key] = w_new
                    print(f"[LOAD] adapt first conv: {first_key} {w_ckpt.shape} -> {w_new.shape}")

    for stem_key_candidate in ["down_projection.proj.weight", "encoder.down_projection.proj.weight"]:
        ckpt_key = stem_key_candidate.replace("encoder.", "") if "encoder." in stem_key_candidate else stem_key_candidate
        model_key = stem_key_candidate

        if ckpt_key in sd and model_key in model_sd:
            w_ckpt = sd[ckpt_key]
            w_model = model_sd[model_key]
            if hasattr(w_ckpt, "shape") and hasattr(w_model, "shape") and w_ckpt.ndim == 5 and w_model.ndim == 5:
                oc1, ic1 = w_ckpt.shape[:2]
                oc2, ic2 = w_model.shape[:2]
                if oc1 == oc2 and ic1 != ic2:
                    if ic1 == 1 and ic2 > 1:
                        w_new = w_ckpt.repeat(1, ic2, 1, 1, 1) / float(ic2)
                        matched[model_key] = w_new
                        print(f"[LOAD] adapt PrimusM stem: {model_key} {w_ckpt.shape} -> {w_new.shape}")
                    elif ic1 > 1 and ic2 == 1:
                        w_new = w_ckpt.mean(dim=1, keepdim=True)
                        matched[model_key] = w_new
                        print(f"[LOAD] adapt PrimusM stem: {model_key} {w_ckpt.shape} -> {w_new.shape}")

    encoder_loaded = [k for k in matched if _is_encoder_key(k)]
    decoder_loaded = [k for k in matched if not _is_encoder_key(k)]
    ratio = len(matched) / max(len(model_sd), 1)

    # Expose loading status to downstream initialisation code. This is
    # especially important for the openmind_hybrid policy: reconstruction-
    # based checkpoints intentionally reuse their pretrained decoder, so a
    # later unconditional decoder_init would silently destroy the hybrid
    # setting reported in the paper.
    model._decoder_loading_policy_trace = policy_trace
    model._effective_encoder_only = bool(effective_encoder_only)
    model._encoder_keys_loaded = list(encoder_loaded)
    model._decoder_keys_loaded = list(decoder_loaded)
    model._pretrained_decoder_loaded = len(decoder_loaded) > 0

    print(f"[LOAD] policy={policy_trace} | effective_encoder_only={effective_encoder_only}")
    print(f"[LOAD] {os.path.basename(path)} | matched={len(matched)}/{len(model_sd)} ({ratio:.2%})")
    print(f"[LOAD]   encoder keys loaded: {len(encoder_loaded)}")
    print(f"[LOAD]   decoder keys loaded: {len(decoder_loaded)}")
    if effective_encoder_only and skipped_decoder:
        print(f"[LOAD]   decoder keys SKIPPED (random init): {len(skipped_decoder)}")
        for k in skipped_decoder[:5]:
            print(f"[LOAD]     - {k}")
        if len(skipped_decoder) > 5:
            print(f"[LOAD]     ... and {len(skipped_decoder) - 5} more")

    if ratio < min_match_ratio:
        raw_keys = list(sd_raw.keys())[:20]
        model_keys = list(model_sd.keys())[:20]
        raise RuntimeError(
            f"Checkpoint seems incompatible (matched ratio {ratio:.2%} < {min_match_ratio:.2%}).\n"
            f"Example ckpt keys: {raw_keys}\nExample model keys: {model_keys}\n"
            f"ckpt file: {path}"
        )

    model_sd.update(matched)
    model.load_state_dict(model_sd, strict=False)
    return model
