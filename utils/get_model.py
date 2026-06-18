import os
import sys
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from Models.nets.generic_UNet import Generic_UNet, InitWeights_He, InitWeights_Gaussian, InitWeights_Xavier
from Models.nets.unetr import UNETR
from Models.nets.swin_unetr import SwinUNETR
from Models.nets.vnet import VNet


@contextmanager
def _temporary_seed(seed: int):
    """
    Temporarily set RNG seed and fully restore outer RNG states afterwards.
    This isolates init-only randomness from the global randomness used elsewhere.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None

    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass


class RandomConvClsHead3D(nn.Module):
    """
    A small multi-layer 3D conv classification head (random init) for feature extraction.

    Module names are kept stable for cfg hooks:
        cls_head.conv1 / cls_head.act1 / cls_head.conv2 / cls_head.act2 / cls_head.conv_out

    Uses LazyConv3d so in_channels is inferred from the attached feature map.
    """

    def __init__(self, out_channels: int = 2, mid_channels: int = 128, act: str = "relu"):
        super().__init__()

        act_l = str(act).lower()
        if act_l in ("relu",):
            Act = lambda: nn.ReLU(inplace=True)
        elif act_l in ("leakyrelu", "lrelu"):
            Act = lambda: nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        elif act_l in ("gelu",):
            Act = lambda: nn.GELU()
        else:
            Act = lambda: nn.ReLU(inplace=True)

        self.conv1 = nn.LazyConv3d(mid_channels, kernel_size=1, bias=True)
        self.act1 = Act()
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.act2 = Act()
        self.conv_out = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv_out(x)
        return x


class RandomProjHead3D(nn.Module):
    """随机初始化的 1×1×1 线性投影头，用于 TLC 特征采样。

    对 encoder 输出 (B, C, Z, Y, X) 做线性投影 → (B, out_dim, Z, Y, X)。
    使用 LazyConv3d 自动推断输入通道数，无需手动指定 in_channels。

    Config 字段 (random_proj):
        enabled      : true
        attach_layer : "encoder.stages.4"   # hook 挂载的 encoder 层
        out_dim      : 128                  # 投影后的特征维度
        seed         : 42                   # 隔离的随机初始化种子
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        # 1×1×1 conv = linear projection, no nonlinearity — pure random projection
        self.proj = nn.LazyConv3d(out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _init_one_module(
    m: nn.Module,
    init_type: str = "xavier",
    neg_slope: float = 1e-2,
    gain: float = 1.0,
    std: float = 0.02,
):
    init_type = str(init_type).lower()

    if isinstance(m, (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.Linear
    )):
        if getattr(m, "weight", None) is not None:
            if init_type in ("he", "kaiming"):
                nn.init.kaiming_normal_(m.weight, a=neg_slope)
            elif init_type in ("gaussian", "normal"):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif init_type in ("xavier_uniform",):
                nn.init.xavier_uniform_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(m.weight, gain=gain)

        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.GroupNorm, nn.LayerNorm
    )):
        if getattr(m, "weight", None) is not None:
            nn.init.constant_(m.weight, 1.0)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)


def _get_model_cfg(configs: dict) -> dict:
    if "model" in configs and isinstance(configs["model"], dict):
        return configs["model"]
    return configs


def _install_random_cls_head(model: nn.Module, configs: dict) -> nn.Module:
    ch = configs.get("cls_head", {}) if isinstance(configs.get("cls_head", {}), dict) else {}
    if not ch.get("enabled", False):
        return model

    attach_layer = ch.get("attach_layer", "encoder.stages.3")
    out_channels = int(ch.get("out_channels", configs.get("num_classes", 2)))
    mid_channels = int(ch.get("mid_channels", 128))
    act = ch.get("act", "relu")
    init_seed = int(ch.get("init_seed", 0))

    # 关键：cls_head 初始化 seed 与全局 seed 隔离
    with _temporary_seed(init_seed):
        model.cls_head = RandomConvClsHead3D(
            out_channels=out_channels,
            mid_channels=mid_channels,
            act=act,
        )

    name2mod = dict(model.named_modules())
    if attach_layer not in name2mod:
        near = [k for k in name2mod.keys() if k.startswith("encoder.stages.3")][:30]
        raise KeyError(
            f"cls_head.attach_layer='{attach_layer}' not found in model.named_modules(). "
            f"Example candidates: {near}"
        )

    if hasattr(model, "_cls_head_hook_handle") and model._cls_head_hook_handle is not None:
        try:
            model._cls_head_hook_handle.remove()
        except Exception:
            pass
        model._cls_head_hook_handle = None

    def _hook(_m, _inp, _out):
        x = _out
        if isinstance(x, (tuple, list)):
            for t in x:
                if torch.is_tensor(t):
                    x = t
                    break
        if not torch.is_tensor(x) or x.dim() != 5:
            return
        model._last_cls_logits = model.cls_head(x)

    model._cls_head_hook_handle = name2mod[attach_layer].register_forward_hook(_hook)
    model._cls_head_attached_to = attach_layer
    print(f"[INIT] cls_head initialized with isolated seed={init_seed}")
    return model


def _install_random_proj_head(model: nn.Module, configs: dict) -> nn.Module:
    """在 encoder 指定层后挂一个随机线性投影头，供 TLC 特征采样使用。

    Config 字段（JSON 顶层或 model 字典里）：
        random_proj:
            enabled      : true
            attach_layer : "encoder.stages.4"   # hook 挂载点（encoder 最深层）
            out_dim      : 128                   # 投影维度
            seed         : 42                    # 隔离的随机初始化种子

    挂载后模型新增子模块 `rand_proj`，可在 config 的 layers 里直接写
    "rand_proj" 来提取投影后的特征。
    """
    rp_cfg = configs.get("random_proj", {})
    if not isinstance(rp_cfg, dict) or not rp_cfg.get("enabled", False):
        return model

    attach_layer = rp_cfg.get("attach_layer", "encoder.stages.4")
    out_dim      = int(rp_cfg.get("out_dim", 128))
    seed         = int(rp_cfg.get("seed", 42))

    # 随机初始化 seed 与全局 seed 隔离，保证可复现
    with _temporary_seed(seed):
        model.rand_proj = RandomProjHead3D(out_dim=out_dim)

    name2mod = dict(model.named_modules())
    if attach_layer not in name2mod:
        candidates = [k for k in name2mod.keys() if "encoder" in k][:30]
        raise KeyError(
            f"random_proj.attach_layer='{attach_layer}' not found in model.named_modules(). "
            f"Encoder layer candidates: {candidates}"
        )

    # 移除旧 hook（重复调用时防止 hook 累积）
    if hasattr(model, "_rand_proj_hook") and model._rand_proj_hook is not None:
        try:
            model._rand_proj_hook.remove()
        except Exception:
            pass
        model._rand_proj_hook = None

    def _hook(_m, _inp, _out):
        x = _out
        if isinstance(x, (tuple, list)):
            for t in x:
                if torch.is_tensor(t):
                    x = t
                    break
        if not torch.is_tensor(x) or x.dim() != 5:
            return
        # 存储投影结果，FeatureExtractor 通过 "rand_proj" 层名 hook 读取
        model._last_rand_proj = model.rand_proj(x)

    model._rand_proj_hook = name2mod[attach_layer].register_forward_hook(_hook)
    model._rand_proj_attached_to = attach_layer
    print(
        f"[INIT] rand_proj installed | attach={attach_layer} | "
        f"out_dim={out_dim} | seed={seed}"
    )
    return model


def reinit_decoder_modules(model: nn.Module, configs: dict) -> nn.Module:
    """
    Reinitialize decoder-related modules for ResEncL / PrimusM
    AFTER encoder checkpoint loading.

    Recommended call order:
        model = get_model(...)
        model = load_pretrained_model(..., model)   # encoder_only=True by default
        model = reinit_decoder_modules(model, configs)
    """
    c_model = _get_model_cfg(configs)

    dec_cfg = configs.get("decoder_init", None)
    if dec_cfg is None and isinstance(c_model, dict):
        dec_cfg = c_model.get("decoder_init", None)

    if not isinstance(dec_cfg, dict) or not dec_cfg.get("enabled", False):
        return model

    # Hybrid-decoder safety guard.
    # If load_pretrained_model has already loaded decoder-side weights
    # (e.g. openmind_hybrid reconstruction models), do NOT reinitialize the
    # decoder unless the config explicitly opts out. Otherwise the hybrid
    # pretrained decoder is overwritten by a random decoder and the experiment
    # degenerates back into the fully-random-decoder setting.
    skip_if_pretrained = bool(dec_cfg.get("skip_if_pretrained_decoder", True))
    pretrained_decoder_loaded = bool(getattr(model, "_pretrained_decoder_loaded", False))
    decoder_keys_loaded = getattr(model, "_decoder_keys_loaded", []) or []
    policy_trace = str(getattr(model, "_decoder_loading_policy_trace", ""))
    if skip_if_pretrained and pretrained_decoder_loaded:
        print(
            "[INIT] skip decoder reinit: pretrained decoder already loaded "
            f"({len(decoder_keys_loaded)} decoder keys, policy={policy_trace}). "
            "Set decoder_init.skip_if_pretrained_decoder=false to force reinit."
        )
        return model

    model_name = c_model.get("name", c_model.get("model_name", ""))
    model_name = str(model_name)

    init_type = dec_cfg.get("type", "xavier")
    seed = int(dec_cfg.get("seed", 0))
    neg_slope = float(dec_cfg.get("neg_slope", 1e-2))
    gain = float(dec_cfg.get("gain", 1.0))
    std = float(dec_cfg.get("std", 0.02))

    include_seg_head = bool(dec_cfg.get("include_seg_head", False))
    include_internal_decoder = bool(dec_cfg.get("include_internal_decoder", False))

    prefixes = []
    if model_name in ["ResEncL", "ResEnc-L", "ResEnc_L"]:
        prefixes = ["decoder.stages."]
        if include_seg_head:
            prefixes.extend(["decoder.seg_layers.", "seg_layers."])

    elif model_name in ["PrimusM", "Primus-M", "Primus_M", "primus_m"]:
        prefixes = ["up_projection.decode."]
        if include_internal_decoder:
            prefixes.append("decoder.")
        if include_seg_head:
            prefixes.extend(["seg_layers.", "decoder.seg_layers.", "up_projection.seg_layers."])

    else:
        print(f"[INIT] Skip decoder reinit: unsupported model_name={model_name}")
        return model

    touched = []
    # 关键：decoder 初始化 seed 与全局 seed 隔离
    with _temporary_seed(seed):
        for name, module in model.named_modules():
            if not name:
                continue
            if any(name.startswith(pfx) for pfx in prefixes):
                _init_one_module(
                    module,
                    init_type=init_type,
                    neg_slope=neg_slope,
                    gain=gain,
                    std=std,
                )
                touched.append(name)

    print(f"[INIT] decoder reinitialized | model={model_name} | type={init_type} | isolated_seed={seed}")
    print(f"[INIT] prefixes={prefixes}")
    print(f"[INIT] touched modules={len(touched)}")
    for n in touched[:20]:
        print(f"[INIT]   - {n}")
    if len(touched) > 20:
        print(f"[INIT]   ... and {len(touched) - 20} more")

    return model


def get_model(args, configs):
    if "model" in configs and isinstance(configs["model"], dict):
        c_model = configs["model"]
    else:
        c_model = configs

    model_name = c_model.get("name", c_model.get("model_name"))
    if model_name is None:
        raise KeyError("无法在配置文件中找到 'name' 或 'model_name' 字段")

    in_ch = c_model.get("num_input_channels", configs.get("num_input_channels", 1))

    num_classes = configs.get("num_classes", 2)
    out_ch = num_classes + 1

    print(f"Loading Model: {model_name} | Input Channels: {in_ch} | Output Channels: {out_ch}")

    if model_name == "UNETR":
        model = UNETR(
            in_channels=in_ch,
            out_channels=out_ch,
            img_size=(configs["roi_z"], configs["roi_y"], configs["roi_x"]),
            feature_size=48,
            norm_name="instance",
        )

    elif model_name == "S4D2W64":
        net_numpool = c_model.get("num_pool", configs.get("num_pool"))

        def get_cfg(key):
            return c_model.get(key, configs.get(key))

        model = Generic_UNet(
            in_ch,
            get_cfg("base_num_features"),
            out_ch,
            get_cfg("num_pool"),
            get_cfg("conv_per_stage"),
            2,
            nn.Conv3d,
            nn.InstanceNorm3d,
            {"eps": 1e-5, "affine": True},
            nn.Dropout3d,
            {"p": 0, "inplace": True},
            nn.LeakyReLU,
            {"negative_slope": 1e-2, "inplace": True},
            get_cfg("deep_supervision"),
            get_cfg("dropout_in_localization"),
            lambda x: x,
            InitWeights_Xavier(1e-2),
            get_cfg("pool_op_kernel_sizes")[:net_numpool],
            get_cfg("conv_kernel_sizes")[: net_numpool + 1],
            False,
            True,
            True,
            get_cfg("max_num_features"),
        )

    elif model_name == "SWUNETR":
        model = SwinUNETR(
            in_channels=in_ch,
            out_channels=out_ch,
            img_size=(configs["roi_z"], configs["roi_y"], configs["roi_x"]),
            feature_size=24,
            norm_name="instance",
        )

    elif model_name == "VNET":
        model = VNet(spatial_dims=3, in_channels=in_ch, out_channels=out_ch)

    elif model_name in ["ResEncL", "ResEnc-L", "ResEnc_L"]:
        nnssl_src = os.environ.get("NNSSL_SRC", "")
        if not nnssl_src:
            try:
                if "/root/autodl-tmp/nnssl-openneuro/src" not in sys.path:
                    sys.path.insert(0, "/root/autodl-tmp/nnssl-openneuro/src")
            except Exception:
                pass

        if not nnssl_src and "/root/autodl-tmp/nnssl-openneuro/src" not in sys.path:
            sys.path.insert(0, "/root/autodl-tmp/nnssl-openneuro/src")

        try:
            from nnssl.architectures.architecture_registry import get_res_enc_l
        except ImportError:
            raise RuntimeError("Could not import nnssl. Please check NNSSL_SRC environment variable.")

        model = get_res_enc_l(num_input_channels=in_ch, num_output_channels=out_ch, deep_supervision=True)

    elif model_name in ["PrimusM", "Primus-M", "Primus_M", "primus_m"]:
        nnssl_src = os.environ.get("NNSSL_SRC", "")
        if nnssl_src and nnssl_src not in sys.path:
            sys.path.insert(0, nnssl_src)
        if "/root/autodl-tmp/nnssl-openneuro/src" not in sys.path:
            sys.path.insert(0, "/root/autodl-tmp/nnssl-openneuro/src")

        input_shape = (configs.get("roi_z", 160), configs.get("roi_y", 160), configs.get("roi_x", 160))
        patch_embed_size = tuple(c_model.get("patch_embed_size", configs.get("patch_embed_size", [8, 8, 8])))

        try:
            from dynamic_network_architectures.architectures.primus import PrimusM as PrimusMClass
        except ImportError:
            raise RuntimeError(
                "Could not import PrimusM from dynamic_network_architectures. "
                "Install: pip install git+https://github.com/TaWald/dynamic-network-architectures@main"
            )

        model = PrimusMClass(
            input_channels=in_ch,
            output_channels=out_ch,
            input_shape=input_shape,
            patch_embed_size=patch_embed_size,
        )
        print(f"  PrimusM created with input_shape={input_shape}, patch_embed_size={patch_embed_size}")

        layer_names = [n for n, _ in model.named_modules() if n and n.count(".") <= 1]
        print(f"  Top-level modules: {layer_names[:20]}")

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = _install_random_cls_head(model, configs)
    model = _install_random_proj_head(model, configs)
    return model
