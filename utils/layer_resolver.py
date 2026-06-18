from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch.nn as nn


def _list_candidates(named_keys: List[str], keyword: str, limit: int = 120) -> str:
    cand = [k for k in named_keys if keyword in k]
    if not cand:
        return "(none)"
    return "\n".join(cand[:limit])


def _first_existing(named: Dict[str, nn.Module], candidates: List[str]) -> str:
    for c in candidates:
        if c in named:
            return c
    raise KeyError(f"None of the candidate layer names exist: {candidates[:8]} ...")


def resolve_layers_and_sample_num(configs: dict, model: nn.Module) -> Tuple[List[str], Dict[str, int]]:

    layer_mode = str(configs.get("layer_mode", "")).strip().lower()

    if not layer_mode:
        layers = list(configs.get("layers", []))
        sample_num = {k: int(v) for k, v in dict(configs.get("sample_num", {})).items()}
        return layers, sample_num

    named = dict(model.named_modules())
    keys = list(named.keys())

    if layer_mode == "decoder_sym_from_encoder":
        encoder_layers = list(configs.get("encoder_layers", configs.get("layers", [])))
        if not encoder_layers:
            raise ValueError("layer_mode=decoder_sym_from_encoder 需要提供 encoder_layers 或 layers")

        # 找 encoder stage 最大索引
        stage_ids = []
        for l in encoder_layers:
            m = re.search(r"encoder\.stages\.(\d+)", str(l))
            if m:
                stage_ids.append(int(m.group(1)))
        if not stage_ids:
            raise ValueError(
                "encoder_layers 里没有发现 'encoder.stages.<k>'，无法做对称映射。"
            )
        max_stage = max(stage_ids)

        enc2dec: Dict[str, str] = {}

        # stem 可选
        include_stem = bool(configs.get("decoder_include_stem", False))
        for l in encoder_layers:
            l = str(l)
            if l == "encoder.stem":
                if not include_stem:
                    continue
                # 尝试常见 stem 命名
                stem_candidates = [
                    "decoder.stem",
                    "decoder.up_stem",
                    "decoder.upstem",
                    "decoder.initial",
                ]
                # 也有人把最高分辨率 up-block 当成 stem
                stem_candidates += [f"decoder.stages.{max_stage}", f"decoder.blocks.{max_stage}"]
                try:
                    enc2dec[l] = _first_existing(named, stem_candidates)
                except KeyError:
                    # stem 不存在就跳过，不影响 stages
                    continue
                continue

            m = re.search(r"encoder\.stages\.(\d+)", l)
            if m:
                e = int(m.group(1))
                d = max_stage - e
                candidates = [
                    f"decoder.stages.{d}",
                    f"decoder.stage.{d}",
                    f"decoder.blocks.{d}",
                    f"decoder.block.{d}",
                    f"decoder.up_blocks.{d}",
                    f"decoder.up.{d}",
                    f"decoder.up_stages.{d}",
                ]
                try:
                    enc2dec[l] = _first_existing(named, candidates)
                except KeyError as ex:
                    raise KeyError(
                        f"无法把 {l} 对称映射到 decoder 层（尝试 idx={d} 失败）。\n"
                        f"你当前模型里 decoder 相关模块示例：\n{_list_candidates(keys, 'decoder')}\n"
                        f"若命名不叫 decoder.*，也可把 layer_mode 关掉，直接在 cfg 里写 layers。"
                    ) from ex

        # 生成最终 layers
        layers: List[str] = []
        for l in encoder_layers:
            if l in enc2dec:
                layers.append(enc2dec[l])

        # sample_num：用 encoder 的采样预算映射过去；缺省则给一个安全值
        enc_sample_num = dict(configs.get("sample_num", {}))
        sample_num: Dict[str, int] = {}
        for enc_l, dec_l in enc2dec.items():
            if enc_l in enc_sample_num:
                sample_num[dec_l] = int(enc_sample_num[enc_l])
            else:
                sample_num[dec_l] = 100

        # 校验 layers 全部存在
        missing = [l for l in layers if l not in named]
        if missing:
            raise KeyError(
                f"Resolved decoder layers contain missing entries: {missing}.\n"
                f"decoder candidates:\n{_list_candidates(keys, 'decoder')}"
            )

        return layers, sample_num

    raise ValueError(f"Unknown layer_mode: {layer_mode}")
