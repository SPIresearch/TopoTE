from __future__ import annotations

import os
import re
from typing import Optional


def infer_openmind_method_key(ckpt_path_or_name: str) -> Optional[str]:

    if not ckpt_path_or_name:
        return None

    name = os.path.basename(ckpt_path_or_name).lower()
    stem = os.path.splitext(name)[0]
    s = re.sub(r"[^a-z0-9]+", "_", stem)

    # nnU-Net
    if re.search(r"nnunet.*def.*1k|nnunet_def_1k|nnunetdef_1k", s):
        return "nnunet_def_1k"
    if re.search(r"nnunet.*def|nnunet_def|nnunetdef", s):
        return "nnunet_def"

    # scratch
    if re.search(r"scratch.*1k|scratch_1k|from_scratch_1k|fromscratch_1k", s):
        return "scratch_1k"
    if re.search(r"(^|_)scratch($|_)|from_scratch|fromscratch", s):
        return "scratch"

    # self-supervised
    if "simmim" in s:
        return "simmim"
    if "swinunetr" in s or "swin_unetr" in s:
        return "swinunetr"
    if "simclr" in s:
        return "simclr"
    if "voco" in s:
        return "voco"

    # VF / MG
    if re.search(r"(^|_)vf($|_)", s):
        return "vf"
    if re.search(r"(^|_)mg($|_)", s):
        return "mg"

    # MAE / S3D
    if re.search(r"(^|_)mae($|_)|mae_", s):
        return "mae"
    if "s3d" in s:
        return "s3d"

    return None
