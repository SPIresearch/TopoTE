import json
import torch
import numpy as np
import random

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
)
from monai.data import DataLoader, Dataset


def get_classification_loader(args, configs):

    data_list = json.load(open(args.data_list_file))
    val_list = data_list["val"]
    
    for item in val_list:
        if "label" not in item:
            raise ValueError(
                f"分类任务需要'label'字段，但在数据项中未找到: {item}\n"
                "正确格式: {{'data': '/path/to/image.nii.gz', 'label': 0}}"
            )
    
    val_transforms = Compose([
        LoadImaged(keys=["data"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["data"]),

        NormalizeIntensityd(keys=["data"], nonzero=True, channel_wise=True),
    ])
    
    val_ds = Dataset(data=val_list, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # 分类任务通常batch_size=1
        num_workers=4,
        shuffle=False,
        drop_last=False
    )
    
    print(f"✅ 加载分类数据集: {len(val_list)} 张图像")
    
    # 统计类别分布
    label_counts = {}
    for item in val_list:
        lb = int(item["label"])
        label_counts[lb] = label_counts.get(lb, 0) + 1
    
    print("📊 类别分布:")
    for lb, count in sorted(label_counts.items()):
        print(f"  类别 {lb}: {count} 张图像 ({count/len(val_list)*100:.1f}%)")
    
    return val_loader


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 复用原有的load_pretrained_model函数
from utils.utility import load_pretrained_model

__all__ = ['get_classification_loader', 'setup_seed', 'load_pretrained_model']
