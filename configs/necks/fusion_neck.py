# resnext/configs/neck/feature_fusion_neck.py
import torch
import torch.nn as nn
from ..common.base_module import BaseModule

class FeatureFusionNeck(BaseModule):
    """融合原图和轮廓图的特征 Neck"""
    def __init__(self, in_channels, fusion_method='concat', init_cfg=None):
        super(FeatureFusionNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.out_channels = in_channels * 2  # 拼接后通道翻倍
        elif fusion_method == 'add':
            self.out_channels = in_channels  # 相加后通道不变
        else:
            raise ValueError(f"不支持的融合方式: {fusion_method}")

    def forward(self, x):
        # x 是一个元组 (img_feats, contour_feats)
        img_feats, contour_feats = x
        if self.fusion_method == 'concat':
            fused_feats = torch.cat([img_feats, contour_feats], dim=1)
        elif self.fusion_method == 'add':
            fused_feats = img_feats + contour_feats
        return fused_feats