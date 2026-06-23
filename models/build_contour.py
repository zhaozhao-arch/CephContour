from configs.backbones import *
from configs.necks import *
from configs.heads import *
from configs.common import BaseModule, Sequential
import torch.nn as nn
import torch
import torch.nn.functional as F


def build_model(cfg):
    if isinstance(cfg, list):
        modules = [eval(cfg_.pop("type"))(**cfg_) for cfg_ in cfg]
        return Sequential(*modules)
    else:
        return eval(cfg.pop("type"))(**cfg)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.fc_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.fc_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.fc_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv):
        B, N, C = x_q.shape

        q = self.norm_q(x_q)
        k = self.norm_k(x_kv)
        v = self.norm_k(x_kv)

        q = self.fc_q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.fc_k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.fc_v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x_q + x


class HierarchicalGeometricInjectionModule(nn.Module):
    def __init__(self, in_channels_list, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        self.img_projs = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list])
        self.geo_projs = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list])
        self.cross_attns = nn.ModuleList([CrossAttention(dim=embed_dim, num_heads=8) for _ in in_channels_list])

        self.align_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2 ** (len(in_channels_list) - 1 - i),
                      padding=1) if i < len(in_channels_list) - 1
            else nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            for i in range(len(in_channels_list))
        ])

        self.concat_proj = nn.Conv2d(embed_dim * len(in_channels_list) + in_channels_list[-1], in_channels_list[-1],
                                     kernel_size=1)

    def forward(self, img_feats, geo_feats):
        rectified_feats = []

        for i in range(len(img_feats)):
            B, C, H, W = img_feats[i].shape

            feat_img = self.img_projs[i](img_feats[i]).flatten(2).transpose(1, 2)
            feat_geo = self.geo_projs[i](geo_feats[i])

            if feat_geo.shape[2:] != (H, W):
                feat_geo = F.interpolate(feat_geo, size=(H, W), mode='bilinear', align_corners=False)

            feat_geo = feat_geo.flatten(2).transpose(1, 2)

            feat_rect = self.cross_attns[i](x_q=feat_img, x_kv=feat_geo)
            feat_rect = feat_rect.transpose(1, 2).reshape(B, self.embed_dim, H, W)

            feat_aligned = self.align_convs[i](feat_rect)

            if feat_aligned.shape[2:] != img_feats[-1].shape[2:]:
                feat_aligned = F.adaptive_avg_pool2d(feat_aligned, img_feats[-1].shape[2:])

            rectified_feats.append(feat_aligned)

        fused_rect = torch.cat(rectified_feats, dim=1)
        final_feat = torch.cat([fused_rect, img_feats[-1]], dim=1)

        return self.concat_proj(final_feat)


class BuildNet(BaseModule):
    def __init__(self, cfg):
        super(BuildNet, self).__init__()
        self.neck_cfg = cfg.get("neck")
        self.head_cfg = cfg.get("head")

        self.backbone = build_model(cfg.get("backbone"))
        self.contour_backbone = build_model(cfg.get("contour_backbone"))

        self.hgim = HierarchicalGeometricInjectionModule(
            in_channels_list=[512, 1024, 2048],
            embed_dim=256
        )

        if self.neck_cfg is not None:
            self.neck = build_model(cfg.get("neck"))

        if isinstance(self.head_cfg, list):
            self.heads = nn.ModuleList([build_model(h) for h in self.head_cfg])
            self.head = None
        elif self.head_cfg is not None:
            self.head = build_model(self.head_cfg)
            self.heads = None
        else:
            self.head = None
            self.heads = None

    def freeze_layers(self, names):
        assert isinstance(names, tuple)
        for name in names:
            layers = getattr(self, name)
            for param in layers.parameters():
                param.requires_grad = False

    def extract_feat(self, img, contour):
        img_feats = self.backbone(img)
        contour_feats = self.contour_backbone(contour)

        fused_feat = self.hgim(img_feats, contour_feats)
        fused_feat = (fused_feat,)

        if hasattr(self, 'neck') and self.neck is not None:
            fused_feat = self.neck(fused_feat)

        return tuple(fused_feat)

    def forward(self, img, contour, return_loss=True, train_statu=False, **kwargs):
        x = self.extract_feat(img, contour)
        if not train_statu:
            if return_loss:
                return self.forward_train(x, **kwargs)
            else:
                return self.forward_test(x, **kwargs)
        else:
            return self.forward_test(x), self.forward_train(x, **kwargs)

    def forward_train(self, x, targets, **kwargs):
        losses = dict()

        if self.heads is not None:
            for i, head in enumerate(self.heads):
                label_i = targets[:, i]
                head_loss = head.forward_train(x, label_i, **kwargs)
                for k, v in head_loss.items():
                    losses[f"branch{i}_{k}"] = v
        else:
            if self.head is not None:
                loss = self.head.forward_train(x, targets, **kwargs)
                losses.update(loss)

        return losses

    def forward_test(self, x, **kwargs):
        if self.heads is not None:
            outs = []
            for head in self.heads:
                out = head.simple_test(x, **kwargs)
                outs.append(out)
            return torch.stack(outs, dim=1)
        else:
            if self.head is not None:
                return self.head.simple_test(x, **kwargs)
            else:
                return x