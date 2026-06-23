import torch
import torch.nn as nn
import torch.nn.functional as F
from .cls_head import ClsHead
from ..common.base_module import BaseModule


class EightHeadLinearClsHead(ClsHead):
    """八个全连接层分类头，每个输出3维向量并通过SoftMax获取概率

    Args:
        in_channels (int): 输入特征通道数
        init_cfg (dict | optional): 初始化配置，默认使用正态分布初始化
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(EightHeadLinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_heads = 8  # 八个完全连接层
        self.num_classes_per_head = 3
        # self.num_classes_per_head = 3  # 每个层输出3维向量

        # 初始化8个全连接层
        self.heads = nn.ModuleList()
        for _ in range(self.num_heads):
            self.heads.append(nn.Linear(in_channels, self.num_classes_per_head))

    def pre_logits(self, x):
        """处理输入特征，取最后一个阶段的特征"""
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=False):
        """推理过程：计算每个头的概率并取最高概率类别"""
        x = self.pre_logits(x)

        # 存储每个头的预测结果
        all_preds = []
        for head in self.heads:
            # print("Input shape to head:", x.shape)  # 应输出 (batch_size, 2048)
            # print("Head weight shape:", head.weight.shape)  # 应输出 (3, 2048)
            cls_score = head(x)
            if softmax:
                # 对每个头的输出应用SoftMax获取概率
                pred_prob = F.softmax(cls_score, dim=1)
                # 获取最高概率的类别索引
                # pred_label = torch.argmax(pred_prob, dim=1, keepdim=True)
                # all_preds.append(pred_prob)
                all_preds.append(pred_prob.unsqueeze(1))

                # 堆叠所有头的预测结果 (num_samples, 8)
        final_pred = torch.cat(all_preds, dim=1)

        if post_process:
            return self.post_process(final_pred)
        else:
            return final_pred

    def forward_train(self, x, gt_label, **kwargs):
        """计算8个头部的损失并融合"""
        x = self.pre_logits(x)  # 处理输入特征
        total_loss = 0.0

        # 遍历8个头部，每个头部计算独立损失
        for i, head in enumerate(self.heads):
            cls_score = head(x)  # 第i个头部的输出 (batch_size, 3)
            # 取出第i个标签（每个样本的第i个标签对应第i个头部）
            head_label = gt_label[:, i]  # (batch_size,)
            # 计算当前头部的交叉熵损失
            loss = F.cross_entropy(cls_score, head_label)
            total_loss += loss

        # 总损失为8个头部损失的平均值
        total_loss /= self.num_heads  # self.num_heads=8
        return dict(loss=total_loss)