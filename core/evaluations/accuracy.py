# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from torch.nn.functional import one_hot
import numpy as np
import torch
import torch.nn as nn


def accuracy_numpy(pred, target, topk=(1,), thrs=0.):
    # 保持原有实现不变
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1,), thrs=0.):
    # 保持原有实现不变
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch_multi_head(pred, target, topk=(1,), thrs=0.):
    """适配8个分支头的topk计算"""
    # pred形状: [batch_size, 8, 3]，target形状: [batch_size, 8]
    num_heads = pred.size(1)  # 8个分支
    all_head_acc = []

    for head_idx in range(num_heads):
        # 提取单个分支的预测和标签
        head_pred = pred[:, head_idx, :]  # [batch_size, 3]
        head_target = target[:, head_idx]  # [batch_size]

        # 对当前分支执行topk操作
        maxk = max(topk)
        pred_score, pred_label = head_pred.topk(maxk, dim=1)  # 按类别维度取topk
        pred_label = pred_label.t()  # 转置为 [maxk, batch_size]

        # 计算当前分支的准确率
        correct = pred_label.eq(head_target.view(1, -1).expand_as(pred_label))
        head_acc = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            head_acc.append(correct_k.mul_(100.0 / head_target.size(0)))

        all_head_acc.append(head_acc)

    return all_head_acc


def accuracy(pred, target, topk=1, thrs=0.):
    """计算单个分支的准确率（保持原有实现）"""
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    to_tensor = (lambda x: torch.from_numpy(x)
    if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)
    # res = accuracy_torch_multi_head(pred, target, topk, thrs)

    return res[0] if return_single else res


def multi_head_accuracy(preds, target, topk=1, thrs=0., num_heads=8):
    """
    计算多个分支头的准确率

    Args:
        preds (torch.Tensor | list[torch.Tensor]): 多个分支的预测结果
            形状应为 (num_heads, batch_size, num_classes) 或列表形式
        target (torch.Tensor | np.ndarray): 目标标签
        topk (int | tuple[int]): 与单头准确率参数相同
        thrs (Number | tuple[Number]): 与单头准确率参数相同
        num_heads (int): 分支头数量，默认为8

    Returns:
        list: 每个分支头的准确率结果列表
    """
    # 确保预测输入格式正确
    if isinstance(preds, torch.Tensor):
        # 检查维度是否符合 (num_heads, batch_size, num_classes)
        assert preds.dim() == 3 and preds.size(0) == num_heads, \
            f"preds should be 3D tensor with shape (num_heads, batch, classes), got {preds.shape}"
        preds = [preds[i] for i in range(num_heads)]
    else:
        assert isinstance(preds, list) and len(preds) == num_heads, \
            f"preds should be list of {num_heads} tensors, got {len(preds)}"

    # 统一目标标签格式
    to_tensor = (lambda x: torch.from_numpy(x)
    if isinstance(x, np.ndarray) else x)
    target = to_tensor(target)

    # 为每个分支计算准确率
    head_accuracies = []
    for head_pred in preds:
        head_acc = accuracy(head_pred, target, topk, thrs)
        head_accuracies.append(head_acc)

    return head_accuracies


class MultiHeadAccuracy(nn.Module):
    """计算多个分支头准确率的模块"""

    def __init__(self, topk=(1,), num_heads=8):
        """
        Args:
            topk (tuple): 准确率计算的topk参数
            num_heads (int): 分支头数量，默认为8
        """
        super().__init__()
        self.topk = topk
        self.num_heads = num_heads

    def forward(self, preds, target):
        """
        Args:
            preds (torch.Tensor | list[torch.Tensor]): 多个分支的预测结果
            target (torch.Tensor): 目标标签

        Returns:
            list: 每个分支头的准确率结果列表
        """
        return multi_head_accuracy(preds, target, self.topk, num_heads=self.num_heads)