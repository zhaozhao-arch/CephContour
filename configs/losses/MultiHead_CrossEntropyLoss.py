# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss


def multi_head_cross_entropy(pred,
                             label,
                             weight=None,
                             reduction='mean',
                             avg_factor=None,
                             class_weight=None):
    """Calculate the CrossEntropy loss for multi-head scenario.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 8, 3),
            where 8 is the number of heads and 3 is the number of classes per head.
        label (torch.Tensor): The gt label with shape (N, 8), each element is in [0, 2].
        weight (torch.Tensor, optional): Sample-wise loss weight with shape (N, 8).
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
        class_weight (torch.Tensor, optional): The weight for each class with shape (3,).

    Returns:
        torch.Tensor: The calculated loss
    """
    # Reshape for F.cross_entropy: (N*8, 3) and (N*8,)
    pred_reshaped = pred.view(-1, 3)
    label_reshaped = label.view(-1)

    # Calculate element-wise losses
    loss = F.cross_entropy(
        pred_reshaped,
        label_reshaped,
        weight=class_weight,
        reduction='none'
    )

    # Reshape back to (N, 8)
    loss = loss.view(pred.size(0), 8)

    # Apply sample-wise weights if provided
    if weight is not None:
        weight = weight.float()
        loss = loss * weight

    # Reduce loss
    loss = weight_reduce_loss(
        loss,
        reduction=reduction,
        avg_factor=avg_factor if avg_factor is not None else pred.numel() / 3
    )

    return loss


class MultiHeadCrossEntropyLoss(nn.Module):
    """Cross entropy loss for multi-head classification.

    Each sample has 8 labels corresponding to 8 heads, each head predicts 3 classes.

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with shape (3,).
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None):
        super(MultiHeadCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.cls_criterion = multi_head_cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): Prediction scores with shape (N, 8, 3).
            label (torch.Tensor): Ground truth labels with shape (N, 8).
            weight (torch.Tensor, optional): Sample-wise weight with shape (N, 8).
            avg_factor (int, optional): Average factor for loss.
            reduction_override (str, optional): Override reduction method.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # Check input shapes
        assert cls_score.dim() == 3 and cls_score.shape[1:] == (8, 3), \
            f"cls_score must be shape (N, 8, 3), got {cls_score.shape}"
        assert label.dim() == 2 and label.shape[1] == 8, \
            f"label must be shape (N, 8), got {label.shape}"

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls