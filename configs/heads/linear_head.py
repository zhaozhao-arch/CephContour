import torch
import torch.nn as nn
import torch.nn.functional as F
from .cls_head import ClsHead


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha).float()
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, pred, target):
        if self.alpha is not None:
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)


        log_pt = F.log_softmax(pred, dim=1)
        pt = torch.exp(log_pt)

        target = target.long()
        log_pt = log_pt.gather(1, target.view(-1, 1))
        pt = pt.gather(1, target.view(-1, 1))


        if self.alpha is not None:
            at = self.alpha.gather(0, target.view(-1))
            log_pt = log_pt * at.view(-1, 1)

        loss = -1 * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean() * self.loss_weight
        elif self.reduction == 'sum':
            return loss.sum() * self.loss_weight
        else:
            return loss * self.loss_weight



class LinearClsHead(ClsHead):
    """Linear classifier head with Focal Loss support."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1,),
                 dropout=0.0,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):

        dummy_loss = dict(type='nn.CrossEntropyLoss', reduction='mean')

        super(LinearClsHead, self).__init__(loss=dummy_loss, init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.topk = topk

        # === 处理 Dropout ===
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if self.num_classes <= 0:
            raise ValueError(f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

        loss_type = loss.get('type')
        loss_weight = loss.get('loss_weight', 1.0)

        if loss_type == 'CrossEntropyLoss':
            class_weight = loss.get('class_weight', None)
            if class_weight is not None:
                class_weight = torch.tensor(class_weight).float()

            self.compute_loss = nn.CrossEntropyLoss(
                weight=class_weight,
                reduction='mean'
            )
            self.loss_weight = loss_weight

        elif loss_type == 'FocalLoss':
            self.compute_loss = FocalLoss(
                gamma=loss.get('gamma', 2.0),
                alpha=loss.get('alpha', None),
                reduction='mean',
                loss_weight=loss_weight
            )
            self.loss_weight = 1.0
        else:
            try:
                if not loss_type.startswith('nn.'):
                    loss_type = 'nn.' + loss_type
                self.compute_loss = eval(loss_type)(reduction='mean')
                self.loss_weight = loss_weight
            except:
                raise TypeError(f'Unsupported loss type: {loss_type}')

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, **kwargs):
        x = self.pre_logits(x)
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc(x)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if isinstance(cls_score, tuple):
            cls_score = cls_score[0]

        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc(x)

        if gt_label.dtype != torch.long:
            gt_label = gt_label.long()

        loss_val = self.compute_loss(cls_score, gt_label)

        if isinstance(self.compute_loss, nn.CrossEntropyLoss):
            loss_val = loss_val * self.loss_weight

        losses = dict(loss=loss_val)
        return losses