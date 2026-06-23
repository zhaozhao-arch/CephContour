model_cfg = dict(
    backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        groups=32,
        width_per_group=4,
        style='pytorch'),
    contour_backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        groups=32,
        width_per_group=4,
        style='pytorch'
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=[
        dict(
            type='LinearClsHead',
            num_classes=3,
            in_channels=2048,
            loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0),
            topk=(1, )
        ) for _ in range(6)
    ]
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadDoubleImageFromFile'),
    dict(type='Resize', size=(512, 512), backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=[0.6, 1.4], contrast=[0.7, 1.3]),
    dict(type='RandomRotation', degrees=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadDoubleImageFromFile'),
    dict(type='Resize', size=(512, 512), backend='pillow'),
    dict(type='CenterCrop', crop_size=512),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_cfg = dict(
    batch_size = 32,
    num_workers = 4,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = '',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 150,
    ),
    test=dict(
        ckpt = '',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,),
            thrs = None,
            average_mode='none'
        )
    )
)

optimizer_cfg = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    grad_clip=dict(max_norm=5.0, norm_type=2)
)

lr_config = dict(
    type='StepLrUpdater',
    step=[50, 100],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)
model_cfg = dict(
    backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        groups=32,
        width_per_group=4,
        style='pytorch'),
    contour_backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        groups=32,
        width_per_group=4,
        style='pytorch'
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=[
        dict(
            type='LinearClsHead',
            num_classes=3,
            in_channels=2048,
            loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0),
            topk=(1, )
        ) for _ in range(6)
    ]
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadDoubleImageFromFile'),
    dict(type='Resize', size=(224, 224), backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=[0.6, 1.4], contrast=[0.7, 1.3]),
    dict(type='RandomRotation', degrees=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadDoubleImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_cfg = dict(
    batch_size = 32,
    num_workers = 4,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = '',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 150,
    ),
    test=dict(
        ckpt = '',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,),
            thrs = None,
            average_mode='none'
        )
    )
)

optimizer_cfg = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    grad_clip=dict(max_norm=5.0, norm_type=2)
)

lr_config = dict(
    type='StepLrUpdater',
    step=[50, 100],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)