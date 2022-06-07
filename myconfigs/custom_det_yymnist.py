dataset_type = 'yyminstDataset'
data_root = '/home/chen/OD/mmdet_tutorial/datasets/yymnist'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = 640
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(640, 640),
        multiscale_mode='range',
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640), allow_negative_crop=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='yyminstDataset',
        ann_file='/home/chen/OD/mmdet_tutorial/datasets/yymnist/train.txt',
        img_prefix='/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=(640, 640),
                multiscale_mode='range',
                ratio_range=(0.1, 2.0),
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(640, 640),
                allow_negative_crop=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(640, 640)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        data_root='/home/chen/OD/mmdet_tutorial/datasets/yymnist',
        classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')),
    val=dict(
        type='yyminstDataset',
        ann_file='/home/chen/OD/mmdet_tutorial/datasets/yymnist/val.txt',
        img_prefix='/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640)),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/home/chen/OD/mmdet_tutorial/datasets/yymnist',
        classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')),
    test=dict(
        type='yyminstDataset',
        ann_file='/home/chen/OD/mmdet_tutorial/datasets/yymnist/test.txt',
        img_prefix='/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640)),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/home/chen/OD/mmdet_tutorial/datasets/yymnist',
        classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')))
evaluation = dict(interval=5, metric='mAP')
custom_imports = dict(
    imports=[
        'custom_detector.models.backbones.custom_backbone',
        'custom_detector.models.necks.bifpn',
        'custom_detector.models.heads.custom_head',
        'custom_detector.models.detectors.custom_det'
    ],
    allow_failed_imports=False)
pretrained = None
norm_cfg = dict(type='BN', momentum=0.01, eps=0.001)
model = dict(
    type='CUSTOM_DET',
    pretrained=None,
    scale=1,
    backbone=dict(
        type='CUSTOMBackBone',
        in_channels=3,
        n_classes=1000,
        se_rate=0.25,
        frozen_stages=-1,
        norm_cfg=dict(type='BN', momentum=0.01, eps=0.001)),
    neck=dict(
        type='BiFPN',
        norm_cfg=dict(type='BN', momentum=0.01, eps=0.001),
        upsample_cfg=dict(mode='nearest')),
    bbox_head=dict(
        type='CustomHead',
        num_classes=10,
        num_ins=5,
        norm_cfg=dict(type='BN', momentum=0.01, eps=0.001),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=5,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    min_lr=1e-05)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=5, max_keep_ckpts=2)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'tutorial_exps/custom_det/latest.pth'
workflow = [('train', 1)]
dist_params = dict(backend='nccl')
custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, interval=1, warm_up=4000)
]
work_dir = './tutorial_exps/custom_det'
seed = 0
gpu_ids = range(0, 1)
