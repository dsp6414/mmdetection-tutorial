model = dict( #model部分
    type='MaskRCNN',  #model类型
    backbone=dict(   #主干特征提取网络的配置
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), #权值初始化
    neck=dict(   #neck网络的配置
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict( #region proposal net部分的配置
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1, #类数目要和样本匹配
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1, #类数目要和样本匹配
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict( #模型train阶段的配置
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict( #model推断或测试阶段的配置信息
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000, 
            nms=dict(type='nms', iou_threshold=0.7), #可以自己修改阈值参数
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))  #model配置部分 end
dataset_type = 'CocoDataset' #设置数据集类型 
data_root = '/home/chen/OD/mmdet_tutorial/datasets/balloon/' #设置数据集根目录
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #归一化配置
train_pipeline = [ #train阶段dataloader transform
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [ #test阶段dataloader transform
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict( #数据集配置
    samples_per_gpu=4, #批的大小
    workers_per_gpu=4, #每gpu多少个worker
    train=dict( #训练集train的配置
        type='CocoDataset', #数据集类型
        ann_file= 
        '/home/chen/OD/mmdet_tutorial/datasets/balloon/annotations/custom_train.json', #标签文件
        img_prefix='/home/chen/OD/mmdet_tutorial/datasets/balloon/train/', #图像目录
        pipeline=train_pipeline, #train阶段dataloader transform
        classes=('balloon', )), #类名称
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/chen/OD/mmdet_tutorial/datasets/balloon/annotations/custom_train.json',
        img_prefix='/home/chen/OD/mmdet_tutorial/datasets/balloon/train/',
        pipeline=test_pipeline,
        classes=('balloon', )),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/chen/OD/mmdet_tutorial/datasets/balloon/annotations/custom_train.json',
        img_prefix='/home/chen/OD/mmdet_tutorial/datasets/balloon/train/',
        pipeline=test_pipeline,
        classes=('balloon', )))
evaluation = dict(metric=['bbox', 'segm'], interval=2) #多少个批次或epoch输出评估结果
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) #优化器配置
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]) #学习率调度策略，hook
runner = dict(type='EpochBasedRunner', max_epochs=100) #设置runner类型，训练多少epoch
checkpoint_config = dict(interval=10) #checkpoint时间间隔
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth' #加载预训练权值
resume_from = None #恢复训练时的预训练权值
workflow = [('train', 1)] #只做train
classes = ('balloon', ) #
work_dir = './tutorial_exps' #工作目录
seed = 0
gpu_ids = range(0, 1) #gpu
