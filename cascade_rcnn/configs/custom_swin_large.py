# -------------------------------------------------------------------------
# 1. Base Config 로드 (모델 파일은 제외하고, 데이터/스케줄/런타임만 로드)
# -------------------------------------------------------------------------
# 경로가 확실하지 않다면 터미널에서 `find`로 찾은 절대 경로를 넣으세요.
_base_ = [
    '/data/ephemeral/home/code/baseline/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/data/ephemeral/home/code/baseline/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/data/ephemeral/home/code/baseline/mmdetection/configs/_base_/default_runtime.py'
]

# -------------------------------------------------------------------------
# 2. 모델 정의 (Flattening: 상속 없이 직접 정의하여 에러 방지)
# -------------------------------------------------------------------------
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'

model = dict(
    type='CascadeRCNN',
    # [v3 필수] 데이터 전처리 설정 (Mean/Std 정규화가 여기서 수행됨)
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    # Backbone: Swin-Large
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        #with_cp=True,  # Checkpointing (VRAM 절약)
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    
    # Neck: FPN (채널 수 매칭)
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536], # Swin-L Output
        out_channels=256,
        num_outs=5),
    
    # Head: Cascade R-CNN
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes는 train.py에서 덮어씌워지지만, 기본값으로 설정해둠
                num_classes=80, 
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
        
    # RPN 설정 (Cascade R-CNN 필수)
    rpn_head=dict(
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
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        
    # Train/Test 설정
    train_cfg=dict(
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
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))


# -------------------------------------------------------------------------
# 3. Optimizer 설정 (수정됨)
# -------------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,  # [핵심] 상속받은 optim_wrapper 설정을 아예 무시하고 새로 씁니다.
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    # Swin Transformer용 파라미터 설정 (필수)
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    # Gradient Clipping 설정
    clip_grad=dict(max_norm=35, norm_type=2)
)
# -------------------------------------------------------------------------
# 4. 기타 설정 (train.py에서 대부분 덮어쓰므로 기본값만 유지)
# -------------------------------------------------------------------------
# MMDetection v3에서는 data = dict(...) 대신 dataloader를 각각 정의합니다.
# 하지만 train.py에서 cfg.train_dataloader = ... 로 덮어쓰고 있으므로
# 여기서는 에러가 나지 않을 정도의 기본값만 있으면 됩니다.
train_dataloader = dict(batch_size=1, num_workers=1) 
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = dict(batch_size=1, num_workers=1)

# FP16 설정 (v3에서는 이렇게 씁니다)
# GPU가 Ampere(3090, A100 등) 이상이면 자동으로 bfloat16 등을 쓸 수도 있지만
# 명시적으로 amp를 켜려면 아래 설정을 사용합니다.
# (필요없으면 주석 처리)
# optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')