import os
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from src.config import *
from src.utils import set_seed, get_base_config

# 이 클래스를 MMDetection의 Hook 레지스트리에 등록
@HOOKS.register_module()  
class ProgressiveMosaicHook(Hook):
    def __init__(self, mosaic_end_epoch=40, total_epochs=50):
        self.mosaic_end_epoch = mosaic_end_epoch        # Mosaic 종료 시점
        self.total_epochs = total_epochs                # 최대 훈련 에폭
        self.mosaic_disabled = False                    # Mosaic 비활성화되었는지 추적
    
    def before_train_epoch(self, runner):
        # 매 에폭 시작 전에 자동으로 호출되는 메서드
        current_epoch = runner.epoch

        # 지정된 에폭에 도달했고, 아직 Mosaic 을 비활성화 안했다면
        if current_epoch >= self.mosaic_end_epoch and not self.mosaic_disabled:
            print(f"\n{'='*60}")
            print(f"[Epoch {current_epoch}] Mosaic Augmentation 비활성화!")
            print(f"이제부터 원본 이미지로 fine-tuning합니다.")
            print(f"{'='*60}\n")
            self._switch_to_normal_pipeline(runner)
            self.mosaic_disabled = True
    
    def _switch_to_normal_pipeline(self, runner):
        # 변경 파이프라인 설정
        normal_pipeline = [
            # 1. 이미지 로드
            dict(type='LoadImageFromFile', backend_args=None),

            # 2. Annotation(bbox) 로드
            dict(type='LoadAnnotations', with_bbox=True),
            
            # 3. 이미지 리사이즈(keep_ratio : 종횡비 유지 여부 True 면 padding 추가)
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),

            # 4. 색상 변환 augmentation
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,                # 밝기 변화 범위 [-32, 32]
                contrast_range=(0.5, 1.5),          # 대비 변화 범위 [0.5 배, 1.5배]
                saturation_range=(0.5, 1.5),        # 채도 변화 범위
                hue_delta=18                        # 색상 (Hue) 변화 범위 [-18, 18]
            ),

            # 5. Albumentation 라이브러리 사용
            dict(
                type='Albu',                                                    # Albumentation wrapper
                transforms=[
                    dict(type='Blur', blur_limit=7, p=0.2),                     # blur_limit : 블러 커널 최대 크기
                    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.2),     # var_limit : 가우시안 노이즈 분산 범위
                    #dict(type='RandomBrightnessContrast', p=0.2),              # 밝기/대비 랜덤 변화
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',                                        # bbox 형식 (x_min, y_min, x_max, y_max)
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],       # 함께 변환할 레이블 필드
                    min_visibility=0.0,                                         # 최소 가시성 (0이면 조금이라도 보이면 유지)
                    filter_lost_elements=True                                   # 완전히 사라진 bbox 제거
                ),
                keymap={'img': 'image', 'gt_bboxes': 'bboxes'},                 # MMDet 과 Albu 간의 키 매핑
                skip_img_without_anno=True                                      # annotation 이 없는 이미지 스킵
            ),
            dict(type='RandomFlip', prob=0.5),                                          # 좌우 반전
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),    # 작은 박스 필터링
            dict(type='PackDetInputs')                                                  # 최종 패킹 -> 학습에 필요한 형태로 데이터 패킹
        ]
        
        # 기존 dataset의 pipeline만 변경
        from mmengine.dataset import Compose
        
        # 현재 사용 중인 데이터셋 로드
        dataset = runner.train_loop.dataloader.dataset
        
        # MultiImageMixDataset인 경우 내부 dataset에 접근
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        
        # Pipeline 교체
        # Compose : 여러 transform 을 순차적으로 실행하는 wrapper
        dataset.pipeline = Compose(normal_pipeline)
        
        print("[INFO] 파이프라인 변경 완료: Mosaic → 원본 이미지")

def main():
    set_seed(SEED)                                              # 재현성을 위한 랜덤 시드 고정
    register_all_modules(init_default_scope=True)               # MMDet 모듈 등록

    cfg = get_base_config(MODEL_CONFIG_PATH)                    # 모델 config 로드
    cfg.default_scope = "mmdet"                                 # 기본 스코프 설정
    cfg.work_dir = WORK_DIR                                     # 체크포인트, 로그 저장 디렉토리 설정
    cfg.randomness = dict(seed=SEED, deterministic=False)       # deterministic : False 면 약간의 속도 향상(완전 재현성 포기)

    metainfo = dict(classes=CLASSES)                            # 클래스 메타 정보 입력
    
    # AMP를 켜면 메모리 여유 공간을 확보할 수 있음
    cfg.train_dataloader.batch_size = 5
    cfg.train_dataloader.num_workers = 4
    
    # Train Pipeline (Mosaic)
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True)
    ]
    
    train_pipeline_stage2 = [
        dict(
            type='Mosaic',                      # Mosaic 증강
            img_scale=(1024, 1024),             # 최종 이미지 크기
            center_ratio_range=(0.5, 1.5),      # 중심점 위치 범위 (랜덤)
            bbox_clip_border=True,              # 이미지 경계 밖 bbox 잘라내기
            pad_val=114.0,                      # 패딩 픽셀 값(회색)
            prob=1.0                            # 모든 데이터에 적용
        ),
        dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
        dict(
            type='PhotoMetricDistortion',
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5), 
            hue_delta=18
        ),
        dict(
            type='Albu',
            transforms=[
                dict(type='Blur', blur_limit=7, p=0.2),
                dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.2),
                dict(type='RandomBrightnessContrast', p=0.2),
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
                min_visibility=0.0,
                filter_lost_elements=True
            ),
            keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
            skip_img_without_anno=True
        ),
        dict(type='RandomFlip', prob=0.5),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='PackDetInputs')
    ]
    
    cfg.train_dataloader.dataset = dict(
        type='MultiImageMixDataset',                                # 여러 이미지를 섞는 데이터셋(Mosaic 용)
        dataset=dict(
            type='CocoDataset',                                     # COCO 포맷 데이터셋
            data_root=DATA_ROOT,                                    # 데이터 루트 경로
            ann_file=TRAIN_ANN,                                     # annotation 파일 경로
            data_prefix=dict(img=''),                               # 이미지 경로 prefix
            filter_cfg=dict(filter_empty_gt=True, min_size=32),     # flitter_empty_gt : bbox 가 없는 이미지 제거, min_size : 최소 이미지 크기
            pipeline=train_pipeline,                                # 기본 파이프라인(이미지 로드)
            metainfo=metainfo,                                      # 클래스 정보
            backend_args=None
        ),
        pipeline=train_pipeline_stage2                              # Mosaic 이 포함된 파이프라인
    )
    
    # 검증 파이프라인 이미지 크기만 조절
    val_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
        )
    ]

    # 검증 데이터로더
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            ann_file=VAL_ANN,
            data_prefix=dict(img=''),
            test_mode=True,
            pipeline=val_pipeline,
            metainfo=metainfo,
            backend_args=None
        )
    )

    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            ann_file=TEST_ANN,
            data_prefix=dict(img=''),
            test_mode=True,
            pipeline=val_pipeline,
            metainfo=metainfo,
            backend_args=None
        )
    )

    cfg.val_evaluator = dict(
        type='CocoMetric',                              # COCO 평가 메트릭
        ann_file=os.path.join(DATA_ROOT, VAL_ANN),
        metric='bbox',                                  # bbox AP 계산
        classwise=True,                                 # 클래스별 AP 도 계산
        format_only=False,                              # 결과 파일만 생성할지 (False 평가도 수행)
        backend_args=None
    )

    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=os.path.join(DATA_ROOT, TEST_ANN),
        metric='bbox',
        format_only=True,
        outfile_prefix=os.path.join(WORK_DIR, 'test_result'),
        backend_args=None
    )

    cfg.train_cfg.max_epochs = 50                       # 최대 에폭 설정
    cfg.train_cfg.val_interval = 1                      # 검증 시행 에폭

    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1,                                     # 1 에폭마다 체크포인트 저장
        max_keep_ckpts=1,                               # 최대 1개만 유지
        save_best='coco/bbox_mAP',                      # mAP 기준 최고 모델 저장
        rule='greater',
        save_last=False
    )

    cfg.custom_hooks = [
        dict(
            type='ProgressiveMosaicHook',
            mosaic_end_epoch=15,                        # 특정 에폭부터 Mosaic 끄기
            total_epochs=50
        ),
        dict(
            type="EarlyStoppingHook",                   # 조기 종료
            monitor="coco/bbox_mAP",                    # 모니터링 메트릭
            patience=5,                                 # 특정 에폭 동안 개선 없으면 종료
            min_delta=0.001,                            # 최소 계선폭 0.001
            strict=True
        )
    ]

    cfg.default_hooks.logger.interval = 10              # 10 iter 마다 로그 출력

    # Cascade R-CNN 은 여러 개의 bbox head 를 가지므로 각 헤드의 클래스 수를 따로 설정해줘야 함
    if isinstance(cfg.model.roi_head.bbox_head, list):
        for head in cfg.model.roi_head.bbox_head:
            head.num_classes = len(CLASSES)
    else:
        cfg.model.roi_head.bbox_head.num_classes = len(CLASSES)

    # Mask RCNN 모델 사용시 MASK Head 제거(객체 검출)
    if "mask_head" in cfg.model.roi_head:
        cfg.model.roi_head.pop("mask_head", None)
        cfg.model.roi_head.pop('mask_roi_extractor', None)

    # ------------------------------------------------------------------
    # [수정 3] AMP (Mixed Precision) 활성화 -> 속도/메모리 최적화
    # ------------------------------------------------------------------
    cfg.optim_wrapper = dict(
        type='AmpOptimWrapper',                                         # Mixed Precision 학습
        loss_scale='dynamic',                                           # 동적 loss scaling
        optimizer=dict(
            type='AdamW',
            lr=0.00005,
            betas=(0.9, 0.999),
            weight_decay=0.05
        ),
        accumulative_counts=4,
        # Swin Transformer 전용 파라미터 설정
        # 위치 임베딩과 정규화 계층은 weight decay 안 함
        paramwise_cfg=dict(
            custom_keys={
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)
            }),
        clip_grad=dict(max_norm=35, norm_type=2)
    )

    cfg.param_scheduler = [
        # Warmup 설정
        dict(
            type='LinearLR',
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=100
        ),
        # 메인 스케줄러 (Cosine Annealing)
        dict(
            type='CosineAnnealingLR',
            T_max=30,          # 전체 Epoch 수와 맞춰야 함
            by_epoch=True,
            begin=20,
            end=50,
            eta_min=5e-6       # 최소 LR (너무 0으로 떨어지지 않게 방지)
        )
    ]

    # wandb 설정
    cfg.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
            init_kwargs=dict(
                project='cascade_mask_rcnn_project',
                name='swin_large_final',
                entity='cv_12'
            )
        )
    ]
    
    cfg.visualizer = dict(type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer')
    
    print(f"[INFO] Progressive Mosaic Training 시작")
    print(f"[INFO] Validation Scale: 1024x1024 (Matched with Train)")
    print(f"[INFO] AMP Enabled: True")
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    main()