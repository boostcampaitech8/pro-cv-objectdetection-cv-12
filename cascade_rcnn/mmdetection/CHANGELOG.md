# Changelog

## 주요 변경사항 (old_baseline → baseline)

### 1. faster_rcnn_train.ipynb 변경사항

#### Import 변경
```python
# 이전 (MMDetection 2.x)
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device

# 현재 (MMDetection 3.x)
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
import pandas as pd
from collections import Counter
```

#### Config 설정 방식 변경
```python
# 이전: 직접 속성 할당
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train.json'
cfg.data.train.pipeline[2]['img_scale'] = (512,512)

# 현재: metainfo와 data_prefix 사용
register_all_modules(init_default_scope=True)
cfg.default_scope = "mmdet"

for ds_key in ["train_dataloader", "test_dataloader"]:
    ds = cfg[ds_key]["dataset"] if "dataset" in cfg[ds_key] else cfg[ds_key]
    ds.metainfo = dict(classes=classes)
    ds.data_root = root
    ds.ann_file = train_ann if ds_key == "train_dataloader" else test_ann
    ds.data_prefix = dict(img="")
```

#### Training 방식 변경
```python
# 이전: 함수 기반 훈련
datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model)
model.init_weights()
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

# 현재: Runner 기반 훈련
runner = Runner.from_cfg(cfg)
runner.train()
```

#### Validation 설정 변경
```python
# 이전: validate=False 파라미터로 제어
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

# 현재: 설정에서 검증 관련 키 제거
for k in ("val_dataloader", "val_evaluator", "val_cfg", "val_loop"):
    cfg.pop(k, None)
cfg.train_cfg = cfg.get("train_cfg", {})
cfg.train_cfg["val_interval"] = 0
```

#### 새로운 기능 추가
```python
# 데이터셋 분석 기능 추가
def summarize_dataset(ds):
    ds.full_init()
    num_images = len(ds)
    classes = list(ds.metainfo.get("classes", []))
    
    counts = Counter()
    for i in range(num_images):
        info = ds.get_data_info(i)
        for inst in info.get("instances", []):
            lbl = inst.get("bbox_label", None)
            if lbl is not None:
                counts[lbl] += 1
    
    df = pd.DataFrame({
        "category": [f"{i} [{c}]" for i, c in enumerate(classes)],
        "count": [counts.get(i, 0) for i in range(len(classes))]
    })
    display(df)
```

### 2. faster_rcnn_inference.ipynb 변경사항

#### Import 변경
```python
# 이전 (MMDetection 2.x)
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

# 현재 (MMDetection 3.x)
from mmengine.config import Config
from mmdet.apis import DetInferencer
```

#### Inference 방식 재설계
```python
# 이전: 수동 추론 파이프라인
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])
output = single_gpu_test(model, data_loader, show_score_thr=0.05)

# 현재: DetInferencer 통합 API
inferencer = DetInferencer(
    model=cfg,
    weights=checkpoint_path,
    device=device
)
results = inferencer(
    img_paths,
    batch_size=batch_size,
    return_datasamples=True,
    no_save_vis=True,
)
```

#### 결과 처리 방식 변경
```python
# 이전: 복잡한 출력 구조 처리
for i, out in enumerate(output):
    prediction_string = ''
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(o[2]) + ' ' + str(o[3]) + ' '

# 현재: DataSample 기반 처리
for data_sample in results["predictions"]:
    inst = data_sample.pred_instances
    bboxes = inst.bboxes.cpu().numpy() if hasattr(inst.bboxes, "cpu") else np.asarray(inst.bboxes)
    scores = inst.scores.cpu().numpy() if hasattr(inst.scores, "cpu") else np.asarray(inst.scores)
    labels = inst.labels.cpu().numpy() if hasattr(inst.labels, "cpu") else np.asarray(inst.labels)
    
    keep = scores >= score_thr
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
```
