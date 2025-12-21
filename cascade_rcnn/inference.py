import os
import os.path as osp
import pandas as pd
import torch

from mmengine.config import Config
from mmdet.apis import DetInferencer
from pycocotools.coco import COCO
from src.config import *

def main():
    # 설정
    BATCH_SIZE = 32
    SCORE_THR = 0.01
    SAVED_EPOCH = "best_coco_bbox_mAP_epoch_22"
    CHECKPOINT_PATH = osp.join(WORK_DIR, f"{SAVED_EPOCH}.pth")

    print(f"[INFO] Loading Checkpoint form: {CHECKPOINT_PATH}")

    # Config 로드 및 수정
    cfg = Config.fromfile(MODEL_CONFIG_PATH)
    cfg.randomness = dict(seed=SEED, deterministic=False)

    # 모델 Head 클래스 수 맞추기
    if isinstance(cfg.model.roi_head.bbox_head, list):
        for head in cfg.model.roi_head.bbox_head:
            head.num_classes = len(CLASSES)
    else:
        cfg.model.roi_head.bbox_head.num_classes = len(CLASSES)

    # Test Dataset Pipeline 수정 (Resize)
    cfg.test_dataloader.dataset.pipeline[1]["scale"] = (1024, 1024)

    # COCO 객체 생성 (이미지 리스트 로드)
    coco = COCO(osp.join(DATA_ROOT, TEST_ANN))
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)

    # 경로 생성 함수
    img_paths = [osp.join(DATA_ROOT, im["file_name"]) for im in imgs]
    file_names = [im["file_name"] for im in imgs]

    # 추론기 초기화
    inferencer = DetInferencer(model=cfg, weights=CHECKPOINT_PATH, device="cuda:0")

    final_results = []
    print(f"[Info] Starting Inference on {len(img_paths)} images...")

    # Batch Inference
    for i in range(0, len(img_paths), BATCH_SIZE):
        batch_imgs = img_paths[i : i + BATCH_SIZE]
        
        batch_results = inferencer(
            batch_imgs,
            batch_size=BATCH_SIZE,
            return_datasamples=True,
            no_save_vis=True,
        )

        for output in batch_results['predictions']:
            pred_instances = output.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            mask = scores > SCORE_THR
            final_results.append({
                "bboxes": bboxes[mask],
                "scores": scores[mask],
                "labels": labels[mask]
            })

        # 메모리 정리
        del batch_results
        if i % 100 == 0:
            torch.cuda.empty_cache()
            print(f"Progress: {i}/{len(img_paths)} done.")

    # Submission CSV 생성
    print("[Info] Creating submission.csv...")
    prediction_strings = []
    
    for img_name, data_sample in zip(file_names, final_results):
        bboxes = data_sample["bboxes"]
        scores = data_sample["scores"]
        labels = data_sample["labels"]
        
        prediction_string = ""
        for idx in range(len(bboxes)):
            label = int(labels[idx])
            score = float(scores[idx])
            xmin, ymin, xmax, ymax = bboxes[idx]
            prediction_string += f"{label} {score:.6f} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f} "
        
        prediction_strings.append(prediction_string.strip())

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv("swin_cascade.csv", index=None)
    print("[Info] Saved to submission.csv successfully.")

if __name__ == "__main__":
    main()