# 재활용 품목 분류를 위한 Object Detection

## 1. 프로젝트 개요

<img src="./assets/introduce.png" alt="introduce" style="zoom:50%;" />

대량 생산/소비 시대의 '쓰레기 대란' 문제에 직면하여, 환경 부담을 줄이기 위한 분리수거의 중요성이 강조되고 있습니다. 정확한 분리배출이 재활용률을 높이는 핵심이므로, 이미지 내의 다양한 쓰레기 객체를 정확히 탐지하고 분류하는 Object Detection 모델을 개발하여 이 문제를 해결하고자 했습니다.

### 프로젝트 구조

**Overall Structure**

<img src="./assets/model_architectrue.png" alt="model" style="zoom:50%;" />

**Repo**

```
pro-cv-objectdetection-cv-12/
├── dataset
│   ├── train
│   ├── valid
│   └── test
├── cascade_rcnn
│   ├── inference.py
│   ├── swin.py
│   └── mmdetection
├── D-FINE
│   ├── D-FINE_inference.ipynb
│   ├── train.py
│   └── train_origin.py
├── DEIMv2_1
│   ├── DEIM_inference.ipynb
│   └── train.py
├── DEIMv2_2
│   ├── DEIM_inference.ipynb
│   ├── DEIM_inference_TTA.ipynb
│   └── train.py
├── yolo12x
│   └── yolo12x_Iterative_Stratification.ipynb
├── RF-DETR
│   ├── rfdetr_inference.ipynb
│   ├── rfdetr_train.ipynb
│   └── train_val_split.ipynbest_check.ipynb
│   ├── train_val_split.ipynb
│   └── wbf.ipynb
└── EDA
    ├── EDA_test.ipynb
    └── EDA_train.ipynb
```



### Contributor

|                            김범진                            |                            김준수                            |                            김한준                            |                            남현지                            |                            송예림                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./assets/rock.png" alt="public" style="zoom:25%;" /><br />[Github](https://github.com/kimbum1018) | <img src="./assets/water.png" alt="public" style="zoom:25%;" /><br />[Github](https://github.com/0129jonsu) | <img src="./assets/normal.png" alt="public" style="zoom:1%;" /><br />[Github](https://github.com/hanjun0126) | <img src="./assets/elec.png" alt="public" style="zoom:25%;" /><br />[Github](https://github.com/yujh5537) | <img src="./assets/fire.png" alt="public" style="zoom:25%;" /><br />[Github](https://github.com/SongYerim) |
|                                                              |                                                              |                                                              |                                                              |                                                              |



### Leaderboard Score

**Public**

<img src="./assets/public.png" alt="public" style="zoom:50%;" />

**Private**

<img src="./assets/private.png" alt="public" style="zoom:50%;" />

개발 환경

하드웨어 : GPU 서버(V100)

운영체제 : Linux

주요 라이브러리 및 프레임워크





## Reference

[Wrapup-Report](https://drive.google.com/file/d/1IfaxKJvA1vDrNyw19lI8GyTUCF0vhZBq/view?usp=share_link)