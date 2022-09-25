# mmformer


# Usage



#### Build Dependencies
```
cd model/ops/
bash make.sh
cd ../../
```

### Data Preparation

+ Please refer to [CyCTR](https://github.com/YanFangCS/CyCTR-Pytorch) to prepare the datasets 
```
${YOUR_PROJ_PATH}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- instances_train2017.json
        |   `-- instances_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```

Then, run  
```
python prepare_coco_data.py
```
to prepare COCO-20^i data.

### Train
Run this command for training:
```
    python TRAIN.py --config-file configs/DATASET/STEP.yaml
```
For example
1. Modify `DATASETS.SPLIT` in `configs/coco/step1.yaml` and run this command for training **step1** of COCO: 
```
    python train_step1.py --config-file configs/coco/step1.yaml --num-gpus 1
```

2. Modify `DATASETS.SPLIT` and `MODEL.WEIGHTS` in `configs/coco/step2.yaml` and run this command for training **step2** of COCO: 
```
    python train.py --config-file configs/coco/step2.yaml --num-gpus 1
```



### Test Only
Modify `eval.yaml` file (`DATASETS.SPLIT` and `MODEL.WEIGHTS`)
Run the following command: 
```
    python test_.py --config-file configs/DATASET/eval.yaml --num-gpus 1 --eval-only
```

For example, 
```
    python test_.py --config-file configs/pascal/eval.yaml --num-gpus 1 --eval-only
```

### Pretrained models
[models]()
