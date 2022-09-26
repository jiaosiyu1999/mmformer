# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

from .data import (
    build_detection_test_loader,
    build_detection_train_loader,
    dataset_sample_per_class,
)
from .evaluation.fewshot_sem_seg_evaluation import (
    FewShotSemSegEvaluator,
)

# models

from .mask2_oracle_ori_fsloss import mask2_oracle_ori_fsloss
from .new_1_sat_cLoss_new import new_1_sat_cLoss_new 


from .test_time_augmentation import SemanticSegmentorWithTTA
# evaluation
# from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.fewshot_sem_seg_evaluation import FewShotSemSegEvaluator