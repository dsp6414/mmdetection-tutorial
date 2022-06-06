import copy
import os.path as osp
import os

import numpy as np
import mmcv
import sys
sys.path.insert(0,'.')
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv import Config


#cfg = Config.fromfile("myconfigs/yolox_s_8x8_300e_coco_yymnist.py") 
cfg = Config.fromfile("configs/tood/tood_r50_fpn_1x_coco.py")
from mmdet.apis import set_random_seed


cfg.load_from = None
# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps/tood_card'


# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 5
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.data.samples_per_gpu=4
cfg.data.workers_per_gpu=2
cfg.runner = dict(type='EpochBasedRunner', max_epochs=100)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# with open("myconfigs/tood_r50_fpn_1x_coco_card.py","w+") as f:
#     f.writelines(cfg.pretty_text)

if __name__ == '__main__':  

    
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from train.train import train_detector
    
    cfg = Config.fromfile("myconfigs/tood_r50_fpn_1x_coco_licence.py")
    datasets = [build_dataset(cfg.data.train)]    
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
    
    for dataset in datasets:
        dataset.CLASSES = ('licence',)
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)




