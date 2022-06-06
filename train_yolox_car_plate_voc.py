import copy
import os.path as osp
import os

import numpy as np
import mmcv
import sys
sys.path.insert(0,'.')
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.xml_style import XMLDataset
from mmcv import Config

@DATASETS.register_module()
class Car_Plate(XMLDataset):    

    CLASSES = ('licence', )

    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

cfg = Config.fromfile("configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py") 
# with open("myconfigs/myconfigs/yolox_s_8x8_300e_voc_car_plate.py","w+") as f:
#     f.writelines(cfg.pretty_text)

if __name__ == '__main__':  

    from mmcv import Config
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from train.train import train_detector
    
    cfg = Config.fromfile("myconfigs/yolox_s_8x8_300e_voc_car_plate.py") 
    datasets = [build_dataset(cfg.data.train)]    
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
    
    for dataset in datasets:
        dataset.CLASSES =  ('licence', )
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    #mmdetection/mmdet/datasets/xml_style.py 修改支持.png
    train_detector(model, datasets, cfg, distributed=False, validate=True)




