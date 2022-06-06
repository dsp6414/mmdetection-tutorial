import os
import sys

from os.path import exists, join, basename, splitext


project_name=os.path.abspath(os.getcwd())
mmdetection_dir = os.path.join(project_name, "mmdetection")
mmcv_dir = os.path.join(project_name, "mmcv")
sys.path.insert(0,project_name)
#sys.path.insert(1,mmcv_dir)

sys.path.append(mmdetection_dir)



import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed


# cfg = Config.fromfile("configs/queryinst/queryinst_r50_fpn_1x_coco.py") 
# print(f'Config:\n{cfg.pretty_text}')
# with open("myconfigs/queryinst_r50_fpn_1x_coco_balloon.py","w+") as f:
#     f.writelines(cfg.pretty_text)

if __name__ == '__main__':  


    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from train.train import train_detector

    cfg = Config.fromfile("myconfigs/queryinst_r50_fpn_1x_coco_balloon.py") 
    print(f'Config:\n{cfg.pretty_text}')
    
    datasets = [build_dataset(cfg.data.train)]    
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
    
    model.CLASSES = ('balloon', )
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)




