import os
import sys

from os.path import exists, join, basename, splitext


project_name=os.path.abspath(os.getcwd())
mmdetection_dir = os.path.join(project_name, "mmdetection")
mmcv_dir = os.path.join(project_name, "mmcv")
sys.path.insert(0,project_name)
#sys.path.insert(1,mmcv_dir)

sys.path.append(mmdetection_dir)


MODELS_CONFIG = {
    'yolox_s_8x8_300e_coco': {
        'config_file': 'configs/yolox/yolox_s_8x8_300e_coco.py'
    }
}

selected_model = 'yolox_s_8x8_300e_coco' 
total_epochs = 20
config_file = MODELS_CONFIG[selected_model]['config_file']
config_fname = os.path.join(project_name, config_file)

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed
cfg = Config.fromfile(config_fname)    
cfg.work_dir = './tutorial_exps'
cfg.load_from = None


cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.evaluation.interval = 5
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10
print(f'Config:\n{cfg.pretty_text}')

# with open("myconfigs/yolox_s_8x8_300e_coco_card.py","w+") as f:
#     f.writelines(cfg.pretty_text)

if __name__ == '__main__':  


    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from train.train import train_detector
    
    cfg = Config.fromfile("myconfigs/yolox_s_8x8_300e_coco_licence.py") 
    datasets = [build_dataset(cfg.data.train)]    
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
    
    for dataset in datasets:
        dataset.CLASSES = ('licence', )
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)




