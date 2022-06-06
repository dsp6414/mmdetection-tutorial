import copy
import os.path as osp


import numpy as np
import mmcv
import sys
sys.path.insert(0,'.')
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class yyminstDataset(CustomDataset):

    CLASSES = ('0', '1', '2','3','4','5','6','7','8','9')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        ann_list = mmcv.list_from_file(self.ann_file)        
    
        data_infos = []
        # convert annotations to middle format
        for ann in ann_list:
            image_id,_,label_info= ann.partition(" ")   

            filename = osp.join(self.img_prefix,image_id)
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}', width=width, height=height)
    
            # load annotations
            
            lines = label_info.split()    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            for line in lines:
                content = line.split(",")
                bbox_name= content[-1]
                bbox = content[0:-1]                
                bbox = [float(e) for e in bbox]    
  
                
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                    dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos

from mmcv import Config


cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
from mmdet.apis import set_random_seed

classes = ('0', '1', '2','3','4','5','6','7','8','9')

# Modify dataset type and path
cfg.dataset_type = 'yyminstDataset'
cfg.data_root = '/home/chen/OD/mmdet_tutorial/datasets/yymnist'

cfg.data.test.type = 'yyminstDataset'
cfg.data.test.data_root = '/home/chen/OD/mmdet_tutorial/datasets/yymnist'
cfg.data.test.ann_file = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/test.txt'
cfg.data.test.classes = classes
cfg.data.test.img_prefix = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/'

cfg.data.train.type = 'yyminstDataset'
cfg.data.train.data_root = '/home/chen/OD/mmdet_tutorial/datasets/yymnist'
cfg.data.train.ann_file = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/train.txt'
cfg.data.train.classes = classes
cfg.data.train.img_prefix = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/'

cfg.data.val.type = 'yyminstDataset'
cfg.data.val.data_root = '/home/chen/OD/mmdet_tutorial/datasets/yymnist'
cfg.data.val.ann_file = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/val.txt'
cfg.data.val.classes = classes
cfg.data.val.img_prefix = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 10
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from =  'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 10
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.data.samples_per_gpu=4
cfg.data.workers_per_gpu=4
cfg.runner = dict(type='EpochBasedRunner', max_epochs=100)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

#with open("myconfigs/faster_rcnn_r50_fpn_1x_coco_yymnist.py","w+") as f:
    #f.writelines(cfg.pretty_text)

if __name__ == '__main__':  

    
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from train.train import train_detector

    #cfg = Config.fromfile("myconfigs/faster_rcnn_r50_fpn_1x_coco_yymnist.py") 
    #cfg = Config.fromfile("myconfigs/faster_rcnn_r50_fpn_1x_coco_yymnist_mmcls.py") 
    #cfg = Config.fromfile("myconfigs/faster_rcnn_convnext_fpn_1x_coco_yymnist.py") 
    cfg = Config.fromfile("myconfigs/faster_rcnn_r50_cbam_fpn_1x_coco_yymnist.py") 
    
 
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    model.CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    # Add an attribute for visualization convenience

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)


  

