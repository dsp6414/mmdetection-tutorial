import copy
import os.path as osp
import os

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

            filename = f'{self.img_prefix}{image_id}'
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

if __name__ == '__main__':  

    from mmcv import Config
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from train.train import train_detector
    
    cfg = Config.fromfile("myconfigs/yolox_s_8x8_300e_coco_yymnist.py") 
    datasets = [build_dataset(cfg.data.train)]    
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))  
    
    for dataset in datasets:
        dataset.CLASSES = ('0', '1', '2','3','4','5','6','7','8','9')
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)




