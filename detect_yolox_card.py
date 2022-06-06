import os
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '/home/chen/OD/mmdet_tutorial/myconfigs/yolox/yolox_s_8x8_300e_coco_card.py'
checkpoint_file = '/home/chen/OD/mmdet_tutorial/checkpoints/yolox/best.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = ('nine', 'ten', 'jack', 'queen', 'king', 'ace')

# test a single image and show the results
img = '/home/chen/OD/mmdet_tutorial/datasets/data_coco/train/cam_image46.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)


  

