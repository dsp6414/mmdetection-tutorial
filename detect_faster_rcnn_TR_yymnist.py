import os
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '/home/chen/OD/mmdet_tutorial/myconfigs/faster_rcnn/faster_rcnn_TR_fpn_1x_coco_yymnist.py'
checkpoint_file = '/home/chen/OD/mmdet_tutorial/checkpoints/faster_rcnn/yymnist/faster_rcnn_swin_TR_yymnist.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# test a single image and show the results
img = '/home/chen/OD/mmdet_tutorial/datasets/yymnist/images/000625.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result,show=True)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)


  

