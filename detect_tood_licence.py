import os
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
# config_file = 'myconfigs/yolox_s_8x8_300e_voc_car_plate.py'
# checkpoint_file = 'tutorial_exps/car_plate/latest.pth'

config_file = 'myconfigs/tood_r50_fpn_1x_coco_licence.py'
checkpoint_file = './tutorial_exps/tood_licence/ciou/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = ('license', )


# test a single image and show the results
img = 'datasets/Car_License_Plate/VOC2007/JPEGImages/Cars379.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result,show=True)
# or save the visualization results to image files
#model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)


  

