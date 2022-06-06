import os
from mmdet.apis import init_detector, inference_detector
import mmcv

# maskrcnn
# config_file = '/home/chen/OD/mmdet_tutorial/myconfigs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_balloon.py'
# checkpoint_file = '/home/chen/OD/mmdet_tutorial/checkpoints/mask_rcnn/best.pth'

#queryinst
# config_file = 'myconfigs/queryinst_r50_fpn_1x_coco_balloon.py'
# checkpoint_file = 'tutorial_exps/queryinst/balloon/latest.pth'

# config_file = 'myconfigs/solo_r50_fpn_1x_coco_balloon.py'
# checkpoint_file = 'tutorial_exps/solo/balloon/latest.pth'

config_file = 'myconfigs/solov2_r50_fpn_1x_coco.py'
checkpoint_file = 'tutorial_exps/solov2/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test a single image and show the results
img = 'datasets/balloon/val/14898532020_ba6199dd22_k.jpg'  # or img = mmcv.imread(img), which will only load it once
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


  

