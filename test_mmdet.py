from mmdet.apis import init_detector, inference_detector
import mmcv
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

img = mmcv.imread('mmdetection/demo/demo.jpg')
result = inference_detector(model,img )


model.show_result(img, result,show=True)
