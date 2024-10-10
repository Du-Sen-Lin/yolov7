# Yolov7

# 一、Code

```python
git clone git@github.com:Du-Sen-Lin/yolov7.git
# 测试
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
# 测试
python detect.py --weights yolov7-d6.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val

# 训练
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
# 报错：TypeError: No loop matching the specified signature and casting was found for ufunc greater 解决方案：https://github.com/WongKinYiu/yolov7/issues/1537
"""
all          21         131       0.997           1       0.996       0.812
obj          21          21       0.999           1       0.995       0.918
滑套          21          13       0.994           1       0.995       0.792
弹桥          21          34           1           1       0.996       0.799
螺丝          21          21       0.999           1       0.996       0.706
螺母          21          21       0.997           1       0.996       0.749
调节块          21          21       0.995           1       0.996       0.908
Optimizer stripped from runs/train/yolov7-custom6/weights/last.pt, 74.8MB
Optimizer stripped from runs/train/yolov7-custom6/weights/best.pt, 74.8MB
"""

# 训练 1280 x 1280
# Single GPU: 显存不够
CUDA_VISIBLE_DEVICES=1 python train.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-d6custom.yaml --weights 'yolov7-d6_training.pt' --name yolov7-d6-custom --hyp data/hyp.scratch.custom.yaml
# Single GPU: train_aux
CUDA_VISIBLE_DEVICES=1,2,3 python train_aux.py --workers 4 --device 1 --batch-size 4 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-d6custom.yaml --weights 'yolov7-d6_training.pt' --name yolov7-d6-custom --hyp data/hyp.scratch.custom.yaml
# 报错：from_which_layer = from_which_layer[fg_mask_inboxes] RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu) 解决：https://github.com/WongKinYiu/yolov7/issues/1101
python train_aux.py --workers 4 --device 4 --batch-size 4 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-d6custom.yaml --weights 'yolov7-d6_training.pt' --name yolov7-d6-custom --hyp data/hyp.scratch.custom.yaml

"""
Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95:
all          21         131       0.997           1       0.996       0.849
obj          21          21       0.995           1       0.996       0.996
滑套          21          13       0.996           1       0.995       0.798
弹桥          21          34       0.998           1       0.996       0.798
螺丝          21          21           1           1       0.996       0.781
螺母          21          21       0.997           1       0.996       0.777
调节块          21          21       0.998           1       0.996       0.942
Optimizer stripped from runs/train/yolov7-d6-custom13/weights/last.pt, 306.9MB
Optimizer stripped from runs/train/yolov7-d6-custom13/weights/best.pt, 306.9MB
"""
```



```python

```



# 二、Paper Reading
