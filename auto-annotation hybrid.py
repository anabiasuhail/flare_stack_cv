# 1. predict an image
import torch
from ultralytics.yolo.data.annotator import auto_annotate
from ultralytics.vit import SAM
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.vit.sam import PromptPredictor, build_sam
from ultralytics.yolo.utils.torch_utils import select_device
import matplotlib.pyplot as plt
import os
import cv2


# root = "D:\\2PAPER_DATA\\ADNOC Subset\\ADNOC 5002\\images"
root = r'D:\2PAPER_DATA\S_images\images'


# 装一张图，看怎么remove 不必要路径,看怎么一步步跑的
import glob
train_img_root = glob.glob(root + '/*.' + 'JPG')  # 读取视角-种类-图片,用glob.glob遍历
for i in train_img_root:
    # auto_annotate(data=i,
    #               det_model=r'D:\PycharmDproject\ultralytics-main\runs\best.pt', sam_model='sam_b.pt')
    auto_annotate(data=i,
                  det_model=r'runs/detect/train15/weights/best.pt', sam_model='sam_b.pt')  # Said 权重有问题，否则不可能上X就出错

# the function within annotator, putting the mask on the dat
#####################following is the auto-annotation############################
###1. use the YOLO to get a bounding box
####  2. the bbox input to the SAM to get the mask (know the label)
###### 3. Point: SAM can segment but gets no label (also can't operate on the video)
#######3. create a label folder, within the folder, see the annotation file
###########4. YOLO creat the .txt: label, polygon
def auto_annotate(data, det_model='yolov8x.pt', sam_model='sam_b.pt', device='0', output_dir=None): # Vm
    # """
    det_model = YOLO(det_model)  # get both detection and segmentation models
    sam_model = build_sam(sam_model)
    det_model.to(device)
    sam_model.to(device)

    if not output_dir:
        output_dir = Path(str(data)).parent / 'labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    prompt_predictor = PromptPredictor(sam_model)
    det_results = det_model(data, stream=True)  # data stream to YOLO
    # move the SAM to the video by getting each frame
    for result in det_results:  # for each frame: get the bbox
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs, 封装函数，得到box
        class_ids = result.boxes.cls.int().tolist()  # 封装函数，从结果得到class label
        if len(class_ids):  # within the video length, get all labels
            prompt_predictor.set_image(result.orig_img)  # for each image, use SAM
            #  set_image from the SAM model
            masks, _, _ = prompt_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=prompt_predictor.transform.apply_boxes_torch(boxes, result.orig_shape[:2]),
                multimask_output=False,
            )  # append the mask with a label

            result.update(masks=masks.squeeze(1))
            segments = result.masks.xyn  # get the annotated msk with label

            # write the annotation info. into the .txt
            with open(str(Path(output_dir) / Path(result.path).stem) + '.txt', 'w') as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f'{class_ids[i]} ' + ' '.join(segment) + '\n')

