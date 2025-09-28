import json

import os
import random

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

random.seed(0)

json_path = "./Datasets/MS_COCO/annotations/person_keypoints_train2014.json"
img_path = "./Datasets/MS_COCO/train2014"

annos = json.loads(open(json_path).read())
print(type(annos))  # <class 'dict'>
print(len(annos))  # 5
for item in annos:
    print(item)
# print(annos["annotations"])


# load coco data
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# 遍历前5张图片中的人体关键点信息(注意，并不是每张图片里都有人体信息)
for img_id in ids[:5]:
    idx = 0
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    print(f"annotations {anns}")
    for ann in anns:
        xmin, ymin, w, h = ann['bbox']
        # 打印人体bbox信息
        print(
            f"[image id: {img_id}] person {idx} bbox: [{xmin:.2f}, {ymin:.2f}, {xmin + w:.2f}, {ymin + h:.2f}]")
        keypoints_info = np.array(ann["keypoints"]).reshape([-1, 3])
        visible = keypoints_info[:, 2]
        keypoints = keypoints_info[:, :2]
        # 打印关键点信息以及可见度信息
        print(
            f"[image id: {img_id}] person {idx} keypoints: {keypoints.tolist()}")
        print(
            f"[image id: {img_id}] person {idx} keypoints visible: {visible.tolist()}")
        idx += 1

# # get all coco class labels
# coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

# # 查看第3张图像
# img_id = ids[0]
# print(f"img_id: {img_id}")
# # 获取对应图像id的所有annotations idx信息
# ann_ids = coco.getAnnIds(imgIds=img_id)
# print(f"ann_ids: {ann_ids}")

# # 根据annotations idx信息获取所有标注信息
# targets = coco.loadAnns(ann_ids)
# print(f"targets: {targets}")

# # get image file name
# path = coco.loadImgs(img_id)[0]['file_name']

# # read image
# img = Image.open(os.path.join(img_path, path)).convert('RGB')
# draw = ImageDraw.Draw(img)
# # draw box to image
# for target in targets:
#     x, y, w, h = target["bbox"]
#     print(f"bbox: {x, y, w, h}")
#     x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
#     draw.rectangle((x1, y1, x2, y2))
#     draw.text((x1, y1), coco_classes[target["category_id"]])

# # show image
# plt.imshow(img)
# plt.show()
