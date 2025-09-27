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

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

# 遍历前三张图像
img_id = ids[100]
print(f"img_id: {img_id}")
# 获取对应图像id的所有annotations idx信息
ann_ids = coco.getAnnIds(imgIds=img_id)
print(f"ann_ids: {ann_ids}")

# 根据annotations idx信息获取所有标注信息
targets = coco.loadAnns(ann_ids)
print(f"targets: {targets}")

# get image file name
path = coco.loadImgs(img_id)[0]['file_name']

# read image
img = Image.open(os.path.join(img_path, path)).convert('RGB')
draw = ImageDraw.Draw(img)
# draw box to image
for target in targets:
    x, y, w, h = target["bbox"]
    print(f"bbox: {x, y, w, h}")
    x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
    draw.rectangle((x1, y1, x2, y2))
    draw.text((x1, y1), coco_classes[target["category_id"]])

# show image
plt.imshow(img)
plt.show()
