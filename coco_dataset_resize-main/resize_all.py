#################################################################################
#                                                                               #
#                               coco_dataset_resize                             #
#                 Author : Michaël Scherer (schererm8791@gmail.com)             #
#                                                                               #
#                              License : GPL v3.0                               #
#                                                                               #
#################################################################################

import argparse
import json
import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
from pycocotools import mask as mask_util
from collections import defaultdict


from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
from pycocotools import mask as mask_util
import numpy as np
import copy


def resizeImageAndAnnotations(imgFile, annotations, inputW, inputH, targetImgW, targetImgH, outputImgFile):
    print("Reading image {0} ...".format(imgFile))
    img = cv2.imread(imgFile)
    if img is None:
        raise FileNotFoundError(f"Image not found: {imgFile}")

    H, W = img.shape[:2]
    assert H == inputH and W == inputW, f"Image size mismatch: {imgFile}"

    # 分离 bbox、segmentation 类型
    bboxesList = []
    segsList = []
    is_rle_list = []
    for ann in annotations:
        # bbox
        x, y, w, h = ann['bbox']
        bboxesList.append(BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h))
        # segmentation
        seg = ann['segmentation']
        segsList.append(seg)
        is_rle = isinstance(seg, dict) and 'counts' in seg
        is_rle_list.append(is_rle)

    # === Step 1: 处理图像和 bbox（同前）===
    if targetImgH == targetImgW:
        if inputW > inputH:
            seq = iaa.Sequential([
                iaa.Resize(
                    {"height": "keep-aspect-ratio", "width": targetImgW}),
                iaa.PadToFixedSize(
                    width=targetImgW, height=targetImgH, position="center")
            ])
        else:
            seq = iaa.Sequential([
                iaa.Resize(
                    {"height": targetImgH, "width": "keep-aspect-ratio"}),
                iaa.PadToFixedSize(
                    width=targetImgW, height=targetImgH, position="center")
            ])
    elif targetImgH > targetImgW:
        seq = iaa.Sequential([
            iaa.Resize({"height": "keep-aspect-ratio", "width": targetImgW}),
            iaa.PadToFixedSize(
                width=targetImgW, height=targetImgH, position="center")
        ])
    else:
        seq = iaa.Sequential([
            iaa.Resize({"height": targetImgH, "width": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(
                width=targetImgW, height=targetImgH, position="center")
        ])

    bbs_on_image = ia.BoundingBoxesOnImage(bboxesList, shape=img.shape)
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs_on_image)

    cv2.imwrite(outputImgFile, image_aug)

    # === Step 2: 处理 segmentation ===
    new_segs = []

    for i, (seg, is_rle) in enumerate(zip(segsList, is_rle_list)):
        if not is_rle:
            # Polygon: 使用 Keypoints（需单独处理）
            # 我们稍后统一用 keypoints 处理所有 polygon
            new_segs.append(None)  # 占位，后面填充
        else:
            # RLE: decode → resize → pad → encode
            try:
                rle = seg
                mask = mask_util.decode(rle)  # (H, W), uint8, 0/1
                # 转为 float for resize
                mask = mask.astype(np.float32)

                # 手动模拟相同变换
                # Step A: resize keeping aspect ratio
                if targetImgH == targetImgW:
                    if inputW > inputH:
                        new_w = targetImgW
                        scale = new_w / inputW
                        new_h = int(inputH * scale)
                    else:
                        new_h = targetImgH
                        scale = new_h / inputH
                        new_w = int(inputW * scale)
                elif targetImgH > targetImgW:
                    new_w = targetImgW
                    scale = new_w / inputW
                    new_h = int(inputH * scale)
                else:
                    new_h = targetImgH
                    scale = new_h / inputH
                    new_w = int(inputW * scale)

                resized_mask = cv2.resize(
                    mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Step B: pad to target size (center)
                pad_h = targetImgH - new_h
                pad_w = targetImgW - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                padded_mask = np.pad(
                    resized_mask,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=0
                )

                # 转回 uint8 和 RLE
                padded_mask = (padded_mask > 0.5).astype(np.uint8)
                rle_new = mask_util.encode(np.asfortranarray(padded_mask))
                rle_new['counts'] = rle_new['counts'].decode(
                    'utf-8')  # COCO JSON 要求 counts 是 str
                new_segs.append(rle_new)

            except Exception as e:
                print(
                    f"Warning: Failed to process RLE segmentation for annotation, keeping original. Error: {e}")
                new_segs.append(seg)

    # === Step 3: 处理所有 polygon segmentation 用 keypoints ===
    all_kps = []
    seg_lengths = []
    poly_indices = []  # 记录哪些是 polygon（用于填回 new_segs）

    for i, (seg, is_rle) in enumerate(zip(segsList, is_rle_list)):
        if not is_rle and isinstance(seg, list):
            poly_indices.append(i)
            for polygon in seg:
                if len(polygon) == 0:
                    continue
                pts = np.array(polygon).reshape(-1, 2)
                for pt in pts:
                    all_kps.append(Keypoint(x=pt[0], y=pt[1]))
                seg_lengths.append(len(pts))
        elif not is_rle:
            # segmentation is [] or invalid
            poly_indices.append(i)
            seg_lengths.append(0)

    if all_kps:
        kps_on_image = KeypointsOnImage(all_kps, shape=img.shape)
        kps_aug = seq(image=img, keypoints=kps_on_image)[1]  # 只取 keypoints
        kp_idx = 0
        seg_idx = 0
        for i in poly_indices:
            seg = segsList[i]
            new_poly_list = []
            if isinstance(seg, list):
                for polygon in seg:
                    if len(polygon) == 0:
                        new_poly_list.append([])
                        continue
                    n_pts = len(polygon) // 2
                    transformed_pts = []
                    for _ in range(n_pts):
                        kp = kps_aug.keypoints[kp_idx]
                        x = float(kp.x)
                        y = float(kp.y)
                        # 可选 clamp
                        x = max(0.0, min(targetImgW, x))
                        y = max(0.0, min(targetImgH, y))
                        transformed_pts.extend([x, y])
                        kp_idx += 1
                    new_poly_list.append(transformed_pts)
            new_segs[i] = new_poly_list
    else:
        # 填充空 polygon
        for i in poly_indices:
            new_segs[i] = segsList[i]  # keep original if error

    return bbs_aug, new_segs


if __name__ == "__main__":

    ia.seed(1)

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--images_dir", required=True,
                    help="Directory where are located the images referenced in the annotations file")
    ap.add_argument("-a", "--annotations_file", required=True,
                    help="COCO JSON format annotations file")
    ap.add_argument("-w", "--image_width", required=True,
                    help="Target image width")
    ap.add_argument("-t", "--image_height", required=True,
                    help="Target image height")
    ap.add_argument("-o", "--output_ann_file", required=True,
                    help="Output annotations file")
    ap.add_argument("-f", "--output_img_dir", required=True,
                    help="Output images directory")

    args = vars(ap.parse_args())

    imageDir = args['images_dir']
    annotationsFile = args['annotations_file']
    targetImgW = int(args['image_width'])
    targetImgH = int(args['image_height'])
    outputImageDir = args['output_img_dir']
    outputAnnotationsFile = args['output_ann_file']

    print("Loading annotations file...")
    data = json.load(open(annotationsFile, 'r'))
    print("Annotations file loaded.")

    print("Building dictionnaries...")
    anns = defaultdict(list)
    annsIdx = dict()
    for i in range(0, len(data['annotations'])):
        anns[data['annotations'][i]['image_id']].append(data['annotations'][i])
        annsIdx[data['annotations'][i]['id']] = i
    print("Dictionnaries built.")

    for img in data['images']:
        print("Processing image file {0} and its annotations...".format(
            img['file_name']))

        annList = anns[img['id']]
        if not annList:
            # No annotations, just resize image
            imgFullPath = os.path.join(imageDir, img['file_name'])
            outputImgFullPath = os.path.join(outputImageDir, img['file_name'])
            outputDir = os.path.dirname(outputImgFullPath)
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            img_raw = cv2.imread(imgFullPath)
            # Apply same resize logic
            if targetImgH == targetImgW:
                if img['width'] > img['height']:
                    seq = iaa.Sequential([
                        iaa.Resize(
                            {"height": "keep-aspect-ratio", "width": targetImgW}),
                        iaa.PadToFixedSize(
                            width=targetImgW, height=targetImgH, position="center")
                    ])
                else:
                    seq = iaa.Sequential([
                        iaa.Resize(
                            {"height": targetImgH, "width": "keep-aspect-ratio"}),
                        iaa.PadToFixedSize(
                            width=targetImgW, height=targetImgH, position="center")
                    ])
            elif targetImgH > targetImgW:
                seq = iaa.Sequential([
                    iaa.Resize(
                        {"height": "keep-aspect-ratio", "width": targetImgW}),
                    iaa.PadToFixedSize(
                        width=targetImgW, height=targetImgH, position="center")
                ])
            else:
                seq = iaa.Sequential([
                    iaa.Resize(
                        {"height": targetImgH, "width": "keep-aspect-ratio"}),
                    iaa.PadToFixedSize(
                        width=targetImgW, height=targetImgH, position="center")
                ])
            img_aug = seq(image=img_raw)
            cv2.imwrite(outputImgFullPath, img_aug)
            img['width'] = targetImgW
            img['height'] = targetImgH
            continue

        imgFullPath = os.path.join(imageDir, img['file_name'])
        outputImgFullPath = os.path.join(outputImageDir, img['file_name'])
        outputDir = os.path.dirname(outputImgFullPath)
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        outNewBBoxes, outNewSegs = resizeImageAndAnnotations(
            imgFullPath, annList,
            int(img['width']), int(img['height']),
            targetImgW, targetImgH,
            outputImgFullPath
        )

        for i, ann in enumerate(annList):
            annId = ann['id']
            idx = annsIdx[annId]
            # Update bbox
            data['annotations'][idx]['bbox'][0] = outNewBBoxes[i].x1
            data['annotations'][idx]['bbox'][1] = outNewBBoxes[i].y1
            data['annotations'][idx]['bbox'][2] = outNewBBoxes[i].x2 - \
                outNewBBoxes[i].x1
            data['annotations'][idx]['bbox'][3] = outNewBBoxes[i].y2 - \
                outNewBBoxes[i].y1
            # Update segmentation
            data['annotations'][idx]['segmentation'] = outNewSegs[i]

        img['width'] = targetImgW
        img['height'] = targetImgH

    print("Writing modified annotations to file...")
    with open(outputAnnotationsFile, 'w') as outfile:
        json.dump(data, outfile)

    print("Finished.")
