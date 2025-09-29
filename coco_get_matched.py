from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab  # matplotlib的一个模块，用于二维、三维图像绘制
import os
import json
pylab.rcParams['figure.figsize'] = (8.0, 10.0)  # 设置画布大小


class CocoApi:
    def __init__(self, dataDir, dataType, annFile):
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = annFile
        self.coco = self.init_coco_api(self.annFile)

    # 初始化COCO API

    def init_coco_api(self, annFile):
        coco = COCO(annFile)
        return coco

    def print_coco_cats(self):
        self.catIds = self.coco.getCatIds()
        print("Categories Ids: \n", self.catIds)
        self.cats = self.coco.loadCats(self.catIds)
        print("Categories: \n", self.cats)
        self.cat_names = [cat['name'] for cat in self.cats]
        # print("Categories Names: \n", self.cat_names)
        self.super_cat_names = [cat['supercategory'] for cat in self.cats]
        print("Super Categories Names: \n", self.super_cat_names)

    def get_imgs_all(self):
        imgIds = self.coco.getImgIds()
        print(f"total num of Image Ids: \n", len(imgIds))
        imgs = self.coco.loadImgs(imgIds)
        return imgIds, imgs

    def get_imgs_by_cat_names(self, cat_names):
        imgIds = self.coco.getImgIds(
            catIds=self.coco.getCatIds(catNms=cat_names))
        print(f"cat_names{cat_names} total num of Image Ids: \n", len(imgIds))
        imgs = self.coco.loadImgs(imgIds)
        # print(f"cat_names{cat_names} Images: \n", imgs)

        return imgIds, imgs

    def get_imgs_by_img_ids(self, imgIds):
        imgs = self.coco.loadImgs(imgIds)
        # print(f"imgIds{imgIds} Images: \n", imgs)

        return imgs

    def get_anns_by_img_ids(self, imgIds):
        annIds = self.coco.getAnnIds(imgIds=imgIds, iscrowd=None)
        print(f"imgIds{imgIds} total num of Ann Ids: \n", len(annIds))
        anns = self.coco.loadAnns(annIds)
        # print(f"imgIds{imgIds} Annotations: \n", anns)

        return annIds, anns

    def show_imgs_by_img_ids(self, imgIds):
        imgs = self.get_imgs_by_img_ids(imgIds)
        for img in imgs:
            I = io.imread('%s/%s/%s' %
                          (self.dataDir, self.dataType, img['file_name']))
            plt.axis('off')
            plt.imshow(I)
            plt.show()

    def show_imgs_with_anns(self, imgIds):
        imgs = self.get_imgs_by_img_ids(imgIds)
        for img in imgs:
            I = io.imread('%s/%s/%s' %
                          (self.dataDir, self.dataType, img['file_name']))
            plt.axis('off')
            plt.imshow(I)
            annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            self.coco.showAnns(anns, draw_bbox=True)
            plt.show()


def test_100matched():
    # 示例使用，获取100对包含指定类别名称的图像和注释，并保存
    dataDir = './Datasets/MS_COCO'
    dataType = 'val2014'
    annFile = './Datasets/MS_COCO/annotations/instances_val2014.json'
    coco_api = CocoApi(dataDir, dataType, annFile)
    coco_api.print_coco_cats()
    # cat_names = ['person', 'dog', 'cat'] 可以是一个list
    cat_names = ['person']
    imgIds, imgs = coco_api.get_imgs_by_cat_names(cat_names)

    # 显示第一个图像和注释
    coco_api.show_imgs_with_anns(imgIds[:1])

    # 取前100个单独保存imgs和anns
    imgIds = imgIds[:100]
    imgs = imgs[:100]
    annIds, anns = coco_api.get_anns_by_img_ids(imgIds)

    test_100_imgpath = "./Datasets/MS_COCO/test_100_val2014_imgs"
    test_100_annpath = "./Datasets/MS_COCO/test_100_val2014_anns"

    # 保存100对imgs和anns
    for i, img in enumerate(imgs):
        I = io.imread('%s/%s/%s' %
                      (dataDir, dataType, img['file_name']))
        plt.axis('off')
        plt.imshow(I)
        if not os.path.exists(test_100_imgpath):
            os.makedirs(test_100_imgpath)
        plt.savefig(f"{test_100_imgpath}/{img['id']}.jpg")
        plt.close()

        annIds = coco_api.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco_api.coco.loadAnns(annIds)
        coco_api.coco.showAnns(anns)
        if not os.path.exists(test_100_annpath):
            os.makedirs(test_100_annpath)

        # ann保存为json格式,增加写入
        with open(f"{test_100_annpath}/{img['id']}.json", 'w') as f:
            json.dump(anns, f)


def test_resized():
    # 示例使用，获取resized后的图片并显示
    dataDir = './resized_dataset/images'
    dataType = 'val2014'
    annFile = './resized_dataset/annotations_val2014.json'
    coco_api = CocoApi(dataDir, dataType, annFile)
    coco_api.print_coco_cats()
    # cat_names = ['person', 'dog', 'cat'] 可以是一个list
    cat_names = ['person']
    imgIds, imgs = coco_api.get_imgs_by_cat_names(cat_names)

    # 显示第一个图像和注释
    print(imgs[:1])
    annIds, anns = coco_api.get_anns_by_img_ids(imgIds[:1])
    print(anns)
    coco_api.show_imgs_with_anns(imgIds[:1])


def test_resized_all():
    # 示例使用，获取resized后的所有图片并显示
    dataDir = './resized_dataset_all/images'
    dataType = 'val2014'
    annFile = './resized_dataset_all/annotations_val2014.json'
    coco_api = CocoApi(dataDir, dataType, annFile)
    coco_api.print_coco_cats()
    # cat_names = ['person', 'dog', 'cat'] 可以是一个list
    cat_names = ['person']
    imgIds, imgs = coco_api.get_imgs_by_cat_names(cat_names)

    print(imgs[:1])
    annIds, anns = coco_api.get_anns_by_img_ids(imgIds[:1])
    print(anns)
    coco_api.show_imgs_with_anns(imgIds[:1])


def test_resized_caption():
    # 示例使用，获取resized后的图片并显示
    dataDir = './resized_dataset_captions/images'
    dataType = 'val2014'
    annFile = './resized_dataset_captions/annotations_captions_val2014.json'
    coco_api = CocoApi(dataDir, dataType, annFile)
    imgIds, imgs = coco_api.get_imgs_all()

    print(imgs[:1])
    annIds, anns = coco_api.get_anns_by_img_ids(imgIds[:1])
    print(anns)
    coco_api.show_imgs_with_anns(imgIds[:1])


def test_resized_caption_keywords():
    # 示例使用，获取resized后的图片并显示
    dataDir = './resized_dataset_captions/images'
    dataType = 'val2014'
    annFile = './resized_dataset_captions/captions_val2014_keywords.json'
    coco_api = CocoApi(dataDir, dataType, annFile)
    imgIds, imgs = coco_api.get_imgs_all()

    print(imgs[:1])
    annIds, anns = coco_api.get_anns_by_img_ids(imgIds[:1])
    print(f"anns: {anns}")
    coco_api.show_imgs_with_anns(imgIds[:1])


if __name__ == '__main__':
    # test_resized()
    # test_resized_all()
    # test_resized_caption()
    test_resized_caption_keywords()
