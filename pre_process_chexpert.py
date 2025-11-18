'''Data Preprocessing'''

###################
## Prerequisites ##
###################
import json
import argparse
import pandas as pd
from easydict import EasyDict as edict
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
import torch
from PIL import Image
import csv

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg_path', metavar='CFG_PATH', type=str,
                    help='Path to the config file in yaml format.')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))


###################
## Preprocessing ##
###################
# Each file contains pairs (path to image, output vector)
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''

Traindata = pd.read_csv('./CheXpert-v1.0{0}/train.csv'.format(img_type))
Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')].copy()
Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')].copy()
Traindata_frt.to_csv(
    './CheXpert-v1.0{0}/train_frt.csv'.format(img_type), index=False)
Traindata_lat.to_csv(
    './CheXpert-v1.0{0}/train_lat.csv'.format(img_type), index=False)
print('Train data length(frontal):', len(Traindata_frt))
print('Train data length(lateral):', len(Traindata_lat))
print('Train data length(total):', len(Traindata_frt) + len(Traindata_lat))

Validdata = pd.read_csv('./CheXpert-v1.0{0}/valid.csv'.format(img_type))
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')].copy()
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')].copy()
Validdata_frt.to_csv(
    './CheXpert-v1.0{0}/valid_frt.csv'.format(img_type), index=False)
Validdata_lat.to_csv(
    './CheXpert-v1.0{0}/valid_lat.csv'.format(img_type), index=False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))
print('Valid data length(total):', len(Validdata_frt) + len(Validdata_lat))

Testdata = pd.read_csv('./CheXpert-v1.0{0}/valid.csv'.format(img_type))
Testdata_frt = Testdata[Testdata['Path'].str.contains(
    'frontal')].copy()  # to avoid SettingWithCopyWarning
Testdata_lat = Testdata[Testdata['Path'].str.contains('lateral')].copy()
Testdata_frt.to_csv(
    './CheXpert-v1.0{0}/test_frt.csv'.format(img_type), index=False)
Testdata_lat.to_csv(
    './CheXpert-v1.0{0}/test_lat.csv'.format(img_type), index=False)
print('Test data length(frontal):', len(Testdata_frt))
print('Test data length(lateral):', len(Testdata_lat))
print('Test data length(total):', len(Testdata_frt) + len(Testdata_lat))

# Make testset for 200 studies (use given valid set as test set)
Testdata_frt.loc[:, 'Study'] = Testdata_frt.Path.str.split(
    '/').str[2] + '/' + Testdata_frt.Path.str.split('/').str[3]
Testdata_frt_agg = Testdata_frt.groupby('Study').agg('first').reset_index()
Testdata_frt_agg = Testdata_frt_agg.sort_values('Path')
Testdata_frt_agg = Testdata_frt_agg.drop('Study', axis=1)
Testdata_frt_agg.to_csv(
    './CheXpert-v1.0{0}/test_agg.csv'.format(img_type), index=False)
print('Test data length(study):', len(Testdata_frt_agg))


######################
## Create a Dataset ##
######################


class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, nnClassCount, policy, transform=None):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None)  # skip the header
            for line in csvReader:
                image_name = line[0]
                npline = np.array(line)
                idx = [7, 10, 11, 13, 15]
                label = list(npline[idx])
                for i in range(nnClassCount):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == 'diff':
                                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            elif policy == 'ones':              # All U-Ones
                                label[i] = 1
                            else:
                                label[i] = 0                    # All U-Zeroes
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


#######################
## Pre-define Values ##
#######################
# Paths to the files with training, validation, and test sets.
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''

Traindata_frt = pd.read_csv(
    './CheXpert-v1.0{0}/train_frt.csv'.format(img_type))
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)
Traindata_frt.to_csv(
    './CheXpert-v1.0{0}/train_frt.csv'.format(img_type), index=False)
Traindata_lat = pd.read_csv(
    './CheXpert-v1.0{0}/train_lat.csv'.format(img_type))
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)
Traindata_lat.to_csv(
    './CheXpert-v1.0{0}/train_lat.csv'.format(img_type), index=False)

pathFileTrain_frt = './CheXpert-v1.0{0}/train_frt.csv'.format(img_type)
pathFileTrain_lat = './CheXpert-v1.0{0}/train_lat.csv'.format(img_type)
pathFileValid_frt = './CheXpert-v1.0{0}/valid_frt.csv'.format(img_type)
pathFileValid_lat = './CheXpert-v1.0{0}/valid_lat.csv'.format(img_type)
pathFileTest_frt = './CheXpert-v1.0{0}/test_frt.csv'.format(img_type)
pathFileTest_lat = './CheXpert-v1.0{0}/test_lat.csv'.format(img_type)
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)

# Neural network parameters
nnIsTrained = cfg.pre_trained  # if pre-trained by ImageNet

# Training settings
trBatchSize = cfg.batch_size  # batch size
trMaxEpoch = cfg.epochs      # maximum number of epochs

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = cfg.imgtransResize

# Class names
# dimension of the output - 5: only competition obs.
nnClassCount = cfg.nnClassCount
class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]


######################
## Create a Dataset ##
######################
# Tranform data
transformList = []
transformList.append(transforms.Resize(
    (imgtransResize, imgtransResize)))  # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

# Create a dataset
'''See 'materials.py' to check the class 'CheXpertDataSet'.'''
datasetTrain_frt = CheXpertDataSet(
    pathFileTrain_frt, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat = CheXpertDataSet(
    pathFileTrain_lat, nnClassCount, cfg.policy, transformSequence)
datasetValid_frt = CheXpertDataSet(
    pathFileValid_frt, nnClassCount, cfg.policy, transformSequence)
datasetValid_lat = CheXpertDataSet(
    pathFileValid_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_frt = CheXpertDataSet(
    pathFileTest_frt, nnClassCount, cfg.policy, transformSequence)
datasetTest_lat = CheXpertDataSet(
    pathFileTest_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_agg = CheXpertDataSet(
    pathFileTest_agg, nnClassCount, cfg.policy, transformSequence)

# Use subset of datasetTrain for training ###
# use subset of original training dataset
train_num_frt = round(len(datasetTrain_frt) * cfg.train_ratio)
# use subset of original training dataset
train_num_lat = round(len(datasetTrain_lat) * cfg.train_ratio)
datasetTrain_frt, _ = random_split(
    datasetTrain_frt, [train_num_frt, len(datasetTrain_frt) - train_num_frt])
datasetTrain_lat, _ = random_split(
    datasetTrain_lat, [train_num_lat, len(datasetTrain_lat) - train_num_lat])
print('<<< Data Information >>>')
print('Train data (frontal):', len(datasetTrain_frt))
print('Train data (lateral):', len(datasetTrain_lat))
print('Valid data (frontal):', len(datasetValid_frt))
print('Valid data (lateral):', len(datasetValid_lat))
print('Test data (frontal):', len(datasetTest_frt))
print('Test data (lateral):', len(datasetTest_lat))
print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt = DataLoader(dataset=datasetTrain_frt, batch_size=trBatchSize,
                                 shuffle=True, num_workers=2, pin_memory=True)
dataLoaderTrain_lat = DataLoader(dataset=datasetTrain_lat, batch_size=trBatchSize,
                                 shuffle=True, num_workers=2, pin_memory=True)
dataLoaderVal_frt = DataLoader(dataset=datasetValid_frt, batch_size=trBatchSize,
                               shuffle=False, num_workers=2, pin_memory=True)
dataLoaderVal_lat = DataLoader(dataset=datasetValid_lat, batch_size=trBatchSize,
                               shuffle=False, num_workers=2, pin_memory=True)
dataLoaderTest_frt = DataLoader(
    dataset=datasetTest_frt, num_workers=2, pin_memory=True)
dataLoaderTest_lat = DataLoader(
    dataset=datasetTest_lat, num_workers=2, pin_memory=True)
dataLoaderTest_agg = DataLoader(
    dataset=datasetTest_agg, num_workers=2, pin_memory=True)
