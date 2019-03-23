# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Goal
# - Create a COCO Dataset
# - Create a DataLoader
# - Use loader to load image, mask and bound boxes

# %matplotlib inline
import os
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

COCO_DIR = "/home/ubuntu/Quan/datasets/coco"
TRAIN_DIR = os.path.join(COCO_DIR, "train2014")
VALID_DIR = os.path.join(COCO_DIR, "val2014")
ANN_DIR = os.path.join(COCO_DIR, "annotations")
TRAIN_ANN_FILE = os.path.join(ANN_DIR, "instances_train2014.json")
VALID_ANN_FILE = os.path.join(ANN_DIR, "instances_valid2014.json")

coco_dataset = CocoDetection(root=TRAIN_DIR,
                            annFile=TRAIN_ANN_FILE,
                            transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))

dataloader = DataLoader(coco_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print("i_batch", i_batch)
    print(sample_batched)
    if i_batch == 1:
        break


