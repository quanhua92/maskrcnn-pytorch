# -*- coding: utf-8 -*-
import pathmagic # noqa
import os
import torch

from maskrcnn.lib.config import cfg
from maskrcnn.lib.utils.io_utils import read_image
from maskrcnn.lib.models.mask_rcnn import MaskRCNN

from maskrcnn.lib.data.preprocessing import mold_inputs

image_path = os.path.join("..", "data", "COCO_val2014_000000018928.jpg")
image = read_image(image_path)

print("image shape", image.shape)

model = MaskRCNN(config=cfg)

a = model.anchors

print(cfg.MODEL.RPN)

images = [image]

molded_images, image_metas = mold_inputs(images, cfg)

# Convert to Tensor
# [N, C, H, W] ----- [1, 3, 1024, 1024]
molded_images = torch.from_numpy(molded_images).float().to(model.device)

# Features Extraction
features = model.extractor(molded_images)
# [N, 512, 32, 32]

print("features", features.size())

# RPN
rpn_logits, rpn_probs, rpn_bbox = model.rpn(features)

print("rpn_probs", rpn_probs.size(), "rpn_bbox", rpn_bbox.size())
probs = rpn_probs.detach().cpu().numpy()
bbox = rpn_bbox.detach().cpu().numpy()
logits = rpn_logits.detach().cpu().numpy()

# Calculate RPN Loss
# 15360 = 32 * 32 * 3 * 5
# probs, logits : 15360, 2 FG / BG (probs is softmax(logits))
# bbox: 15360, 4

# Binary Class Loss
# positive label: 1 - anchors with highest IoU with a ground truth box
#                   - anchors with IoU overlap > 0.7 with any ground truth box
# negative label: 0 - non-positive label with IoU lower than 0.3
# -> Loss cls = log loss over 2 classes 

# Box Regression Loss
# Loss reg = R(t_i - t*_i)
# R: Smooth L1
# t_i
# For positive anchor only:
# - Positive anchor means: we have an anchor & a ground truth box 
# 
# Loop through anchors
# - assign positive, negative
#       - remove cross-boundary
#       - find labels
#       - random sample anchors
# - calc loss
#       positive & negative -> binary class loss
#       positive: regression loss
