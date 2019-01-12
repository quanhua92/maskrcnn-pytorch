import os
import torch as th
from maskrcnn.lib.utils import io_utils
from maskrcnn.lib.models.mask_rcnn import MaskRCNN
from maskrcnn.lib.config import cfg


def test_mask_rcnn():
    cfg.MODEL.IMAGENET_MODEL_PRETRAINED = False
    model = MaskRCNN(config=cfg)
    assert model.extractor is not None

    image_path = os.path.join("data", "COCO_val2014_000000018928.jpg")
    image = io_utils.read_image(image_path)

    model.predict([image])
