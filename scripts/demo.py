import pathmagic # noqa
import os
from maskrcnn.lib.config import cfg
from maskrcnn.lib.utils.io_utils import read_image
from maskrcnn.lib.models.mask_rcnn import MaskRCNN

image_path = os.path.join("..", "data", "COCO_val2014_000000018928.jpg")
image = read_image(image_path)

print("image shape", image.shape)

model = MaskRCNN(config=cfg)

model.predict([image])
