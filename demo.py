import os
from maskrcnn.lib.utils.io_utils import read_image
from maskrcnn.lib.models.faster_rcnn import FasterRCNN

image_path = os.path.join("data", "COCO_val2014_000000018928.jpg")
image = read_image(image_path)

print(image.shape)
