import torch as th
from maskrcnn.lib.models.faster_rcnn import FasterRCNN
from maskrcnn.lib.config import cfg


def test_faster_rcnn():
    model = FasterRCNN(extractor=None, config=cfg)
    # model.predict([th.zeros(10)])
