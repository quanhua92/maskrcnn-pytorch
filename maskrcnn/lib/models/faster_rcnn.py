import torch as t
from torch import nn
from maskrcnn.lib.utils.torch_utils import no_grad
from maskrcnn.lib.data.preprocessing import mold_inputs


class FasterRCNN(nn.Module):

    def __init__(self, extractor, config):
        """

        Args:
            extractor:
        """
        super(FasterRCNN, self).__init__()

        self.extractor = extractor
        self.config = config

    def forward(self, x):
        return x

    @no_grad
    def predict(self, images):
        """
        Run the detection pipeline.

        Args:
            images: list of images, can have different sizes

        Returns:

        """

        molded_images, image_metas = mold_inputs(images, self.config)
