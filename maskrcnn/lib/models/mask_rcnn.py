from __future__ import division
import torch
from torch import nn

from maskrcnn.lib.utils.torch_utils import no_grad
from maskrcnn.lib.data.preprocessing import mold_inputs
from maskrcnn.lib.models.imagenet_models import initialize_imagenet_model
from maskrcnn.lib.rpn.region_proposal_network import RPN


class MaskRCNN(nn.Module):

    def __init__(self, config):
        super(MaskRCNN, self).__init__()

        # Modules
        self.extractor = None
        self.rpn = None

        self.config = config
        self.build(config=config)

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def build(self, config):
        """
        Build network architecture including:
        - Extractor: Feature Extractor
        - RPN: Region Proposal Network

        Args:
            config:

        Returns:

        """

        # Extractor
        imagenet_model = initialize_imagenet_model(model_name=config.MODEL.IMAGENET_MODEL_NAME,
                                                   use_pretrained=config.MODEL.IMAGENET_MODEL_PRETRAINED)
        self.extractor = imagenet_model.features.to(self.device)

        # Region Proposal Network
        self.rpn = RPN(1000, 512, 512).to(self.device)

    @no_grad
    def predict(self, images):
        """
        Run the detection pipeline.

        Args:
            images: list of images, can have different sizes. images in HWC and RGB format.
                    Range of its values is [0, 255]

        Returns:

        """

        molded_images, image_metas = mold_inputs(images, self.config)

        # Convert to Tensor
        molded_images = torch.from_numpy(molded_images).float().to(self.device)

        # Features Extraction
        features = self.extractor(molded_images)

        print("features", features.size())

        # RPN
        rpn_logits, rpn_probs, rpn_bbox = self.rpn(features)

        print("rpn_probs", rpn_probs.size(), "rpn_bbox", rpn_bbox.size())

        # ROI Pooling / ROI Align

        # Classification

