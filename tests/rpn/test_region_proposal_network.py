import torch
from maskrcnn.lib.rpn.region_proposal_network import RPN


def test_rpn_forward():
    n_anchors_per_location = 100
    n_batch = 1
    n_width = 32
    n_height = 32
    n_in_channels = 512
    n_mid_channel = 1024

    # Create a test tensor of size (n_batch, n_in_channels, n_height, n_width)
    test_tensor = torch.zeros((n_batch, n_in_channels, n_height, n_width))

    rpn = RPN(n_anchors_per_location=n_anchors_per_location, in_channels=n_in_channels, mid_channels=n_mid_channel)

    rpn_logits, rpn_probs, rpn_bbox = rpn(test_tensor)

    assert rpn_logits.size() == torch.Size([n_batch, n_anchors_per_location * n_height * n_width, 2])
    assert rpn_probs.size() == torch.Size([n_batch, n_anchors_per_location * n_height * n_width, 2])
    assert rpn_bbox.size() == torch.Size([n_batch, n_anchors_per_location * n_height * n_width, 4])



