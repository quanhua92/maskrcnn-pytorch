from torch import nn


class RPN(nn.Module):
    """
    Region Proposal Network introduced in Faster-RCNN.
    """

    def __init__(self, n_anchors_per_location, in_channels, mid_channels):
        """
        Build the Region proposal network including:
        - A conv layer to generate intermediate feature map
        - A conv layer to generate objectiveness score for each anchor (foreground / background)
        - A conv layer to generate bounding boxes offset for each anchor

        Args:
            n_anchors_per_location (int): total number of anchors for each location
            in_channels (int): The channel size of the input
            mid_channels (int): The channel size of the intermediate conv layer
        """
        super(RPN, self).__init__()

        self.conv_shared = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_score = nn.Conv2d(mid_channels, n_anchors_per_location * 2, kernel_size=1, stride=1, padding=0)
        self.conv_bbox = nn.Conv2d(mid_channels, n_anchors_per_location * 4, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Forward the Region Proposal Network

        Notations:
        N : batch size
        C : channel size
        H and W: height and width of the input feature
        A: number of anchors per location

        Args:
            x (Tensor): The feature maps extracted from the images. Shape: (N, C, H, W)

        Returns:
            rpn_logits: [N, H * W * A, 2] Anchor FG/BG classifier logits (before softmax)
            rpn_probs: [N, H * W * A, 2] Anchor FG/BG probabilities for each anchor
            rpn_bbox: [N, H * W * A, (dy, dx, log(dh), log(dw)] Anchor bounding box offsets for each anchor
        """

        # Shared convolution base of the RPN [batch, mid_channels, height, width]
        x = self.relu(self.conv_shared(x))

        # Anchor scores [batch, n_anchors_per_location * 2, height, width]
        rpn_logits = self.conv_score(x)

        # Reshape rpn_logits to [batch, n_anchors, 2] with n_anchors = n_anchors_per_location * height * width
        rpn_logits = rpn_logits.permute(0, 2, 3, 1)
        rpn_logits = rpn_logits.contiguous()
        rpn_logits = rpn_logits.view(x.size()[0], -1, 2)

        # Softmax on the logits to get probabilities
        rpn_probs = self.softmax(rpn_logits)

        # Bounding Box regression [batch, n_anchors_per_location * 4, height, width)
        rpn_bbox = self.conv_bbox(x)

        # Reshape rpn_bbox to [batch, n_anchors, 4] where 4 is (dy, dx, log(dh), log(dw))
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return rpn_logits, rpn_probs, rpn_bbox
