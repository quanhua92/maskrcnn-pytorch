import pathmagic # noqa
from maskrcnn.lib.rpn.anchors import generate_anchors
from maskrcnn.lib.config import cfg as config


#Debug
config.MODEL.RPN.SCALES = [512]
config.MODEL.RPN.RATIOS = [1]
config.MODEL.RPN.ANCHOR_STRIDE = 10

anchors, n_anchors_per_location = generate_anchors(scales=config.MODEL.RPN.SCALES,
                                                   ratios=config.MODEL.RPN.RATIOS,
                                                   anchor_stride=config.MODEL.RPN.ANCHOR_STRIDE,
                                                   height=config.MODEL.RPN.IN_HEIGHT,
                                                   width=config.MODEL.RPN.IN_WIDTH)

print(config.MODEL.RPN)
print("n_anchors_per_location", n_anchors_per_location)
print("anchors", anchors.shape)

n_anchors = anchors.shape[0]

for i in range(n_anchors):
    print("i", i)

    print(anchors[i, :])

    if i > 10:
        break

