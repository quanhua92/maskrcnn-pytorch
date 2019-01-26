from yacs.config import CfgNode as CN

_C = CN()

# MODEL
_C.MODEL = CN()

# IMAGENET_MODEL_NAME must be selected from this list :
# [vgg16, squeezenet]
# Must change the _C.MODEL.RPN.IN_HEIGHT, WIDTH, CHANNELS when changing model name
_C.MODEL.IMAGENET_MODEL_NAME = "vgg16"
_C.MODEL.IMAGENET_MODEL_PRETRAINED = True

# RPN
_C.MODEL.RPN = CN()
_C.MODEL.RPN.SCALES = [32, 64, 128]
_C.MODEL.RPN.RATIOS = [0.5, 1, 2]
_C.MODEL.RPN.ANCHOR_STRIDE = 1

# RPN: input feature maps height, width, channels. These are different for each pretrained imagenet model
_C.MODEL.RPN.IN_HEIGHT = 32
_C.MODEL.RPN.IN_WIDTH = 32
_C.MODEL.RPN.IN_CHANNELS = 512

_C.MODEL.RPN.MID_CHANNELS = 512

# IMAGE
_C.IMAGE = CN()
# Images are resized such that the smallest side >= MIN_DIM and the longest side <= MAX_DIM
_C.IMAGE.MIN_DIM = 800
_C.IMAGE.MAX_DIM = 1024

# Images are padded so that their sizes will be (MAX_DIM x MAX_DIM).
# Must be True because we create anchors beforehand in the build() function so we can't deal with dynamic size
_C.IMAGE.PADDING = True
