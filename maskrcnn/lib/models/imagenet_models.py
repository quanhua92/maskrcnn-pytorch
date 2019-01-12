from torchvision import models


def initialize_imagenet_model(model_name, use_pretrained):
    model = None
    if model_name == "vgg16":
        model = models.vgg16(pretrained=use_pretrained)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=use_pretrained)
    return model
