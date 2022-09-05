# returns the input's grad
from copy import deepcopy
import torch

# module, weight_grad, output_grad -> input_grad
from torch import nn
from torch.nn import ModuleList

from YoloLRP.filter import relevance_filter
from model import YOLOv3, CNNBlock, ResidualBlock, ScalePrediction

from Global import activations

top_k_percent = 0.04




def lookup_hook_fn():
    return {
        nn.Linear: Linear_hook,
        nn.Conv2d: Conv2d_hook,
        nn.ReLU: NonLinearity,
        nn.LeakyReLU: NonLinearity,
        nn.Dropout: Dropout_hook,
        nn.AdaptiveAvgPool2d: AdaptiveAvgPool2D_hook,
        nn.MaxPool2d: MaxPool2D_hook,
        nn.BatchNorm2d: BatchNorm_hook,
        YOLOv3: Identity_hook,
        # CNNBlock: CNNBlock_hook,
        ResidualBlock: Identity_hook,
        ModuleList: Identity_hook,
        nn.Sequential: Identity_hook,
        ScalePrediction: Identity_hook,
        nn.Upsample: Identity_hook
    }


def CNNBlock_hook(module: CNNBlock, weight_grad, relevance):
    eps = 1e-5
    activation = activations[module]
    # copied module
    CNNBlock = deepcopy(module)
    CNNBlock.layer.weight = torch.nn.Parameter(CNNBlock.weight.clamp(min=0.0))
    CNNBlock.layer.bias = torch.nn.Parameter(torch.zeros_like(CNNBlock.layer.bias))




def BatchNorm_hook(module, weight_grad, relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    batchnorm = deepcopy(module)
    batchnorm.layer.weight = torch.nn.Parameter(batchnorm.weight.clamp(min=0.0))
    batchnorm.layer.bias = torch.nn.Parameter(torch.zeros_like(batchnorm.layer.bias))
    relevance = relevance[0]


    running_mean = batchnorm.running_mean
    running_var = batchnorm.running_var

    # relevance score ok megszurve
    r = relevance_filter(relevance, top_k_percent=top_k_percent)
    # elozo aktivaciok ujboli atkuldese es eps vel megtoldasa
    z = batchnorm.forward(activation) + eps
    s = r / z

    c = torch.mm(s, batchnorm.weight)
    r = (activation * c).data

    r = r * torch.rsqrt(running_var + eps)
    # r = r + self.running_mean # ezzel nem kell foglalkozni, mert a bias nem modositja a reteg erzekenyseget a kimenetre

    return r


def Conv2d_hook(module, weight_grad, relevance) -> torch.Tensor:
    global activations
    relevance = relevance[0] # 1-tuple-be van csomagolva az elozo layer gradiense
    eps = 1e-5
    activation = activations[module]
    # copied module
    conv = deepcopy(module)
    conv.weight = torch.nn.Parameter(conv.weight.clamp(min=0.0))
    conv.bias = torch.nn.Parameter(torch.zeros_like(conv.bias))

    relevance = relevance_filter(relevance, top_k_percent=top_k_percent)  # kiveszi a 0.04 edet a relevance nak
    z = conv.forward(activation) + eps
    s = (relevance / z).data

    z.requires_grad = True
    s.requires_grad = False
    relevance.requires_grad = False

    torch.sum(z * s).backward()
    c = activation.grad
    r = (activation * c).data
    return r



def Linear_hook(module, weight_grad, relevance) -> torch.Tensor:
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    linear = deepcopy(module)
    linear.weight = torch.nn.Parameter(linear.weight.clamp(min=0.0))
    linear.bias = torch.nn.Parameter(torch.zeros_like(linear.bias))
    relevance = relevance[0]  # 1-tuple-be van csomagolva az elozo layer gradiense

    r = relevance_filter(relevance, top_k_percent=top_k_percent)
    z = linear.forward(activation) + eps
    s = r / z
    c = torch.mm(s, linear.weight)
    r = (activation * c).data
    return r

def NonLinearity(module, weight_grad, relevance) -> torch.Tensor:
    return relevance


def Dropout_hook(module, weight_grad, relevance):
    return relevance

def Identity_hook(module, weight_grad, relevance):
    return relevance


def AdaptiveAvgPool2D_hook(module, weight_grad, relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    avgpool2d = deepcopy(module)
    avgpool2d.layer.weight = torch.nn.Parameter(avgpool2d.weight.clamp(min=0.0))
    avgpool2d.layer.bias = torch.nn.Parameter(torch.zeros_like(avgpool2d.layer.bias))

    z = module.forward(activation) + eps
    s = (relevance / z).data
    # az adott szinten az osszrelevance z * s
    (z * s).sum().backward()
    c = activation.grad
    r = (activation * c).data
    return r


def AvgPool2D_hook(module, weight_grad, relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    avgpool2d = deepcopy(module)
    avgpool2d.layer.weight = torch.nn.Parameter(avgpool2d.weight.clamp(min=0.0))
    avgpool2d.layer.bias = torch.nn.Parameter(torch.zeros_like(avgpool2d.layer.bias))

    z = module.forward(activation) + eps
    s = (relevance / z).data
    # az adott szinten az osszrelevance z * s
    (z * s).sum().backward()
    c = activation.grad
    r = (activation * c).data
    return r




def MaxPool2D_hook(module, weight_grad, relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    avgpool2d = deepcopy(module)
    avgpool2d.layer.weight = torch.nn.Parameter(avgpool2d.weight.clamp(min=0.0))
    avgpool2d.layer.bias = torch.nn.Parameter(torch.zeros_like(avgpool2d.layer.bias))

    z = module.forward(activation) + eps
    s = (relevance / z).data
    # az adott szinten az osszrelevance z * s
    (z * s).sum().backward()
    c = activation.grad
    r = (activation * c).data
    return r



