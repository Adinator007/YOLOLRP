# returns the input's grad
from copy import deepcopy
import torch

# module, input_relevance, output_grad -> input_grad
from torch import nn
from torch.nn import ModuleList

from YoloLRP.filter import output_relevance_filter
from model import YOLOv3, CNNBlock, ResidualBlock, ScalePrediction

from Global import grads, activationss

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
        nn.BatchNorm2d: BatchNorm_hook
    }


'''
def CNNBlock_hook(module: CNNBlock, input_relevance, output_relevance):
    eps = 1e-5
    activation = activations[module]
    # copied module
    CNNBlock = deepcopy(module)
    CNNBlock.layer.weight = torch.nn.Parameter(CNNBlock.weight.clamp(min=0.0))
    CNNBlock.layer.bias = torch.nn.Parameter(torch.zeros_like(CNNBlock.layer.bias))
'''



def BatchNorm_hook(module, input_relevance, output_relevance):
    eps = 1e-5
    activation = activationss[module]
    # copied module
    batchnorm = deepcopy(module)
    output_relevance = output_relevance[0]


    running_mean = batchnorm.running_mean
    running_var = batchnorm.running_var

    # output_relevance score ok megszurve
    r = output_relevance_filter(output_relevance, top_k_percent=top_k_percent)
    # elozo aktivaciok ujboli atkuldese es eps vel megtoldasa
    # csatornankent kell ezt megcsinalni

    input_relevance = (r * torch.reshape(batchnorm.weight, (1, r.shape[1], 1, 1)),)

    # input_relevance = r
    return input_relevance


def Conv2d_hook(module, input_relevance, output_relevance) -> torch.Tensor:
    output_relevance = output_relevance[0]  # 1-tuple-be van csomagolva az elozo layer gradiense
    eps = 1e-5
    activation = activationss[module]
    # copied module
    conv = deepcopy(module)
    conv.weight = torch.nn.Parameter(conv.weight.clamp(min=0.0))
    conv.bias = torch.nn.Parameter(torch.zeros_like(conv.bias)) if conv.bias is not None else None

    output_relevance = output_relevance_filter(output_relevance,
                                               top_k_percent=top_k_percent)  # kiveszi a 0.04 edet a output_relevance nak
    z = conv.forward(activation) + eps
    s = (output_relevance / z).data

    z.requires_grad = True
    s.requires_grad = False
    output_relevance.requires_grad = False

    v = torch.sum(z * s)
    v.requires_grad = True

    v.backward(retain_graph=True)
    c = grads[id(activation)]
    r = (activation * c).data
    input_relevance = (r, )
    return input_relevance


def Linear_hook(module, input_relevance, output_relevance) -> torch.Tensor:
    eps = 1e-5
    activation = activationss[module]
    # copied module
    linear = deepcopy(module)
    linear.weight = torch.nn.Parameter(linear.weight.clamp(min=0.0))
    linear.bias = torch.nn.Parameter(torch.zeros_like(linear.bias))
    output_relevance = output_relevance[0]  # 1-tuple-be van csomagolva az elozo layer gradiense

    r = output_relevance_filter(output_relevance, top_k_percent=top_k_percent)
    z = linear.forward(activation) + eps
    s = r / z
    c = torch.mm(s, linear.weight)
    input_relevance = (activation * c).data
    return input_relevance

def NonLinearity(module, input_relevance, output_relevance) -> torch.Tensor:
    input_relevance = output_relevance
    return input_relevance


def Dropout_hook(module, input_relevance, output_relevance):
    input_relevance = output_relevance
    return input_relevance

def Identity_hook(module, input_relevance, output_relevance):
    input_relevance = output_relevance
    return input_relevance


def AdaptiveAvgPool2D_hook(module, input_relevance, output_relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    avgpool2d = deepcopy(module)
    avgpool2d.layer.weight = torch.nn.Parameter(avgpool2d.weight.clamp(min=0.0))
    avgpool2d.layer.bias = torch.nn.Parameter(torch.zeros_like(avgpool2d.layer.bias))

    z = module.forward(activation) + eps
    s = (output_relevance / z).data
    # az adott szinten az osszrelevance z * s
    (z * s).sum().backward()
    c = activation.grad
    r = (activation * c).data
    input_relevance = r
    return input_relevance


def AvgPool2D_hook(module, input_relevance, output_relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    avgpool2d = deepcopy(module)
    avgpool2d.layer.weight = torch.nn.Parameter(avgpool2d.weight.clamp(min=0.0))
    avgpool2d.layer.bias = torch.nn.Parameter(torch.zeros_like(avgpool2d.layer.bias))

    z = module.forward(activation) + eps
    s = (output_relevance / z).data
    # az adott szinten az osszrelevance z * s
    (z * s).sum().backward()
    c = activation.grad
    r = (activation * c).data
    input_relevance = r
    return input_relevance




def MaxPool2D_hook(module, input_relevance, output_relevance):
    global activations
    eps = 1e-5
    activation = activations[module]
    # copied module
    avgpool2d = deepcopy(module)
    avgpool2d.layer.weight = torch.nn.Parameter(avgpool2d.weight.clamp(min=0.0))
    avgpool2d.layer.bias = torch.nn.Parameter(torch.zeros_like(avgpool2d.layer.bias))

    z = module.forward(activation) + eps
    s = (output_relevance / z).data
    # az adott szinten az osszrelevance z * s
    (z * s).sum().backward()
    c = activation.grad
    r = (activation * c).data
    input_relevance = r
    return input_relevance



