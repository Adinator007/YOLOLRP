from torch import nn

import torch
import config
from LRPBackwardHooks import lookup_hook_fn
from YoloLRP.lrp import LRPModel
from model import YOLOv3

from Global import activations

def store_activations(x, y, z): # maga a modell, a bemenete a layer nek
    activations[x] = y[0]

def b_hook(x, y, z):
    return torch.tensor([[99, 98, 97, 96, 95]]).float()
    # return torch.tensor([95]).float()


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    input = torch.ones(5)
    input.requires_grad = True


    lut = lookup_hook_fn()
    for name, module in model.named_modules():
        module.register_forward_hook(store_activations)
        if module.__class__ in lut.keys():
            module.register_full_backward_hook(lut[module.__class__])

    # net.weight = nn.Parameter(torch.zeros_like(net.weight))
    yoloInput = torch.ones(2, 3, 416, 416).float().cuda()
    res = model(yoloInput)
    res[0][:, :, 2, 2].sum().backward()
    print("ok")


    '''
    for name, module in net.named_children():
        if 'Linear' in module.__class__.__name__:
            # module.register_forward_hook(hook_fn)
            module.register_backward_hook(b_hook)
    '''

    # model.register_forward_hook(hook_fn)
    # lrp_model = LRPModel(model=model)

    '''
    lin1 = nn.Linear(5, 10)
    lin2 = nn.Linear(10, 20, bias=False)
    lin3 = nn.Linear(20, 20, bias=False)
    lin4 = nn.Linear(20, 30, bias=False)
    net = nn.Sequential(
        lin1,
        lin2,
        nn.Sequential(
            lin3,
            lin4
        )
    )
    net.register_full_backward_hook(b_hook)
    a = torch.ones(10)
    sol = net(a)

    sol.sum().backward()
    '''


def main2():
    def backward_hook(m, input_gradients, output_gradients):
        print('input_gradients {}'.format(input_gradients))
        print('output_gradients {}'.format(output_gradients))
        input_gradients = (torch.ones_like(input_gradients[0]),)
        return input_gradients

    conv = nn.Conv2d(1, 1, 3)
    conv.register_full_backward_hook(backward_hook)

    x = torch.randn(1, 1, 3, 3).requires_grad_()
    out = conv(x)
    out.mean().backward()
    print(x.grad.shape)  # ones


if __name__ == '__main__':
    main()