from netron import onnx
from torch import nn
import onnx

import torch
import config
from LRPBackwardHooks import lookup_hook_fn
from YoloLRP.lrp import LRPModel
from model import YOLOv3

from Global import activations


lista = []

def store_activations(x, y, z): # maga a modell, a bemenete a layer nek
    lista.append(y[0])
    activations[x] = y[0]

def b_hook(x, y, z):
    return torch.tensor([[99, 98, 97, 96, 95]]).float()
    # return torch.tensor([95]).float()


def main():
    from torch.profiler import profile, record_function, ProfilerActivity

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.train()

    #torch torch.save(model, "bestModel.pth")
    '''torch.onnx.export(model,  # model being run
                      torch.zeros(2, 3, 416, 416).cuda(),  # model input (or a tuple for multiple inputs)
                      "model.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    onnx_model = onnx.load("bestModel.pth")
    onnx.checker.check_model(onnx_model)

    return
    '''

    input = torch.ones(5)
    input.requires_grad = True


    lut = lookup_hook_fn()
    for name, module in model.named_modules():
        module.register_forward_hook(store_activations)
        if module.__class__ in lut.keys():
            module.register_full_backward_hook(lut[module.__class__])

    # optimizer = nn.SGD()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):

            # net.weight = nn.Parameter(torch.zeros_like(net.weight))
            yoloInput = torch.ones(2, 3, 416, 416).float().to(config.DEVICE)
            # yoloInput.requires_grad = True
            res = model(yoloInput)

            global activations
            activations = [activations[a].data.requires_grad_(True) for a in list(activations)]
            # lista = [a.requires_grad_]

            # torch.autograd.gradcheck(model, yoloInput)
            torch.sum(res[0][:, :, :, :]).backward(retain_graph=True)
            # optimizer.step()
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
    lin1.register_full_backward_hook(b_hook)
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
    # net.register_full_backward_hook(b_hook)
    a = nn.Parameter(torch.ones(5))
    sol = net(a)
    sol.sum().backward()

    print(a.grad)

    '''


def main2():
    def backward_hook(m, input_gradients, output_gradients):
        print('input_gradients {}'.format(input_gradients))
        print('output_gradients {}'.format(output_gradients))
        input_gradients = (torch.ones_like(input_gradients[0]),)
        return input_gradients


    '''
    l = nn.Linear(2, 20)
    l.register_full_backward_hook(backward_hook)
    a = nn.Parameter(torch.randn(2))
    a = nn.Parameter(torch.tensor([3, 4]))
    out = l(a)
    out.mean().backward()


    '''
    conv = nn.Conv2d(1, 1, 3, bias = False)
    conv.register_full_backward_hook(backward_hook)

    x = torch.randn(1, 1, 3, 3).requires_grad_(True)
    out = conv(x)
    out.mean().backward()
    print(x.grad.shape)  # ones

if __name__ == '__main__':
    main()