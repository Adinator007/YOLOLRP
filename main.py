from netron import onnx
from torch import nn
import onnx

import torch
import config
from LRPBackwardHooks import lookup_hook_fn
from YoloLRP.lrp import LRPModel
from model import YOLOv3, CNNBlock

from Global import grads, activationss
from utils import eval_lrp, get_loaders, original_get_evaluation_bboxes, plot_image, plot_couple_examples2, \
    load_checkpoint


def store_activations(x, y, z): # maga a modell, a bemenete a layer nek
    activationss[x] = y[0]

def b_hook(x, y, z):
    return torch.tensor([[99, 98, 97, 96, 95]]).float()
    # return torch.tensor([95]).float()


IMAGE_SIZE = (416, 416)


class YOLOBox:
    def __init__(self, t, mode='YOLO'):
        if mode == 'YOLO':
            x, y, w, h = t
            x = x*IMAGE_SIZE[1]
            y = y*IMAGE_SIZE[0]
            w = w*IMAGE_SIZE[0]
            h = h*IMAGE_SIZE[1]

            self.xmin = x - w/2
            self.xmax = x + w/2
            self.ymin = y - h/2
            self.ymax = y + h/2

        elif mode == 'COCO':
            self.xmin = t[0]
            self.ymin = t[1]
            self.xmax = t[2]
            self.ymax = t[3]
        else:
            print('Invalid model parameter')

idx = 0

def main():
    def save_grad(activation):
        def hook(grad):
            grads[id(activation)] = grad

        return hook

    '''
    # conv = nn.Conv2d(10, 20, 3, 1)
    f = CNNBlock(10, 20, kernel_size=3)

    lut = lookup_hook_fn()
    for name, module in f.named_modules():
        module.register_forward_hook(store_activations)
        if module.__class__ in lut.keys():
            module.register_full_backward_hook(lut[module.__class__])

    # x = torch.randn(1, 10, 30, 30).requires_grad_(True)
    x = torch.randn(1, 10, 30, 30).requires_grad_(True)
    out = f(x)

    for activation in activationss.values():
        activation.register_hook(save_grad(activation))
    out.mean().backward()
    print(grads[id(out)].shape)

    return
    '''

    from torch.profiler import profile, record_function, ProfilerActivity

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.train()

    '''
    loader, _, _ = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")
    imgs, boxes = eval_lrp(model, loader)

    classes = [torch.tensor([1])]

    import wandb
    wandb.init("yolo_visualization")

    table = wandb.Table(columns=["id", "image_and_boxes"])

    def bounding_boxes(raw_image, v_boxes, v_clsids, v_scores, log_width=1024, log_height=1024):
        global idx
        # load raw input photo
        all_boxes = []
        # plot each bounding box for this image
        for b_i, box in enumerate(v_boxes):
            # get coordinates and labels
            box_data = {
                "position": {
                    "minX": int(box.xmin.item()),
                    "maxX": int(box.xmax.item()),
                    "minY": int(box.ymin.item()),
                    "maxY": int(box.ymax.item())
                },
                "class_id": int(v_clsids[b_i].item()),
                # optionally caption each box with its class and score
                "box_caption": "%s (%.3f)" % (v_clsids[b_i], v_scores[b_i] if v_scores is not None else 1),
                "domain": "pixel",
                # "scores": {"score": v_scores[b_i].item()}
                "scores": {"score": 1}
            }
            all_boxes.append(box_data)

        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image,
                                boxes={"predictions": {"box_data": all_boxes, "class_labels": {k: v for k, v in enumerate(config.PASCAL_CLASSES)}}})

        # ID -> LABEL

        table.add_data(idx, box_image)
        idx += 1

        return box_image

    def visualize_wandb(imgs, boxes, classes, scoress):
        if scoress is not None:
            for img, bbs, clss, scores in zip(imgs, boxes, classes, scoress):
                bounding_boxes(img, bbs, clss, scores)
        else:
            for img, bbs, clss in zip(imgs, boxes, classes):
                bounding_boxes(img, bbs, clss, None)

    bboxes = []

    for box in boxes:
        bboxes.append(
            [YOLOBox(
                (box[3].cpu(), box[4].cpu(), box[5].cpu(), box[6].cpu()),
                mode='YOLO'
            )] # TODO rewrite for generic bb
        )

    visualize_wandb(imgs, bboxes, classes, None) # imgs, boxes, classes, scoress

    wandb.log(
        {
            "table": table
        }
    )

    wandb.finish()

    return
    '''

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
            yoloInput = nn.Parameter(torch.ones(2, 3, 416, 416).float().to(config.DEVICE))
            # yoloInput.requires_grad = True
            res = model(yoloInput)

            # activationss = [activationss[a].data.requires_grad_(True) for a in list(activationss)]
            for activation in activationss.values():
                activation.register_hook(save_grad(activation))
            '''
            lista = [a.data.requires_grad_(True) for a in lista]
            '''
            # lista = [a.requires_grad_]

            # torch.autograd.gradcheck(model, yoloInput)
            torch.sum(res[0][:, :, :, :]).backward(retain_graph=True)
            # optimizer.step()
            # print("ok")




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

    def save_grad(activation):
        def hook(grad):
            grads[activation] = grad

        return hook


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
    # conv.register_full_backward_hook(backward_hook)

    x = torch.randn(1, 1, 3, 3).requires_grad_(True)
    out = conv(x)
    out.register_hook(save_grad("out"))
    out.mean().backward()
    print(grads["out"].shape)  # ones



def main3():
    loader, _, _ = get_loaders(train_csv_path=config.DATASET + "/1examples.csv", test_csv_path=config.DATASET + "/1examples.csv")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    pred_boxes, true_boxes, images = original_get_evaluation_bboxes(
        loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )

    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    plot_couple_examples2(model, images[0], 0.6, 0.5, scaled_anchors)


def main4():

    def save_grad(activation):
        def hook(grad):
            grads[id(activation)] = grad

        return hook

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    load_checkpoint(config.CHECKPOINT_FILE, model, None, None)  # checkpoint_file, model, optimizer, lr

    lut = lookup_hook_fn()
    for name, module in model.named_modules():
        module.register_forward_hook(store_activations)
        if module.__class__ in lut.keys():
            module.register_full_backward_hook(lut[module.__class__])

    yoloInput = nn.Parameter(torch.ones(2, 3, 416, 416).float().to(config.DEVICE))
    res = model(yoloInput)
    for activation in activationss.values():
        activation.register_hook(save_grad(activation))
    torch.sum(res[0][:, :, :, :]).backward(retain_graph=True)


if __name__ == '__main__':
    main4()