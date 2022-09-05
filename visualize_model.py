from tensorboardX import SummaryWriter
import torch

# from ML.Pytorch.object_detection.YOLOv3 import config
from torch import nn

import tensorboard

from model import YOLOv3, ResidualBlock


def main():
    writer = SummaryWriter()
    net = YOLOv3(num_classes=20)
    input = torch.randn(10, 3, 416, 416)

    a = nn.Conv2d(3, 3, 3, 3)

    # [ a for a in net.named_modules()], net.layers[10].layers[0][1]
    # [name for name in net.named_parameters()] ez egy tuple, benne van a layer neve es parameterei

    # [f for f in net.named_modules()][10][1]
    # 10. elem, 0 -> a modulnak a neve
    # 1 -> maga a modul. Az is lehet nested

    writer.add_graph(net, input)
    writer.close()


def main2():
    a = ResidualBlock(10, True, 2)
    print(len([f for f in a.named_modules()]))
    # a = nn.BatchNorm2d(10)
    # print(a.state_dict())

if __name__ == '__main__':
    main2()