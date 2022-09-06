# import wandb
from PIL.Image import Image
from tensorboardX import SummaryWriter
import torch

# from ML.Pytorch.object_detection.YOLOv3 import config
from torch import nn

import tensorboard
from torch.utils.data import DataLoader

import config
from dataset import YOLODataset
from model import YOLOv3, ResidualBlock, IMAGE_SIZE
from utils import get_loaders

classes = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}

# this is the order in which my classes will be displayed
display_ids = {"car" : 0, "truck" : 1, "person" : 2, "traffic light" : 3, "stop sign" : 4,
               "bus" : 5, "bicycle": 6, "motorbike" : 7, "parking meter" : 8, "bench": 9,
               "fire hydrant" : 10, "aeroplane" : 11, "boat" : 12, "train": 13}
# this is a revese map of the integer class id to the string class label
class_id_to_label = { int(v) : k for k, v in classes.items()}

def bounding_boxes(raw_image, v_boxes, v_labels, v_scores, log_width, log_height):
    # load raw input photo
    # raw_image = load_img(filename, target_size=(log_height, log_width))

    loader, test_loader = get_loaders(r"D:\Object_detection\PASCAL_VOC\8examples.csv", r"D:\Object_detection\PASCAL_VOC\8examples.csv")

    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        box_data = {"position" : {
          "minX" : box.xmin,
          "maxX" : box.xmax,
          "minY" : box.ymin,
          "maxY" : box.ymax},
          "class_id" : display_ids[v_labels[b_i]],
          # optionally caption each box with its class and score
          "box_caption" : "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
          "domain" : "pixel",
          "scores" : { "score" : v_scores[b_i] }}
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(
        raw_image,
        boxes = {
            "predictions": {
                "box_data": all_boxes,
                "class_labels" : class_id_to_label
            }
        }
    )

    return box_image



def main():
    writer = SummaryWriter()
    net = YOLOv3(num_classes=20)
    input = torch.randn(10, 3, 416, 416)

    # a = nn.Conv2d(3, 3, 3, 3)

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