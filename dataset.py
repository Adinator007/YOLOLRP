"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""


import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import iou_width_height, non_max_suppression, cells_to_bboxes, plot_image

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # put classes to the and of bb s, for albumentations
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # feltesszuk hogy minden scale en ugyanannyi anchor van, nekunk most 9 anchors in total van
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # azert 6 mert [p_o, x, y, w, h, class label] # ezek a gt bb k. S -> 13, 26, 52 -> az, hogy mekkora a grid
        for box in bboxes: # for each scale, which anchor should be responsible and for which particular cell
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) # calculate iou for this particular box and ALL the anchors
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # first -> best anchor
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # there must be an anchor for each bb in each scale, there is an anchor for each scale for each bb. Vagyis azt akarjuk, hogy minden gt boxot minden scale tudjon predikalni
            for anchor_idx in anchor_indices: # megyunk vegig a legjobban illeszkedo anchor okon es mindig az elso kerul a scale hez
                # which scale
                scale_idx = anchor_idx // self.num_anchors_per_scale # melyik scale hez tartozik az anchor. Pl, ha 8 az anchor_idx akkor 8//3(mert ennyi van egy scale en) az 2 ot ad, ami az utolso scale
                # which anchor on that particular scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # [0, 1, 2]. Ezen az adott scale en melyik anchor hoz rendeljuk ezt a bb t
                S = self.S[scale_idx] # how many cells in this particular scale that we are looking at
                i, j = int(S * y), int(S * x)  # melyik cell a grid ben az amiben benne van a bb kozepe
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # ez az anchor foglalt e mar ez a cella
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # conf ertek
                    # ezek, hogy a cell hez kepest hogyan van ez a bb
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    # ezekre fogunk predikalni exp vel
                    width_cell, height_cell = (
                        width * S, # ezek mindd gt-k
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: # ha ugyanabban a cell ben van egy masik anchor, aminek eleg jo az iou ja, akkor azt disregard oljuk, ezeket nem kell megbuntetni
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        r"D:\Object_detection\PASCAL_VOC\8examples.csv",
        r"D:\Object_detection\PASCAL_VOC\images",
        r"D:\Object_detection\PASCAL_VOC\labels",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = non_max_suppression(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)

if __name__ == "__main__":
    test()
