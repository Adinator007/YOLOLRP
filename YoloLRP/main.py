import time

import torch.cuda

import config
from YoloLRP.lrp import LRPModel
from YoloLRP.visualize import plot_relevance_scores
from model import YOLOv3
from utils import get_loaders


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    lrp_model = LRPModel(model=model)

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        # y = y.to(device)  # here not used as method is unsupervised.

        t0 = time.time()
        r = lrp_model.forward(x)
        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

        plot_relevance_scores(x=x, r=r, name=str(i), config=config)

if __name__ == '__main__':
    main()