import os
import torch
import numpy as np
from PIL import Image

from Datasets.seg_transfo import SegTransformCompose, ToTensor


class WoodScape(torch.utils.data.Dataset):
    colors = np.array([
         [  0,   0,   0],   # "void"
         [128,  64, 128],   # "road",
         [ 69,  76,  11],   # "lanemarks",
         [  0, 255,   0],   # "curb",
         [220,  20,  60],   # "person",
         [255,   0,   0],   # "rider",
         [  0,   0, 142],   # "vehicles",
         [119,  11,  32],   # "bicycle",
         [  0,   0, 230],   # "motorcycle",
         [220, 220,   0],   # "traffic_sign",
    ])

    cmap = dict(zip(range(len(colors)), colors))
    class_name = ["void", "road", "lanemarks", "curb", "person", "rider", "vehicles",
                   "bicycle", "motorcycle", "traffic_sign"]
    class_rate = torch.FloatTensor([0.08740, 0.02362, 0.37615, 0.9003, 2.15570,
                                    3.03690, 0.15685, 1.00000, 2.7519, 7.5890])

    def __init__(self, folder, split, transforms=SegTransformCompose(ToTensor())):
        super(WoodScape, self).__init__()
        self.folder = folder
        self.split = split
        with open(os.path.join(self.folder, split + ".txt")) as f:
            self.img_set = np.array([l.split() for l in f.readlines()])
        self.x = self.img_set[:, 0]
        self.y = self.img_set[:, 1]

        self.transforms = transforms

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, id):
        imgx = Image.open(self.x[id])
        imgy = Image.open(self.y[id])
        imgx, imgy = self.transforms(imgx, imgy)
        return imgx, imgy
