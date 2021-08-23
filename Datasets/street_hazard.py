import json
import torch
from PIL import Image

from Datasets.seg_transfo import SegTransformCompose, ToTensor


class StreetHazard(torch.utils.data.Dataset):
    """ Class to load the Streethazards dataset"""

    color = [[  0,   0,   0],                  # unlabeled
             [ 70,  70,  70],                  # building
             [190, 153, 153],                  # fence
             [250, 170, 160],                  # other
             [220,  20,  60],                  # pedestrian
             [153, 153, 153],                  # pole
             [157, 234,  50],                  # road line
             [128,  64, 128],                  # road
             [244,  35, 232],                  # sidewalk
             [107, 142,  35],                  # vegetation
             [  0,   0, 142],                  # car
             [102, 102, 156],                  # wall
             [220, 220,   0],                  # traffic sign
             [ 60, 250, 240],                  # anomaly
           ]

    cmap = dict(zip(range(len(color)), color))

    class_name = ["unlabeled", "building", "fence", "other", "pedestrian", "pole", "street line", "road",
                    "side walk", "vegetation", "vehicule", "wall", "trafic sign", "anomaly"]

    def __init__(self, imgFolder, split, transforms=SegTransformCompose(ToTensor())):
        super(StreetHazard, self).__init__()
        self.imgFolder = imgFolder
        self.split = split
        f = imgFolder + split + ".odgt"
        self.img_set = [json.loads(x.rstrip()) for x in open(f, 'r')][0]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, i):
        filex = self.imgFolder + self.img_set[i]["fpath_img"]
        filey = self.imgFolder + self.img_set[i]["fpath_segm"]

        imgx = Image.open(filex).convert('RGB')
        imgy = Image.open(filey)
        imgx, imgy = self.transforms(imgx, imgy)
        imgy -= 1                                   # shift label from 1-15 to 0-14
        return imgx, imgy
