import json
import torch
from PIL import Image

from Datasets.seg_transfo import SegTransformCompose, ToTensor


class BddAnomaly(torch.utils.data.Dataset):
    """ Class to load the BddAnomaly dataset"""

    cmap = [[128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic_light
            [220, 220, 0],    # traffic_sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [0, 130, 180],    # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
            [0, 0, 0]]        # unlabelled

    cmap = dict(zip(range(len(cmap)), cmap))

    class_name = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                   "bicycle", "other"]

    def __init__(self, imgFolder, split, transforms=SegTransformCompose(ToTensor())):
        super(BddAnomaly, self).__init__()
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

        return imgx, imgy
