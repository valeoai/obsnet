import os
import torch
import numpy as np
from PIL import Image


class CityScapes(torch.utils.data.Dataset):

    colors = np.array([[128, 64, 128],  # 0: road
                       [244, 35, 232],  # 1: sidewalk
                       [70, 70, 70],  # 2: building
                       [102, 102, 156],  # 3: wall
                       [190, 153, 153],  # 4: fence
                       [153, 153, 153],  # 5: pole
                       [250, 170, 30],  # 6: traffic_light
                       [220, 220, 0],  # 7: traffic_sign
                       [107, 142, 35],  # 8: vegetation
                       [152, 251, 152],  # 9: terrain
                       [0, 130, 180],  # 10: sky
                       [220, 20, 60],  # 11: person
                       [255, 0, 0],  # 12: rider
                       [0, 0, 142],  # 13: car
                       [0, 0, 70],  # 14: truck
                       [0, 60, 100],  # 15: bus
                       [0, 80, 100],  # 16: train
                       [0, 0, 230],  # 17: motorcycle
                       [119, 11, 32],  # 18: bicycle
                       [0, 0, 0]])  # 19: unlabelled

    cmap = dict(zip(range(len(colors)), colors))

    nclass = 19
    class_rate = torch.FloatTensor([1] * nclass)
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_name = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                  "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                   "truck", "bus", "train", "motorcycle", "bicycle", "unlabelled"]

    class_map = dict(zip(valid_classes, range(len(valid_classes))))

    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                            "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
                            "train", "motorcycle", "bicycle", "unlabelled"]

        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.transforms:
            img, lbl = self.transforms(img, lbl)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        lbl = lbl.astype(float)
        return img, lbl

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 19
        # Put all void classes to zero
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    @staticmethod
    def recursive_glob(rootdir=".", suffix=""):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]
