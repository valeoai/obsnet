import os
import numpy as np
import torch
from PIL import Image
import cv2
import base64
import json


class CEA(torch.utils.data.Dataset):
    colors = np.array([
        [0, 0, 255],      # Road
        [0, 255, 0],      # Curb
        [255, 0, 0],      # Sidewalk
        [127, 127, 127],  # Background
        [50, 50, 50],     # Void
        [0, 50, 255],     # Parking space
        [69, 76, 11],     # Lane
        [89, 96, 103],    # Stop yield
        [76, 11, 103],    # Cross_walk
        [89, 96, 50],     # Other
        [89, 255, 103],   # Text
        [0, 255, 255]     # Object
    ])

    cmap = dict(zip(range(len(colors)), colors))
    class_names = ["road", "curb", "sidewalk", "background", "void", "parking_space",
                   "lane", "stop_yield", "cross_walk", "other", "text", "object"]
    class_rate = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def __init__(self, root_folder, split="train", transforms=None):
        self.root_folder = root_folder
        self.transforms = transforms
        act_split = "train" if split == "val" else split
        with open(os.path.join(root_folder, "segmentation_"+act_split+".txt")) as f:
            self.img_set = np.array([l.split() for l in f.readlines()])

        if split == 'train':
            self.img_set = self.img_set[:int(len(self.img_set) * 0.8)]
        elif split == "val":
            self.img_set = self.img_set[int(len(self.img_set) * 0.8):]

        self.mapping_id = np.array(
        [3  , 6  , 6  ,11  , 7  , 4  , 1  , 2  , 1  , 0  , 3  , 7  , 4  , 7  , 7  , 6  , 6  , 5  ,
         5  , 8  , 9  , 9  , 9  , 9  , 8  , 8  , 0  , 8  , 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255,   3, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255,   3], dtype=np.uint8)

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, idx):
        img_path, annot_path, ind_seq = self.img_set[idx]
        img = Image.open(os.path.join(self.root_folder, img_path))

        seg_json = open(os.path.join(self.root_folder, annot_path))
        seg_json = json.load(seg_json)

        mask_encoded = seg_json["annotations"][int(ind_seq)]["mask"].replace('data:image/png;base64,', '')
        mask_decoded = base64.b64decode(mask_encoded)
        mask_decoded = np.fromstring(mask_decoded, dtype='uint8')
        mask_decoded = cv2.imdecode(mask_decoded, cv2.IMREAD_UNCHANGED)[:, :, 0]

        labels = self.mapping_id[mask_decoded]
        labels = labels.reshape(mask_decoded.shape)
        labels = Image.fromarray(labels)

        img, labels = self.transforms(img, labels)
        return img, labels
