import json
import torch
from PIL import Image

from Datasets.seg_transfo import SegTransformCompose, ToTensor


class BddAnomaly(torch.utils.data.Dataset):
    """ Class to load the BddAnomaly dataset"""

    colors = [[0, 0, 0],  # 0. unlabelled
              [128, 64, 128],  # 1. road
              [244, 35, 232],  # 2. sidewalk
              [70, 70, 70],  # 3. building
              [102, 102, 156],  # 4. wall
              [190, 153, 153],  # 5 fence
              [153, 153, 153],  # 6. pole
              [250, 170, 30],  # 7. traffic_light
              [220, 220, 0],  # 8. traffic_sign
              [107, 142, 35],  # 9. vegetation
              [152, 251, 152],  # 10. terrain
              [0, 130, 180],  # 11. sky
              [220, 20, 60],  # 12. person
              [255, 0, 0],  # 13. rider
              [0, 0, 142],  # 14. car
              [0, 0, 70],  # 15. truck
              [0, 60, 100],  # 16. bus
              [0, 80, 100],  # 17. train
              [0, 0, 230],  # 18. motorcycle
              [119, 11, 32]]  # 19. bicycle

    cmap = dict(zip(range(len(colors)), colors))

    class_name = ["other", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                  "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                  "bicycle"]

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
