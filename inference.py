import glob
import time

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import numpy as np
from Models.segnet import SegNet
from Models.obsnet import Obsnet_Seg as ObsNet
from Models.deeplab_v3plus import deeplab_v3plus

def img2tensor(file, args):
    img = Image.open(file)
    img = img.resize((args.w, args.h), Image.BILINEAR)
    img = tf.to_tensor(img)
    img = tf.normalize(img, args.mean, args.std, False)
    img = img.unsqueeze(0).to(args.device)
    return img


def test(args):

    if args.model == "segnet":
        segnet = SegNet(3, args.nclass, init_vgg=False).to(args.device)
        obsnet = ObsNet(input_channels=3, output_channels=1).to(args.device)

    elif args.model == "deeplabv3plus":
        segnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True).to(args.device)
        obsnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True, obsnet=True).to(args.device)

    segnet.load_state_dict(torch.load(args.segnet_file))
    segnet.eval()

    obsnet.load_state_dict(torch.load(args.obsnet_file))
    obsnet.eval()

    total_time = 0
    with torch.no_grad():
        for i, file in enumerate(args.imgs):
            img = img2tensor(file, args)

            # Inference
            start = time.time()
            seg_pred, obs_pred = _inference(img, segnet, obsnet)
            total_time += time.time() - start
            img = (img - img.min()) / (img.max() - img.min())
            # Post processing
            seg_pred = torch.argmax(seg_pred, dim=1)
            seg_pred = Image.fromarray(seg_pred[0].byte().cpu().numpy()).resize((args.w, args.h))
            seg_pred.putpalette(args.colors.astype("uint8"))
            seg_pred = Image.blend(transforms.ToPILImage()(img[0]), seg_pred.convert("RGB"), alpha=0.5)

            obs_pred = (obs_pred - obs_pred.min()) / (obs_pred.max() - obs_pred.min())
            obs_pred = obs_pred * args.yellow + (1 - obs_pred) * args.blue
            # obs_pred = F.interpolate(obs_pred, (args.h//50, args.w//50))
            # obs_pred = torch.where(obs_pred < 0.2, args.zero, obs_pred)
            # obs_pred = F.interpolate(obs_pred, (args.h, args.w), mode='bilinear', align_corners=False)
            obs_pred = Image.blend(transforms.ToPILImage()(img[0]), transforms.ToPILImage()(obs_pred[0]), alpha=0.9)

            # Concat and Save visualization
            res = Image.new('RGB', (3 * args.w, args.h))
            res.paste(transforms.ToPILImage()(img[0]), (0, 0))
            res.paste(seg_pred, (args.w, 0))
            res.paste(obs_pred, (2 * args.w, 0))
            res.save(f'./save_img/img_{i:03d}.png')

            print(f"Test: {i/len(args.imgs)*100:.1f} %")
    print(f"Inference run time: {len(args.imgs)/total_time:.1f} FPS")


def _inference(img, segnet, obsnet):
    seg_feat = segnet(img, return_feat=True)
    obs_pred = torch.sigmoid(obsnet(img, seg_feat)).squeeze()
    return seg_feat[-1], obs_pred


if __name__ == '__main__':
    class Arguments:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.segnet_file = "/root/workspace/Projects/Stuff/Git_repo/Segmentation/ckpt/Bdd/DeepLabv3plus.pth"
            self.obsnet_file = "./ckpt/BddAnomaly/best.pth"
            self.imgs = glob.glob("/datasets_master/dataset_RoadAnomalyTrack/images/*")
            self.data = "BddAnomaly"
            self.model = "deeplabv3plus"
            self.one = torch.FloatTensor([1.]).to(self.device)
            self.zero = torch.FloatTensor([0.]).to(self.device)

    args = Arguments()

    if args.data == "CamVid":
        args.mean = [0.4108, 0.4240, 0.4316]
        args.std = [0.3066, 0.3111, 0.3068]
        args.h, args.w = [360, 480]
        args.nclass = 12
        args.colors = np.array([
            [128, 128, 128],  # sky
            [128, 0, 0],  # building
            [192, 192, 128],  # column_pole
            [128, 64, 128],  # road
            [0, 0, 192],  # sidewalk
            [128, 128, 0],  # Tree
            [192, 128, 128],  # SignSymbol
            [64, 64, 128],  # Fence
            [64, 0, 128],  # Car
            [64, 64, 0],  # Pedestrian
            [0, 128, 192],  # Bicyclist
            [0, 0, 0],  # Void
        ])

    elif args.data == "BddAnomaly":
        args.mean = [0.3698, 0.4145, 0.4247]
        args.std = [0.2525, 0.2695, 0.2870]
        args.h, args.w = [360, 640]  # Original size [720, 1280]
        args.nclass = 19
        args.colors = np.array([
            [128, 64, 128],  # road
            [244, 35, 232],  # sidewalk
            [70, 70, 70],  # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],  # traffic_light
            [220, 220, 0],  # traffic_sign
            [107, 142, 35],  # vegetation
            [152, 251, 152],  # terrain
            [0, 130, 180],  # sky
            [220, 20, 60],  # person
            [255, 0, 0],  # rider
            [0, 0, 142],  # car
            [0, 0, 70],  # truck
            [0, 60, 100],  # bus
            [0, 80, 100],  # train
            [0, 0, 230],  # motorcycle
            [119, 11, 32],  # bicycle
            [0, 0, 0]])  # unlabelled
    else:
        raise NameError("Data not known")
    args.cmap = dict(zip(range(len(args.colors)), args.colors))
    args.yellow = torch.FloatTensor([1, 1, 0]).to(args.device).view(1, 3, 1, 1).expand(1, 3, args.h, args.w)
    args.blue = torch.FloatTensor([0, 0, .4]).to(args.device).view(1, 3, 1, 1).expand(1, 3, args.h, args.w)
    test(args)

