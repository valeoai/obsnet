import os
import torch
import torch.nn as nn
from Models.segnet import SegNet
from Models.obsnet import Obsnet_Seg, Obsnet_Small
from Models.deeplab_v3plus import deeplab_v3plus


def net_loader(args):
    """ load the observer network and the segmentation network """

    if args.model == "segnet":
        segnet = SegNet(3, args.nclass, init_vgg=False).to(args.device)
        if args.obs_mlp:
            obsnet = Obsnet_Small(input_channels=512, output_channels=1).to(args.device)
        else:
            obsnet = Obsnet_Seg(input_channels=3, output_channels=1).to(args.device)

    elif args.model == "deeplabv3plus":
        segnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True).to(args.device)
        obsnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True, obsnet=True).to(args.device)
    else:
        raise NameError("Model not known")

    segnet.load_state_dict(torch.load(args.segnet_file))
    segnet.eval()

    if args.test_only and "obsnet" in args.test_multi:
        obsnet.load_state_dict(torch.load(os.path.join(args.obsnet_file, "best.pth")))
        obsnet.eval()

    if not args.test_only:
        segnet = nn.DataParallel(segnet)
        obsnet = nn.DataParallel(obsnet)

    return obsnet, segnet

