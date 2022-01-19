from torch.utils.data import DataLoader

from Datasets.camvid import CamVid
from Datasets.cea import CEA
from Datasets.street_hazard import StreetHazard
from Datasets.BDD_anomaly import BddAnomaly
from Datasets.woodscape import WoodScape
from Datasets.seg_transfo import HFlip, ToTensor, Resize, RandomCrop, SegTransformCompose
from Datasets.seg_transfo import Normalize, AdjustContrast, AdjustBrightness, AdjustSaturation


def data_loader(args):
    """ Load the dataloard corresponding to the args.data parameters
        args -> Argparse: global arguments"""
    t_train = SegTransformCompose(RandomCrop(args.crop), Resize(size=args.size), HFlip(0.5),
                                  AdjustBrightness(0.5), AdjustContrast(0.5),
                                  AdjustSaturation(0.5), ToTensor(),
                                  Normalize(mean=args.mean, std=args.std)
                                  )

    t_eval = SegTransformCompose(Resize(size=args.size),
                                 ToTensor(),
                                 Normalize(mean=args.mean, std=args.std)
                                 )

    if args.data == "CamVid":
        train_set = CamVid(args.dset_folder, split="train", transforms=t_train)
        val_set = CamVid(args.dset_folder, split="val", transforms=t_eval)
        test_set = CamVid(args.dset_folder, split="test_ood", transforms=t_eval)

    elif args.data == "StreetHazard":
        train_set = StreetHazard(args.dset_folder, split="train", transforms=t_train)
        val_set = StreetHazard(args.dset_folder, split="validation", transforms=t_eval)
        test_set = StreetHazard(args.dset_folder, split="test", transforms=t_eval)

    elif args.data == "BddAnomaly":
        train_set = BddAnomaly(args.dset_folder, split="train", transforms=t_train)
        val_set = BddAnomaly(args.dset_folder, split="validation", transforms=t_eval)
        test_set = BddAnomaly(args.dset_folder, split="test", transforms=t_eval)

    elif args.data == "CEA":
        train_set = CEA(args.dset_folder, split="train", transforms=t_train)
        val_set = CEA(args.dset_folder, split="val", transforms=t_eval)
        test_set = CEA(args.dset_folder, split="test", transforms=t_eval)

    elif args.data == "WoodScape":
        train_set = WoodScape(folder=args.dset_folder, split="train", transforms=t_train)
        val_set = WoodScape(folder=args.dset_folder, split="val", transforms=t_eval)
        print("The test split of WoodScape is not the official one, (it's the val set)")
        test_set = WoodScape(folder=args.dset_folder, split="val", transforms=t_eval)

    else:
        raise NameError('Unknown dataset')

    train_loader = DataLoader(train_set, batch_size=args.bsize, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=args.bsize, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, shuffle=True)

    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
