import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch.utils.tensorboard import SummaryWriter

from Utils.utils import reinitialize
from Datasets.load_data import data_loader
from Models.load_net import net_loader

from train import training
from evaluation import evaluate


def main(args):
    """ Manage the training, the testing and the declaration of the 'global parameters'
        such as optimizer, scheduler, the SumarryWriter, etc. """

    writer = SummaryWriter(args.tboard)
    writer.add_text("option", str(args), 0)

    # Load Dataset
    train_loader, val_loader, test_loader = data_loader(args)
    args.cmap = train_loader.dataset.cmap
    args.class_name = train_loader.dataset.class_name

    # Load Networks
    obsnet, segnet = net_loader(args)
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(obsnet.parameters(), lr=args.lr)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(obsnet.parameters(), lr=args.lr)
    sched = scheduler.MultiStepLR(optimizer, milestones=[args.epoch // 2, args.epoch-5], gamma=0.2)

    if args.test_only:
        evaluate(0, obsnet, segnet, test_loader, "Test", writer, args)
    else:
        if not args.no_pretrained:
            reinitialize(obsnet, args.segnet_file)
        best = 0
        start = time.time()
        for epoch in range(0, args.epoch+1):
            print(f"######## Epoch: {epoch} || Time: {(time.time() - start)/60:.2f} min ########")

            train_loss, obsnet_acc = training(epoch, obsnet, segnet, train_loader, optimizer, writer, args)
            val_loss, results_obs = evaluate(epoch, obsnet, segnet, val_loader, "Val", writer, args)

            if epoch % 5 == 0:               # save ckpt
                model_to_save = obsnet.module.state_dict()
                torch.save(model_to_save, os.path.join(args.obsnet_file, f"epoch{epoch:03d}.pth"))

            if results_obs["auroc"] > best:  # Save Best model
                print("save best net!!!")
                best = results_obs["auroc"]
                model_to_save = obsnet.module.state_dict()
                torch.save(model_to_save, os.path.join(args.obsnet_file, "best.pth"))
            sched.step()
    writer.close()


if __name__ == '__main__':
    ### Argparse ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_folder",   type=str,   default="",       help="path to dataset")
    parser.add_argument("--segnet_file",   type=str,   default="",       help="path to segnet")
    parser.add_argument("--obsnet_file",   type=str,   default="",       help="path to obsnet")
    parser.add_argument("--data",          type=str,   default="",       help="CamVid|StreetHazard|BddAnomaly")
    parser.add_argument("--tboard",        type=str,   default="",       help="path to tensorboeard log")
    parser.add_argument("--model",         type=str,   default="segnet", help="Segnet|Deeplabv3")
    parser.add_argument("--optim",         type=str,   default="SGD",    help="type of optimizer SGD|AdamW")
    parser.add_argument("--T",             type=int,   default=50,       help="number of forward pass for ensemble")
    parser.add_argument("--seed",          type=int,   default=-1,       help="seed, if -1 no seed is use")
    parser.add_argument("--bsize",         type=int,   default=8,        help="batch size")
    parser.add_argument("--lr",            type=float, default=2e-2,     help="learning rate of obsnet")
    parser.add_argument("--Temp",          type=float, default=1.2,      help="temperature scaling ratio")
    parser.add_argument("--noise",         type=float, default=0.,       help="noise injection in the img data")
    parser.add_argument("--epsilon",       type=float, default=0.1,      help="epsilon for adversarial attacks")
    parser.add_argument("--gauss_lambda",  type=float, default=0.002,    help="lambda parameters for gauss params")
    parser.add_argument("--epoch",         type=int,   default=50,       help="number of epoch")
    parser.add_argument("--num_workers",   type=int,   default=0,        help="number of workers")
    parser.add_argument("--num_nodes",     type=int,   default=1,        help="number of node")
    parser.add_argument("--adv",           type=str,   default="none",   help="type of adversarial attacks")
    parser.add_argument("--test_multi",    type=str,   default="obsnet", help="test all baseline, split by comma")
    parser.add_argument("--drop",          action='store_true',          help="activate dropout in segnet")
    parser.add_argument("--no_img",        action='store_true',          help="use image for obsnet")
    parser.add_argument("--obs_mlp",       action='store_true',          help="use a smaller archi for obsnet")
    parser.add_argument("--test_only",     action='store_true',          help="evaluate methods")
    parser.add_argument("--no_residual",   action='store_true',          help="remove residual connection for obsnet")
    parser.add_argument("--no_pretrained", action='store_true',          help="load segnet weight for the obsnet")
    args = parser.parse_args()

    # Setting multi GPUs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "CamVid":
        args.size = [360, 480]
        args.crop = (50, 80)
        args.pos_weight = torch.tensor([2]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [128, 128, 60, 80]
        args.mean = [0.4108, 0.4240, 0.4316]
        args.std = [0.3066, 0.3111, 0.3068]
        args.nclass = 12
    elif args.data == "StreetHazard":
        args.size = [720, 1280]
        args.crop = (150, 250)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [300, 360, 160, 200]
        args.mean = [0.3301, 0.3457, 0.3728]
        args.std = [0.1773, 0.1767, 0.1900]
        args.nclass = 14
    elif args.data == "BddAnomaly":
        args.size = [360, 640]  # Original size [720, 1280]
        args.crop = (80, 150)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [300, 360, 160, 200]
        args.mean = [0.3698, 0.4145, 0.4247]
        args.std = [0.2525, 0.2695, 0.2870]
        args.nclass = 19
    elif args.data == "CEA":
        args.size = [600, 960]  # Original size [800, 1280]
        args.crop = (150, 250)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [300, 360, 160, 200]
        args.mean = [0.3307, 0.3650, 0.3722]
        args.std = [0.4279, 0.4612, 0.5387]
        args.nclass = 12
    elif args.data == "WoodScape":
        args.size = [725, 960]  # Original size [966, 1280]
        args.crop = (150, 250)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [300, 360, 160, 200]
        args.mean = [0.3213, 0.3335, 0.3400]
        args.std = [0.1368, 0.1376, 0.1419]
        args.nclass = 10
    else:
        raise NameError("Dataset not understand")

    args.one = torch.FloatTensor([1.]).to(args.device)
    args.zero = torch.FloatTensor([0.]).to(args.device)

    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        random.seed(args.seed)

    args.test_multi = args.test_multi.split(",")

    main(args)
