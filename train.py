import torch
from torchvision.utils import make_grid
from Utils.affichage import draw, plot
from Utils.utils import transform_output
from Utils.adv_attacks import select_attack


def training(epoch, obsnet, segnet, train_loader, optimizer, writer, args):
    """ Train the observer network for one epoch
        epoch        ->  int: current epoch
        obsnet       ->  torch.Module: the observer to train
        segnet       ->  torch.Module: the segnet pretrained and freeze
        train_loader ->  torch.DataLoader: the training dataloader
        optimizer    ->  torch.optim: optimizer to train observer
        writer       ->  SummaryWriter: for tensorboard log
        args         ->  Argparse: global parameters
    return:
        avg_loss -> float : average loss on the dataset
    """

    obsnet.train()
    avg_loss, nb_sample, obsnet_acc, segnet_acc = 0, 0, 0., 0.
    for i, (images, target) in enumerate(train_loader):
        bsize, channel, width, height = images.size()
        nb_sample += bsize * width * height

        images = images.to(args.device)
        target = target.to(args.device)

        if args.adv != "none":                                                             # perform the LAA
            images, mask = select_attack(images, target, segnet, args)

        with torch.no_grad():
            segnet_feat = segnet(images, return_feat=True)                                 # SegNet forward
            segnet_pred = transform_output(pred=segnet_feat[-1], bsize=bsize, nclass=args.nclass)

            error = segnet_pred.max(-1)[1].view(-1) != target.view(-1)               # GT for observer training
            # attack = mask.sum(1).reshape(-1) != 0
            # supervision = error + attack
            supervision = torch.where(error, args.one, args.zero).to(args.device)

        obs_pred = obsnet(images, segnet_feat, no_residual=args.no_residual, no_img=args.no_img)
        obs_pred = transform_output(pred=obs_pred, bsize=bsize, nclass=1)

        loss = args.criterion(obs_pred.view(-1), supervision.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.cpu().item()
        segnet_acc += segnet_pred.max(-1)[1].view(-1).eq(target.view(-1)).sum()
        obsnet_acc += torch.round(torch.sigmoid(obs_pred)).view(-1).eq(supervision.view(-1)).sum()

        print(f"\rTrain loss: {loss.cpu().item():.4f}, "
              f"Progress: {100*(i/len(train_loader)):.2f}%",
              end="")

        writer.add_scalars('data/TrainLoss', {"loss": loss}, (epoch * len(train_loader)) + i)

        if i == 0:                                                                      # Visualization
            with torch.no_grad():
                sm = segnet_feat[-1]                                                    # MCP visualization
                sm = 1 - torch.softmax(sm, 1).max(1)[0][0]
                sm_uncertainty = draw(sm, args).cpu()

                obs_pred = obs_pred.view(bsize, -1)                                     # ObsNet visualization
                obsnet_uncertainty = draw(torch.sigmoid(obs_pred[0]), args).cpu()

                obs_label = supervision.view(bsize, -1)                                 # GT visualization
                label = draw(obs_label[0], args).cpu()

                uncertainty_map = torch.cat((obsnet_uncertainty, sm_uncertainty, label), dim=0)
                uncertainty_map = make_grid(uncertainty_map, normalize=False)
                segmentation_map = plot(images + (10 * mask), segnet_feat[-1], target, args)

            writer.add_image("Image/Train/segmentation", segmentation_map, epoch)
            writer.add_image("Image/Train/uncertainty", uncertainty_map, epoch)

    avg_loss /= len(train_loader)

    obsnet_acc = 100 * (obsnet_acc / nb_sample)
    segnet_acc = 100 * (segnet_acc / nb_sample)

    print(f"\rEpoch Summary: Train Avg loss: {avg_loss:.4f}, "
          f"ObsNet acc: {obsnet_acc:.2f}, "
          f"SegNet acc: {segnet_acc:.2f}"
          )

    return avg_loss, obsnet_acc
