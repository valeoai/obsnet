import random
import torch
import torch.nn.functional as F
from skimage.draw import random_shapes

from Utils.utils import transform_output
from Datasets.load_data import data_loader


def add_shape(args):
    """ return a Tensor with a random shape
        1 for the shape
        0 for the background
        The size is controle by args.patch_size
    """
    image, _ = random_shapes(args.patch_size[0:2], min_shapes=6, max_shapes=10,
                             intensity_range=(0, 50),  min_size=args.patch_size[2],
                             max_size=args.patch_size[3], allow_overlap=True, num_channels=1)
    image = torch.round(1 - torch.FloatTensor(image)/255.)
    return image.squeeze().to(args.device)


def fgsm_attack(images, target, segnet, mask, mode, args):
    """ Perform a FGSM attack on the image
        images -> Tensor: (b,c,w,h) the batch of the images
        target -> Tensor: the label
        segnet -> torch.Module: The segmentation network
        mask   -> Tensor: (b,1,w,h) binary mask where to perform the attack
        mode   -> str: either to minimize the True class or maximize a random different class
        args   ->  Argparse: global arguments
    return
        images       -> Tensor: the image attacked
        perturbation -> Tensor: the perturbation
    """
    bsize, channel, width, height = images.size()
    images += 0.03 * torch.randn_like(images) * mask     # add some noise for more diverse attack
    images.requires_grad = True
    segnet_pred = segnet(images, return_feat=False)
    segnet_pred = transform_output(segnet_pred, bsize, args.nclass)
    if mode == "max":
        fake_target = torch.randint(args.nclass - 1, size=(bsize, 1)).expand_as(target).to(args.device)
        loss = F.cross_entropy(segnet_pred, fake_target.reshape(-1))
    elif mode == "min":
        loss = F.cross_entropy(segnet_pred, target.long().reshape(-1))

    segnet.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()

    sign_data_grad *= mask
    if mode == "max":
        perturbed = - args.epsilon * sign_data_grad
    elif mode == "min":
        perturbed = args.epsilon * sign_data_grad

    images.data = torch.clamp(images + perturbed, -3, 3)

    return images.detach(), perturbed


def generate_mask_all_images(images, args):
    return torch.ones_like(images)


def generate_mask_pixels_sparse(images, args):
    """ generate a mask to attack random pixel on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    return torch.where(torch.rand_like(images) < 0.05, args.one, args.zero)


def generate_mask_class(images, target, segnet_pred, args):
    """ generate a mask to attack a random class on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width * height).to(args.device)
    for i in range(len(mask)):
        valid_classes, count = torch.unique(target[i], return_counts=True)
        chosen_classes = valid_classes[random.randint(0, len(valid_classes)-1)]
        mask[i][target[i].view(-1) == chosen_classes] = 1
        if args.adv.startswith("min"):   # do not attack if the sample are already badly classify
            mask[i][target[i].view(-1) != segnet_pred[i].view(-1)] = 0
    return mask.view(bsize, 1, width, height).expand_as(images)


def generate_mask_square_patch(images, args):
    """ Generate a mask to attack a random square patch on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).to(args.device)
    for i in range(len(mask)):
        x = random.randint(0, width - args.patch_size[0])
        y = random.randint(0, height - args.patch_size[1])
        h = args.patch_size[1]
        w = args.patch_size[0]
        mask[i, x:x + w, y:y + h] = 1.
    return mask.view(bsize, 1, width, height).expand_as(images)


def generate_mask_random_patch(images, args):
    """ Generate a mask to attack a random shape on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).to(args.device)
    for i in range(len(mask)):
        x = random.randint(0, width - args.patch_size[0])
        y = random.randint(0, height - args.patch_size[1])
        h = args.patch_size[1]
        w = args.patch_size[0]
        shape = add_shape(args)
        mask[i, x:x + w, y:y + h] = shape == 1.
    return mask.view(bsize, 1, width, height).expand_as(images)


def generate_mask_fractal(images, args):
    """ Generate a mask to attack a fractal shape on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, 3, width, height).to(args.device)

    try:
        fractal = next(args.fractal)
    except StopIteration:
        args.fractal = iter(data_loader("Fractal", args))
        fractal = next(args.fractal, None)
    fractal = fractal.to(args.device)

    for i in range(len(mask)):
        x = random.randint(50, width - args.patch_size[0])
        y = random.randint(0, height - args.patch_size[1])
        h = args.patch_size[1]
        w = args.patch_size[0]
        patch = torch.where(fractal[i].unsqueeze(0) > 0, args.one, args.zero).to(args.device)
        mask[i, :, x:x + w, y:y + h] = patch
        # texture[i] += images[i, :, x:x + w, y:y + h].mean() - texture.mean() - 0.2
        # images[i, :] = torch.where(mask[i, :] > 0, texture[i, :], images[i, :])

    return mask, images


def select_attack(images, target, segnet, args):
    """ Select the right attack given args.adv """
    if args.adv.startswith("min_"):
        if args.adv.endswith("all_image"):
            mask = generate_mask_all_images(images, args)
        elif args.adv.endswith("pixels_sparse"):
            mask = generate_mask_pixels_sparse(images, args)
        elif args.adv.endswith("class"):
            segnet_pred = segnet(images, mc_drop=False)[-1].detach().max(1)[1]
            mask = generate_mask_class(images, target, segnet_pred, args)
        elif args.adv.endswith("square_patch"):
            mask = generate_mask_square_patch(images, args)
        elif args.adv.endswith("random_patch"):
            mask = generate_mask_random_patch(images, args)
        return fgsm_attack(images, target, segnet, mask, "min", args)
    elif args.adv.startswith("max_"):
        if args.adv.endswith("all_image"):
            mask = generate_mask_all_images(images, args)
        elif args.adv.endswith("pixels_sparse"):
            mask = generate_mask_pixels_sparse(images, args)
        elif args.adv.endswith("class"):
            segnet_pred = segnet(images, mc_drop=False)[-1].detach()
            mask = generate_mask_class(images, target, segnet_pred, args)
        elif args.adv.endswith("square_patch"):
            mask = generate_mask_square_patch(images, args)
        elif args.adv.endswith("random_patch"):
            mask = generate_mask_random_patch(images, args)
        return fgsm_attack(images, target, segnet, mask, "max", args)
    elif args.adv == "fractal":
        mask, images = generate_mask_fractal(images, args)
        return fgsm_attack(images, target, segnet, mask, "max", args)
    else:
        raise NameError('Unknown attacks, please check args.adv arguments')
