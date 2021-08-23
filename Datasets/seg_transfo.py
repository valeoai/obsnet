import random
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image

# Here are custom transformation for Semgmentation Data Augmentation #


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))

    def __call__(self, imgx, imgy):
        imgy = imgy.resize(self.size, Image.NEAREST)
        imgx = imgx.resize(self.size, Image.BILINEAR)
        return imgx, imgy


class Rotate(object):
    def __init__(self, rotation=2):
        self.rot = rotation

    def __call__(self, imgx, imgy):
        rotate = np.random.randint(-self.rot, self.rot, 1)
        imgx = imgx.rotate(rotate, Image.BILINEAR)
        imgy = imgy.rotate(rotate, Image.NEAREST, fillcolor=1)
        return imgx, imgy


class HFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgx, imgy):
        if np.random.random() > self.p:
            imgx.transpose(Image.FLIP_LEFT_RIGHT)
            imgy.transpose(Image.FLIP_LEFT_RIGHT)
        return imgx, imgy


class RandomCrop(object):
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, imgx, imgy):
        w = imgx.width
        h = imgx.height
        cropx, cropw = np.random.randint(0, self.crop[0], 2)
        cropy, croph = np.random.randint(0, self.crop[1], 2)
        imgx = imgx.crop((cropx, cropy, w - cropw, h - croph))
        imgy = imgy.crop((cropx, cropy, w - cropw, h - croph))
        return imgx.resize((w, h), Image.BILINEAR), imgy.resize((w, h), Image.NEAREST)


class ToTensor(object):
    def __call__(self, imgx, imgy):
        return tf.to_tensor(imgx), (tf.to_tensor(imgy)*255).long().view(-1)


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, imgx, imgy):
        return tf.normalize(imgx, self.mean, self.std, self.inplace), imgy


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, imgy):
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), imgy


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, imgy):
        return tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)), imgy


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, imgy):
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), imgy


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, imgy):
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), imgy


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, imgy):
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), imgy


class SegTransformCompose(object):
    def __init__(self, *args):
        self.list_transfo = args

    def __call__(self, imgx, imgy):
        for t in self.list_transfo:
            imgx, imgy = t(imgx, imgy)
        return imgx, imgy
