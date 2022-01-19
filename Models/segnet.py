import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, is_batchnorm=True):
        super().__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride,
                             bias=bias, dilation=dilation)

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels, init_vgg):
        super().__init__()
        self.in_channels = input_channels
        self.n_classes = output_channels

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.dropout_down3 = nn.Dropout(0.5)
        self.down4 = segnetDown3(256, 512)
        self.dropout_down4 = nn.Dropout(0.5)
        self.down5 = segnetDown3(512, 512)
        self.dropout_down5 = nn.Dropout(0.5)

        self.up5 = segnetUp3(512, 512)
        self.dropout_up5 = nn.Dropout(0.5)
        self.up4 = segnetUp3(512, 256)
        self.dropout_up4 = nn.Dropout(0.4)
        self.up3 = segnetUp3(256, 128)
        self.dropout_up3 = nn.Dropout(0.3)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, self.n_classes)

        if init_vgg:
            self.init_vgg16_params()

    def forward(self, inputs, mc_drop=False, return_feat=False):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)

        down4 = F.dropout(down3, 0.5) if mc_drop else self.dropout_down3(down3)
        down4, indices_4, unpool_shape4 = self.down4(down4)

        down5 = F.dropout(down4, 0.5) if mc_drop else self.dropout_down4(down4)
        down5, indices_5, unpool_shape5 = self.down5(down5)

        up5 = F.dropout(down5, 0.5) if mc_drop else self.dropout_down5(down5)
        up5 = self.up5(up5, indices_5, unpool_shape5)

        up4 = F.dropout(up5, 0.5) if mc_drop else self.dropout_up5(up5)
        up4 = self.up4(up4, indices_4, unpool_shape4)

        up3 = F.dropout(up4, 0.4) if mc_drop else self.dropout_up4(up4)
        up3 = self.up3(up3, indices_3, unpool_shape3)

        up2 = F.dropout(up3, 0.3) if mc_drop else self.dropout_up3(up3)
        up2 = self.up2(up2, indices_2, unpool_shape2)

        up1 = self.up1(up2, indices_1, unpool_shape1)

        return [down1, down2, down3, down4, down5, up5, up4, up3, up2, up1] if return_feat else up1

    def init_vgg16_params(self):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
