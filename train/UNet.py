import torch
import torch.nn as nn

# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# Defines the Unet.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
class RecursiveUNet(nn.Module):
    def __init__(self,
                num_classes=2,
                in_channels=1,
                initial_filter_size=64,
                kernel_size=3,
                num_downs=4,
                norm_layer=nn.InstanceNorm2d,
                activation=nn.LeakyReLU(inplace=True)):
#       InstancNorm performs better than BatchNorm for airway segmentation
        super(RecursiveUNet, self).__init__()
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1),
                                             out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes,
                                             kernel_size=kernel_size,
                                             norm_layer=norm_layer,
                                             innermost=True,
                                             activation=activation)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes,
                                                 kernel_size=kernel_size,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 activation=activation)

        unet_block = UnetSkipConnectionBlock(in_channels=in_channels,
                                             out_channels=initial_filter_size,
                                             num_classes=num_classes,
                                             kernel_size=kernel_size,
                                             submodule=unet_block,
                                             norm_layer=norm_layer,
                                             outermost=True,
                                             activation=activation)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 num_classes=1,
                 kernel_size=3,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 activation=nn.LeakyReLU(inplace=True)):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # downconv
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              norm_layer=norm_layer,
                              activation=activation)
        conv2 = self.contract(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              norm_layer=norm_layer,
                              activation=activation)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            activation=activation)
        conv4 = self.expand(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            activation=activation)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d,activation=nn.LeakyReLU(inplace=True)):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            activation)
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3, activation=nn.LeakyReLU(inplace=True)):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            activation,
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)

