import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode="reflect"),
        nn.InstanceNorm2d(out_c),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode="reflect"),
        nn.InstanceNorm2d(out_c),
        nn.LeakyReLU(inplace=True),
    )
    return conv


def crop_img(tensor, target_tensor):
    # batch_size, channel, height, width
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta : tensor_size - delta, delta : tensor_size - delta]


class ZUNet_v1(nn.Module):
    def __init__(self, in_channels=3,num_c=1):
        super(ZUNet_v1, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            # in_channels=1024,
            in_channels=1034,
            out_channels=512,
            kernel_size=2,
            stride=2,
        )
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=num_c, kernel_size=1)

    def forward(self, image, z):
        # encoder
        x1 = self.down_conv_1(image)  # ---->
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  # --->
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  # -->
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)  # ->
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # add z
        one_hot_z = nn.functional.one_hot(z, num_classes=10)
        one_hot_z = torch.unsqueeze(torch.unsqueeze(one_hot_z, -1), -1)
        x9_z = torch.cat([x9, one_hot_z.repeat(1, 1, 32, 32)], dim=1)
        # decoder
        x = self.up_trans_1(x9_z)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        x = self.out(x)
        return x


class ZUNet_v2(nn.Module):
    def __init__(self, in_channels=1):
        super(ZUNet_v2, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_8x8 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.down_conv_1 = double_conv(in_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        self.preclassifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=8192, out_features=10)

    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)  # ---->
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  # --->
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  # -->
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)  # ->
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        x10 = self.max_pool_8x8(x9)
        preclassifier = self.preclassifier(x10)
        z = self.classifier(preclassifier.view(-1, 512 * 4 * 4))

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        x = self.out(x)
        return x, z
