import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(input_channels, output_channels, stride=1, padding=0, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)    

def conv3x3(input_channels, output_channels, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.ReflectionPad2d(padding),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=0, bias=bias)
    )

def conv5x5(input_channels, output_channels, stride=1, padding=2, bias=False):
    return nn.Sequential(
        nn.ReflectionPad2d(padding),
        nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=stride, padding=0, bias=bias)
    )

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(input_channels, output_channels),
            nn.LeakyReLU(0.1),
            conv3x3(output_channels, output_channels),
            nn.LeakyReLU(0.1),
            conv3x3(output_channels, output_channels),
            nn.LeakyReLU(0.1)
        )

    # not RNN here
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, is_deconv=False, is_out=False):
        super(UpBlock, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        blocks = [
            conv3x3(input_channels*2, output_channels),
            nn.LeakyReLU(0.1),
            conv3x3(output_channels, output_channels),
            nn.LeakyReLU(0.1)
        ]
        if is_out:
            blocks += [conv1x1(output_channels, output_channels)]
        self.block = nn.Sequential(*blocks)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        # for padding
        offset1 = outputs2.size()[-1] - inputs1.size()[-1]
        offset2 = outputs2.size()[-2] - inputs1.size()[-2]
        padding1 = []
        padding2 = []
        if offset1 > 0:
            padding2 += [0, 0]
            if offset1 % 2 == 0:
                padding1 += [offset1 // 2, offset1 // 2]
            else:
                padding1 += [offset1 // 2 + 1, offset1 // 2]
        else:
            offset1 = -offset1
            padding1 += [0, 0]
            if offset1 % 2 == 0:
                padding2 += [offset1 // 2, offset1 // 2]
            else:
                padding2 += [offset1 // 2 + 1, offset1 // 2]

        if offset2 > 0:
            padding2 += [0, 0]
            if offset2 % 2 == 0:
                padding1 += [offset2 // 2, offset2 // 2]
            else:
                padding1 += [offset2 // 2 + 1, offset2 // 2]

        else:
            offset2 = -offset2
            padding1 += [0, 0]
            if offset2 % 2 == 0:
                padding2 += [offset2 // 2, offset2 // 2]
            else:
                padding2 += [offset2 // 2 + 1, offset2 // 2]

        outputs1 = nn.functional.pad(inputs1, padding1, mode='replicate')
        outputs2 = nn.functional.pad(outputs2, padding2, mode='replicate')

        output = torch.cat([outputs1, outputs2], dim=1)
        output = self.block(output)

        return output

class Autoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(Autoencoder, self).__init__()
        self.conv0 = ConvBlock(input_channels, 13)
        self.conv1 = ConvBlock(13, 32)
        self.conv2 = ConvBlock(32, 43)
        self.conv3 = ConvBlock(43, 57)
        self.conv4 = ConvBlock(57, 76)
        self.conv5 = ConvBlock(76, 101)

        self.bottleneck = ConvBlock(101, 101)
        self.up1 = UpBlock(101, 76)
        self.up2 = UpBlock(76, 57)
        self.up3 = UpBlock(57, 43)
        self.up4 = UpBlock(43, 32)
        self.up5 = UpBlock(32, 13)
        self.conv_out = ConvBlock(13, 1)

    # U-Net
    def forward(self, x):
        x = self.conv0(x)
        skip1 = self.conv1(x)
        x = nn.functional.max_pool2d(input=skip1, kernel_size=2)
        skip2 = self.conv2(x)
        x = nn.functional.max_pool2d(input=skip2, kernel_size=2)
        skip3 = self.conv3(x)
        x = nn.functional.max_pool2d(input=skip3, kernel_size=2)
        skip4 = self.conv4(x)
        x = nn.functional.max_pool2d(input=skip4, kernel_size=2)
        skip5 = self.conv5(x)
        x = nn.functional.max_pool2d(input=skip5, kernel_size=2)

        x = self.bottleneck(x)
        x = self.up1(skip5, x)
        x = self.up2(skip4, x)
        x = self.up3(skip3, x)
        x = self.up4(skip2, x)
        x = self.up5(skip1, x)
        x = self.conv_out(x)
        return x

# First UNet
# source: https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    import time
    # for i in range(100):
    test_in = torch.rand(1, 3, 480, 640).to('cuda:1')
        # t1 = time.time()
    net = UNet(3, 1).to('cuda:1')
    pytorch_total_params = sum(p.numel() for p in net.parameters())
        # out = net(test_in)
    # print(time.time()-t1)
    print(pytorch_total_params)