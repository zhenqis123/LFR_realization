import torch
import torch.nn as nn

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        before_pool = x
        x = self.pool(x)
        return x, before_pool

# 定义U-net的解码器部分
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, before_pool):
        x = self.upconv(x)
        x = torch.cat([x, before_pool], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

# 定义完整的U-net模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = UNetEncoderBlock(in_channels, 64)
        self.enc2 = UNetEncoderBlock(64, 128)
        self.enc3 = UNetEncoderBlock(128, 256)
        self.enc4 = UNetEncoderBlock(256, 512)

        # 中间层
        self.middle = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 解码器部分
        self.dec4 = UNetDecoderBlock(1024, 512)
        self.dec3 = UNetDecoderBlock(512, 256)
        self.dec2 = UNetDecoderBlock(256, 128)
        self.dec1 = UNetDecoderBlock(128, 64)

        # 输出层
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1, before_pool1 = self.enc1(x)
        enc2, before_pool2 = self.enc2(enc1)
        enc3, before_pool3 = self.enc3(enc2)
        enc4, before_pool4 = self.enc4(enc3)

        middle = self.middle(enc4)
        middle = self.relu(middle)

        dec4 = self.dec4(middle, before_pool4)
        dec3 = self.dec3(dec4, before_pool3)
        dec2 = self.dec2(dec3, before_pool2)
        dec1 = self.dec1(dec2, before_pool1)

        out = self.outconv(dec1)
        return out
