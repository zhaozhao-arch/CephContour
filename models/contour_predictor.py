import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


class ImprovedEstimatorDecoder(nn.Module):
    def __init__(self, boundary_num):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, boundary_num, 3, stride=1, padding=1),
            nn.BatchNorm2d(boundary_num)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.relu(x)


class ImprovedContourDecoder(nn.Module):
    def __init__(self, contour_num=1):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.cbam1 = CBAM(256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.cbam2 = CBAM(128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.cbam3 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, contour_num, 3, stride=1, padding=1)

    def forward(self, x):
        x_up1 = self.up1(x)
        x_up1 = self.cbam1(x_up1)
        x = self.conv1(x_up1) + x_up1

        x_up2 = self.up2(x)
        x_up2 = self.cbam2(x_up2)
        x = self.conv2(x_up2) + x_up2

        x_up3 = self.up3(x)
        x_up3 = F.relu(self.bn3(x_up3))
        x_up3 = self.cbam3(x_up3)
        x = self.conv3(x_up3) + x_up3
        x = self.final_conv(x)

        return x


class Estimator(nn.Module):
    def __init__(self, image_encoder, boundary_num):
        super().__init__()
        self.image_encoder = image_encoder
        self.decoder = ImprovedEstimatorDecoder(boundary_num)

    def forward(self, x):
        output = self.image_encoder(x)
        return self.decoder(output)


class ContourPredictor(nn.Module):
    def __init__(self, image_encoder, contour_num):
        super().__init__()
        self.image_encoder = image_encoder
        self.decoder = ImprovedContourDecoder(contour_num)
        self.relu = nn.ReLU()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.image_encoder(x)
        upper_maps = self.decoder(output)
        return self.relu(upper_maps)