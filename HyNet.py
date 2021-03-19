import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torchvision

def conv3x3(in_ch, out_ch):
    # padding=0 in original paper
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.drop = nn.Dropout(0.2)
        self.bn = nn.BatchNorm2d(out)
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DecoderBlock_Unet(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock_Unet, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            ConvRelu(in_channels, in_channels),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2*scale, stride=scale, padding=scale//2),
            nn.ReLU(inplace=True),
            ConvRelu(in_channels, out_channels),
        )
    def forward(self, x):
        return self.block(x)

# General Models
class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock_Unet(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock_Unet(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock_Unet(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock_Unet(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock_Unet(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return (x_out, x_out, x_out, x_out)

      
      
class HyNet(nn.Module):
    def __init__(self):
        super(UpNet_N4_mobile, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.GAP = nn.AvgPool2d(kernel_size=512 // 32)
        net = models.mobilenet_v2(pretrained=True)
        self.net0_2_contains = list(list(net.children())[0].children())[:3]
        self.net0_2 = nn.Sequential(*self.net0_2_contains)  # (bs, 24, 128, 128)
        self.net3_6_contains = list(list(net.children())[0].children())[3:7]
        self.net3_6 = nn.Sequential(*self.net3_6_contains)  # (bs, 32, 64, 64)
        self.net7_13_contains = list(list(net.children())[0].children())[7:14]
        self.net7_13 = nn.Sequential(*self.net7_13_contains)  # (bs, 96, 32, 32)
        self.net8_19_contains = list(list(net.children())[0].children())[14:-1]
        self.net8_19 = nn.Sequential(*self.net8_19_contains)  # (bs, 320, 16, 16)

        # Encoder
        self.encoder1 = self.net0_2
        self.encoder2 = self.net3_6
        self.encoder3 = self.net7_13
        self.encoder4 = self.net8_19

        # Decoder
        self.center = nn.Sequential(ConvRelu(320, 320))
        self.decoder4 = DecoderBlock(320, 320)
        self.decoder3 = DecoderBlock(96, 96)
        self.decoder2 = DecoderBlock(32, 32)
        self.decoder1 = DecoderBlock(24, 24)

        self.up4 = nn.Sequential(ConvRelu(640, 96))
        self.up3 = nn.Sequential(ConvRelu(96*2, 32))
        self.up2 = nn.Sequential(ConvRelu(32*2, 24))
        self.up1 = nn.Sequential(ConvRelu(24*2, 24))

        # Final Classifier
        # self.fc_32 = nn.Conv2d(32, 2, 1, padding=0)
        self.fc_64 = nn.Conv2d(96 + 32 + 24 + 24, 1, 1, padding=0)
        # self.fc_128 = nn.Conv2d(128 + 512 + 256, 2, 1, padding=0)
        # self.fc_256 = nn.Conv2d(256 + 512, 2, 1, padding=0)
        # self.fc_512 = nn.Conv2d(512, 2, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear')

        # self._init_weight()

    def forward(self, x):
        # Encoder
        e1_ = self.encoder1(x)  # (bs, 24, 128, 128)
        e2_ = self.encoder2(e1_)  # (bs, 32, 64, 64)
        e3_ = self.encoder3(e2_)  # (bs, 96, 32, 32)
        e4_ = self.encoder4(e3_)  # (bs, 320, 16, 16)


        center = self.center(self.maxpool(e4_))

        # UpSample
        e4__ = self.up4(torch.cat([e4_, self.decoder4(center)], 1))  # (bs, 512, 16, 16)
        e3__ = self.up3(torch.cat([e3_, self.decoder3(e4__)], 1))  # (bs, 256, 32, 32)
        e2__ = self.up2(torch.cat([e2_, self.decoder2(e3__)], 1))  # (bs, 128, 64, 64)
        e1__ = self.up1(torch.cat([e1_, self.decoder1(e2__)], 1))  # (bs, 64, 128, 128)

        # e4 = e4__
        # e3 = torch.cat([e3__, self.upsample2(e4__)], 1)  # 512 + 256
        # e2 = torch.cat([e2__, self.upsample2(e3__), self.upsample4(e4__)], 1)
        e1 = torch.cat([e1__, self.upsample2(e2__), self.upsample4(e3__), self.upsample8(e4__)], 1)

        # Using 1x1 Conv as Scoring
        score_map_1 = self.upsample4(self.fc_64(e1))  # (bs, 1, 64, 64)
        # score_map_2 = self.upsample8(self.fc_128(e2))
        # score_map_3 = self.upsample16(self.fc_256(e3))
        # score_map_4 = self.upsample32(self.fc_512(e4))

        return score_map_1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


  
