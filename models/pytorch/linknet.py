
import torch.nn as nn

from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self, m, n, stride=2):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(m, m // 4, 1)
        self.norm1 = nn.BatchNorm2d(m // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.conv2 = nn.ConvTranspose2d(m // 4, m // 4, 3, stride=stride, padding=1)
        self.norm2 = nn.BatchNorm2d(m // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(m // 4, n, 1)
        self.norm3 = nn.BatchNorm2d(n)
        self.relu3 = nn.ReLU(inplace=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight = init.xavier_uniform(m.weight, gain=init.calculate_gain('relu'))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight = init.constant(m.weight, 1)
        #         m.bias = init.constant(m.bias, 0)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x, output_size=(inputs.size(-2) * 2, inputs.size(-1) * 2))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class FinalBlock(nn.Module):
    def __init__(self, num_filters, num_classes=2):
        super().__init__()

        # Final Classifier
        self.finalconv1 = nn.ConvTranspose2d(num_filters, num_filters // 2, 3, stride=2, padding=1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(num_filters // 2, num_filters // 2, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        # self.finalconv3 = nn.ConvTranspose2d(num_filters // 2, num_classes, 2, stride=2)
        self.finalconv3 = nn.Conv2d(num_filters // 2, num_classes, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight = init.xavier_uniform(m.weight, gain=init.calculate_gain('relu'))

    def forward(self, inputs):
        x = self.finalconv1(inputs, output_size=(inputs.size(-2) * 2, inputs.size(-1) * 2))
        x = self.finalrelu1(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        # x = self.finalconv3(x, output_size=(inputs.size(-2) // 2, inputs.size(-1) // 2))
        x = self.finalconv3(x)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class LinkNet(nn.Module):
    def __init__(self, num_classes, depth=18):
        super().__init__()

        filters = [64, 128, 256, 512]

        if depth == 18:
            print('=> Building LinkNet from ResNet-18')
            resnet = models.resnet18(pretrained=True)
        elif depth == 34:
            print('=> Building LinkNet from ResNet-34')
            resnet = models.resnet34(pretrained=True)
        else:
            raise ValueError(f'Unexcpected LinkNet depth: {depth}')

        self.intro = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final = FinalBlock(filters[0], num_classes)

    def forward(self, inputs):
        x = self.intro(inputs)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        return self.final(d1)


def linknet18(num_classes=2):
    return LinkNet(num_classes=num_classes, depth=18)


def linknet34(num_classes=2):
    return LinkNet(num_classes=num_classes, depth=34)
