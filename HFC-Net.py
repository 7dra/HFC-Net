import torch
import torch.nn as nn
import torch.nn.functional as F


class FR(nn.Module):
    def __init__(self, channels):
        super(FR, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):

        mask = self.mask_generator(x)

        recon = self.reconstructor(x)

        masked_features = x * mask

        reconstructed_features = self.reconstructor(masked_features)


        return x + reconstructed_features


class GCA(nn.Module):
    def __init__(self, channels):
        super(GCA, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4)  
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).permute(2, 0, 1)  # (H*W, B, C)
        x_flat = self.norm(x_flat)

        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)

        out = x + attn_output  

        out = out + self.ffn(out)  
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class HFC(nn.Module):
    def __init__(self, num_classes):
        super(HFC, self).__init__()
        # encoder
        self.encoder1 = ConvBlock(3, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)


        self.fr = FR(512)
        self.gca = GCA(512)

        # decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):

        e1 = self.encoder1(x)

        e2 = self.encoder2(F.max_pool2d(e1, 2))

        e3 = self.encoder3(F.max_pool2d(e2, 2))

        e4 = self.encoder4(F.max_pool2d(e3, 2))



        e4 = self.fr(e4)

        e4 = self.gca(e4)

        d3 = self.upconv3(e4)

        d3 = torch.cat([d3, e3], dim=1)

        d3 = self.decoder3(d3)


        d2 = self.upconv2(d3)

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.decoder1(d1)
        out = self.final(d1)
        return out

x = torch.randn(2, 3, 224, 224)
net = HFC(3)
out = net(x)
print(out.shape)

