import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.op(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch)
        self.conv2 = ConvNormAct(out_ch, out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    def forward(self, x):
        return self.conv1(x) + self.shortcut(x)

class MedicalResUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_filters=32):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8) 
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)
        self.head = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(dim * 4, dim, 1), nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        x = x + self.proj(out)
        
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        x = x + self.mlp(x_norm)
        return x

class HybridViTSegmenter(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, deep_supervision=True, img_size=256, patch_size=16, embed_dim=512, depth=4, heads=8):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        
        self.trans_in = nn.Conv2d(nb_filter[3], nb_filter[4], 1)
        self.transformer = nn.Sequential(*[TransformerBlock(nb_filter[4], heads=heads) for _ in range(depth)])
        self.trans_out = nn.Conv2d(nb_filter[4], nb_filter[4], 1)
        
        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3])
        
        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])
        
        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])
        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        self.final1 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final3 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final4 = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x4_0 = self.pool(x3_0)
        x4_0 = self.trans_in(x4_0)
        x4_0 = self.transformer(x4_0)
        x4_0 = self.trans_out(x4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        output4 = self.final4(x0_4)
        if self.deep_supervision and self.training:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output4, output3, output2, output1]
        else:
            return output4


StrongHybridViT = HybridViTSegmenter