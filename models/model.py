from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
import torch
from torch import nn
import torch.nn.functional as F


class ELGANet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Network = FeatureNet()
        self.P2SP = P2SP()

    def forward(self, x):
        Patch_feature, Pixel_feature = self.Network(x)
        assign = self.P2SP(Pixel_feature,Patch_feature)
        return assign

    def weight_parameters(self):
        #print("param in model")
        #for name, param in self.named_parameters():
        #    print(name)
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class GlobalAttention_MH(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim / self.num_heads) ** -0.5

        self.conv_patch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size, padding=0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        patch_feat = self.conv_patch(x)

        B, C, H, W = x.shape
        N = H * W
        pixel_query = self.q(x).reshape(B, self.num_heads, C // self.num_heads, N)  # B,head,C_head,N

        B, C, H_p, W_p = patch_feat.shape
        N_patch = H_p * W_p
        patch_key, patch_value = self.kv(patch_feat).reshape(B, self.num_heads, C // self.num_heads * 2, N_patch).chunk(
            2, dim=2)  # B,head,C_head,N_patch

        attn = pixel_query.transpose(-1, -2) @ patch_key * self.scale
        attn = self.softmax(attn)  # B,head,N,N_patch

        out = (patch_value @ attn.transpose(-1, -2))  # B,head,C_head,N
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return out


class LocalAttention_MH(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim / self.num_heads) ** -0.5

        self.conv_patch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size, padding=0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.window_size
        w = self.window_size
        #Patch_pixels = h * w
        hh = H // h
        ww = W // w
        #N_patch = hh * ww
        patch_feat = self.conv_patch(x)  # B,C,hh,ww

        pixel_query = self.q(x).reshape(B * self.num_heads, C // self.num_heads, hh, h, ww, w).permute(0, 2, 4, 3, 5,
                                                                                                       1).reshape(
            B * self.num_heads, hh * ww, h * w, C // self.num_heads)  # B*head,N_patch,Patch_pixels,C//self.num_heads
        patch_key, patch_value = self.kv(patch_feat).reshape(B * self.num_heads, C // self.num_heads * 2, hh, ww).chunk(
            2, dim=1)

        patch_key_unfold = F.unfold(patch_key, (3, 3),
                                    padding=1)  # (Bhead*head, Chead*9, hh*ww)  Bhead*head,Chead*9,N_patch
        patch_key_unfold = patch_key_unfold.transpose(1, 2).reshape(B * self.num_heads, hh * ww, C // self.num_heads,
                                                                    9)  # B*head,N_patch,C,9

        patch_value_unfold = F.unfold(patch_value, (3, 3),
                                      padding=1)  # (B*self.num_heads, C//self.num_heads*9, hh*ww)  B*self.num_heads,C//self.num_heads*9,N_patch
        patch_value_unfold = patch_value_unfold.transpose(1, 2).reshape(B * self.num_heads, hh * ww,
                                                                        C // self.num_heads,
                                                                        9)  # B*self.num_heads,N_patch,C//self.num_heads,9

        attn = pixel_query @ patch_key_unfold * self.scale  # (B, hh*ww, h*w, C) (B, N_patch, patch_pixel,C, 9)
        attn = self.softmax(attn)

        out = attn @ patch_value_unfold.transpose(-1, -2)  # (B, N_patch, patch_pixel, C)
        out = (out.reshape(B * self.num_heads, hh, ww, h, w, C // self.num_heads).permute(0, 5, 1, 3, 2, 4)).reshape(B,
                                                                                                                     C,
                                                                                                                     H,
                                                                                                                     W)

        out = self.proj(out)

        return out



class Sobel_H(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(Sobel_H, self).__init__()
        #H
        #self.Sobel_kernel = torch.tensor([[-1., 0., 1.],
        #                                  [-2., 0., 2.],
        #                                  [-1., 0., 1.]], dtype=torch.float32, requires_grad=False)
        self.Sobel_kernel = torch.tensor([[-1., 0., 1.],
                                          [-2., 0., 2.],
                                          [-1., 0., 1.]], dtype=torch.float32)
        self.Sobel_kernel = self.Sobel_kernel.repeat(out_channels, in_channels // out_channels, 1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        with torch.no_grad():
            self.conv.weight.copy_(self.Sobel_kernel)
        #self.conv.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x




class Sobel_V(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(Sobel_V, self).__init__()
        #V
        #self.Sobel_kernel = torch.tensor([[1., 2., 1.],
        #                                  [0., 0., 0.],
        #                                  [-1., -2., -1.]], dtype=torch.float32, requires_grad=False)

        self.Sobel_kernel = torch.tensor([[1., 2., 1.],
                                          [0., 0., 0.],
                                          [-1., -2., -1.]], dtype=torch.float32)
        self.Sobel_kernel = self.Sobel_kernel.repeat(out_channels, in_channels // out_channels, 1, 1)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        with torch.no_grad():
            self.conv.weight.copy_(self.Sobel_kernel)
        #self.conv.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class MaxP_AvgP(nn.Module):
    def __init__(self, kernel_size=3, dim=32, stride=3, padding=1):
        super(MaxP_AvgP, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.avgPool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        mapimg = self.maxpool(x)
        avpimg = self.avgPool(x)
        out = self.bn(abs(mapimg - avpimg))
        return out


class EeEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EeEM, self).__init__()
        self.SobelH = Sobel_H(in_channels, out_channels, stride=1, padding=1)
        self.SobelV = Sobel_V(in_channels, out_channels, stride=1, padding=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)  # 321->311
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.mp_ap = MaxP_AvgP(kernel_size=3, dim=out_channels, stride=1, padding=1)
        self.convFu = BasicConv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = self.SobelH(x)
        y2 = self.SobelV(x)
        x3 = self.gelu(self.bn(self.conv3x3(x)))
        y3 = self.mp_ap(x3)
        y = self.convFu(torch.cat([y1, y2, y3, x3], dim=1))

        return y  # B,C,H,W

def concatenate_channels(tensors):
    if len(tensors) < 2:
        raise ValueError("At least two tensors are required for concatenation.")

    channel_tensors = []

    for channel_idx in range(tensors[0].size(1)):
        for tensor in tensors:
            channel_map = tensor[:, channel_idx:channel_idx + 1, :, :]
            channel_tensors.append(channel_map)
    concatenated_channel = torch.cat(channel_tensors, dim=1)

    return concatenated_channel



class ConvAttentionM(nn.Module):
    def __init__(self, in_dim, out_dim, keep_size=False):
        super().__init__()
        if not keep_size:
            self.catten = nn.Sequential(
                conv(True, in_dim, in_dim, kernel_size=3, stride=2),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1,
                          groups=in_dim, bias=False),
                nn.GELU(),
                SE(out_dim, out_dim),
                nn.Conv2d(out_dim, out_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_dim)
                )
        else:
            self.catten = nn.Sequential(
                conv(True, 3, 16),
                conv(True, 16, 16)
            )

        self.keep_size = keep_size

    def forward(self, x):
        x = self.catten(x)
        return x


class FeatureNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FeatureNet, self).__init__()
        self.batchNorm = batchNorm

        self.Patch_H2L = nn.Sequential(conv(True, 16 * 16, 16 * 8, kernel_size=1), conv(True, 16 * 8, 16, kernel_size=1))

        self.conv0 = ConvAttentionM(3,16,keep_size=True)

        self.conv1a = ConvAttentionM(16,32)
        self.conv1b = EeEM(32, 32)

        self.conv2a = ConvAttentionM(32,64)
        self.conv2b = LocalAttention_MH(64, 1, 4)
        self.atten2 = GlobalAttention_MH(64, 1, 4)
        self.conv2c = conv(self.batchNorm, 64 * 3, 64, kernel_size=1)

        self.conv3a = ConvAttentionM(64,128)
        self.conv3b = LocalAttention_MH(128, 1, 2)
        self.atten3 = GlobalAttention_MH(128, 1, 2)
        self.conv3c = conv(self.batchNorm, 128 * 3, 128, kernel_size=1)

        self.conv4a = ConvAttentionM(128,256)
        self.conv4b = LocalAttention_MH(256, 1, 1)
        self.atten4 = GlobalAttention_MH(256, 1, 1)
        self.conv4c = conv(self.batchNorm, 256 * 3, 256, kernel_size=1)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256 + 32, 128)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128 + 32, 64)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64 + 32, 32)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 + 32, 16)

        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(256, 32, kernel_size=8, stride=8, padding=0), nn.GELU())
        self.conv5_1 = conv(self.batchNorm, 32, 32)
        self.conv5a = conv(self.batchNorm, 32 * 2, 32, kernel_size=3)
        self.conv5b = conv(self.batchNorm, 32, 32, kernel_size=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        out1 = self.conv0(x)

        out2 = self.conv1a(out1)
        out2 = self.conv1b(out2)

        out3 = self.conv2a(out2)
        local_out3 = self.conv2b(out3)
        global_out3 = self.atten2(out3)
        out3 = self.conv2c(torch.cat((local_out3, global_out3,out3), 1))

        out4 = self.conv3a(out3)
        local_out4 = self.conv3b(out4)
        global_out4 = self.atten3(out4)
        out4 = self.conv3c(torch.cat((local_out4, global_out4,out4), 1))

        out5 = self.conv4a(out4)
        local_out5 = self.conv4b(out5)
        global_out5 = self.atten4(out5)
        out5 = self.conv4c(torch.cat((local_out5, global_out5,out5), 1))


        out5_up = self.conv5_1(self.deconv5(out5))
        concat = concatenate_channels([out2, out5_up])
        d2 = self.conv5b(self.conv5a(concat))
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.interpolate(d2, scale_factor=0.5, mode='bilinear', align_corners=False)
        d4 = F.interpolate(d3, scale_factor=0.5, mode='bilinear', align_corners=False)


        out_deconv3 = self.deconv3(out5)

        concat3 = torch.cat((out4, d4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, d3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, d2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, d1, out_deconv0), 1)
        Pixel_feature = self.conv0_1(concat0)

        Patch_feature = self.Patch_H2L(out5)

        return Patch_feature, Pixel_feature

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]







