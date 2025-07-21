import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):


    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class P2SP(nn.Module):
    def __init__(self, dim=16, stoken_size=[16,16]):
        super().__init__()
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.para = torch.nn.Parameter(
            torch.tensor([[[[0.8, 0.9, 0.8, 0.9, 1, 0.9, 0.8, 0.9, 0.8]]]]).expand((1, 1, 16 * 16, 9)).cuda())
        self.para.retain_grad()


    def stoken_forward(self, x,stoken_features):
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        stoken_features = F.unfold(stoken_features, (3, 3), padding=1)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)


        affinity_matrix = pixel_features @ stoken_features * self.scale   # (B, hh*ww, h*w, 9)
        affinity_matrix = affinity_matrix * self.para
        affinity_matrix = affinity_matrix.softmax(-1)

        affinity_matrix = affinity_matrix.reshape(B, hh , ww, h , w, 9).permute(0,5,1,3,2,4).reshape(B,9,hh*h,ww*w)

        return affinity_matrix

    def forward(self, x,stoken_features):
        return self.stoken_forward(x,stoken_features)


#***********************************************************************

#set padding by myself
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
#***********************************************************************
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.GELU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.GELU()
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.GELU()
    )

