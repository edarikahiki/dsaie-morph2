import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Core Blocks --------------------

class DoubleConv3D(nn.Module):
    """(3D convolution => BN => ReLU => squeeze D => 2D Convolution)
    use this as initial convolution that use 3D Convolution
    3D Convolution means: including temporal scale"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, temporal_kernel=1, drop_channels=False, p_drop=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        # if temporal_kernel == 1:
        #     temporal_pad = 0
        # else:
        #     temporal_pad = 1

        # 3D conv on (B, C, T, H, W)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(temporal_kernel, kernel_size, kernel_size),
                      padding=(0, kernel_size//2, kernel_size//2), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 2D conv on (B, C, H, W)
        self.conv2d = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size,
                      padding=(kernel_size//2), bias=False
                      )
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)
        
        if drop_channels and p_drop is not None:
            self.dropout = nn.Dropout3d(p = p_drop)
        else:
            self.dropout = None

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.conv3d(x)
        if self.dropout is not None:
            x = self.dropout(x)
        #squeeze depth dimension
        x = x.squeeze(2)            # (N, mid_channels, H, W)
        x = self.conv2d(x)          # (N, out_channels, H, W)
        return x


class DoubleConv2D(nn.Module):
    """Standard 2D double conv"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, temporal_kernel=1, drop_channels=False, p_drop=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
    
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size,
                      padding= kernel_size//2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size,
                      padding=(kernel_size//2), bias=False)
        )

        if drop_channels and p_drop is not None:
            self.double_conv.add_module("dropout", nn.Dropout2d(p=p_drop))

    def forward(self, x):
        # x: (N, C, H, W)
        return self.double_conv(x)


class Down2D(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv2D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, drop_channels=False, p_drop=None, 
                 pool_temporal=False, pooling='max'):
        super().__init__()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=2)

        self.conv = DoubleConv2D(in_channels, out_channels, kernel_size=kernel_size,
                                 drop_channels=drop_channels, p_drop=p_drop)
    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return x


class Up2D(nn.Module):
    """Upscaling then DoubleConv2D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, drop_channels=False, p_drop=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_out_channels = in_channels // 2
            
        self.conv = DoubleConv2D(in_channels, out_channels, mid_channels=in_channels // 2,
                                 kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad spatial dimensions if needed
        # diffD = x2.size(2) - x1.size(2)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        # diffD // 2, diffD - diffD // 2
                        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------- UNet2.5D --------------------

class UNet25D_addlayer(nn.Module):
    def __init__(self, n_channels, n_classes, init_hid_dim=16, kernel_size=3, temporal_kernel=1, pooling='max', bilinear=False,
                 drop_channels=False, p_drop=None, pool_temporal=False):
        """
        n_channels: input channels (usually 1 for grayscale)
        n_classes: output channels (e.g., segmentation classes)
        init_hid_dim: initial hidden feature maps
        pool_temporal: whether to pool over temporal dimension
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_hid_dim = init_hid_dim 
        self.bilinear = bilinear
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.drop_channels = drop_channels
        self.p_drop = p_drop
        self.temporal_kernel = temporal_kernel
        hid_dims = [init_hid_dim * (2**i) for i in range(6)]
        self.hid_dims = hid_dims

        self.inc = DoubleConv3D(n_channels, hid_dims[0], kernel_size=kernel_size, temporal_kernel=temporal_kernel, drop_channels=drop_channels, p_drop=p_drop)
        self.down1 = Down2D(hid_dims[0], hid_dims[1], kernel_size, drop_channels, p_drop, pool_temporal, pooling)
        self.down2 = Down2D(hid_dims[1], hid_dims[2], kernel_size, drop_channels, p_drop, pool_temporal, pooling)
        self.down3 = Down2D(hid_dims[2], hid_dims[3], kernel_size, drop_channels, p_drop, pool_temporal, pooling)
        self.down4 = Down2D(hid_dims[3], hid_dims[4], kernel_size, drop_channels, p_drop, pool_temporal, pooling)
        self.down5 = Down2D(hid_dims[4], hid_dims[5], kernel_size, drop_channels, p_drop, pool_temporal, pooling)

        self.up1 = Up2D(hid_dims[5], hid_dims[4], kernel_size, bilinear, drop_channels, p_drop)
        self.up2 = Up2D(hid_dims[4], hid_dims[3], kernel_size, bilinear, drop_channels, p_drop)
        self.up3 = Up2D(hid_dims[3], hid_dims[2], kernel_size, bilinear, drop_channels, p_drop)
        self.up4 = Up2D(hid_dims[2], hid_dims[1], kernel_size, bilinear, drop_channels, p_drop)
        self.up5 = Up2D(hid_dims[1], hid_dims[0], kernel_size, bilinear, drop_channels, p_drop)

        self.outc = OutConv2D(hid_dims[0], n_classes)
        self.sigmoid = nn.Sigmoid()  # for binary segmentation; use Softmax(dim=1) for multiclass

    def forward(self, x):
        # x: (B, T, H, W) -> add channel dim
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, T, H, W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        x = self.outc(x)
        x = self.sigmoid(x)

        if self.n_classes == 1:
            x = x.squeeze(1)
        return x
