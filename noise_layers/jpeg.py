import torch.nn as nn
import torch
import torch.nn.functional as F

class UNetJPEG(nn.Module):
    def __init__(self, num_channels=3):
        super(UNetJPEG, self).__init__()
        
        self.encoder1 = self.conv_block(num_channels + 1, 64)  # Adding +1 to account for the quality factor
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.middle = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, num_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )

    def forward(self, x, quality_factor):
        # Add quality factor to the input as a channel (reshape and repeat it to match the input size)
        b, c, h, w = x.shape
        quality_factor = quality_factor.view(b, 1, 1, 1).repeat(1, 1, h, w)
        x = torch.cat([x, quality_factor], dim=1)

        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        middle = self.middle(F.max_pool2d(e4, 2))

        d4 = self.decoder4(torch.cat([F.interpolate(middle, scale_factor=2), e4], dim=1))
        d3 = self.decoder3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        output = self.final(d1)
        return output

class Jpeg(nn.Module):
    def __init__(self, quality_range, use_sim=False):
        super(Jpeg, self).__init__()
        self.quality_min = quality_range[0]
        self.quality_max = quality_range[1]
        self.device = None
        self.use_sim = use_sim
        self.is_hidden = False

        md_name = f'models/jpegsim_r0.10l0.00010b256_0912-115748-model.pth'
        print(f'load model from {md_name}')
        checkpoint = torch.load(md_name)

        self.jpegsim = UNetJPEG().to(torch.device('cuda'))
        self.jpegsim.load_state_dict(checkpoint['model'])
        for param in self.jpegsim.parameters():
            param.requires_grad = False

    def __str__(self):
        return f'Jpeg({self.quality_min},{self.quality_max})'
    
    def unet_jpeg(self, noised_and_cover):
        images = noised_and_cover[1]
        q = torch.randint(int(self.quality_min), int(self.quality_max), (images.shape[0],)).to(images.device)
        out = self.jpegsim(images, q/100)
        noised_and_cover[0] = out
        return noised_and_cover

    def forward(self, noised_and_cover):
        return self.unet_jpeg(noised_and_cover)
