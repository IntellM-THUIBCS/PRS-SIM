import torch
import torch.nn as nn

"""
# --------------------------------------------
https://github.com/LabForComputationalVision/bias_free_denoising/blob/master/models/unet.py
# --------------------------------------------
"""


# --------------------------------------------
# unet in noise2noise paper
# --------------------------------------------

class UNet(nn.Module):

    def __init__(self, in_nc=1, out_nc=1):
        '''
        initialize the unet
        '''
        super(UNet, self).__init__()
        in_channels = in_nc
        out_channels = out_nc
        self.encode1 = nn.Sequential(
            nn.Conv3d(in_channels, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(48, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((2,2,2)))

        self.encode2 = nn.Sequential(
            nn.Conv3d(48, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((2,2,2)))

        self.encode3 = nn.Sequential(
            nn.Conv3d(48, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1,2,2)))

        self.encode4 = nn.Sequential(
            nn.Conv3d(48, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1,2,2)))

        self.encode5 = nn.Sequential(
            nn.Conv3d(48, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d((1,2,2)))

        self.encode6 = nn.Sequential(
            nn.Conv3d(48, 48, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.ConvTranspose3d(48, 48, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1), bias=False))

        self.decode1 = nn.Sequential(
            nn.Conv3d(96, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1), bias=False))
        
        self.decode2 = nn.Sequential(
            nn.Conv3d(144, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1), bias=False))
        
        self.decode3 = nn.Sequential(
            nn.Conv3d(144, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1), bias=False))
        
        self.decode4 = nn.Sequential(
            nn.Conv3d(144, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(96, 96, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose3d(96, 96, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1), bias=False))
        
        self.decode5 = nn.Sequential(
            nn.Conv3d(96 + in_channels, 64, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(64, 32, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.01))

        self.output_layer = nn.Conv3d(32, out_channels, (3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)

        self._init_weights()

    def forward(self, x): # z_size is 4
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)
        # print(pool5.shape)
        upsample5 = self.encode6(pool5)
        # print(upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode1(concat5)
        # print(upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode2(concat4)
        # print(upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode3(concat3)
        # print(upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode4(concat2)
        # print(upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=1)
        upsample0 = self.decode5(concat1)
        # print(upsample0.shape)
        output = self.output_layer(upsample0)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d) :
                nn.init.kaiming_normal_(m.weight.data)