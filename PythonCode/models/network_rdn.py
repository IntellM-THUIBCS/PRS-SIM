import torch.nn as nn
import torch

'''
https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
https://arxiv.org/abs/1802.08797
'''

class RDB_Con(nn.Module):
    def __init__(self, inChannels, growRate):
        super(RDB_Con, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv_layer = nn.Sequential(*[
            nn.Conv2d(Cin, G, (3,3), padding=(1,1), stride=(1,1)),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv_layer(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Con(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, (1,1), padding=(0,0), stride=(1,1))

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self,  in_nc=1, out_nc=1, para1=3):
        super(RDN, self).__init__()
        G0 = 64
        if para1 == 1:
            RDNconfig = 'A'
        elif para1 == 2:
            RDNconfig = 'B'
        else:
            RDNconfig = 'C'

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (10, 6, 32),
        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_nc, G0, (3,3), padding=(1,1), stride=(1,1))
        self.SFENet2 = nn.Conv2d(G0, G0, (3,3), padding=(1,1), stride=(1,1))

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, (1,1), padding=(0,0), stride=(1,1)),
            nn.Conv2d(G0, G0, (3,3), padding=(1,1), stride=(1,1))
        ])

        # output net
        self.UPNet = nn.Sequential(*[nn.Conv2d(G0, out_nc, (3,3), padding=(1,1), stride=(1,1))])


    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return self.UPNet(x)