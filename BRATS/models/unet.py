import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y

class KnetU(nn.Module):
    def __init__(self, planes, norm='bn', first=False):
        super(KnetU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        # self.conv3 = nn.Conv3d(planes//2, planes//2, 3, 1, 1, bias=False)
        # self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        a,b,c,d,e = x.shape
        # print(a,b,c,d,e)
        y = F.upsample(x, size = (2*c,2*d,e), mode='trilinear', align_corners=False)
        # print(y.shape)
        y = self.relu(self.conv2(y))

        # y = torch.cat([prev, y], 1)
        # y = self.relu(self.bn3(self.conv3(y)))

        return y

class KnetD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='in', first=False):
        super(KnetD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0,1,4,2,3)
        if not self.first:
            x = self.maxpool(x.squeeze(0))
        x = x.unsqueeze(0)
        x = x.permute(0,1,3,4,2)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)
class Unet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=5):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm, True)
        self.convu2 = ConvU(4*n, norm, True)
        self.convu1 = ConvU(2*n, norm, True)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        # print(x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        # y3 = self.seg3(y3)
        # y2 = self.seg2(y2) + self.upsample(y3)
        # y1 = self.seg1(y1) + self.upsample(y2)

        return y1

class segnet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=5):
        super(segnet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm,True)
        self.convu2 = ConvU(4*n, norm,True)
        self.convu1 = ConvU(2*n, norm,True)

        self.seg3 = nn.Conv3d(4*n, num_classes, 1)
        self.seg2 = nn.Conv3d(2*n, num_classes, 1)
        self.seg1 = nn.Conv3d(n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        # print(x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + self.upsample(y2)

        return y1

class kiunet3d(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=5):
        super(kiunet3d, self).__init__()
        # print("HERE")
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.kconvu1 = KnetU(n, norm, first = True)
        self.kconvu2 = KnetU(n//2, norm, first = False)
        self.kconvu3 = KnetU(n//2, norm, first = False)
        # self.kconvu4 = KnetU(n//4, norm, first = False)
        # self.kconvd1 = KnetD(n//2,n//2,dropout,norm)
        self.kconvd2 = KnetD(1,n//2,dropout,norm)
        self.kconvd3 = KnetD(n//2,n,dropout,norm)
        self.kconvd4 = KnetD(n,n*2,dropout,norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        self.segk = nn.Conv3d(n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        # print(x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        k1 = self.kconvu1(x1)
        # print(k1.shape)
        k2 = self.kconvu2(k1)
        # print(k2.shape)
        k3 = k2
        # k3 = self.kconvu3(k2)
        # print(k3.shape)
        # k4 = self.kconvu4(k3)

        # k3 = self.kconvd1(k3)
        # print(k3.shape)
        # k3 = torch.add(k3,k2)
        k3 = self.kconvd2(k3)
        k3 = torch.add(k3,k1)
        k3 = self.kconvd3(k3)
        # k3 = self.kconvd4(k3)
        k3 = self.segk(k3)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        # print(y1.shape)


        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + self.upsample(y2)
        # print(y1.shape,k3.shape)
        y1 = torch.add(y1,k3)

        return y1

class kiunet3dwcrfb(nn.Module): #

    def __init__(self, c=4,n=1,channels=128,groups = 16,norm='bn', num_classes=5):
        super(kiunet3dwcrfb, self).__init__()

        # Entry flow
        self.encoder1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=1, bias=False)# H//2
        self.encoder2 = nn.Conv3d( n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.encoder3 = nn.Conv3d( 2*n, 4*n, kernel_size=3, padding=1, stride=1, bias=False)
        
        self.kencoder1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kencoder2 = nn.Conv3d( n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kencoder3 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)

        self.downsample1 = nn.MaxPool3d(2, stride=2)
        self.downsample2 = nn.MaxPool3d(2, stride=2)
        self.downsample3 = nn.MaxPool3d(2, stride=2)
        self.kdownsample1 = nn.MaxPool3d(2, stride=2)
        self.kdownsample2 = nn.MaxPool3d(2, stride=2)
        self.kdownsample3 = nn.MaxPool3d(2, stride=2)
        

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.kupsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8        
        self.kupsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.kupsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        
        self.decoder1 = nn.Conv3d( 4*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)        
        self.decoder2 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)        
        self.decoder3 = nn.Conv3d( 2*n, c, kernel_size=3, padding=1, stride=1, bias=False)        
        self.kdecoder1 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)        
        self.kdecoder2 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)        
        self.kdecoder3 = nn.Conv3d( 2*n, c, kernel_size=3, padding=1, stride=1, bias=False)        
        
        self.intere1_1 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv3d(2*n,4*n,3, stride=1, padding=1)
        # self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv3d(4*n,2*n,3, stride=1, padding=1)
        # self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.intd3_2bn = nn.BatchNorm2d(64)

        self.seg = nn.Conv3d(c, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))  #U-Net branch
        out1 = F.relu(F.interpolate(self.kencoder1(x),scale_factor=2,mode ='trilinear')) #Ki-Net branch
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intere1_1(out1)),scale_factor=0.25,mode ='trilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.intere1_2(tmp)),scale_factor=4,mode ='trilinear')) #CRFB
        
        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.kencoder2(out1),scale_factor=2,mode ='trilinear'))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intere2_1(out1)),scale_factor=0.0625,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intere2_2(tmp)),scale_factor=16,mode ='trilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.kencoder3(out1),scale_factor=2,mode ='trilinear'))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intere3_1(out1)),scale_factor=0.015625,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intere3_2(tmp)),scale_factor=64,mode ='trilinear'))
        
        ### End of encoder block

        ### Start Decoder
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=2,mode ='trilinear'))  #U-NET
        out1 = F.relu(F.max_pool3d(self.kdecoder1(out1),2,2)) #Ki-NET
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.interd1_1(out1)),scale_factor=0.0625,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.interd1_2(tmp)),scale_factor=16,mode ='trilinear'))
        
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=2,mode ='trilinear'))
        out1 = F.relu(F.max_pool3d(self.kdecoder2(out1),2,2))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.interd2_1(out1)),scale_factor=0.25,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.interd2_2(tmp)),scale_factor=4,mode ='trilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=2,mode ='trilinear'))
        out1 = F.relu(F.max_pool3d(self.kdecoder3(out1),2,2))

        

        out = torch.add(out,out1) # fusion of both branches

        out = F.relu(self.seg(out))  #1*1 conv
        

        # out = self.soft(out)
        return out