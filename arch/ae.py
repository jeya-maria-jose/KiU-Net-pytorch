# Code for KiU-Net
# Author: Jeya Maria Jose
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        

        self.encoder1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(1024, 512, 3, stride=1,padding=2)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(512, 256, 3, stride=1, padding=2)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(64, 2, 3, stride=1, padding=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape)
        out = self.soft(out)
        return out

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        
        self.encoder1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        t4 = out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        
        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape,t4.shape)
        out = torch.add(out,t4)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape)
        
        out = self.soft(out)
        return out

class kinetwithsk(nn.Module):
    def __init__(self):
        super(kinetwithsk, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        out = F.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        t1 = out
        out = F.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        t2 = out
        out = F.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape)

        out = F.relu(F.max_pool2d(self.decoder3(out),2,2))
        out = torch.add(out,t2)
        out = F.relu(F.max_pool2d(self.decoder4(out),2,2))
        out = torch.add(out,t1)
        out = F.relu(F.max_pool2d(self.decoder5(out),2,2))
        
        out = self.soft(out)
        return out

class kinet(nn.Module):
    
    def __init__(self):
        super(kinet, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 15, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 8, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 5, stride=1, padding=1)
        
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        
        self.decoder3 =   nn.Conv2d(128, 64, 5, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 8, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 15, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        
        out = F.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))

        out = F.relu(F.max_pool2d(self.decoder3(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder4(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder5(out),2,2))

        out = self.soft(out)
        
        return out

class kiunet(nn.Module):
    
    def __init__(self):
        super(kiunet, self).__init__()
        

        self.encoder1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)   
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(1, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(8,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x),2,2)))  #U-Net branch
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))) #Ki-Net branch
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bilinear')) #CRFB
        
        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        ### Start Decoder
        
        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear')))  #U-NET
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1),2,2))) #Ki-NET
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1),2,2)))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1),2,2)))

        

        out = torch.add(out,out1) # fusion of both branches

        out = F.relu(self.final(out))  #1*1 conv
        

        out = self.soft(out)
        
        return out

