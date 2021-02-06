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
        # out = self.soft(out)
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
        
        # out = self.soft(out)
        return out

class kinetwithsk(nn.Module):
    def __init__(self):
        super(kinetwithsk, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        # self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        # self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        # self.decoderf3 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        # self.encoderf1 =   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        # self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
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
        
        # out = self.soft(out)
        return out

class kitenet(nn.Module):
    
    def __init__(self):
        super(kitenet, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        # self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        # self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        
        out = F.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))

        out = F.relu(F.max_pool2d(self.decoder3(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder4(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder5(out),2,2))

        # out = self.soft(out)
        
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
        

        # out = self.soft(out)
        
        return out

class reskiunet(nn.Module):
    
    def __init__(self):
        super(reskiunet, self).__init__()
        

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1) 
        self.en1 = nn.Conv2d(3, 16, 1, stride=1, padding=0) # b, 16, 10, 10
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.en2=   nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3=   nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.de1 =   nn.Conv2d(64, 32, 1, stride=1, padding=0) 
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2 =   nn.Conv2d(32,16, 1, stride=1, padding=0)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3 =   nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1 =   nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2=   nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3 =   nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.enf1 =   nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2=   nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3 =   nn.Conv2d(32, 64, 1, stride=1, padding=0)
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

        out = torch.add(self.en1(x),self.encoder1(x))  #init
        out = F.relu(self.en1_bn(F.max_pool2d(out,2,2))) # U-Net
        out1 = torch.add(self.enf1(x),self.encoder1(x)) #init
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))) # ki-net

        tmp = out

        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1


        out = torch.add(self.en2(out),self.encoder2(out)) #res
        out1 = torch.add(self.enf2(out1),self.encoderf2(out1)) #res

        out = F.relu(self.en2_bn(F.max_pool2d(out,2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(out1,scale_factor=(2,2),mode ='bilinear')))
        
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = torch.add(self.en3(out),self.encoder3(out)) #res
        out1 = torch.add(self.enf3(out1),self.encoderf3(out1)) #res

        out = F.relu(self.en3_bn(F.max_pool2d(out,2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(out1,scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = torch.add(self.de1(out),self.decoder1(out)) #res
        out1 = torch.add(self.def1(out1),self.decoderf1(out1)) #res
        out = F.relu(self.de1_bn(F.interpolate(out,scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def1_bn(F.max_pool2d(out1,2,2)))
        

        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = torch.add(self.de2(out),self.decoder2(out)) #res
        out1 = torch.add(self.def2(out1),self.decoderf2(out1)) #res
        out = F.relu(self.de2_bn(F.interpolate(out,scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool2d(out1,2,2)))

        tmp = out
        
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = torch.add(self.de3(out),self.decoder3(out)) #res
        out1 = torch.add(self.def3(out1),self.decoderf3(out1)) #res
        out = F.relu(self.de3_bn(F.interpolate(out,scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(out1,2,2)))

        

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        

        # out = self.soft(out)
        # print(out.shape)
        return out

class DenseBlock(nn.Module):

    def __init__(self, in_planes):
        super(DenseBlock, self).__init__()
        # print(int(in_planes/4))
        self.c1 = nn.Conv2d(in_planes,in_planes,1,stride=1, padding=0)
        self.c2 = nn.Conv2d(in_planes,int(in_planes/4),3,stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(in_planes)
        self.b2 = nn.BatchNorm2d(int(in_planes/4))
        self.c3 = nn.Conv2d(in_planes+int(in_planes/4),in_planes,1,stride=1, padding=0)
        self.c4 = nn.Conv2d(in_planes,int(in_planes/4),3,stride=1, padding=1)        

        self.c5 = nn.Conv2d(in_planes+int(in_planes/2),in_planes,1,stride=1, padding=0)
        self.c6 = nn.Conv2d(in_planes,int(in_planes/4),3,stride=1, padding=1)        

        self.c7 = nn.Conv2d(in_planes+3*int(in_planes/4),in_planes,1,stride=1, padding=0)
        self.c8 = nn.Conv2d(in_planes,int(in_planes/4),3,stride=1, padding=1)        
                    
    def forward(self, x):
        org = x
        # print(x.shape)
        x= F.relu(self.b1(self.c1(x)))
        # print(x.shape)
        x= F.relu(self.b2(self.c2(x)))
        d1 = x
        # print(x.shape)
        x = torch.cat((org,d1),1)
        x= F.relu(self.b1(self.c3(x)))
        x= F.relu(self.b2(self.c4(x)))
        d2= x
        x = torch.cat((org,d1,d2),1)
        x= F.relu(self.b1(self.c5(x)))
        x= F.relu(self.b2(self.c6(x)))
        d3= x
        x = torch.cat((org,d1,d2,d3),1)
        x= F.relu(self.b1(self.c7(x)))
        x= F.relu(self.b2(self.c8(x)))
        d4= x
        x = torch.cat((d1,d2,d3,d4),1)
        x = torch.add(org,x)
        return x


class densekiunet(nn.Module):
    
    def __init__(self):
        super(densekiunet, self).__init__()
        

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1) 
        self.en1 =  DenseBlock(in_planes = 16) # b, 16, 10, 10
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.en2=    DenseBlock(in_planes = 32)
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3=    DenseBlock(in_planes = 64)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.de1 =   DenseBlock(in_planes = 32)
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2 =    DenseBlock(in_planes = 16)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3 =   DenseBlock(in_planes = 8)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1 =    DenseBlock(in_planes = 32)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2=    DenseBlock(in_planes = 16)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3 =    DenseBlock(in_planes = 8)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.enf1 =    DenseBlock(in_planes = 16)
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2=    DenseBlock(in_planes = 32)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3 =    DenseBlock(in_planes = 64)
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

        out = F.relu(self.en1_bn(F.max_pool2d(self.en1(self.encoder1(x)),2,2)))
        out1 = F.relu(self.enf1_bn(F.interpolate(self.enf1(self.encoderf1(x)),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(self.en2_bn(F.max_pool2d(self.en2(self.encoder2(out)),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.enf2(self.encoderf2(out1)),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.en3(self.encoder3(out)),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.enf3(self.encoderf3(out1)),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(self.de1_bn(F.interpolate(self.de1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.def1(self.decoderf1(out1)),2,2)))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(self.de2_bn(F.interpolate(self.de2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.def2(self.decoderf2(out1)),2,2)))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(self.de3_bn(F.interpolate(self.de3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.def3(self.decoderf3(out1)),2,2)))

        

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        

        # out = self.soft(out)
        # print(out.shape)
        return out

class kiunet3d(nn.Module): #

    def __init__(self, c=4,n=1,channels=128,groups = 16,norm='bn', num_classes=5):
        super(kiunet3d, self).__init__()

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

