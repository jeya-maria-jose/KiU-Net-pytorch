import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

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
        x = torch.cat((org,d1,d2,d3,d4),1)

        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3,stride = 1,padding=1)
        self.conv1_2 = nn.Conv2d(8, 8, 3,stride = 1,padding=1)
        # self.conv1_decon = nn.ConvTranspose2d(8,8,2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, 3,stride = 1,padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, 3,stride=1,padding=1)
        # self.conv2_decon = nn.ConvTranspose2d(16,16,2,stride=2)

        self.conv3 = nn.Conv2d(16, 32, 3,stride = 1,padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, 3,stride=1,padding=1)
        # self.conv3_decon = nn.ConvTranspose2d(64,64,2,stride=2)

        self.conv4 = nn.Conv2d(32, 64, 3,stride = 1,padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, 3,stride = 1,padding=1)
        # self.conv4_decon = nn.ConvTranspose2d(128,128,2,stride=2)

        self.conv5 = nn.Conv2d(64, 64, 3,stride = 1,padding=1)
        self.conv5_2 = nn.Conv2d(64, 64, 3,stride = 1,padding=1)

        
        self.deconv1 = nn.Conv2d(128,64,3,stride = 1, padding = 1)
        self.deconv1_2 = nn.Conv2d(64,32,3,stride = 1, padding = 1)

        # self.deconv2_decon = nn.Conv2d(128,64,2)
        self.deconv2 = nn.Conv2d(64,32,3,stride = 1, padding = 1)
        self.deconv2_2 = nn.Conv2d(32,16,3,stride = 1, padding = 1)

        # self.deconv3_decon = nn.Conv2d(64,32,2)
        self.deconv3 = nn.Conv2d(32,16,3,stride = 1, padding = 1)
        self.deconv3_2 = nn.Conv2d(16,8,3,stride = 1, padding = 1)

        # self.deconv4_decon = nn.Conv2d(32,16,2)
        self.deconv4 = nn.Conv2d(16,8,3,stride = 1, padding = 1)
        self.deconv4_2 = nn.Conv2d(8,8,3,stride = 1, padding = 1)

        self.deconv5 = nn.Conv2d(8,2,3,stride = 1, padding = 1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_2(x))
        
        q1 = x
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_2(x))
        q2 = x
        
        x = F.max_pool2d(x,2)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv3_2(x))
        # print(x.shape)
        q3 = x
        # x = self.conv3_decon(x)
        x = F.max_pool2d(x,2)
        # print(x.shape)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4_2(x))
        q4 = x
        # x = self.conv4_decon(x)
        x = F.max_pool2d(x,2)
        # print(x.shape)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv5_2(x))
        # print(x.shape)
        
        x = F.upsample(x,scale_factor=(2,2))
        
        # print(x.shape)
        x = torch.cat((x,q4),1)
        # print(x.shape)

        x = F.relu(self.deconv1(x))
        x= F.relu(self.deconv1_2(x))
        x = F.upsample(x,scale_factor=(2,2))
        
        x = torch.cat((x,q3),1)

        x = F.relu(self.deconv2(x))
        x= F.relu(self.deconv2_2(x))
        x = F.upsample(x,scale_factor=(2,2))
        
        x = torch.cat((x,q2),1)

        x = F.relu(self.deconv3(x))
        x= F.relu(self.deconv3_2(x))
        x = F.upsample(x,scale_factor=(2,2))
        
        x = torch.cat((x,q1),1)

        x = F.relu(self.deconv4(x))
        x=F.relu(self.deconv4_2(x))

        x = F.relu(self.deconv5(x))     

        output = self.soft(x)
        return output
