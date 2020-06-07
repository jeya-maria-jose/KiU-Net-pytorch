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
        self.conv1 = nn.Conv2d(1, 16, 3,stride = 1,padding=1)
        self.conv1_2 = DenseBlock(in_planes = 16)

        self.conv2 = nn.Conv2d(32, 32, 3,stride = 1,padding=1)
        self.conv2_2 = DenseBlock(in_planes = 32)

        self.conv3 = nn.Conv2d(64, 32, 3,stride = 1,padding=1)
        self.conv3_2 = DenseBlock(in_planes = 32)


        self.deconv2 = nn.Conv2d(128,16,3,stride = 1, padding = 1)
        self.deconv2_2 = DenseBlock(in_planes = 16)

        self.deconv3 = nn.Conv2d(64,16,3,stride = 1, padding = 1)
        self.deconv3_2 = DenseBlock(in_planes = 16)

        self.deconv5 = nn.Conv2d(32,2,3,stride = 1, padding = 1)

        self.nconv1 = nn.Conv2d(1, 16, 3,stride = 1,padding=1)
        self.nconv1_2 = DenseBlock(in_planes = 16)

        self.nconv2 = nn.Conv2d(32, 32, 3,stride = 1,padding=1)
        self.nconv2_2 = DenseBlock(in_planes = 32)

        self.nconv3 = nn.Conv2d(64, 32, 3,stride = 1,padding=1)
        self.nconv3_2 = DenseBlock(in_planes = 32)


        self.ndeconv2 = nn.Conv2d(128,16,3,stride = 1, padding = 1)
        self.ndeconv2_2 = DenseBlock(in_planes = 16)

        self.ndeconv3 = nn.Conv2d(64,16,3,stride = 1, padding = 1)
        self.ndeconv3_2 = DenseBlock(in_planes = 16)

        self.ndeconv5 = nn.Conv2d(32,2,3,stride = 1, padding = 1)
        self.final = nn.Conv2d(4,2,3,stride = 1, padding = 1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        org = x
        x = F.relu(self.conv1(x))
        # print(x.shape[1])
        x = self.conv1_2(x)
        # print(x.shape)
        q1 = x

        x = F.upsample(x,scale_factor=(2,2))
        # x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv2_2(x))
        # print(x.shape)
        q2 = x

        x = F.upsample(x,scale_factor=(2,2))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv3_2(x))


        x = F.max_pool2d(x,2)
        x = torch.cat((x,q2),1)

        x = F.relu(self.deconv2(x))
        x= F.relu(self.deconv2_2(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,q1),1)

        x = F.relu(self.deconv3(x))
        x= F.relu(self.deconv3_2(x))
 
        x = F.relu(self.deconv5(x))     


        # output = self.soft(x)

        x2 = F.relu(self.nconv1(org))
        # print(x2.shape[1])
        x2 = self.nconv1_2(x2)
        # print(x2.shape)
        q1 = x2

        # x2 = F.upsample(x2,scale_factor=(2,2))
        x2 = F.max_pool2d(x2,2)
        # x2 = F.max2_pool2d(x2, 2)
        # print(x2.shape)
        x2 = F.relu(self.nconv2(x2))
        # print(x2.shape)
        x2 = F.relu(self.nconv2_2(x2))
        # print(x2.shape)
        q2 = x2
        x2 = F.max_pool2d(x2,2)
        # x2 = F.upsample(x2,scale_factor=(2,2))
        # print(x2.shape)
        x2 = F.relu(self.nconv3(x2))
        # print(x2.shape)
        x2 = F.relu(self.nconv3_2(x2))


        # x2 = F.max2_pool2d(x2,2)
        x2 = F.upsample(x2,scale_factor=(2,2))
        x2 = torch.cat((x2,q2),1)

        x2 = F.relu(self.ndeconv2(x2))
        x2= F.relu(self.ndeconv2_2(x2))
        
        x2 = F.upsample(x2,scale_factor=(2,2))
        x2 = torch.cat((x2,q1),1)

        x2 = F.relu(self.ndeconv3(x2))
        x2= F.relu(self.ndeconv3_2(x2))
 
        x2 = F.relu(self.ndeconv5(x2))

        x = torch.cat((x,x2),1)     
        x = self.final(x)

        output = self.soft(x)



        return output

