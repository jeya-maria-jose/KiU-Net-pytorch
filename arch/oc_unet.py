import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

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
        # self.deconv5_2 = nn.Conv2d(256,128,3,stride = 1, padding = 0)

        # self.encoder1 = nn.Conv2d(1, 16, 3, stride = 1,padding=1)  
        # self.encoder1_2 = nn.Conv2d(16, 16, 3, stride = 1,padding=1)  
                
        # self.encoder2 =   nn.Conv2d(16, 32, 3, stride  =1,padding=1)
        # self.encoder2_2 =   nn.Conv2d(32, 32, 3, stride  =1,padding=1)
        
        # self.encoder3 =   nn.Conv2d(32, 64, 3, stride  =1,padding=1)
        # self.encoder3_2 =   nn.Conv2d(64, 64, 3, stride  =1,padding=2)
        
        # self.encoder4 =   nn.Conv2d(64, 128, 3, stride  =1,padding=1)  
        # self.encoder4_2 =   nn.Conv2d(128, 128, 3, stride  =1,padding=2)
                
        # self.encoder5 =   nn.Conv2d(128, 256, 3, stride  =1,padding=1)  
        # self.encoder5_2 =   nn.Conv2d(256, 256, 3, stride  =1,padding=2)

        # self.decoder1_decon = nn.ConvTranspose2d(256,128,2,padding=0)        
        # self.decoder1 = nn.Conv2d(256, 128, 3, stride = 1,padding=2)  
        # self.decoder1_2 = nn.Conv2d(128, 64, 3, stride = 1,padding=1)          

        # self.decoder2_decon = nn.ConvTranspose2d(64,64,2,padding=0)
        # self.decoder2 =   nn.Conv2d(128, 64, 3, stride =1,padding=2)
        # self.decoder2_2 =   nn.Conv2d(64, 32, 3, stride  =1,padding=2)
        
        # self.decoder3_decon = nn.ConvTranspose2d(32,32,2,padding=0)
        # self.decoder3 =   nn.Conv2d(64, 32, 3, stride  =1,padding=4)
        # self.decoder3_2 =   nn.Conv2d(32, 16, 3, stride  =1,padding=4)
        
        # self.decoder4_decon = nn.ConvTranspose2d(16,16,2,padding=0)
        # self.decoder4 =   nn.Conv2d(32, 16, 3, stride  =1,padding=1)  
        # self.decoder4_2 =   nn.Conv2d(16, 16, 3, stride  =1,padding=1)
                
        # self.decoder5 =   nn.Conv2d(16, 2, 1, stride  =1,padding=0)  
        # self.decoder5_2 =   nn.Conv2d(256, 256, 3, stride  =1,padding=2)
        self.soft = nn.Softmax(dim =1)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv1_2(x))
        # print(x.shape)
        q1 = x
        # x = self.conv1_decon(x)
        x = F.upsample(x,scale_factor=(2,2))
        # x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv2_2(x))
        # print(x.shape)
        q2 = x
        # x = self.conv2_decon(x)
        x = F.upsample(x,scale_factor=(2,2))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv3_2(x))
        # print(x.shape)
        q3 = x
        # x = self.conv3_decon(x)
        x = F.upsample(x,scale_factor=(2,2))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4_2(x))
        q4 = x
        # x = self.conv4_decon(x)
        x = F.upsample(x,scale_factor=(2,2))
        # print(x.shape)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv5_2(x))
        # print(x.shape)
        
        x = F.max_pool2d(x,2)
        # print(x.shape)
        x = torch.cat((x,q4),1)
        # print(x.shape)

        x = F.relu(self.deconv1(x))
        x= F.relu(self.deconv1_2(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,q3),1)

        x = F.relu(self.deconv2(x))
        x= F.relu(self.deconv2_2(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,q2),1)

        x = F.relu(self.deconv3(x))
        x= F.relu(self.deconv3_2(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,q1),1)

        x = F.relu(self.deconv4(x))
        x=F.relu(self.deconv4_2(x))

        x = F.relu(self.deconv5(x))     

        output = self.soft(x)
        return output

