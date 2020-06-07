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

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        # self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        # self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        # self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        # self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        # self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        # self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        # self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

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

class unetautoencoder(nn.Module):
    def __init__(self):
        super(unetautoencoder, self).__init__()
        
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

class ocwithsk(nn.Module):
    def __init__(self):
        super(ocwithsk, self).__init__()
        
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

class upautoencoder(nn.Module):
    
    def __init__(self):
        super(upautoencoder, self).__init__()
        
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
        
        # self.decoderf4 =   nn.Conv2d(8, 4, 3, stride=1, padding=1)
        # self.decoderf5 =   nn.Conv2d(4, 2, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        
        # out2,indices1 = F.max_pool2d(self.encoder1(out2),2,2,return_indices=True)
        # out2,indices2 = F.max_pool2d(self.encoder2(out2),2,2,return_indices=True)
        # out2,indices3 = F.max_pool2d(self.encoder3(out2),2,2,return_indices=True)
        # print(indices3.shape)
        # out = F.relu(F.max_pool2d(self.encoder1(x),4,4))
                
        # out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        # out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        # out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        
        # out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        # out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))


        out = F.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape)

        out = F.relu(F.max_pool2d(self.decoder3(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder4(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder5(out),2,2))

        # out = F.relu(self.decoderf1(out))
        # out = F.relu(self.decoderf2(out))
        # out = F.relu(self.decoderf3(out))
        # out = F.relu(self.decoderf4(out))
        # out = F.relu(self.decoderf5(out))

        out = self.soft(out)
        
        return out


class combautoencoder(nn.Module):
    
    def __init__(self):
        super(combautoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
                
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class unetcombautoencoder(nn.Module):
    
    def __init__(self):
        super(unetcombautoencoder, self).__init__()
        
        # self.encoder1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        # self.encoder2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        # self.encoder3=   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        # self.decoder1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        # self.decoder2 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoder3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.decoderf2=   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.encoderf1 =   nn.Conv2d(1, 64, 3, stride=1, padding=1)
        # self.encoderf2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoderf3 =   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.encoder1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.final = nn.Conv2d(8,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class knet(nn.Module):
    
    def __init__(self):
        super(knet, self).__init__()
        
        # self.encoder1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        # self.encoder2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        # self.encoder3=   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        # self.decoder1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        # self.decoder2 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoder3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.decoderf2=   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.encoderf1 =   nn.Conv2d(1, 64, 3, stride=1, padding=1)
        # self.encoderf2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoderf3 =   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.encoder1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)

        
        
        self.final = nn.Conv2d(8,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class knetplus(nn.Module):
    
    def __init__(self):
        super(knetplus, self).__init__()
        
        # self.encoder1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        # self.encoder2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        # self.encoder3=   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        # self.decoder1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        # self.decoder2 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoder3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.decoderf2=   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.encoderf1 =   nn.Conv2d(1, 64, 3, stride=1, padding=1)
        # self.encoderf2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoderf3 =   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.encoder1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.intere1_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intere2_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intere3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        
        self.intere1_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intere2_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intere3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        
        self.interd1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.interd2_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        
        self.interd1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.interd2_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        
        self.final = nn.Conv2d(8,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(self.intere1_1(out1),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.intere1_2(out),scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(self.intere2_1(out1),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.intere2_2(out),scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(self.intere3_1(out1),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.intere3_2(out),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out = torch.add(out,F.interpolate(self.interd1_1(out1),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.interd1_2(out),scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out = torch.add(out,F.interpolate(self.interd2_1(out1),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.interd2_2(out),scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class knetplusbn(nn.Module):
    
    def __init__(self):
        super(knetplusbn, self).__init__()
        
        # self.encoder1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        # self.encoder2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        # self.encoder3=   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        # self.decoder1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        # self.decoder2 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoder3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.decoderf2=   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.decoderf3 =   nn.Conv2d(64, 16, 3, stride=1, padding=1)

        # self.encoderf1 =   nn.Conv2d(1, 64, 3, stride=1, padding=1)
        # self.encoderf2=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoderf3 =   nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  # b, 1, 28, 28
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

        self.encoderf1 =   nn.Conv2d(3, 16, 3, stride=1, padding=1)
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

        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x),2,2)))
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear')))
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(out))),scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear')))
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(out))),scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')))
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(out))),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1),2,2)))
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(out))),scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1),2,2)))
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(out))),scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1),2,2)))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out


class deepunetcombautoencoder(nn.Module):
    
    def __init__(self):
        super(deepunetcombautoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        


        # self.decoderf1=   nn.Conv2d(4, 8, 3, stride=1, padding=1)
        # self.decoderf2 =   nn.Conv2d(8, 128, 3, stride=1, padding=1)
        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        
        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoderf4 =   nn.Conv2d(128, 8, 3, stride=1, padding=1)
        # self.encoderf5 =   nn.Conv2d(8, 4, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16,2,1,stride=1,padding=0)

        # self.tmp1 = nn.Conv2d(8, 32, 3, stride=1, padding=1)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(64,64),mode ='bilinear'))
        
        u3 = out
        # o3 = out1

        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        # out1 = F.relu(F.interpolate(self.encoderf4(out1),scale_factor=(2,2),mode ='bilinear'))
        u4 = out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        
        # print(out.shape,u4.shape)
        out = torch.add(out,u4)
        
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,u3)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)
        # print(out.shape,out1.shape)
        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out


class fulldeepunetcombautoencoder(nn.Module):
    
    def __init__(self):
        super(fulldeepunetcombautoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv2d(128, 200, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(200, 300, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(300, 200, 3, stride=1, padding=1)
        self.decoder2 =   nn.Conv2d(200, 128, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(1.5,1.5),mode ='bilinear'))
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(out1,scale_factor=((1/3),(1/3)),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(3,3),mode ='bilinear'))
        
        u1 = out
        o1 = out1

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(1.5,1.5),mode ='bilinear'))
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(out1,scale_factor=((1/9),(1/9)),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(9,9),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(1.5,1.5),mode ='bilinear'))
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(out1,scale_factor=((1/27),(1/27)),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(27,27),mode ='bilinear'))
        
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        # print(out.shape)
        ### End of encoder block

        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape)
        
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out1.shape)
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(out1,scale_factor=((1/6.75),(1/6.75)),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(6.75,6.75),mode ='bilinear'))
        # print(out.shape,out1.shape)
        
        out = torch.add(out,u2)
        out1 = F.interpolate(out1,scale_factor = ((1/0.75),(1/0.75)),mode = 'bilinear')
        out1 = torch.add(out1,o2)

        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        
        # print(out.shape,out1.shape)
        
        out = torch.add(out,F.interpolate(out1,scale_factor=((1/2.25),(1/2.25)),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(2.25,2.25),mode ='bilinear'))
        # print(out.shape,out1.shape)
        
        out = torch.add(out,u1)
        out1 = F.interpolate(out1,scale_factor=((1/0.75),(1/0.75)))
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out1.shape)
        out1 = F.interpolate(out1,scale_factor=((1/0.75),(1/0.75)))
        # print(out1.shape)
        # out1 = F.interpolate(out1,scale_factor=((1/0.75),(1/0.75)))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        del out1,u1,u2,o1,o2
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class unet2combautoencoder(nn.Module):
    
    def __init__(self):
        super(unet2combautoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16,2,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        u1 = out
        o1 = out1
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        u2 = out
        o2 = out1
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        

        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        # print(out.shape,out1.shape)
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)

        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        # print(out.shape,out1.shape)

        out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class comb2autoencoder(nn.Module):
    
    def __init__(self):
        super(comb2autoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16,2,1,stride=1,padding=0)

        self.tmp1 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tmp2 = nn.Conv2d(128,32,1,stride=1,padding=0)
        self.tmp3 = nn.Conv2d(64,32,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        # out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        # out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        # out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        
        # out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')
        # out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        # out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
                
        # out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        t3 = F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear')
        # out = torch.add(out,F.interpolate(out1,scale_factor=(0.015625,0.015625),mode ='bilinear'))
        # out1 = torch.add(out1,F.interpolate(out,scale_factor=(64,64),mode ='bilinear'))
        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        
        out = torch.add(out,torch.add(self.tmp2(t3),torch.add(t1,self.tmp1(t2))))
        
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))

        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')
        # out = torch.add(out,F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear'))
        # out1 = torch.add(out1,F.interpolate(out,scale_factor=(16,16),mode ='bilinear'))
        
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        # out = torch.add(out,F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear'))
        # out1 = torch.add(out1,F.interpolate(out,scale_factor=(4,4),mode ='bilinear'))
        # print(t1.shape,t2.shape)
        # out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))
        # print(out1.shape,out.shape)
        out = torch.add(out,torch.add(t1,self.tmp3(t2)))

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        

        # print(out.shape,out1.shape)

        # out = torch.add(out,out1)

        out = F.relu(self.final(out))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out

class unetcomb2autoencoder(nn.Module):
    
    def __init__(self):
        super(unetcomb2autoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.decoder1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.final = nn.Conv2d(16,2,1,stride=1,padding=0)

        self.tmp1 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tmp2 = nn.Conv2d(128,32,1,stride=1,padding=0)
        self.tmp3 = nn.Conv2d(64,32,1,stride=1,padding=0)

        self.tmpf3 = nn.Conv2d(128,32,1,stride=1,padding=0)
        self.tmpf2 = nn.Conv2d(64,32,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        #K-NET encoder
        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        o1 = out1
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')
        o2 = out1        
        
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear'))
        t3 = F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear')
        
        # U-NET encoder start
        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        #Fusing all feature maps from K-NET
        out = torch.add(out,torch.add(self.tmp2(t3),torch.add(t1,self.tmp1(t2))))
        
        u1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        u2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))

        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,u2)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u1)

        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out1 = torch.add(out1,o2)
        
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')

        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out1 = torch.add(out1,o1)
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))
        
        # Fusing all layers at the last layer of decoder
        out = torch.add(out,torch.add(self.tmpf3(t3),torch.add(t1,self.tmpf2(t2))))

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = F.relu(self.final(out))
        
        out = self.soft(out)
        # print(out.shape)
        return out


class rautoencoder(nn.Module):
    def __init__(self):
        super(rautoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(512, 256, 3, stride=2, padding=1)  # b, 16, 10, 10
            
        self.encoder2 = nn.Conv2d(256, 128, 3, stride=2, padding=1)  # b, 8, 3, 3
            
        # self.encoder3 = nn.Conv2d(16, 2, 3, stride=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(128,64, 3, stride=2, padding=1)
        
        self.encoder4 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
        
        self.encoder5 = nn.Conv2d(32, 2, 3, stride=2, padding=1)

        self.decoder1 = nn.ConvTranspose2d(1, 32, 3, stride=2,padding = 1 )  # b, 16, 5, 5
            
        self.decoder2 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        
        self.decoder3 = nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1)
        
        self.decoder4 = nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1)
        # self.decoder3 = nn.ConvTranspose2d(32, 64, 2, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder5= nn.ConvTranspose2d(256, 512, 3, stride=2, padding=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.decoder1(x))
        # print(out.shape)
        out = F.relu(self.decoder2(out))
        # print(out.shape)
        out = F.relu(self.decoder3(out))
        # print(out.shape)
        # out = F.relu(self.decoder4(out))
        # print(out.shape)
        # out = F.relu(self.decoder5(out))
        # print(out.shape)
        # out = F.relu(F.max_pool2d(self.encoder1(out),1))
        # cc
        # out = F.relu(F.max_pool2d(self.encoder2(out),1,1))        
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder3(out),1,1))

        out = F.relu(F.max_pool2d(self.encoder4(out),1,1))

        out = F.relu(F.max_pool2d(self.encoder5(out),1,1))

        out = self.soft(out)     
        # print(out.shape)
        return out

class nautoencoder(nn.Module):
    def __init__(self):
        super(nautoencoder, self).__init__()
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # b, 16, 10, 10
            
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # b, 8, 3, 3
            
        # self.encoder3 = nn.Conv2d(16, 2, 3, stride=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        self.encoder4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.decoder1 = nn.ConvTranspose2d(512, 256, 3, stride=2,padding = 1)  # b, 16, 5, 5
            
        self.decoder2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        
        self.decoder3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        
        self.decoder4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        # self.decoder3 = nn.ConvTranspose2d(32, 64, 2, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder5= nn.ConvTranspose2d(32, 2, 3, stride=2, padding=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(F.max_pool2d(self.encoder1(x),1))
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder2(out),1,1))
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder3(out),1,1))
        # print(out.shape)
        # out = F.relu(F.max_pool2d(self.encoder4(out),1,1))
        # print(out.shape)
        # out = F.relu(F.max_pool2d(self.encoder5(out),1,1))
        # print(out.shape)
        # out = F.relu(self.decoder1(out))
        
        # out = F.pad(out,(0,1,0,1))
        # # print(out.shape)
        # out = F.relu(self.decoder2(out))
        # out = F.pad(out,(0,1,0,1))
        # print(out.shape)
        out = F.relu(self.decoder3(out))
        out = F.pad(out,(0,1,0,1))
        # print(out.shape)
        out = F.relu(self.decoder4(out))
        out = F.pad(out,(0,1,0,1))
        # print(out.shape)
        out = F.relu(self.decoder5(out))
        out = F.pad(out,(0,1,0,1))
        # print(out.shape)  
        out = self.soft(out)     
        # print(out.shape)
        return out

class combnewautoencoder(nn.Module):
    
    def __init__(self):
        super(combnewautoencoder, self).__init__()

        self.nencoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.nencoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.nencoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.nencoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.nencoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.ndecoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        # self.ndecoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        self.ndecoder3 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.ndecoder4 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.ndecoder5 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.ndecoder6 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.ndecoder7 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.ndecoder8 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        # self.decoderf =   nn.Conv2d(4, 2, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        out = F.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))
        
        out = F.relu(F.max_pool2d(self.decoder3(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder4(out),2,2))
        out = F.relu(F.max_pool2d(self.decoder5(out),2,2))

        out = self.soft(out)
        # print(out.shape)
        return out


class combnew2autoencoder(nn.Module):
    
    def __init__(self):
        super(combnew2autoencoder, self).__init__()

        self.nencoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.nencoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.nencoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.nencoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.nencoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.ndecoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        # self.ndecoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=2)  # b, 8, 15, 1
        self.ndecoder3 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        # self.ndecoder4 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.ndecoder5 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.ndecoder6 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.ndecoder7 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.ndecoder8 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=2)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 2, 3, stride=1, padding=1)

        # self.decoderf =   nn.Conv2d(4, 2, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))  
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2)) 
        
        out2 = F.relu(F.upsample(self.nencoder1(x),scale_factor=(2,2)))
        out2 = F.relu(F.upsample(self.nencoder2(out2),scale_factor=(2,2)))  
        out2 = F.relu(F.upsample(self.nencoder3(out2),scale_factor=(2,2))) 
        
        out2 = F.relu(F.max_pool2d(self.ndecoder3(out2),2,2))
        out2 = F.max_pool2d(out2,2,2)
        out2 = F.max_pool2d(out2,2,2)
        out2 = F.max_pool2d(out2,2,2)
        out2 = F.max_pool2d(out2,2,2)
        out2 = F.max_pool2d(out2,2,2)

        out = torch.cat((out,out2),1)
        # print(out.shape)

        out = F.relu((self.decoder2(out)))
        # print(out.shape)
        out = F.relu(F.upsample(self.decoder3(out),scale_factor=(2,2)))
        # print(out.shape)
        out = F.relu(F.upsample(self.decoder4(out),scale_factor=(2,2)))
        out = F.relu(F.upsample(self.decoder5(out),scale_factor=(2,2)))
        # out = F.relu(self.decoderf(out))
        
        # print(out.shape)

        out = self.soft(out)
        # print(out.shape)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

class attnautoencoder(nn.Module):
    def __init__(self):
        super(attnautoencoder, self).__init__()
        
        self.en1 = AttentionStem(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, groups=1)
        self.en2 = AttentionStem(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, groups=1)
        self.en3 = AttentionStem(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1)
        
        self.de1 = AttentionStem(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, groups=1)
        self.de2 = AttentionStem(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1, groups=1)
        self.de3 = AttentionStem(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1, groups=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        out = F.relu(F.max_pool2d(self.en1(x),2,2))
        out = F.relu(F.max_pool2d(self.en2(out),2,2))
        out = F.relu(F.max_pool2d(self.en3(out),2,2))
        out = F.relu(F.interpolate(self.de1(out),scale_factor=(2,2), mode='bilinear'))
        out = F.relu(F.interpolate(self.de2(out),scale_factor=(2,2),mode = 'bilinear'))
        out = F.relu(F.interpolate(self.de3(out),scale_factor=(2,2),mode = 'bilinear'))
        out = self.soft(out)
        
        return out

class ocattnautoencoder(nn.Module):
    def __init__(self):
        super(ocattnautoencoder, self).__init__()
        
        self.en1 = AttentionConv(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, groups=1)
        self.en2 = AttentionConv(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, groups=1)
        self.en3 = AttentionConv(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1)
        
        self.de1 = AttentionConv(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, groups=1)
        self.de2 = AttentionConv(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1, groups=1)
        self.de3 = AttentionConv(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1, groups=1)
        
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        out = F.relu(F.interpolate(self.en1(x),scale_factor=(2,2), mode='bilinear'))
        out = F.relu(F.interpolate(self.en2(out),scale_factor=(2,2),mode = 'bilinear'))
        out = F.relu(F.interpolate(self.en3(out),scale_factor=(2,2),mode = 'bilinear'))
        
        out = F.relu(F.max_pool2d(self.de1(out),2,2))
        out = F.relu(F.max_pool2d(self.de2(out),2,2))
        out = F.relu(F.max_pool2d(self.de3(out),2,2))
        
        out = self.soft(out)
        
        return out
