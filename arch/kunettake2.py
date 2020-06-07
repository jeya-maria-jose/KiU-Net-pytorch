class kunetv2(nn.Module):
    
    def __init__(self):
        super(kunetv2vvv, self).__init__()
        
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

        