import torch
import torch.nn as nn
import torch.nn.functional as F

from Code.Res2Net import res2net50_v1b_26w_4s

class CB3(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB3, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        # x: [B, in_channels, H, W]
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = F.relu(y)
        return y
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.fuse = CB3(2, 1, False, False)
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        avg_out = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W]
        max_out = torch.max(ftr, dim=1, keepdim=True)[0] # [B, 1, H, W]
        cat_out = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        sam = F.sigmoid(self.fuse(cat_out)) # # [B, 1, H, W]
        return sam*ftr + ftr

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class R_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(R_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class DSEL(nn.Module):
    def __init__(self, channel,n_class):
        super(DSEL, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(channel, n_class, 1)
        self.sa = SpatialAttention()
    def forward(self, x4):
        x4_1 = self.upsample(x4)        #2 32 16 16
        x4_1 = self.upsample(x4_1)      #2 32 32 32
        x = self.conv5(self.sa(x4_1))
        return x

class DSER(nn.Module):
    def __init__(self,channel, n_class):
        super(DSER,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2=nn.Conv2d(channel, channel, 3, padding=0, dilation=8)  
        self.conv3=nn.Conv2d(channel, channel, 3, padding=0, dilation=4)  
        self.conv4=nn.Conv2d(channel, channel, 3, padding=0, dilation=2)  
        self.conv5 = nn.Conv2d(3*channel, n_class, 1)
    def forward(self, x2, x3, x4):
        x2_1 = self.conv2(x2)       #2 32 16 16
        x3_1 = self.conv3(x3)       #2 32 8 8
        x4_1 = self.conv4(x4)       #2 32 4 4
        x4_1 = self.upsample(x4_1)  #2 32 8 8
        x = torch.cat((x3_1,x4_1), 1)   #2 64 8 8
        x = self.upsample(x)        #2 64 16 16
        x = torch.cat((x2_1,x), 1)  #2 96 16 16
        x = self.conv5(x)           #2 1 16 16
        return x

class MBGC(nn.Module):
    def __init__(self,in_channel_left,in_channel_down,in_channel_right):
        super(MBGC,self).__init__()
        self.conv0=nn.Conv2d(in_channel_left,256,kernel_size=3,stride=1,padding=1)
        self.bn0=nn.BatchNorm2d(256)
        self.conv1=nn.Conv2d(in_channel_down,256,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(256)
        self.conv2=nn.Conv2d(in_channel_right,256,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.conv_d1=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.conv_d2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.conv_l=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(256*3,256,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(256)


    def forward(self,left,down,right):
        left=F.relu(self.bn0(self.conv0(left)),inplace=True)
        down=F.relu(self.bn1(self.conv1(down)),inplace=True)
        right=F.relu(self.bn2(self.conv2(right)),inplace=True)
        down_1=self.conv_d1(down)
        w1=self.conv_l(left)
        if down.size()[2:]!=left.size()[2:]:
            down_=F.interpolate(down,size=left.size()[2:],mode='bilinear')
            z1=F.relu(w1*down_,inplace=True)
        else:
            z1=F.relu(w1*down,inplace=True)
        if down_1.size()[2:]!=left.size()[2:]:
            down_1=F.interpolate(down_1,size=left.size()[2:],mode='bilinear')
        z2=F.relu(down_1*left,inplace=True) 
        down_2 = self.conv_d2(right)
        if down_2.size()[2:]!=left.size()[2:]:
            down_2=F.interpolate(down_2,size=left.size()[2:],mode='bilinear')
        #z3=F.relu((-1*down_2+1)*left,inplace=True)
        z3=F.relu(down_2,inplace=True)
        out=torch.cat((z1,z2,z3),dim=1)
        return F.relu(self.bn3(self.conv3(out)),inplace=True)
    def initialize(self):
        weight_init(self)


def weight_init(module):
    for n,m in model.named_children():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()

class BSNet(nn.Module):
    def __init__(self, channel=32, n_class=1):
        super(BSNet, self).__init__()
        ch = [64, 256, 512, 1024, 2048]
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = R_modified(512, channel)
        self.rfb3_1 = R_modified(1024, channel)
        self.rfb4_1 = R_modified(2048, channel)
        
        # ----MBG----
        self.fam45=MBGC(64,2048,1)
        self.fam34=MBGC(64,1024,1)
        self.fam23=MBGC(64,512,1)
        self.fam_conv = nn.Conv2d(256, n_class, kernel_size=1)

        # ---- edge ----
        self.edge_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.edge_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)

        # ---- DSE ----
        self.dsel = DSEL(channel, n_class)
        self.dser = DSER(channel, n_class)

    def forward(self, x):               
        x = self.resnet.conv1(x)    
        x = self.resnet.bn1(x)     
        x = self.resnet.relu(x)       
        x = self.resnet.maxpool(x)     
        x1 = self.resnet.layer1(x)    
        x2 = self.resnet.layer2(x1)     
        x3 = self.resnet.layer3(x2)   
        x4 = self.resnet.layer4(x3)    
        x2_rfb = self.rfb2_1(x2)      
        x3_rfb = self.rfb3_1(x3)     
        x4_rfb = self.rfb4_1(x4)    

        x = self.edge_conv1(x1)
        x = self.edge_conv2(x)
        edge_guidance = self.edge_conv3(x)  # torch.Size([2, 64, 64, 64])
        lateral_edge = self.edge_conv4(edge_guidance)  
        lateral_edge = F.interpolate(lateral_edge, scale_factor=4, mode='bilinear')   
 
        # ---- DSE ----
        _test_ra5_feat = self.dsel(x4_rfb)     
        fam5_feat = self.dser(x2_rfb,x3_rfb,x4_rfb)    
        fam5_feat = F.interpolate(fam5_feat,size=_test_ra5_feat.size()[2:],mode='bilinear')+_test_ra5_feat      
        lateral_map_5 = F.interpolate(fam5_feat,scale_factor=8,mode='bilinear')

        # ---- MBG ----
        fam4_feat = self.fam_conv(self.fam45(edge_guidance,x4,fam5_feat))  
        x=fam4_feat+F.interpolate(fam5_feat,size=fam4_feat.size()[2:],mode='bilinear')  
        lateral_map_4 = F.interpolate(x,scale_factor=4,mode='bilinear') 
        
        fam3_feat = self.fam_conv(self.fam34(edge_guidance,x3,fam4_feat))     
        x=fam3_feat+fam4_feat    
        lateral_map_3 = F.interpolate(x,scale_factor=4,mode='bilinear') 

        fam2_feat = self.fam_conv(self.fam23(edge_guidance,x2,fam3_feat))     
        x=fam2_feat+fam3_feat    
        lateral_map_2 = F.interpolate(x,scale_factor=4,mode='bilinear') 
  
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge


if __name__ == '__main__':
    ras = PraNetPlusPlus().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    out = ras(input_tensor)
