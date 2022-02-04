import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.utils.spectral_norm as sn
from .SN import spectral_norm_with_lower_bound as sn

class swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        #return 2.0 * x * Fun.sigmoid(2.0 * x)
        return  x * F.sigmoid(x)

def swish_f(x):
    return x * F.sigmoid(x)

class block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_in, n_out, 3, 1, 1)
        self.res = nn.Conv2d(n_in, n_out, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        h1 = swish_f(self.conv1(x))
        h2 = swish_f(self.conv2(h1))
        h1_res = swish_f(self.res(x))
        return self.pooling(h2 + h1_res)

class EBM_res(nn.Module):
    def __init__(self, n_c, n_f, l=0.2, img_size=32):
        super(EBM_res, self).__init__()
        if img_size == 32:
            self.f = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                swish(),
                block(n_f, n_f * 2),
                block(n_f * 2, n_f * 4),
                block(n_f * 4, n_f * 8),
                nn.Conv2d(n_f * 8, 100, 4, 1, 0))
        elif img_size == 64:
            self.f = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                swish(),
                block(n_f, n_f * 2),
                block(n_f * 2, n_f * 4),
                block(n_f * 4, n_f * 8),
                block(n_f * 8, n_f * 8),
                nn.Conv2d(n_f * 8, 100, 4, 1, 0))
        else:
            raise NotImplementedError
    def forward(self, x):
        return self.f(x).squeeze().sum(-1)

class EBM(nn.Module):
    def __init__(self, n_c, n_f, l=0.2):
        super(EBM, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            #nn.LeakyReLU(l),
            swish(),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            #nn.LeakyReLU(l),
            swish(),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            #nn.LeakyReLU(l),
            swish(),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            #nn.LeakyReLU(l),
            swish(),
            #nn.Conv2d(n_f * 8, 1, 4, 1, 0))
            nn.Conv2d(n_f * 8, 100, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze().sum(-1)

class res_block(nn.Module):
    def __init__(self, n_in, n_out):
        super(res_block, self).__init__()
        #conv1 = nn.Conv2d(n_in, n_out, 3, 1, 1)
        #conv2 = nn.Conv2d(n_out, n_out, 3, 1, 1)
        #torch.nn.init.zeros_(conv1.bias)
        #torch.nn.init.zeros_(conv2.weight)
        #torch.nn.init.zeros_(conv2.bias)
        #self.conv1 = sn(nn.Conv2d(n_in, n_out, 3, 1, 1))
        #self.conv2 = sn(nn.Conv2d(n_out, n_out, 3, 1, 1))

        conv1 = nn.Conv2d(n_in, n_out, 3, 1, 1)
        conv2 = nn.Conv2d(n_out, n_out, 3, 1, 1)
        torch.nn.init.zeros_(conv1.bias)    
        torch.nn.init.zeros_(conv2.weight)
        torch.nn.init.zeros_(conv2.bias)
        self.conv1 = sn(conv1)
        self.conv2 = sn(conv2)
        self.conv2_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float), requires_grad=True)
        if n_in != n_out:
            #short_cut = nn.Conv2d(n_in, n_out, 3, 1, 1)
            #torch.nn.init.zeros_(short_cut.bias)
            #self.short_cut = sn(nn.Conv2d(n_in, n_out, 3, 1, 1))
            short_cut = nn.Conv2d(n_in, n_out, 3, 1, 1)
            torch.nn.init.zeros_(short_cut.bias)
            self.short_cut = sn(short_cut)
        else:
            self.short_cut = nn.Identity()


    def forward(self, x):
        h = nn.functional.leaky_relu(x, 0.2)
        #h = swish_f(x)
        h = self.conv1(h)
        h = nn.functional.leaky_relu(h, 0.2)
        #h = swish_f(h)
        h = self.conv2(h) * self.conv2_scale 
        h = h + self.short_cut(x)  
        return h

class EBM_deep32_3(nn.Module):
    def __init__(self, n_c=3, res_blocks=[128, 256, 256, 256], N=5, n_f=128):
        super(EBM_deep32_3, self).__init__()
        conv_in = nn.Conv2d(n_c, 128, 3, 1, 1)
        torch.nn.init.zeros_(conv_in.bias)
        self.conv_in = sn(nn.Conv2d(n_c, 128, 3, 1, 1))
        #self.conv_in = nn.Conv2d(n_c, 128, 3, 1, 1)
        #torch.nn.init.zeros_(self.conv_in.bias)
        self.res_levels = []
        self.downsamples = []
        self.N_res_block = len(res_blocks)
        self.N = N
        cur_channel = 128
        for i_level in range(self.N_res_block):
            res_s = []
            for i_block in range(N):
                res_s.append(res_block(n_in=cur_channel, n_out=res_blocks[i_level]))
                cur_channel = res_blocks[i_level]
            self.res_levels.append(nn.ModuleList(res_s))
            if i_level < self.N_res_block - 1:
                self.downsamples.append(nn.AvgPool2d(2, 2, 0))
        #fc_out = nn.Linear(cur_channel, 1)
        #torch.nn.init.zeros_(fc_out.bias)
        #self.fc_out = sn(nn.Linear(cur_channel, 1))
        self.res_levels = nn.ModuleList(self.res_levels)  
        fc_out = nn.Conv2d(cur_channel, 100, 4, 1, 0)
        #fc_out = nn.Linear(cur_channel, 100)
        self.fc_out = sn(fc_out)
    
    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.N_res_block):
            for i_block in range(self.N):
                h = self.res_levels[i_level][i_block](h)
            if i_level < self.N_res_block - 1:
                h = self.downsamples[i_level](h)
        h = nn.functional.relu(h)
        #h = swish_f(h)
        #h = torch.sum(h, dim=[1,2,3])
        out = torch.sum(self.fc_out(h), dim=[1,2,3])
        return h

class EBM_deep32(nn.Module):
    def __init__(self, n_c=3, res_blocks=[128, 256, 256, 256], N=5, n_f=128):
        super(EBM_deep32, self).__init__()
        conv_in = nn.Conv2d(n_c, 128, 3, 1, 1)
        torch.nn.init.zeros_(conv_in.bias)
        self.conv_in = sn(nn.Conv2d(n_c, 128, 3, 1, 1))
        #self.conv_in = nn.Conv2d(n_c, 128, 3, 1, 1)
        #torch.nn.init.zeros_(self.conv_in.bias)
        self.res_levels = []
        self.downsamples = []
        self.N_res_block = len(res_blocks)
        self.N = N
        cur_channel = 128
        for i_level in range(self.N_res_block):
            res_s = []
            for i_block in range(N):
                res_s.append(res_block(n_in=cur_channel, n_out=res_blocks[i_level]))
                cur_channel = res_blocks[i_level]
            self.res_levels.append(nn.ModuleList(res_s))
            if i_level < self.N_res_block - 1:
                self.downsamples.append(nn.AvgPool2d(2, 2, 0))
        #fc_out = nn.Linear(cur_channel, 1)
        #torch.nn.init.zeros_(fc_out.bias)
        #self.fc_out = sn(nn.Linear(cur_channel, 1))
        self.res_levels = nn.ModuleList(self.res_levels)  

        #fc_out = nn.Linear(cur_channel, 100)
        self.fc_out = sn(fc_out)
    
    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.N_res_block):
            for i_block in range(self.N):
                h = self.res_levels[i_level][i_block](h)
            if i_level < self.N_res_block - 1:
                h = self.downsamples[i_level](h)
        h = nn.functional.relu(h)
        #h = swish_f(h)
        #h = torch.sum(h, dim=[1,2,3])
        #out = self.fc_out(h).sum(-1)
        return h

class res_block_swish(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_in, n_out, 3, 1, 1)
        self.res = nn.Conv2d(n_in, n_out, 3, 1, 1)

    def forward(self, x):
        h1 = swish_f(self.conv1(x))
        h2 = swish_f(self.conv2(h1))
        h1_res = swish_f(self.res(x))
        return h2 + h1_res

class EBM_deep32_swish(nn.Module):
    def __init__(self, n_c=3, res_blocks=[512, 1024, 1024], N=3, n_f=128):
        super(EBM_deep32_swish, self).__init__()
        self.conv_in = nn.Conv2d(n_c, 256, 3, 1, 1)
        self.res_levels = []
        self.downsamples = []
        self.N_res_block = len(res_blocks)
        self.N = N
        cur_channel = 256
        for i_level in range(self.N_res_block):
            res_s = []
            for i_block in range(N):
                res_s.append(res_block_swish(n_in=cur_channel, n_out=res_blocks[i_level]))
                cur_channel = res_blocks[i_level]
            self.res_levels.append(nn.ModuleList(res_s))
            self.downsamples.append(nn.AvgPool2d(2, 2, 0))
        self.res_levels = nn.ModuleList(self.res_levels)  
        self.fc_out = nn.Conv2d(cur_channel, 100, 4, 1, 0)  
    
    def forward(self, x):
        h = self.conv_in(x)
        h = swish_f(h)
        for i_level in range(self.N_res_block):
            for i_block in range(self.N):
                h = self.res_levels[i_level][i_block](h)
            h = self.downsamples[i_level](h)
        out = self.fc_out(h)
        return out.squeeze().sum(-1) 

class EBM_res_wA(nn.Module):
    def __init__(self, n_c, n_f, l=0.2, img_size=32):
        print("Do ebm with lower bound estimation")
        super(EBM_res_wA, self).__init__()
        self.A = nn.Parameter(torch.tensor(-3.0, dtype=torch.float))
        if img_size == 32:
            self.f = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                swish(),
                block(n_f, n_f * 2),
                block(n_f * 2, n_f * 4),
                block(n_f * 4, n_f * 8),
                nn.Conv2d(n_f * 8, 100, 4, 1, 0))
        elif img_size == 64:
            self.f = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                swish(),
                block(n_f, n_f * 2),
                block(n_f * 2, n_f * 4),
                block(n_f * 4, n_f * 8),
                block(n_f * 8, n_f * 8),
                nn.Conv2d(n_f * 8, 100, 4, 1, 0))
        else:
            raise NotImplementedError
    def forward(self, x):
        return self.f(x).squeeze().sum(-1) + self.A