# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
        
# straight forward shit
class Nerf4D_relight_baseline(nn.Module):
    def __init__(self,D=8,W=256,input_ch=256,skips=[4]):
        super(Nerf4D_relight_baseline, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.views_linears = nn.ModuleList([nn.Linear(3 + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        # self.alpha_linear   = nn.Linear(W, 1)
        self.rgb_linear     = nn.Linear(W//2, 3)

        self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 4)), requires_grad=False)

        self.b_color = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 4)), requires_grad=False)

    def forward(self,x_input):
        x     = x_input[:,0:4]
        x_dir = x_input[:,4:7]

        x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
        o1 = torch.sin(x)
        o2 = torch.cos(x)
        input_pts = torch.cat([o1, o2], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        
        feature = self.feature_linear(h)

        h = torch.cat([feature, x_dir], -1)

        for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

        rgb = self.rgb_linear(h)

        return rgb

class Nerf4D_relu_depth_relight_normalAlbedo_roughness(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, output_ch=4, skips=[4],method="full",pe=True):
        """ 
        """
        super(Nerf4D_relu_depth_relight_normalAlbedo_roughness, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch
        self.skips = skips
        self.method = method
        
        # normal albedo linear
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        # rgb linear
        self.pts_linears_rgb = nn.ModuleList(
        [nn.Linear(input_ch+7, W+7)] + [nn.Linear(W+7, W+7) if i not in self.skips else nn.Linear(W+7 + input_ch+7, W+7) for i in range(D-1)])


        self.pts_linears_rgb_wor = nn.ModuleList(
        [nn.Linear(input_ch+6, W+6)] + [nn.Linear(W+6, W+6) if i not in self.skips else nn.Linear(W+6 + input_ch+6, W+6) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears_normalAlbedo = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch+3, W//2)])
        
        self.albedo_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.normal_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.rough_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])

        # if use_viewdirs:
        self.feature_linear     = nn.Linear(W, W)

        self.feature_linear_rgb = nn.Linear(W+7, W)
        self.feature_linear_rgb_wor = nn.Linear(W+6, W)
        
        self.depth_linear       = nn.Linear(W, 1)

        self.rgb_linear = nn.Linear(W//2, 3) # add lights
        self.albedo_linears2 = nn.Linear(W//2, 3) # add lights
        self.normal_linears2 = nn.Linear(W//2, 3) # add lights
        self.rough_linears2 = nn.Linear(W//2, 1) # add lights

        self.rgb_act   = nn.Sigmoid()
        self.depth_act = nn.Sigmoid() 

        self.normal_act   = nn.Tanh()
        self.albedo_act = nn.Sigmoid() 
        self.rough_act = nn.Sigmoid()

        if pe == False:
            self.input_net =  nn.Linear(4, input_ch)
        # else:
        #     self.output_linear = nn.Linear(W, output_ch)
        self.pe = pe

        self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 4)), requires_grad=False)

        self.b_color = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 4)), requires_grad=False)

    def normalAlbedo_rough_forward(self,input_pts):

        h = input_pts

        for i,l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        feature = self.feature_linear(h)
        h = feature

        # albedo
        for i, l in enumerate(self.albedo_linears):
            h_albedo = self.albedo_linears[i](h)
            h_albedo = F.relu(h_albedo)

        # normal
        for i, l in enumerate(self.normal_linears):
            h_normal = self.normal_linears[i](h)
            h_normal = F.relu(h_normal)

        # roughness
        if(self.method == "full"):
            for i, l in enumerate(self.rough_linears):
                h_rough = self.rough_linears[i](h)
                h_rough = F.relu(h_rough)

        albedo   = self.albedo_act( self.albedo_linears2(h_albedo))       
        normal   = self.normal_act( self.normal_linears2(h_normal))

        if(self.method == "full"):
            rough    = self.rough_act( self.rough_linears2(h_normal))

        if(self.method == "full"):
            return albedo, normal,rough
        elif(self.method == "lambertian"):
            return albedo,normal
        else:
            raise Exception("mode not support")

    def rgbLight_forward(self,input_pts, x_dir,normal, albedo,rough):
        
        input_pts_na = torch.cat([input_pts,normal,albedo,rough],dim=-1)

        h = input_pts_na
        for i, l in enumerate(self.pts_linears_rgb):
            h = self.pts_linears_rgb[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_na, h], -1)

        # if self.use_viewdirs:
        feature = self.feature_linear_rgb(h)
        h = feature
        h = torch.cat([feature, x_dir], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        rgb   = self.rgb_act(rgb)
        # depth = self.depth_act(depth)        

        return rgb

    def rgbLight_forward_WOR(self,input_pts, x_dir,normal, albedo):
        
        input_pts_na = torch.cat([input_pts,normal,albedo],dim=-1)

        h = input_pts_na
        for i, l in enumerate(self.pts_linears_rgb_wor):
            h = self.pts_linears_rgb_wor[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_na, h], -1)

        # if self.use_viewdirs:
        feature = self.feature_linear_rgb_wor(h)
        h = feature
        h = torch.cat([feature, x_dir], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        rgb   = self.rgb_act(rgb)
        # depth = self.depth_act(depth)        

        return rgb

    def forward(self, x_input):
        
        
        x     = x_input[:,0:4]
        x_dir = x_input[:,4:7]

        if self.pe: # if use pe
            x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
            o1 = torch.sin(x)
            o2 = torch.cos(x)
            input_pts = torch.cat([o1, o2], dim=-1)
        else:
            input_pts = self.input_net(x)
            input_pts = F.relu(input_pts)

        if self.method == "full":
            albedo,normal,rough = self.normalAlbedo_rough_forward(input_pts)
            rgb = self.rgbLight_forward(input_pts,x_dir, normal, albedo,rough)
            return rgb, normal, albedo,rough, x_dir
        elif self.method == "lambertian":
            albedo,normal = self.normalAlbedo_rough_forward(input_pts)
            rgb = self.rgbLight_forward_WOR(input_pts,x_dir, normal, albedo)
            return rgb, normal, albedo, x_dir
        else:
            raise Exception("mode not support")
    