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

import os
import torch 
from torch._C import dtype
from torch.utils.data import Dataset
import numpy as np
import math


class Nelf_dataset(Dataset):
    
    def __init__(self,uvst,color,**kvs):
        
        self.uvst  = uvst
        self.color = color #* 2 - 1.0

        # convert to tensor format
        self.uvst  = torch.from_numpy(self.uvst.astype('float32'))
        self.color = torch.from_numpy(self.color.astype('float32'))
        
    
    def __getitem__(self,index):
        
        item = {'input': self.uvst[index],
                'color': self.color[index]}
        
        return item
        
        
    def __len__(self):
        return self.uvst.shape[0]


class Nelf_dataset_depth(Dataset):
    
    def __init__(self,uvst,color,cam_idx,**kvs):
        # super(LightStage2d_data, self).__init__(**kvs)
        
        self.uvst  = uvst
        self.color = color #* 2 - 1.0
        self.cam_idx = cam_idx

        # convert to tensor format
        self.uvst    = torch.from_numpy(self.uvst.astype('float32'))
        self.color   = torch.from_numpy(self.color.astype('float32'))
        self.cam_idx = torch.from_numpy(self.cam_idx.astype('float32'))
        
    
    def __getitem__(self,index):
        
        item = {'input': self.uvst[index],
                'color': self.color[index],
                'cam_idx':self.cam_idx[index]}
        
        return item
        
        
    def __len__(self):
        return self.uvst.shape[0]



class Nelf_dataset_Relight(Dataset):
    
    def __init__(self,uvst,color,light_dir,h,w,**kvs):
        # super(LightStage2d_data, self).__init__(**kvs)
    
        self.uvst  = uvst
        self.color = color #* 2 - 1.0
        self.l_dir = light_dir
        print(self.l_dir.shape)
        self.rayNum_img = h * w # ray count in single image

        # convert to tensor format
        self.uvst  = torch.from_numpy(self.uvst.astype('float32'))
        self.color = torch.from_numpy(self.color.astype('float32'))
        self.light_dir = torch.from_numpy(self.l_dir.astype('float32'))
    
    def __getitem__(self,index):

        img_index = int(index/ self.rayNum_img)

        input_cat = torch.cat([self.uvst[index],self.light_dir[img_index,:]],dim=-1)
    
        item = {'input': input_cat,
                'color': self.color[index]}
        
        return item
        
        
    def __len__(self):
        return self.uvst.shape[0]



class Nelf_dataset_Relight_rough(Dataset):
    
    def __init__(self,uvst,color,light_dir,view_dir,h,w,**kvs):
        # super(LightStage2d_data, self).__init__(**kvs)
    
        self.uvst  = uvst
        self.color = color #* 2 - 1.0
        self.l_dir = light_dir
        self.v_dir = view_dir
        print(self.l_dir.shape)
        self.rayNum_img = h * w # ray count in single image

        # convert to tensor format
        self.uvst  = torch.from_numpy(self.uvst.astype('float32'))
        self.color = torch.from_numpy(self.color.astype('float32'))
        self.light_dir = torch.from_numpy(self.l_dir.astype('float32'))
        self.v_dir = torch.from_numpy(self.v_dir.astype('float32'))
    
    def __getitem__(self,index):

        img_index = int(index/ self.rayNum_img)

        input_cat = torch.cat([self.uvst[index],self.light_dir[img_index,:]],dim=-1)
    
        item = {'input': input_cat,
                'color': self.color[index],
                'view_dir':self.v_dir[index]}
        
        return item
        
        
    def __len__(self):
        return self.uvst.shape[0]
    