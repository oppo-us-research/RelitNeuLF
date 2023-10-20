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

import numpy as np
import os
import glob
import shutil
from sklearn import preprocessing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2,math
import re
from PIL import Image

def load_pfm(filename):
    with open(filename, 'r', newline='', encoding='latin-1') as file:
        header = file.readline().strip()
        color = validate_header(header)

        width, height = parse_dimensions(file)
        scale, endian = parse_scale_and_endian(file)

        data = read_data(file, endian, width, height, color)
        reversed_data = reverse_data_order(data)

    return reversed_data

def validate_header(header):
    if header == 'PF':
        return True
    elif header == 'Pf':
        return False
    else:
        raise ValueError('Not a PFM file.')

def parse_dimensions(file):
    dim_line = file.readline()
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', dim_line)
    if dim_match:
        return map(int, dim_match.groups())
    else:
        raise ValueError('Malformed PFM header.')

def parse_scale_and_endian(file):
    scale_line = file.readline().strip()
    scale = float(scale_line)
    endian = '<' if scale < 0 else '>'
    scale = abs(scale)
    return scale, endian

def read_data(file, endian, width, height, color):
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape)

def reverse_data_order(data):
    reverse_ids = range(len(data)-1, -1, -1)
    return data[reverse_ids]

class microfacetRender():
    
    def __init__(self,batch_size = 8192):

        self.F0   = 0.05
        self.temp = torch.tensor(2.0).cuda()
       

        

    def forward(self,raw_dir,view_dir,preds_norm,preds_albe,preds_rough):
        
        self.h = 0.5 * raw_dir + 0.5 * view_dir

        self.h = self.h / torch.max(torch.sum(self.h*self.h,axis=1))

        diffuseBatch = preds_albe #/ np.pi
        roughBatch   = preds_rough.repeat(1,3)

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(preds_norm * view_dir, dim = 1), 0, 1)
        ndh = torch.clamp(torch.sum(preds_norm * self.h, dim = 1), 0, 1)
        ndl = torch.clamp(torch.sum(preds_norm * raw_dir, dim = 1), 0, 1)

        ndv = torch.transpose(ndv.repeat(3,1),0,1)
        ndh = torch.transpose(ndh.repeat(3,1),0,1)
        ndl = torch.transpose(ndl.repeat(3,1),0,1)

        vdh = torch.sum( (view_dir * self.h), dim = 1)
        self.frac0 = self.F0 + (1-self.F0) * torch.pow(self.temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)
        # self.frac0 = (1-self.F0) * torch.pow(self.temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        self.frac0 = self.frac0.repeat(3,1)
        self.frac0 = torch.transpose(self.frac0,0,1)
        frac = alpha2 * self.frac0

        nom0 = ndh * ndh * (alpha2 - 1) + 1
        nom1 = ndv * (1 - k) + k
        nom2 = ndl * (1 - k) + k
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)

        specPred = frac / nom 

        color = (specPred + diffuseBatch)* ndl

        return torch.clamp(color, 0, 1)

def compute_dist(w, h, fov=57):
    """
    fov: in degree
    """
    fov = fov * np.pi / 180
    return 0.5 * w / np.tan(0.5 * fov)

def rm_folder(path):
    if os.path.exists(path):
       files = glob.glob(path+'*')

       if(len(files)>0):
            for f in files:
                try:
                    shutil.rmtree(f)
                    
                except:
                    os.remove(f)
    #    os.makedirs(path)
    else:
        os.makedirs(path)


def rm_folder_keep(path):
    if os.path.exists(path):
        print("path already exists")
    else:
        os.makedirs(path)

def eval_uv(rays_whole):
    max_u = rays_whole[:,0].max()
    min_u = rays_whole[:,0].min()
    max_v = rays_whole[:,1].max()
    min_v = rays_whole[:,1].min()

    return min_u,max_u,min_v,max_v

def eval_trans(trans):
    max_x = trans[:,0].max()
    min_x = trans[:,0].min()
    max_y = trans[:,1].max()
    min_y = trans[:,1].min()

    return max_x,min_x,max_y,min_y

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d

def eval_uvst(rays_whole):
    max_u = rays_whole[:,0].max()
    min_u = rays_whole[:,0].min()
    max_v = rays_whole[:,1].max()
    min_v = rays_whole[:,1].min()
    max_s = rays_whole[:,2].max()
    min_s = rays_whole[:,2].min()
    max_t = rays_whole[:,3].max()
    min_t = rays_whole[:,3].min()


    return min_u,max_u,min_v,max_v,min_s,max_s,min_t,max_t

def Get_direction_circle360_zigzag(phi_up,phi_down,theta_left,theta_right):
    
    unit_size = 10

    rows    = (phi_up - phi_down + unit_size)//unit_size
    columns = (theta_right - theta_left + unit_size)//unit_size

    visited = [[False] * columns for _ in range(rows)]
    
    # total = int(rows/10 * columns/10)
    total = rows * columns

    dir_unit_group = []

    directions = [[0, 1],[0, -1]]
    row, column = 0, 0
    directionIndex = 0

    row    = 0 
    column = 0 

    for i in range(total):
        theta = column*unit_size + theta_left
        phi   = phi_up  - row*unit_size

        if theta >= -90 and theta <=90:
            dir_unit = np.array([math.cos(math.radians(phi))* math.sin(math.radians(theta)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* math.cos(math.radians(theta))
                                            ])
        elif theta > 90 and theta < 180:
            dir_unit = np.array([math.cos(math.radians(phi))* math.cos(math.radians(theta-90)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* -math.sin(math.radians(theta-90))
                                            ])
        else:
            dir_unit = np.array([math.cos(math.radians(phi))* -math.cos(math.radians(theta+90)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* math.sin(math.radians(theta+90))
                                            ])

        dir_unit_group.append(dir_unit)

        visited[row][column] = True

        nextRow, nextColumn = row + directions[directionIndex][0], column + directions[directionIndex][1]

        if nextColumn == columns or nextColumn == -1:
            row+=1
            column = 0
 
        else:
            row    += directions[directionIndex][0]
            column += directions[directionIndex][1]
    
    dir_unit_group = np.asarray(dir_unit_group)

    return dir_unit_group

def Get_direction_circle360(phi_up,phi_down,theta_left,theta_right):
    
    unit_size = 10

    rows    = (phi_up - phi_down + unit_size)//unit_size
    columns = (theta_right - theta_left + unit_size)//unit_size

    visited = [[False] * columns for _ in range(rows)]
    
    total = rows * columns

    dir_unit_group = []

    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    row, column = 0, 0
    directionIndex = 0

    row    = 0 
    column = 0 

    for i in range(total):
        theta = column*unit_size + theta_left
        phi   = phi_up  - row*unit_size

        if theta >= -90 and theta <=90:
            dir_unit = np.array([math.cos(math.radians(phi))* math.sin(math.radians(theta)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* math.cos(math.radians(theta))
                                            ])
        elif theta > 90 and theta < 180:
            dir_unit = np.array([math.cos(math.radians(phi))* math.cos(math.radians(theta-90)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* -math.sin(math.radians(theta-90))
                                            ])
        else:
            dir_unit = np.array([math.cos(math.radians(phi))* -math.cos(math.radians(theta+90)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* math.sin(math.radians(theta+90))
                                            ])

        dir_unit_group.append(dir_unit)

        visited[row][column] = True

        nextRow, nextColumn = row + directions[directionIndex][0], column + directions[directionIndex][1]

        if not (0 <= nextRow < rows and 0 <= nextColumn < columns and not visited[nextRow][nextColumn]):
            directionIndex = (directionIndex + 1) % 4

        row    += directions[directionIndex][0]
        column += directions[directionIndex][1]

    dir_unit_group = np.asarray(dir_unit_group)

    return dir_unit_group

def Dir_out360():

    theta_left_cir  = -170
    theta_right_cir =  170
    phi_down_cir    = -80
    phi_up_cir      =  80

    # dir_unit_group_circle = Get_direction_circle360_zigzag(phi_up_cir,phi_down_cir,theta_left_cir,theta_right_cir)
    dir_unit_group_circle = Get_direction_circle360(phi_up_cir,phi_down_cir,theta_left_cir,theta_right_cir)
    
    
    return dir_unit_group_circle

def Get_direction_circle(phi_up,phi_down,theta_left,theta_right):
    
    rows    = phi_up - phi_down
    columns = theta_right - theta_left

    visited = [[False] * columns for _ in range(rows)]
    
    # total = int(rows/10 * columns/10)
    total = 12 * 4

    dir_unit_group = []

    directions = [[0, 10], [10, 0], [0, -10], [-10, 0]]
    row, column = 0, 0
    directionIndex = 0

    row    = 0 
    column = 0 

    for i in range(total):
        theta = column+theta_left
        phi   = phi_up - row



        dir_unit = np.array([math.cos(math.radians(phi))* math.sin(math.radians(theta)),
                                        math.sin(math.radians(phi)),
                                        math.cos(math.radians(phi))* math.cos(math.radians(theta))
                                        ])

        dir_unit_group.append(dir_unit)

        visited[row][column] = True

        nextRow, nextColumn = row + directions[directionIndex][0], column + directions[directionIndex][1]

        if not (0 <= nextRow < rows and 0 <= nextColumn < columns and not visited[nextRow][nextColumn]):
            directionIndex = (directionIndex + 1) % 4

        row    += directions[directionIndex][0]
        column += directions[directionIndex][1]

    dir_unit_group = np.asarray(dir_unit_group)

    return dir_unit_group

def Dir_out():

    theta_left_zig  = -60
    theta_right_zig =  60
    phi_down_zig  = -3
    phi_up_zig =  3

    theta_left_cir  = -60
    theta_right_cir =  60
    phi_down_cir    = -60
    phi_up_cir      =  60

    dir_unit_group_circle = Get_direction_circle(phi_up_cir,phi_down_cir,theta_left_cir,theta_right_cir)
    
    return dir_unit_group_circle


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduceBrightness(img_rgb,factor=0.5):
    hsvImg = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)

    # hsvImg= hsvImg.astype('float32')
    # decreasing the V channel by a factor from the original
    hsvImg[...,2] = hsvImg[...,2]*factor

    img_rgb_reduced=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
    cv2.imwrite('reduce_light.png',cv2.cvtColor(img_rgb_reduced,cv2.COLOR_RGB2BGR))
    # plt.subplot(111), plt.imshow(cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB))       

    return img_rgb_reduced

def covertAngle2Dir(theta,phi):
    # theta is horizontal, phi is vertical
    if theta >= -90 and theta <=90:
            dir_unit = np.array([math.cos(math.radians(phi))* math.sin(math.radians(theta)),
                                            math.sin(math.radians(phi)),
                                            math.cos(math.radians(phi))* math.cos(math.radians(theta))
                                            ])
    elif theta > 90 and theta < 180:
        dir_unit = np.array([math.cos(math.radians(phi))* math.cos(math.radians(theta-90)),
                                        math.sin(math.radians(phi)),
                                        math.cos(math.radians(phi))* -math.sin(math.radians(theta-90))
                                        ])
    else:
        dir_unit = np.array([math.cos(math.radians(phi))* -math.cos(math.radians(theta+90)),
                                        math.sin(math.radians(phi)),
                                        math.cos(math.radians(phi))* math.sin(math.radians(theta+90))
                                        ])
    return dir_unit

