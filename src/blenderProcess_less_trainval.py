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

import cv2
import imageio
import numpy as np
import argparse
import os
import sys
cpwd = os.getcwd()
sys.path.append(cpwd)
from src.cam_view import rayPlaneInter,get_rays
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R_func
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default = 'data/BlenderData/FaceBase/multilight/',help = 'exp name')
parser.add_argument('--uv_depth',type = float, default = 2.6)
parser.add_argument('--st_depth',type = float, default = 0.0)
parser.add_argument('--light_rad',type = float, default = 25.0)
parser.add_argument('--load_lights',action='store_true',default= True)
parser.add_argument('--resize_ratio',type=int,default=2)

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def RT2C2W(R,T):
    """
        RT MATRIX TO camera to world matrix
    """

    T = np.expand_dims(T,1)

    R_inv = inv(R)
    T_inv = -R_inv@T

    test = R_func.from_matrix(R)
    d = test.as_euler('zyx',degrees=True)

    c2w = np.concatenate((R_inv,T_inv),1)
   
    return c2w


def test_projection(obj_data, K,RT):
    
    vertex_array = np.asarray(obj_data.vertices)

    pad_one = np.ones((vertex_array.shape[0],1))

    vertex_array_pad = np.concatenate((vertex_array,pad_one),1)
    vertex_array_pad_T = vertex_array_pad.T

    P = np.matmul(K,RT)

    uv_scale = np.matmul(P,vertex_array_pad_T)
    uv_scale = uv_scale.T

    uv_scale = uv_scale / np.broadcast_to(np.expand_dims(uv_scale[:,2],1),np.shape(uv_scale))

    uv_scale_value = uv_scale[:,:2]

    img = np.zeros((800,800))
    img = img.astype(np.uint8)

    for index in range(uv_scale_value.shape[0]):
        
        u = uv_scale_value[index,0]
        v = uv_scale_value[index,1]

        img[v,u] = 255

    cv2.imwrite('test.png',img)


if __name__=="__main__":

    args = parser.parse_args()

    data_dir  = args.data_dir
    label_dir = data_dir + "cam_data_label.npz"
    obj_dir   = args.data_dir + "1_neutral.obj" 
    uv_depth  = args.uv_depth
    st_depth  = args.st_depth
    r_ratio   = args.resize_ratio 

    uvst_path = f"{data_dir}/uvst_{r_ratio}.npy"
    rgb_path = f"{data_dir}/rgb_{r_ratio}.npy"
    albedo_path = f"{data_dir}/albedo_{r_ratio}.npy"

    t_path = f"{data_dir}/trans.npy"
    k_path = f"{data_dir}/k_{r_ratio}.npy"
    l_path = f"{data_dir}/l.npy"
    view_dir_path = f"{data_dir}/view_dir_{r_ratio}.npy"
    
    pose_path    = f"{data_dir}/cam_pose.npy"
    cam_idx_path = f"{data_dir}/cam_idx.npy"

    
    label = np.load(label_dir,allow_pickle=True)   
    
    imgnames = label['imgname']
    Ks       = label['intrinsic'] 
    w2c_R     = label['R_bcam']
    w2c_T     = label['T_bcam']
    l_dirs   = label['light_dir']
    

    # set radius
    # radius = 5
    # train
    uvst = []
    rgb  = []
    albedo = []
    Trans = []
    light_dir = []
    view_dir = []
    # val
    uvst_val  = []
    rgb_val   = []
    albedo_val = []
    Trans_val = []
    light_dir_val = []
    view_dir_val  = []
    # intrinsics=[]
    testskip = 8
    cam_idx = []
    cam_pose = []

    for i, (imgname,K,R,T,l_unit) in tqdm(enumerate(zip(imgnames,Ks,w2c_R,w2c_T,l_dirs))):
        prefix = imgname.split("_")

        # read obj file
        img_albedo_file = "{}albedo_{}_{}_0001.png".format(data_dir,prefix[0],prefix[1])
        
        img_albedo =  cv2.imread(img_albedo_file)
        img_albedo = cv2.cvtColor(img_albedo,cv2.COLOR_BGR2RGB)

        # read original img
        img_bgr = cv2.imread(data_dir+imgname)
        img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

        img_rgb    = img_rgb[::r_ratio,::r_ratio,:]
        img_albedo = img_albedo[::r_ratio,::r_ratio,:]

        K[:2,:2] /= r_ratio

        rgb2d    = np.transpose(img_rgb / 255.0, (1,0,2))
        albedo2d = np.transpose(img_albedo / 255.0, (1,0,2))

        rgb2d    = np.reshape(rgb2d,(-1,3))
        albedo2d = np.reshape(albedo2d,(-1,3))
        # test_projection(obj_data,K,RT)

        img_h = img_rgb.shape[0]
        img_w = img_rgb.shape[1]

        # RT to C2W matrix
        c2w = RT2C2W(R,T)

        ray_o,ray_d = get_rays(img_h,img_w,K,c2w)

        ray_o = np.reshape(ray_o,(-1,3))
        ray_d = np.reshape(ray_d,(-1,3))

        # add view direction vector
        view_ray =  ray_o + ray_d
        data_view_dir = -view_ray / np.broadcast_to(np.expand_dims(np.linalg.norm(view_ray,axis=1),1),view_ray.shape)
        view_dir.append(data_view_dir)

        plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

        # interset radius plane
        p_uv = np.broadcast_to(np.array([0.0,0.0,uv_depth]),np.shape(ray_o))
        p_st = np.broadcast_to(np.array([0.0,0.0,st_depth]),np.shape(ray_o))

        # interset radius plane 
        inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)

        inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)
        # inter_st[:,0]+=img_w/2.0
        # inter_st[:,1]+=img_h/2.0

        data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)
        
        if i < 25:
            cam_pose.append(c2w[:3,3])
        # train and val split
        if i % 8 !=0:
            uvst.append(data_uvst)
            rgb.append(rgb2d)
            albedo.append(albedo2d)
            Trans.append(c2w[:2,3].T)
            light_dir.append(l_unit/args.light_rad)
            intrinsics = K
            cam_idx.append(ray_o)
            
        else:
            uvst_val.append(data_uvst)
            rgb_val.append(rgb2d)
            albedo_val.append(albedo2d)
            Trans_val.append(c2w[:2,3].T)
            light_dir_val.append(l_unit/args.light_rad)
            intrinsics_val = K

    # train and val
    uvst      = np.asarray(uvst)
    rgb       = np.asarray(rgb)
    albedo    = np.asarray(albedo)

    trans     = np.asanyarray(Trans)
    light_dir = np.asarray(light_dir)
    view_dir  = np.asarray(view_dir) 
    
    cam_idx = np.asarray(cam_idx)
    cam_pose = np.asarray(cam_pose)
    
    # val
    uvst_val      = np.asarray(uvst_val)
    rgb_val       = np.asarray(rgb_val)
    albedo_val    = np.asarray(albedo_val)  

    trans_val     = np.asanyarray(Trans_val)
    light_dir_val = np.asarray(light_dir_val)
    view_dir_val  = np.asarray(view_dir_val) 

    # reshape
    uvst_val   = np.array(uvst_val).reshape(-1, 4)
    rgb_val    = np.array(rgb_val).reshape(-1, 3)
    albedo_val = np.array(albedo_val).reshape(-1, 3)
    view_dir_val = np.array(view_dir_val).reshape(-1, 3)
    
    cam_idx = np.reshape(cam_idx,(-1,3))
    cam_pose = np.reshape(cam_pose,(-1,3))

    uvst = np.array(uvst).reshape(-1, 4)
    rgb = np.array(rgb).reshape(-1, 3)
    albedo = np.array(albedo).reshape(-1, 3)
    view_dir = np.array(view_dir).reshape(-1, 3)

    print("uvst_train = ", uvst.shape)
    print("rgb_train = ", rgb.shape)

    print("uvst_val = ", uvst_val.shape)
    print("rgb_val = ", rgb_val.shape)


    # train and val
    np.save(uvst_path.replace('.npy','train.npy'), uvst)
    np.save(rgb_path.replace('.npy','train.npy'), rgb)
    # np.save(albedo_path.replace('.npy','train.npy'), albedo)
    np.save(t_path.replace('.npy','train.npy'), trans)
    np.save(k_path.replace('.npy','train.npy'), intrinsics)
    np.save(l_path.replace('.npy','train.npy'),light_dir)
    np.save(view_dir_path.replace('.npy','train.npy'),view_dir)

    np.save(uvst_path.replace('.npy','val.npy'), uvst_val)
    np.save(rgb_path.replace('.npy','val.npy'), rgb_val)
    np.save(albedo_path.replace('.npy','val.npy'), albedo_val)
    np.save(t_path.replace('.npy','val.npy'), trans_val)
    np.save(k_path.replace('.npy','val.npy'), intrinsics_val)
    np.save(l_path.replace('.npy','val.npy'),light_dir_val)
    np.save(view_dir_path.replace('.npy','val.npy'),view_dir_val)
    
    np.save(pose_path.replace('.npy', f'train.npy'),cam_pose)
    np.save(cam_idx_path.replace('.npy', f'train.npy'),cam_idx)


