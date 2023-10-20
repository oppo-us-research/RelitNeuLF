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

import sys,os
from numpy.lib.stride_tricks import broadcast_to
cpwd = os.getcwd()
sys.path.append(cpwd)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm,trange
import os
import argparse
import cv2
from torch.utils.tensorboard import SummaryWriter
# from sklearn.preprocessing import normalize
from torch.nn.parallel.data_parallel import DataParallel
from src.model import Nerf4D_relight_baseline, Nerf4D_relu_depth_relight_normalAlbedo_roughness
import torchvision
import glob
import torch.optim as optim
from utils import compute_dist,eval_uvst,rm_folder,AverageMeter,rm_folder_keep,eval_uv,eval_trans
from torch.utils.data import Dataset
from src.utils import microfacetRender
from src.dataloader import Nelf_dataset_Relight, Nelf_dataset_Relight_rough
from src.cam_view import rayPlaneInter,get_rays,rotation_matrix_from_vectors
from sklearn.neighbors import NearestNeighbors
from src.utils import Dir_out
import imageio
import logging
from datetime import datetime
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',type=str, default = 'menLazy360_multi_albedo_baseline',help = 'exp name')
parser.add_argument('--data_dir',type=str, default = 'data/BlenderData/menLazy/',help='data folder name')
parser.add_argument('--batch_size',type=int, default = 8192,help='normalize input')

parser.add_argument('--fourier_weight',type=float,default=1,help='fourier weight')
parser.add_argument('--fourier_epoch',type=int,default=40,help='fourier start epoch')
parser.add_argument('--test_freq',type=int,default=3,help='test frequency')
parser.add_argument('--save_checkpoints',type=int,default=3,help='checkpoint frequency')
parser.add_argument('--whole_epoch',type=int,default=1000,help='checkpoint frequency')
parser.add_argument('--gpuid',type=str, default = '2',help='data folder name')

parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
parser.add_argument('--norm_fac',type=int, default= 1000, help= "normalize the data uvst")
parser.add_argument('--work_num',type=int, default= 64, help= "normalize the data uvst")
parser.add_argument('--lr_pluser',type=int, default = 100,help = 'scale for dir')
parser.add_argument('--lr',type=float,default=1e-04,help='learning rate')
parser.add_argument('--loadcheckpoints', action='store_true', default = False)
parser.add_argument('--st_depth',type=float, default= 0.0, help= "st depth")
parser.add_argument('--uv_depth',type=float, default= 2.6, help= "uv depth")
parser.add_argument('--resize_ratio',type=int,default=2)
parser.add_argument('--method',type=str, default = 'baseline',help='baseline, full, lambertian')


class train():
    def __init__(self,args):
         # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))

        # data root
        data_root = args.data_dir

        # fourier epoch
        self.fourier_epoch = args.fourier_epoch

        # method
        self.method = args.method

        # network model
        if self.method == "lambertian" or self.method == "full":
            self.model = Nerf4D_relu_depth_relight_normalAlbedo_roughness(method = self.method)
        elif self.method == "baseline":
            self.model = Nerf4D_relight_baseline()
        else:
            raise Exception("no model found!")

        # self.model = Nerf4D_relu()
        # self.sfft = SFFT()

        # uv and st depth
        self.uv_depth = args.uv_depth
        self.st_depth = args.st_depth
        self.r_ratio  = args.resize_ratio

        # normalize factor
        self.norm_fac = args.norm_fac

        self.exp = 'Exp_'+args.exp_name

        # tensorboard writer
        self.summary_writer = SummaryWriter("src/tensorboard/"+self.exp)

        # save img
        self.save_img = True
        self.img_folder_train = 'result_mm/'+self.exp+'/train/'
        self.img_folder_test = 'result_mm/'+self.exp+'/test/'
        self.checkpoints = 'result_mm/'+self.exp+'/checkpoints/'

        # load checkpoints
        self.iter = 0
        if(args.loadcheckpoints):
            self.load_check_points()

        args.img_scale = args.resize_ratio
        # else:
        # make folder
        rm_folder(self.img_folder_train)
        rm_folder(self.img_folder_test)
        rm_folder_keep(self.checkpoints)

        # logging system
        handlers = [logging.StreamHandler()]
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler('result_mm/'+self.exp+f'/{dt_string}.log', mode='w'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S', handlers=handlers,
        )

        self.model = self.model.cuda()


        # height and width
        image_paths = glob.glob(f"{data_root}/*.png")
        sample_img = cv2.imread(image_paths[0])

        self.h = int(sample_img.shape[0]/args.img_scale)
        self.w = int(sample_img.shape[1]/args.img_scale)

        self.img_scale = args.img_scale

        self.step_count = 0

        # load nelf data
        print("Start loading...")
        self.uvst_whole = np.load(f"{data_root}/uvst_{self.r_ratio}train.npy") / args.norm_fac
        self.color_whole = np.load(f"{data_root}/rgb_{self.r_ratio}train.npy")
        self.trans       = np.load(f"{data_root}/transtrain.npy")
        self.intrinsic       = np.load(f"{data_root}/k_{self.r_ratio}train.npy")
        self.light_dir       = np.load(f"{data_root}/ltrain.npy")
        self.view_dir       = np.load(f"{data_root}/view_dir_{self.r_ratio}train.npy")

        # val
        self.uvst_whole_val   = np.load(f"{data_root}/uvst_{self.r_ratio}val.npy") / args.norm_fac
        self.color_whole_val  = np.load(f"{data_root}/rgb_{self.r_ratio}val.npy")
        self.albedo_whole_val = np.load(f"{data_root}/albedo_{self.r_ratio}val.npy")
        self.trans_val        = np.load(f"{data_root}/transval.npy")
        self.intrinsic_val    = np.load(f"{data_root}/k_{self.r_ratio}val.npy")
        self.light_dir_val    = np.load(f"{data_root}/lval.npy")
        self.view_dir_val     = np.load(f"{data_root}/view_dir_{self.r_ratio}val.npy")

        self.batch_size = args.batch_size
        # indx arrary
        self.ray_num = self.uvst_whole.shape[0]
        self.img_num = self.ray_num / (self.h*self.w)
        self.img_idx = torch.arange(0,self.img_num)
        self.img_idx = torch.broadcast_to(torch.unsqueeze(self.img_idx,1),[int(self.img_num),self.h*self.w])
        self.img_idx = torch.reshape(self.img_idx,(-1,1))
        self.img_idx = torch.squeeze(self.img_idx,1).type(torch.long)

        self.ray_num_val = self.uvst_whole_val.shape[0]
        self.img_num_val = self.ray_num_val / (self.h*self.w)
        self.img_idx_val = torch.arange(0,self.img_num_val)
        self.img_idx_val = torch.broadcast_to(torch.unsqueeze(self.img_idx_val,1),[int(self.img_num_val),self.h*self.w])
        self.img_idx_val = torch.reshape(self.img_idx_val,(-1,1))
        self.img_idx_val = torch.squeeze(self.img_idx_val,1).type(torch.long)
        

        # convert to tensor format
        self.uvst      = torch.from_numpy(self.uvst_whole.astype('float32'))
        self.color     = torch.from_numpy(self.color_whole.astype('float32'))
        self.light_dir = torch.from_numpy(self.light_dir.astype('float32'))
        self.v_dir     = torch.from_numpy(self.view_dir.astype('float32'))

        self.uvst_val      = torch.from_numpy(self.uvst_whole_val.astype('float32'))
        self.color_val     = torch.from_numpy(self.color_whole_val.astype('float32'))
        self.albedo_val    = torch.from_numpy(self.albedo_whole_val.astype('float32'))
        self.light_dir_val = torch.from_numpy(self.light_dir_val.astype('float32'))
        self.v_dir_val     = torch.from_numpy(self.view_dir_val.astype('float32'))

        # self.uv_whole_raw = np.load(f"{data_root}/uv_sparse_4.npy")

        print("Stop loading...")
        rays_whole = np.concatenate([self.uvst_whole, self.color_whole], axis=1)
        self.plane_dist = compute_dist(self.w, self.h)
        print(f"{rays_whole.shape[0]} rays loaded, w={self.w}, h={self.h}, d={self.plane_dist} color[min={rays_whole[:,4:].min()}, max={rays_whole[:,4:].max()}]")

        self.min_u,self.max_u,self.min_v,self.max_v,self.min_s,self.max_s,self.min_t,self.max_t = eval_uvst(rays_whole)
        self.max_x,self.min_x,self.max_y,self.min_y = eval_trans(self.trans)
     
        # assign batch
        self.start, self.end = [], []
        s = 0
        while s < self.uvst_whole.shape[0]:
            self.start.append(s)
            s+= args.batch_size
            self.end.append(min(s,self.uvst_whole.shape[0]))

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        self.step_size_compute = int(self.img_idx.shape[0] / self.batch_size) * args.lr_pluser
        self.vis_step = 1
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)  

        # fourier weight
        self.f_weight = args.fourier_weight

        self.epoch_num = args.whole_epoch
        

        self.radius = self.uv_depth

        self.gif_dir = Dir_out() # use for last visualization
        print("here")

        self.m_render = microfacetRender(batch_size=args.batch_size)

        self.lpips = lpips.LPIPS(net='vgg') 

    def train_step(self,args):
       
        for epoch in range(0,self.epoch_num): 

            if(args.loadcheckpoints):
                epoch = self.iter+1
        

            self.losses = AverageMeter()
            self.losses_rgb = AverageMeter()
            self.losses_photometric = AverageMeter()
            self.losses_normalRegu = AverageMeter()
            
            self.model.train()
            self.step_count +=1

            # get the index
            perm = torch.randperm(self.uvst_whole.shape[0])
            self.uvst = self.uvst[perm]
            self.color = self.color[perm]
            self.v_dir = self.v_dir[perm]
            self.img_idx = self.img_idx[perm]

            self.train_loader = [{'input': torch.cat([self.uvst[s:e],self.light_dir[self.img_idx[s:e]]],dim=-1),
                                  'color': self.color[s:e],
                                  'v_dir': self.v_dir[s:e]} for s, e in zip(self.start, self.end)]

            pbar = self.train_loader
            for i,data_batch in enumerate(pbar):
                self.optimizer.zero_grad()
 
                # load the data
                inputs  = data_batch["input"].cuda()
                color = data_batch["color"].cuda()
                v_dir = data_batch["v_dir"].cuda()

                if self.method == "full" or self.method == "lambertian":
                    if self.method == "full": 
                        preds_color,pred_normal, pred_albedo,pred_rough,light_dir = self.model(inputs.cuda())
                        
                        ones_data = torch.ones(pred_normal.shape[0]).cuda()
                        loss_regu_norm     = (ones_data - torch.norm(pred_normal,dim=1)) * (ones_data - torch.norm(pred_normal,dim=1))
                        loss_regu_norm_sum = torch.mean(loss_regu_norm) * 10
                        
                        # microfacet brdf render
                        pred_color_in = self.m_render.forward(light_dir,v_dir,pred_normal,pred_albedo,pred_rough)

                    
                    elif self.method == "lambertian":
                        preds_color,pred_normal, pred_albedo,light_dir = self.model(inputs.cuda())
                        ones_data = torch.ones(pred_normal.shape[0]).cuda()
                        loss_regu_norm     = (ones_data - torch.norm(pred_normal,dim=1)) * (ones_data - torch.norm(pred_normal,dim=1))
                        loss_regu_norm_sum = torch.mean(loss_regu_norm) * 10

                        # N * L
                        value_photometric = torch.bmm(light_dir.view(light_dir.shape[0],1,light_dir.shape[1]) , pred_normal.view(light_dir.shape[0],light_dir.shape[1],1))
                        value_photometric = torch.clamp(value_photometric,0)   

                        # RGB = Kd* N * L
                        pred_color_in     = torch.bmm(pred_albedo.view(pred_albedo.shape[0],pred_albedo.shape[1],1), value_photometric)
                        pred_color_in     = pred_color_in.view(pred_color_in.shape[0],pred_color_in.shape[1])

                    # loss_photometric
                    photometric_val = (preds_color - pred_color_in) * (preds_color - pred_color_in)
                    loss_photometric  = 1000 * torch.mean(photometric_val)
                    
                    loss_rgb = 1000*torch.mean((preds_color - color) * (preds_color - color))
                    loss = loss_rgb + loss_photometric + loss_regu_norm_sum

                    self.losses.update(loss.item(), inputs.size(0))
                    self.losses_rgb.update(loss_rgb.item(),inputs.size(0))
                    self.losses_photometric.update(loss_photometric,inputs.size(0))
                    self.losses_normalRegu.update(loss_regu_norm_sum,inputs.size(0))
                   
                    log_str = 'epoch {}/{}, {}/{}, learning rate:{}, total loss:{:4f},RGB loss:{:4f},photometric loss:{:4f},normal regu loss:{:4f}'\
                        .format(epoch,self.epoch_num,i+1,len(pbar),\
                        self.optimizer.param_groups[0]['lr'],self.losses.avg,self.losses_rgb.avg,self.losses_photometric.avg,self.losses_normalRegu.avg)

                elif self.method == "baseline":
                    preds_color = self.model(inputs.cuda())
                    loss_rgb = 1000*torch.mean((preds_color - color) * (preds_color - color))

                    loss = loss_rgb
                    self.losses.update(loss.item(), inputs.size(0))
                    self.losses_rgb.update(loss_rgb.item(),inputs.size(0))

                    log_str = 'epoch {}/{}, {}/{}, learning rate:{}, total loss:{:4f},RGB loss:{:4f}'\
                        .format(epoch,self.epoch_num,i+1,len(pbar),self.optimizer.param_groups[0]['lr'],self.losses.avg,self.losses_rgb.avg)
                else:
                    raise Exception("no model name included!")
                # break

                loss.backward()
                self.optimizer.step()
                    
                if (i+1) % 2000 == 0:
                    logging.info(log_str)
                             
                self.train_summaries()
               
            logging.info(log_str)      
            self.scheduler.step()  

            with torch.no_grad():
                self.model.eval()
                if epoch % args.test_freq ==0:

                    save_dir  = self.img_folder_train + f"epoch-{epoch}"

                    total_num = int(self.uvst_whole.shape[0]/(self.h*self.w)-1)

                    cam_num   = np.random.randint(total_num)

                    light_cur = self.light_dir[cam_num,:]

        
                    uvst_cam  = self.uvst_whole[cam_num*self.w*self.h:(cam_num+1)*self.w*self.h,:]
                    gt_colors = self.color_whole[cam_num*self.w*self.h:(cam_num+1)*self.w*self.h,:]

                    # generate predicted camera position
                    cam_x = np.random.uniform(self.min_x,self.max_x)
                    cam_y = np.random.uniform(self.min_y,self.max_y)
                    cam_z = self.radius

                    cam_x = self.max_x
                    cam_y = self.max_y

                    gt_img = gt_colors.reshape((self.w,self.h,3)).transpose((2,1,0))
                    gt_img = torch.from_numpy(gt_img)

                    uvst_random = self.get_uvst(cam_x,cam_y,cam_z)
                    uvst_random_centered = self.get_uvst(cam_x,cam_y,cam_z,True)

                    save_dir = self.img_folder_train + f"epoch-{epoch}"
                    rm_folder(save_dir)

                    self.render_sample_img(self.model, uvst_cam, self.w, self.h, light_cur,f"{save_dir}/recon")#,f"{save_dir}/recon_depth.png")
                    self.render_sample_img(self.model, uvst_random, self.w, self.h, light_cur,f"{save_dir}/random")#,f"{save_dir}/random_depth.png")
                    self.render_sample_img(self.model, uvst_random_centered, self.w, self.h,light_cur, f"{save_dir}/random_centered")#,f"{save_dir}/random_depth.png")
               
                    torchvision.utils.save_image(gt_img, f"{save_dir}/gt.png")

                    if self.method == "full":
                        self.gen_gif_vis(f"{save_dir}/video_ani.gif",f"{save_dir}/video_ani_normal.gif",f"{save_dir}/video_ani_albedo.gif",f"{save_dir}/video_ani_rough.gif")
                    
                    self.val(epoch)

                if epoch % args.save_checkpoints == 0:
                    cpt_path = self.checkpoints + f"nelf-{epoch}.pth"
                    torch.save(self.model.state_dict(), cpt_path)
                self.model.train() 

    def val(self,epoch):
        # pass
        save_dir = self.img_folder_test + f"epoch-{epoch}"
        rm_folder(save_dir)

        count = 0
        i = 0
        p = []
        s = []
        l = []

        p_albedo = []
        s_albedo = []
        l_albedo = []

        while i < self.uvst_whole_val.shape[0]:
            end = i + self.w*self.h

            uvst         = self.uvst_val[i:end]
            l_dir        = self.light_dir_val[self.img_idx_val[i:end]]
            gt_color     = self.color_val[i:end] 
            gt_alebdo = self.albedo_val[i:end] 
            
            input = torch.cat([uvst,l_dir],dim=-1)

            if self.method == "full":
                preds_color,pred_normal, preds_albedo,pred_rough,light_dir = self.model(input.cuda())
            elif self.method == "lambertian":
                preds_color,pred_normal, preds_albedo,light_dir = self.model(input.cuda())
            elif self.method == "baseline":
                preds_color = self.model(input.cuda())

            # for lpips
            pred_img = preds_color.reshape((self.w,self.h,3)).permute((2,1,0))
            gt_img   = torch.tensor(gt_color).reshape((self.w,self.h,3)).permute((2,1,0))

            if self.method == "full" or self.method == "lambertian":
                preds_albedo_img = preds_albedo.reshape((self.w,self.h,3)).permute((2,1,0))
                gt_alebdo_img   = torch.tensor(gt_alebdo).reshape((self.w,self.h,3)).permute((2,1,0))

            if count % 50 ==0:
                torchvision.utils.save_image(pred_img,f"{save_dir}/test_{count}.png")
                torchvision.utils.save_image(gt_img,f"{save_dir}/gt_{count}.png")

                if self.method == "full" or self.method == "lambertian":
                    torchvision.utils.save_image(preds_albedo_img,f"{save_dir}/test_albedo{count}.png")
                    torchvision.utils.save_image(gt_alebdo_img,f"{save_dir}/gt_albedo{count}.png")

            # predict res
            preds_color  = preds_color.cpu().numpy()
            gt_color     = gt_color.cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_color, preds_color, data_range=1)
            ssim = structural_similarity(gt_color.reshape((self.h,self.w,3)), preds_color.reshape((self.h,self.w,3)), data_range=preds_color.max() - preds_color.min(),multichannel=True)
            lsp  = self.lpips(pred_img.cpu(),gt_img) 

            # rgb
            p.append(psnr)
            s.append(ssim)
            l.append(lsp.numpy().item())
            
            if self.method == "full" or self.method == "lambertian":
                preds_albedo = preds_albedo.cpu().numpy()
                gt_alebdo = gt_alebdo.cpu().numpy()
                psnr_albedo = peak_signal_noise_ratio(gt_alebdo, preds_albedo, data_range=1)
                ssim_albedo = structural_similarity(gt_alebdo.reshape((self.h,self.w,3)), preds_albedo.reshape((self.h,self.w,3)), data_range=preds_albedo.max() - preds_albedo.min(),multichannel=True)
                lsp_albedo  = self.lpips(preds_albedo_img.cpu(),gt_alebdo_img) 

                # albedo
                p_albedo.append(psnr_albedo)
                s_albedo.append(ssim_albedo)
                l_albedo.append(lsp_albedo.numpy().item())
            

            i = end
            count+=1

        logging.info(f'>>> RGB val: psnr  mean {np.asarray(p).mean()}')
        logging.info(f'>>> RGB val: ssim  mean {np.asarray(s).mean()}')
        logging.info(f'>>> RGB val: lpips mean {np.asarray(l).mean()}')
        
        if self.method == "full" or self.method == "lambertian":
            logging.info(f'>>> albedo val: psnr  mean {np.asarray(p_albedo).mean()}')
            logging.info(f'>>> albedo val: ssim  mean {np.asarray(s_albedo).mean()}')
            logging.info(f'>>> albedo val: lpips mean {np.asarray(l_albedo).mean()}')

        return p

        
    def get_uvst(self,cam_x,cam_y,cam_z, center_flag=False):
        # np.eye(3,dtype=float)

        t = np.asarray([cam_x,cam_y,cam_z])

        # get rotation matrix
        if(center_flag):
            cur_p = -t
            cur_o = np.array([0,0,-1])
            rot_mat = rotation_matrix_from_vectors(cur_o,cur_p)
        else:
            rot_mat = np.eye(3,dtype=float)

        # c2w = np.concatenate((np.eye(3,dtype=float),np.expand_dims(t,1)),1)
        c2w = np.concatenate((rot_mat,np.expand_dims(t,1)),1)

        ray_o,ray_d = get_rays(self.h,self.w,self.intrinsic,c2w)

        ray_o = np.reshape(ray_o,(-1,3))
        ray_d = np.reshape(ray_d,(-1,3))

        plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

        # interset radius plane
        p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
        p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

        # interset radius plane 
        inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)

        inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)
     
        data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)
        
        data_uvst /= self.norm_fac

        return data_uvst


    def render_sample_img(self,model,uvst, w, h, light_cur,save_path=None,save_depth_path=None,save_flag=True):
        with torch.no_grad():
            
            # generate face 2 face light
            light_sample = light_cur
            light_sample = np.broadcast_to(light_sample,(uvst.shape[0],3))
            light_sample = torch.from_numpy(light_sample.astype(np.float32)).cuda()

            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

            uvst_light = torch.cat([uvst,light_sample],dim=-1)

            if self.method == "full":
                pred_color,pred_normal,pred_albedo,pred_rough,light_dir = model(uvst_light)
                pred_rough_img = pred_rough.reshape((w,h,1)).permute((2,1,0))

            elif self.method == "lambertian":
                pred_color,pred_normal,pred_albedo,light_dir = model(uvst_light)

            elif self.method == "baseline":
                pred_color = model(uvst_light)

            pred_img       = pred_color.reshape((w,h,3)).permute((2,1,0))
            
            if self.method == 'full' or self.method == "lambertian":
                # reshape
                pred_normal = pred_normal.reshape((w,h,3)).permute((1,0,2))
                pred_albedo = pred_albedo.reshape((w,h,3)).permute((1,0,2))
                

                # detach
                pred_normal = pred_normal.detach().cpu().numpy()
                # pred_albedo = pred_albedo.detach().cpu().numpy()

                # save 
                N_img_res = pred_normal
                N_img_res = (N_img_res+1.0)/2.0 * 255.0 
                N_img_res = N_img_res.astype('uint8')

                res_albedo = pred_albedo.detach().cpu().numpy()
                res_img = (255 * res_albedo).astype(np.uint8)
            
                if(save_flag):
                    cv2.imwrite(save_path+"_normal.png",cv2.cvtColor(N_img_res,cv2.COLOR_RGB2BGR ))
                    cv2.imwrite(save_path+"_albedo.png",cv2.cvtColor(res_img,cv2.COLOR_RGB2BGR ))

                    if self.method == "full":
                        torchvision.utils.save_image(pred_rough_img, save_path+"_rough.png")

            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path+".png")
                # torchvision.utils.save_image(pred_dimg, save_depth_path)
                
            if self.method == "full":
                return pred_color.reshape((w,h,3)).permute((1,0,2)),N_img_res,res_img,pred_rough.reshape((w,h,1)).permute((1,0,2)) #,pred_depth_norm.reshape((w,h,1)).permute((1,0,2))
            elif self.method == "lambertain":
                return pred_color.reshape((w,h,3)).permute((1,0,2)),N_img_res,res_img
            elif self.method == "baseline":
                return pred_color.reshape((w,h,3)).permute((1,0,2))

    def train_summaries(self):
            
        self.summary_writer.add_scalar('total loss', self.losses.avg, self.step_count)

    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"{self.checkpoints}nelf-{self.iter}.pth"

            # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
            print(f"Load weights from {ckpt_name}")
            
            ckpt = torch.load(ckpt_name)
            try:
                self.model.load_state_dict(ckpt)
            except:
                tmp = DataParallel(self.model)
                tmp.load_state_dict(ckpt)
                self.model.load_state_dict(tmp.module.state_dict())
                del tmp
        
        else:
            print("non checkpoints exists.")

    def gen_gif_vis(self,savename,savename_normal,savename_albedo,savename_rough):
        
        step_interval = 0.1

        num = (self.max_x - self.min_x) / step_interval * 2 + (self.max_y - self.min_y) / step_interval *2

        directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]

        v_unit,u_unit = self.max_y,self.min_x
        
        view_group = []
        view_group_normal = []
        view_group_albedo = []
        view_group_rough = []

        directionIndex = 0

        for i in range(0,int(num)):

            # index light
            dir_index = i % self.gif_dir.shape[0]
            
            # view_unit = self.render_sample_img(self.model,u_unit,v_unit,self.w,self.h,None,None,False)
            data_uvst = self.get_uvst(u_unit,v_unit,self.radius,True)
            view_unit,view_unit_normal,view_unit_albedo,view_unit_rough = self.render_sample_img(self.model,data_uvst,self.w,self.h,self.gif_dir[dir_index,:],None,None,False)

            view_unit *= 255
            view_unit_rough *=255
            # view_unit_depth *=255

            view_unit       = view_unit.cpu().numpy().astype(np.uint8)
            view_unit_rough = view_unit_rough.cpu().numpy().astype(np.uint8)

            view_unit        = imageio.core.util.Array(view_unit)
            view_unit_normal = imageio.core.util.Array(view_unit_normal)
            view_unit_albedo = imageio.core.util.Array(view_unit_albedo)
            view_unit_rough  = imageio.core.util.Array(view_unit_rough)

            view_group.append(view_unit)
            view_group_normal.append(view_unit_normal)
            view_group_albedo.append(view_unit_albedo)
            view_group_rough.append(view_unit_rough)

            nextRow, nextColumn = v_unit + step_interval * directions[directionIndex][0], u_unit + step_interval * directions[directionIndex][1]

            if not (self.min_y < nextRow <= self.max_y and self.min_x <= nextColumn < self.max_x ):
                directionIndex = (directionIndex + 1) % 4

            v_unit += step_interval*directions[directionIndex][0]
            u_unit += step_interval*directions[directionIndex][1]
                

        imageio.mimsave(savename, view_group,fps=30)
        imageio.mimsave(savename_normal, view_group_normal,fps=30)
        imageio.mimsave(savename_albedo, view_group_albedo,fps=30)
        imageio.mimsave(savename_rough,  view_group_rough,fps=30)
       

if __name__ == '__main__':

    args = parser.parse_args()

    m_train = train(args)

    m_train.train_step(args)
