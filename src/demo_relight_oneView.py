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
cpwd = os.getcwd()
sys.path.append(cpwd)
import os
import cv2
import torch
import argparse
import torchvision
from src.model import Nerf4D_relu_depth_relight_normalAlbedo_roughness
from torch.nn.parallel.data_parallel import DataParallel
import glob
from src.utils import Dir_out360
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
from src.utils import get_rays_np
from src.cam_view import GenNewView_stanford,rayPlaneInter
import imageio
from utils import compute_dist,eval_uvst,rm_folder,covertAngle2Dir,reduceBrightness,rm_folder_keep,load_pfm
from scipy.spatial.transform import Rotation as R_func
from numpy.linalg import inv
from src.cam_view import rayPlaneInter,get_rays,SphToVec,rotateVector,subPixels
from PIL import ImageDraw,Image
from tqdm import tqdm

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=256"

parser = argparse.ArgumentParser()
parser.add_argument('--st_depth',type=float, default= 0.0, help= "st depth")
parser.add_argument('--uv_depth',type=float, default= 5.0, help= "uv depth")
parser.add_argument('--exp_name',type=str, default = 'FaceBase_multi_albedo_full',help = 'exp name')
parser.add_argument('--data_dir',type=str, default = 'data/BlenderData/FaceBase/multilight/',help='data folder name')
parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--resize_ratio',type=int,default=2)
parser.add_argument('--norm_fac',type=int, default= 1000, help= "normalize the data uvst")
parser.add_argument('--env_map_dir', type=str, default = 'env_map/rnl_probe.pbm',help='data folder name')
parser.add_argument('--compute_OLAT', action='store_true', default = False)


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

class Relight():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))

        self.model = Nerf4D_relu_depth_relight_normalAlbedo_roughness()
        self.model = self.model.cuda()

        self.exp = 'Exp_'+args.exp_name
        self.checkpoints = 'result_mm/'+self.exp+'/checkpoints/'

        self.load_check_points()

        # data folder
        self.src = args.data_dir
        self.src_xml   = self.src + "cam_data_label.npz"
        self.label     = np.load(self.src_xml,allow_pickle=True)
        self.imgnames  = self.label['imgname']
        self.Ks        = self.label['intrinsic'] 
        self.w2c_R     = self.label['R_bcam']
        self.w2c_T     = self.label['T_bcam']

        # data root
        data_root = args.data_dir

        #
        self.r_ratio = args.resize_ratio

        # uv and st depth
        self.uv_depth = args.uv_depth
        self.st_depth = args.st_depth

        # img size
        args.img_scale = args.resize_ratio
        image_paths = glob.glob(f"{data_root}/*.png")
        sample_img = cv2.imread(image_paths[0])
        self.h = int(sample_img.shape[0]/args.img_scale)
        self.w = int(sample_img.shape[1]/args.img_scale)
        self.img_scale = args.img_scale

        self.norm_fac = args.norm_fac

        # save place
        self.relight_base_folder = 'result_relight/'+self.exp
        rm_folder_keep(self.relight_base_folder)

        self.OLAT_dir = '/OLAT_img/'
        # filename
        self.OLAT_filefolder = self.relight_base_folder + self.OLAT_dir
        rm_folder_keep(self.OLAT_filefolder)

        self.env_map_file = args.env_map_dir
        self.env_name_base = os.path.splitext(os.path.basename(self.env_map_file))[0]
        self.env_mapImg_dir = '/lightImg'+'/'+self.env_name_base+'/'
        self.env_mapImg_filefolder = self.relight_base_folder + self.env_mapImg_dir
        rm_folder_keep(self.env_mapImg_filefolder)

        # load env map
        env = load_pfm(self.env_map_file)
        env = cv2.resize(env, (int(env.shape[0] / 2), int(env.shape[0] / 2)))
        env = cv2.resize(env, (int(env.shape[0] / 2), int(env.shape[0] / 2)))
        self.env = cv2.resize(env, (int(env.shape[0] / 2), int(env.shape[0] / 2)))
        self.envRes = self.env.shape[0]

        # lighting result folder
        self.res_folder = self.relight_base_folder + '/result/' + self.env_name_base + '/'
        rm_folder_keep(self.res_folder)

        # light res
        self.lightRes = 64
        self.deg =90
        self.nSample =10
        self.flipOpt = None
        self.rotRes = 200
        self.imgRate = 0.5
        # flag 
        self.compute_OLAT = args.compute_OLAT
        self.compute_light= True

    def genViews(self):
        
        img_set = glob.glob(self.src+"00_00_*.png")
        for i, (imgname,K,R,T) in tqdm(enumerate(zip(self.imgnames,self.Ks,self.w2c_R,self.w2c_T))):
            img = self.src+imgname
            if img in img_set:
                # read original img
                img_bgr = cv2.imread(img)
                img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

                img_rgb    = img_rgb[::self.r_ratio,::self.r_ratio,:]
               
                img_h = img_rgb.shape[0]
                img_w = img_rgb.shape[1]

                K[:2,:2] /= self.r_ratio

                c2w = RT2C2W(R,T)

                ray_o,ray_d = get_rays(img_h,img_w,K,c2w)

                ray_o = np.reshape(ray_o,(-1,3))
                ray_d = np.reshape(ray_d,(-1,3))

                # add view direction vector
                view_ray =  ray_o + ray_d
                data_view_dir = -view_ray / np.broadcast_to(np.expand_dims(np.linalg.norm(view_ray,axis=1),1),view_ray.shape)
                plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

                # interset radius plane
                p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
                p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

                # interset radius plane 
                inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)

                inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)

                data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)

                data_uvst/=self.norm_fac
                break

        return data_uvst


    def computePipeline(self):
        # get uvst view of img, eg. front view
        uvst = self.genViews()

        dirs = self.genCanDirs()

        print("uvst and dir generated....")
        # dir to coefs
        self.Coef = []
        for dir_ in dirs:
            self.Coef.append([dir_[0]*0.5+0.5,dir_[1]*0.5+0.5])
        self.Coef = np.asarray(self.Coef)

        # gen OLAT Img
        if self.compute_OLAT:
            self.genImg(uvst,dirs)

            print("compute_OLAT generated....")
        # generate weights
        if self.compute_light:
            weights, bSparse = self._genFrameMaps()
            self._genLightImgs(self.env_mapImg_filefolder,weights)

            print("compute_light generated....")

        # gpu add together
        weights = np.asarray(weights)
        weights_tensor = torch.from_numpy(weights).cuda()
        weights_tensor = torch.unsqueeze(weights_tensor,2) # frame x l_dir x 1 x channel
        weights_tensor = torch.unsqueeze(weights_tensor,3)
        #
        #  

        # torch.from_numpy(weights).cuda()
        results = torch.zeros(weights_tensor.shape[0],self.h,self.w, 3).cuda()
        for i in tqdm(range(weights.shape[1])):
            img = np.asarray(Image.open("{}/{}_{:3f}_{:3f}.png".format(self.OLAT_filefolder,
                    i,self.Coef[i][0], self.Coef[i][1])), np.uint8)

            img_tensor = torch.from_numpy(img).float().cuda()

            inImg = torch.pow(img_tensor / 255.0, 2.2)

            results += weights_tensor[:,i,:,:,:] * inImg

        results = torch.pow(results, 1.0/2.2)*255.0
        results[results > 255] = 255

        results = results.cpu().numpy()

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(self.res_folder+"RelitVid"+self.env_name_base+".mp4", fourcc, 24.0, (self.w,self.h))

        for i, frame in enumerate(results):
                
                res = frame.astype(np.uint8)

                Image.fromarray(res).save(
                    self.res_folder + "/{:4d}.png".format(i))

                out.write(cv2.cvtColor(res,cv2.COLOR_RGB2BGR))
        
        out.release()
        
        # Save as GIF
        gif_images = []
        for i, frame in enumerate(results):
            res = frame.astype(np.uint8)
            gif_images.append(res)
        
        gif_filename = self.res_folder + "RelitVid" + self.env_name_base + ".gif"
        imageio.mimsave(gif_filename, gif_images, fps=24)
            

    def genLightImgs(self):
        pass

    def genCanDirs(self):
        unit = 2.0 / self.lightRes
        uvMap = np.ones((self.lightRes, self.lightRes, 2))
        oneC = np.linspace(-1.0, 1.0, self.lightRes)
        uvMap[:, :, 0] = oneC.reshape((1,-1))/2.0
        uvMap[:, :, 1] = oneC.reshape((-1, 1))*-1.0/2.0

        phis = np.arctan2(uvMap[:,:,1], uvMap[:,:,0]).reshape(-1)
        thetas = np.pi*np.sqrt(uvMap[:,:,1]**2.0 + uvMap[:,:,0]**2.0).reshape(-1)

        self.validMapIds = np.reshape(np.where(thetas < np.pi / 2.0),-1)
        self.chosenIds = np.reshape(np.where(thetas[self.validMapIds] < np.deg2rad(self.deg)),-1)

        dirs = SphToVec(np.column_stack([phis[self.validMapIds], thetas[self.validMapIds]]))
        self.candDirs = dirs

        #fine sample the envmaps to antialias
        self.fineSampleDirs = []
        sampleUnit = unit / (self.nSample + 1)
        for ix in range(self.nSample):
            for iy in range(self.nSample):
                uvMap = np.ones((self.lightRes, self.lightRes, 2))
                oneX = np.linspace(-1.0 - unit/2.0 + (ix+1)*sampleUnit, 1.0 + unit/2.0 - (self.nSample - ix)*sampleUnit, self.lightRes)
                oneY = np.linspace(-1.0 - unit/2.0 + (iy + 1) * sampleUnit, 1.0 + unit/2.0 - (self.nSample - iy) * sampleUnit, self.lightRes)
                uvMap[:, :, 0] = oneX.reshape((1, -1)) / 2.0
                uvMap[:, :, 1] = oneY.reshape((-1, 1)) * -1.0 / 2.0

                curphis = (np.arctan2(uvMap[:, :, 1], uvMap[:, :, 0]).reshape(-1))[self.validMapIds]
                curthetas = (np.pi * np.sqrt(uvMap[:, :, 1] ** 2.0 + uvMap[:, :, 0] ** 2.0).reshape(-1))[self.validMapIds]

                curDirs = SphToVec(np.column_stack([curphis, curthetas]))
                if type(self.flipOpt) != type(None):
                    if self.flipOpt[0] == 1:
                        curDirs[:, 0] *= -1
                    if self.flipOpt[1] == 1:
                        curDirs[:, 1] *= -1
                self.fineSampleDirs.append(curDirs)

        return dirs

    def genImg(self, uvst, dirs):
        uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
        for cnt, dir in tqdm(enumerate(dirs)):
            # print("cnt: ", cnt)
            with torch.no_grad():  # Enclose in no_grad for inference operations.
                img = self.render_sample_img(self.model, uvst, self.w, self.h, dir, cnt)

    def render_sample_img(self, model, uvst, w, h, light_cur, cnt):
        
        # Generate face 2 face light
        light_sample = np.broadcast_to(light_cur, (uvst.shape[0], 3))
        light_sample = torch.from_numpy(light_sample.astype(np.float32)).cuda()

        uvst_light = torch.cat([uvst, light_sample], dim=-1)

        pred_color, pred_normal, pred_albedo, pred_rough, light_dir = model(uvst_light)

        pred_img = pred_color.reshape((w, h, 3)).permute((2, 1, 0))
        filename = "{}_{:3f}_{:3f}.png".format(cnt, light_cur[0] * 0.5 + 0.5, light_cur[1] * 0.5 + 0.5)
        imgname = self.OLAT_filefolder + filename

        # Save image
        torchvision.utils.save_image(pred_img.detach().cpu(), imgname)  # Detach and move to CPU before saving.

        return pred_img.detach().cpu()  # Detach and move to CPU before returning.
            
    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        self.iter=0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"{self.checkpoints}nelf-{self.iter}.pth"

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


    def _genFrameMaps(self):
        weightMaps = []

        rotAxis = (0,1.0,0)

        candDirs = self.candDirs.copy()
        if type(self.flipOpt) != type(None):
            if self.flipOpt[0] == 1:
                candDirs[:,0] *= -1
            if self.flipOpt[1] == 1:
                candDirs[:,1] *= -1

        for i in range(self.rotRes):
            deg = 360.0 / self.rotRes * i


            rotDirs = rotateVector(candDirs, rotAxis, np.deg2rad(deg))

            r = (1.0/np.pi) * np.arccos(rotDirs[:,2]) / ((1.0-rotDirs[:,2]**2)**0.5)

            pos = ((np.column_stack([rotDirs[:, 0] * r, -rotDirs[:, 1] * r]) * 0.5 + 0.5) * self.envRes)
            color = subPixels(self.env, pos[:,0], pos[:,1])

            nS = 1
            for iS in range(len(self.fineSampleDirs)):
                onerotDirs = rotateVector(self.fineSampleDirs[iS], rotAxis, np.deg2rad(deg))

                r = (1.0 / np.pi) * np.arccos(onerotDirs[:, 2]) / ((1.0 - onerotDirs[:, 2] ** 2) ** 0.5)

                pos = ((np.column_stack([onerotDirs[:, 0] * r, -onerotDirs[:, 1] * r]) * 0.5 + 0.5) * self.envRes)


                onecolor = subPixels(self.env, pos[:, 0], pos[:, 1])
                color += onecolor
                nS += 1
            color/=nS
            
            solidAnlgeRate = 0.001
            color *=  self.imgRate * solidAnlgeRate

            weightMaps.append(color)

        return weightMaps, False


    def _genLightImgs(self, outFolder, weightMaps):

        # weights, bSparse = weightMaps
        weights = weightMaps

        lightImg = (np.zeros((self.lightRes, self.lightRes, 3)) * 255).reshape(-1,3)

        if self.chosenIds is None:
            chosenIds = range(len(self.validMapIds))
        else:
            chosenIds = self.chosenIds

        print(f"choosing {len(chosenIds)} from {len(self.validMapIds)}")

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(outFolder+"ligImgVid"+self.env_name_base+".mp4", fourcc, 24.0, (self.lightRes,self.lightRes))

        # Save as GIF
        gif_images = []
        for iw, weight in enumerate(weights):

            color = weight / 0.01

            gc = np.power(color, 1.0/2.2)*255.0
            lightImg[self.validMapIds[chosenIds]] = gc[chosenIds]
            lightImg[lightImg>255] = 255

            res = lightImg.reshape((self.lightRes, self.lightRes, 3)).astype(np.uint8)

            out.write(cv2.cvtColor(res,cv2.COLOR_RGB2BGR))
            gif_images.append(res)

            Image.fromarray(res).save(outFolder+"/%04d.png"%iw)

        out.release()
        
        gif_filename = outFolder + "ligImgVid" + self.env_name_base + ".gif"
        imageio.mimsave(gif_filename, gif_images, fps=24)
        
       

if __name__=="__main__":
    args = parser.parse_args()

    m_eval = Relight(args)

    m_eval.computePipeline()