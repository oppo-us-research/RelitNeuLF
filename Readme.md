#  [Relit-NeuLF: Efficient Relighting and Novel View Synthesis via Neural 4D Light Field](https://oppo-us-research.github.io/RelitNeuLF-website/)  
## ACM Multimedia(ACM MM) 2023 
[Zhong Li](https://sites.google.com/site/lizhong19900216)<sup>1</sup>,
 [Liangchen Song](https://lsongx.github.io/)<sup>2</sup>,
 [Zhang Chen](https://zhangchen8.github.io/)<sup>1</sup>,
 [Xiangyu Du](https://www.linkedin.com/in/xiangyu-du-9b1216113/)<sup>1</sup>,
 [Lele Chen](https://lelechen63.github.io/)<sup>1</sup>,
 [Junsong Yuan](https://cse.buffalo.edu/~jsyuan/)<sup>2</sup>,
 [Yi Xu](https://www.linkedin.com/in/yi-xu-42654823/)<sup>1</sup>,  
 <sup>1</sup>OPPO US Research Center, <sup>2</sup>University at Buffalo


<p align="center">
  <img src='img/teaser.png' width="750"/>
</p>

> In this paper, we address the problem of simultaneous relighting and novel view synthesis of a complex scene from multi-view image with a limited number of light sources. We propose an analysis-synthesis approach called Relit-NeuLF. Following the recent neural 4D light field network (NeuLF) [22 ], Relit-NeuLF first leverages a
two-plane light field representation to parameterize each ray in a 4D coordinate system, enabling efficient learning and inference. Then, we recover the spatially-varying bidirectional reflectance distribu-
tion function (SVBRDF) of a 3D scene in a self-supervised manner. A DecomposeNet learns to map each ray to its SVBRDF components:albedo, normal, and roughness. Based on the decomposed BRDF components and conditioning light directions, a RenderNet learns to synthesize the color of the ray. To self-supervise the SVBRDF decomposition, we encourage the predicted ray color to be close to the physically-based rendering result using the microfacet model. Comprehensive experiments demonstrate that the proposed method is efficient and effective on both synthetic data and real-world human face data, and outperforms the state-of-the-art results

### [Project Page](https://arxiv.org/pdf/2310.14642.pdf) | [Paper](https://arxiv.org/pdf/2310.14642.pdf) | [Dataset](https://huggingface.co/datasets/lizhong323/LumiView) | [Youtube](https://youtu.be/80D2PHTaiLE)
TLDR, [Relit-NeuLF](https://oppo-us-research.github.io/RelitNeuLF-website/) (Relightable Neural 4D Light Field) is a method that achieves state-of-the-art results for simutaneously relighting and  synthesizing novel views of complex scenes. Here are some videos generated by this repository (pre-trained models are provided below):

# 1.Installation
Our code is tested on Ubuntu 18.04 with Pytorch 1.10.1 Python 3.7,CUDA 11.3. We recommend using Anaconda to install the dependencies.

```
conda create -n relitneulf python=3.7
conda activate relitneulf
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install opencv-python
pip install imageio==2.20.0
pip install scipy
pip install tqdm
pip install setuptools==59.5.0
pip install tensorboardX==1.6
pip install tensorboard==2.1.0
pip install scikit-learn
pip install scikit-image
pip install lpips   
```

# 2.How to Run

- 2.1 **Synthetic Dataset**

  We used Blender’s physically based path tracer renderer and rendered 3 textured objects: `synthetic face`, `wood train`, and `face mask`. We set up 5 × 5 camera views on the front hemisphere, set 105 directional light sources around the full sphere, and render at a resolution of 800 × 800 pixels. Each camera differs by 10 degrees and each light source differs by 25 degrees on the sphere. Please download the `LumiView` dataset from this link hosted by huggingface: [BlenderData](https://huggingface.co/datasets/lizhong323/LumiView/resolve/main/data.zip).

  ```
  mkdir data/BlenderData
  wget https://huggingface.co/datasets/lizhong323/LumiView/resolve/main/data.zip
  unzip data.zip
  ```
  unzip it and put them in the folder `data/BlenderData/`.

  For the detailed description of the dataset, please refer to the [dataset page](https://huggingface.co/datasets/lizhong323/LumiView/blob/main/README.md).

- 2.2 **Train the model**
  To train the multiview and multilight model, please run the following command for indivisual scene in `LumiView dataset`:
  
  For `ToyTrain` scene:
  ```
  # preprocessing the data
  python src/blenderProcess_less_trainval.py --data_dir data/BlenderData/toytrain/multilight/ --uv_depth 9.0 --st_depth 0.0 --resize_ratio 2

  python src/train_nelf_blender_multilight_normalalbedo_roughness_Fast_val.py --exp_name toytrain_multi_albedo_full --data_dir data/BlenderData/toytrain/multilight/ --uv_depth 10.0 --st_depth 0.0 --resize_ratio 2 --method full
  ```

  For `FaceCover`scene:
  ```
  # preprocessing the data
  python src/blenderProcess_less_trainval.py --data_dir data/BlenderData/facecover/multilight/ --uv_depth 10.0 --st_depth 0.0 --resize_ratio 2 

  python src/train_nelf_blender_multilight_normalalbedo_roughness_Fast_val.py --exp_name facecover_multi_albedo_full --data_dir data/BlenderData/facecover/multilight/ --uv_depth 10.0 --st_depth 0.0 --resize_ratio 2 --method full
  ```
  For `FaceBase` scene:
  ```
  # preprocessing the data
  python src/blenderProcess_less_trainval.py --data_dir data/BlenderData/FaceBase/multilight/ --uv_depth 5.0 --st_depth 0.0 --resize_ratio 2 

  python src/train_nelf_blender_multilight_normalalbedo_roughness_Fast_val.py --exp_name FaceBase_multi_albedo_full --data_dir data/BlenderData/FaceBase/multilight/ --uv_depth 5.0 --st_depth 0.0 --resize_ratio 2 --method full --gpuid 0
  ```
  The results will in `result_mm/`folder. The expect animation result is `result_mm/Exp_xxxx/epoch-xxx/video_ani.gif`

 <table cellspacing="0" cellpadding="0" style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
      <td><img src="material/video_ani_facecover.gif" alt="GIF1 Description" width="150"/></td>
      <td><img src="material/video_ani_toytrain.gif" alt="GIF2 Description" width="150"/></td>
      <td><img src="material/video_ani.gif" alt="GIF3 Description" width="150"/></td>
    </tr>
  </table>





- 2.3 **Relighting demo**

  Relighting demo on synthetic data for various light probe format environment map. Please download the light probe data `*.pfm` from here: [light probe](https://www.pauldebevec.com/Probes/), and put them in the folder `env_map/`.

  To run the relighting demo, for here, we choose the `grace.pfm` as the environment map, and experiment on the synthetic data `FaceBase` with the light source distance `uv_depth=5.0` and `st_depth=0.0`. If it is your first time to run this experiment, please add `--compute_OLAT` argument to precompute the OLAT imgs. 

  ```
  # grace_probe
  python src/demo_relight_oneView.py --st_depth 0.0 --uv_depth 5.0 --exp_name FaceBase_multi_albedo_full --data_dir data/BlenderData/FaceBase/multilight/ --env_map_dir 'env_map/grace_probe.pfm' --compute_OLAT

  # uffizi_probe
  python src/demo_relight_oneView.py --st_depth 0.0 --uv_depth 5.0 --exp_name FaceBase_multi_albedo_full --data_dir data/BlenderData/FaceBase/multilight/ --env_map_dir 'env_map/uffizi_probe.pbm'
  ```
  Above command will generate the relighting results structure like below
  ```
  - result_relight/
  - Exp_FaceBase_multi_albedo_full/
    - lightImg # store current direction of light probe imgs and mp4 video
    - OLAT_img # store the precomputed OLAT imgs
    - result # store the relight results and mp4 video
  ```
  And expecting relighting result is like below:
  <p align="center">
  <img src="material/relighting.gif" alt="Alternate Text" width="400">
  </p>

# Citation

```
@inproceedings{li2023relitneulf,
  title={Relit-NeuLF: Efficient Novel View Synthesis with Neural 4D Light Field},
  author={Li, Zhong, Song, Liangchen, Chen, Zhang, Du, Xiangyu, Chen, Lele, Yuan, Junsong, Xu, Yi},
  booktitle={Proceedings of the 31th ACM International Conference on Multimedia},
  year={2023}
}
```