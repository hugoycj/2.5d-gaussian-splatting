# 2DGS implementation based on GauStudio [beta version]
## Key Features
1. **Naive 2DGS**: We set the third scale component of 3DGS to 0, to achieve the desired 2D effect without the need for a custom rasterizer. This allows our system to be compatible with a wide range of 3DGS-enabled renderers, providing greater flexibility and ease of integration.

2. **Simplified distortion loss for floater cleaning**: We addresses the floater issues by minimizing the modified distortion loss

3. **Streamlined Mesh Reconstruction with GauStudio**

## Updates
- [x] (**14/04/2024**) Release a baseline version for 2DGS
- [x] (**16/04/2024**) Resolved the critical issue outlined in https://github.com/hugoycj/2dgs-gaustudio/issues/1.
- [x] (**18/04/2024**) Add mask preparation script and mask loss
- [x] (**24/04/2024**) Add a tutorial on how to use the DTU, BlendedMVS, and MobileBrick datasets for training
- [x] (**27/04/2024**) Implemented simplified distortion loss (Latest Version [GauStudio Rasterizer](https://github.com/GAP-LAB-CUHK-SZ/gaustudio/tree/master/submodules/gaustudio-diff-gaussian-rasterization) is needed)
- [ ] Improve mesh quality by integrating monocular prior similar to [dn-splatter](https://github.com/maturk/dn-splatter) and [gaussian_surfels](https://turandai.github.io/projects/gaussian_surfels/)
- [ ] Improve training efficiency.

## Implementation
### Naive 2DGS
We initialize the third component of the 3DGS's scale to  log(0), which effectively sets it to negative infinity (log(0)). Although log(0) is mathematically undefined, the subsequent exponential activation function (exp) maps negative infinity to 0. This means that after the initialization and activation, the third scale component becomes zero.

During optimization, the gradients for this third scale component are not explicitly stopped. However, **since the activation function (exp) is applied, any changes to the pre-activation value (negative infinity) will still result in a zero output after the activation.**

Consequently, the third scale component remains zero throughout the optimization process, effectively zeroing out the scaling along that axis.

### Simplified distortion loss
The original distortion loss is defined as $\sum_{i}\sum_{j} w_i w_j |z_i - z_j|$. To calculate the distortion loss, a custom rasterizer is needed for computing the distortion map. To simplify the implementation, we approximate the original distortion loss by considering only the dominant weight. Our assumption is that if $j \neq \operatorname{argmax}(w_j)$, then $w_i \times w_j \approx 0$, since one of the weights is not the maximum weight. Then, we can write the distortion loss in the form of:
$$
\sum_{i}\sum_{j} w_i w_j |z_i - z_j| \\
\approx \sum_{i} w_i w_j |z_i - z_j|, \quad j = \operatorname{argmax}(w_j) \\
 = w_j \sum_{i} w_i |z_i - z_j|
$$
Using the definitions $\text{Depth} = \sum_{i} w_i z_i$ and $\text{Opacity} = \sum_{i} w_i$, the simplified distortion loss becomes:
$$
w_j * |\text{Depth} - \text{Opacity} * z_j|
$$

## Install
```
# Install GauStudio for training and mesh extraction
%cd /content
!rm -r /content/gaustudio
!pip install pip install -q plyfile torch torchvision tqdm opencv-python-headless omegaconf einops kiui scipy pycolmap==0.4.0 vdbfusion kornia trimesh
!git clone --recursive https://github.com/GAP-LAB-CUHK-SZ/gaustudio.git
%cd gaustudio/submodules/gaustudio-diff-gaussian-rasterization
!python setup.py install
%cd ../../
!python setup.py develop
```

## Training
### Training on Colmap Dataset
#### Prepare Data
```
# generate mask
python preprocess_mask.py --data <path to data>
```
#### Running
```
# 2DGS training
pythont train.py -s <path to data> -m output/trained_result
# 2DGS training with mask
pythont train.py -s <path to data> -m output/trained_result --w_mask #make sure that `masks` dir exists under the data folder
# naive 2DGS training without extra regularization
python train.py -s <path to data>  -m output/trained_result --lambda_normal_consistency 0. --lambda_depth_distortion 0.
```
The results will be saved in `output/trained_result/point_cloud/iteration_{xxxx}/point_cloud.ply`. 

### Training on DTU
#### Prepare Data
Download preprocessed DTU data provided by [NeuS](https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?e=1&dl=0)

The data is organized as follows:
```
<model_id>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000000.png        # target image for each view
    |-- 000001.png
    ...
|-- mask
    |-- 000000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 000001.png
    ...
```
#### Running
```
# 2DGS training
pythont train.py --dataset neus -s <path to DTU data>/<model_id> -m output/DTU-neus/<model_id>
# e.g.
python train.py --dataset neus -s ./data/DTU-neus/dtu_scan105 -m output/DTU-neus/dtu_scan105

# 2DGS training with mask
pythont train.py --dataset neus -s <path to DTU data>/<model_id> -m output/DTU-neus-w_mask/<model_id> --w_mask
# e.g.
python train.py --dataset neus -s ./data/DTU-neus/dtu_scan105 -m output/DTU-neus-w_mask/dtu_scan105 --w_mask
```

### Training on BlendedMVS
Download original [BlendedMVS data](https://github.com/YoYo000/BlendedMVS) which is in MVSNet input format.
The data is organized as follows:
```
<model_id>                
├── blended_images          
│	├── 00000000.jpg        
│	├── 00000000_masked.jpg        
│	├── 00000001.jpg        
│	├── 00000001_masked.jpg        
│	└── ...                 
├── cams                      
│  	├── pair.txt           
│  	├── 00000000_cam.txt    
│  	├── 00000001_cam.txt    
│  	└── ...                 
└── rendered_depth_maps     
  	├── 00000000.pfm        
   	├── 00000001.pfm        
   	└── ...                    
```

#### Running
```
# 2DGS training
pythont train.py --dataset mvsnet -s <path to BlendedMVS data>/<model_id> -m output/BlendedMVS/<model_id>
# e.g.
python train.py --dataset mvsnet -s ./data/BlendedMVS/5a4a38dad38c8a075495b5d2 -m output/BlendedMVS/5a4a38dad38c8a075495b5d2
```


### Training on MobileBrick
Download original [MobileBrick data](http://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip).
The data is organized as follows:
```
SEQUENCE_NAME
├── arkit_depth (the confidence and depth maps provided by ARKit)
|    ├── 000000_conf.png
|    ├── 000000.png
|    ├── ...
├── gt_depth (The high-resolution depth maps projected from the aligned GT shape)
|    ├── 000000.png
|    ├── ...     
├── image (the RGB images)
|    ├── 000000.jpg
|    ├── ...
├── mask (object foreground mask projected from the aligned GT shape)
|    ├── 000000.png
|    ├── ...
├── intrinsic (3x3 intrinsic matrix of each image)
|    ├── 000000.txt
|    ├── ...
├── pose (4x4 transformation matrix from camera to world of each image)
|    ├── 000000.txt
|    ├── ...
├── mesh
|    ├── gt_mesh.ply
├── visibility_mask.npy (the visibility mask to be used for evaluation)
├── cameras.npz (processed camera poses using the format of NeuS)       
```

#### Running
```
# 2DGS training
pythont train.py --dataset mobilebrick -s <path to MobileBrick data>/<model_id> -m output/MobileBrick/<model_id>
# e.g.
python train.py --dataset mobilebrick -s ./data/MobileBrick/test/aston -m output/MobileBrick/aston

# 2DGS training with mask
pythont train.py --dataset mobilebrick -s <path to MobileBrick data>/<model_id> -m output/MobileBrick-w_mask/<model_id> --w_mask
# e.g.
python train.py --dataset mobilebrick -s ./data/MobileBrick/test/aston -m output/MobileBrick-w_mask/aston --w_mask
```

## Mesh Extraction
```
gs-extract-mesh -m output/trained_result -o output/trained_result
```
# Bitex
If you found this library useful for your research, please consider citing:
```
@article{ye2024gaustudio,
  title={GauStudio: A Modular Framework for 3D Gaussian Splatting and Beyond},
  author={Ye, Chongjie and Nie, Yinyu and Chang, Jiahao and Chen, Yuantao and Zhi, Yihao and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2403.19632},
  year={2024}
}
@article{huang20242d,
  title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  journal={arXiv preprint arXiv:2403.17888},
  year={2024}
}
```
# License
This project is licensed under the Gaussian-Splatting License - see the LICENSE file for details.
