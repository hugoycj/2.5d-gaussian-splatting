# 2DGS implementation based on GauStudio [beta version]
<img alt="2dgs" src="assets/2dgs-teaser.jpeg" width="100%">

## Key Features

1. **Simplified 2D Geometry Shader (2DGS) Implementation**: We set the third scale component of 3DGS to 0, to achieve the desired 2D effect without the need for a custom rasterizer. This allows our system to be compatible with a wide range of 3DGS-enabled renderers, providing greater flexibility and ease of integration.

2. **Floater cleaning using simplified distortion loss**: We addresses the floater issues by minimizing the modified distortion loss, which is the gap between the median depth and the rendered depth

3. **Streamlined Mesh Reconstruction with GauStudio**

## TODOs
- [x] **Resolved the critical issue outlined in https://github.com/hugoycj/2dgs-gaustudio/issues/1. The open-source version has some issues compared to our private implementation. It will take a few days to address them. Please check back in 2-3 days.**
- [x] Add mask preparation script and mask loss
- [ ] Add a tutorial on how to use the DTU, BlendedMVS, and MobileBrick datasets for training
- [ ] Implemented distortion loss in 2DGS paper
- [ ] Improve mesh quality by integrating monocular prior similar to [dn-splatter](https://github.com/maturk/dn-splatter) and [gaussian_surfels](https://turandai.github.io/projects/gaussian_surfels/)
- [ ] Improve training efficiency.

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
pythont train.py --dataset neus -s <path to DTU data>/<model_id> -m output/DTU-neus/<model_id> --w_mask
# e.g.
python train.py --dataset neus -s ./data/DTU-neus/dtu_scan105 -m output/DTU-neus/dtu_scan105 --w_mask
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
