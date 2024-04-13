# 2DGS implementation based on GauStudio [beta version]
## Key Features
* naive implementation for 2DGS by setting the third axis of scale to 0
* naive implementation of distortion loss by minimize the gap between median depth and rendered depth
* reconstruct mesh in one line using GauStudio

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

## How to Use
```
# 2DGS training
pythont train.py -s <path to data> -m output/trained_result
# naive 2DGS training without extra regularization
python train.py -s <path to data>  -m output/trained_result --lambda_normal_consistency 0. --lambda_depth_distortion 0.
```
The results will be saved in `output/trained_result/point_cloud/iteration_{xxxx}/point_cloud.ply`. 

## Mesh Extraction
```
gs-extract-mesh -m output/trained_result -o output/trained_result
```

# License
This project is licensed under the Gaussian-Splatting License - see the LICENSE file for details.