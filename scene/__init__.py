#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from gaustudio import datasets

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        self.train_cameras = {}
        self.test_cameras = {}
        
        # Initialize dataset with gaustudio.datasets
        _dataset = datasets.make({"name": args.dataset, "source_path": args.source_path, \
                                            "images": args.images,
                                            'w_mask': args.w_mask,
                                            "resolution":resolution_scales[0], "data_device":"cuda", \
                                            "eval": False})
        _dataset.export(os.path.join(self.model_path, "cameras.json"))
        print("Loading Training Cameras")
        self.train_cameras[resolution_scales[0]] = _dataset.all_cameras
        print("Loading Test Cameras")
        self.test_cameras[resolution_scales[0]] = []
        self.cameras_extent = _dataset.cameras_extent

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # Initialize pcd with gaustudio.initializers
            from gaustudio.pipelines import initializers
            from gaustudio import models
            pcd = models.make("general_pcd")
            if args.dataset != "colmap":
                if args.w_mask:
                    initializer_config = {"name":'colmap', "workspace_dir":os.path.join(args.source_path, 'colmap-w_mask')}
                else:
                    initializer_config = {"name":'colmap', "workspace_dir":os.path.join(args.source_path, 'colmap')}
            else:
                initializer_config = {"name":'colmap', "workspace_dir":args.source_path}
            initializer = initializers.make(initializer_config)
            initializer(pcd, _dataset, overwrite=False)
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]