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

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mask_utils import DINOFeatureExtractor, MLPModel, calculate_residual_mask, interpolation
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_set_gendf_train(model_path, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, mlp_model, features_fine):
    render_path = os.path.join(model_path, "renders", f"render_{iteration}")
    gts_path = os.path.join(model_path, "renders", f"data_{iteration}")
    mask_path = os.path.join(model_path, "renders", f"mask_{iteration}")
    combine_path = os.path.join(model_path, "renders", f"composition_{iteration}")

    makedirs(render_path, exist_ok=True)
    makedirs(combine_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        name = "train_"+view.image_name[:-4]
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        upsample_feature = interpolation(features_fine[view.image_name], gt.shape[1], gt.shape[2])
        mask = mlp_model(upsample_feature)
        loss_mask = mask.clone().detach() > 0.25
        loss_mask = -F.max_pool2d(-(loss_mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{name}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{name}.png"))
        torchvision.utils.save_image(loss_mask, os.path.join(mask_path, f"{name}.png"))

        loss_mask = loss_mask.repeat(3, 1, 1)  # (1,H,W) → (3,H,W)
        combine = torch.cat([gt, loss_mask, rendering], dim=-1)
        torchvision.utils.save_image(combine, os.path.join(combine_path, f"{name}.png"))

def render_set_gendf_test(model_path, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        name = f"val_{idx:04d}"
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        combine = torch.cat([gt, rendering], dim=-1)

        torchvision.utils.save_image(combine, os.path.join(render_path, f"{name}.png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # load model
        ckpt = os.path.join(dataset.model_path, "ckpts", f"mlp_ckpt_{iteration}.pth")
        checkpoint = torch.load(ckpt, map_location="cuda")
        mlp_model = MLPModel().to(device="cuda")
        mlp_model.load_state_dict(checkpoint["mlp"])
        upper_feat_res, lower_feat_res = checkpoint["opt"]

        # dino features
        feature_extractor = DINOFeatureExtractor().cuda()
        features_fine, features_coarse = {}, {}
        for cam in tqdm(scene.getTrainCameras(), desc=f"DINOv2 GT Feature Extraction"):
            features_fine[cam.image_name] = feature_extractor(cam.original_image, upper_feat_res).cpu()
            #features_coarse[cam.image_name] = feature_extractor(cam.original_image, lower_feat_res).cpu()
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             #render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
             render_set_gendf_train(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, mlp_model, features_fine)

        if not skip_test:
             #render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
             render_set_gendf_test(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
