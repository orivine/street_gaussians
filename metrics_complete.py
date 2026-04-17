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
import torch
import torchvision.transforms.functional as tf
import json
from PIL import Image
from pathlib import Path

from tqdm import tqdm
from lib.config import cfg
from lib.utils.loss_utils import ssim, psnr
from lib.utils.lpipsPyTorch import lpips
from lib.datasets.dataset import Dataset
from lib.utils.waymo_psnr_star import compute_psnr_star_masks, psnr_star


def evaluate(split='test'):
    scene_dir = cfg.model_path
    dataset = Dataset()
    if split == 'test':
        test_dir = Path(scene_dir) / "test"
        cam_infos = dataset.test_cameras[1]
    else:
        test_dir = Path(scene_dir) / "train"
        cam_infos = dataset.train_cameras[1]
        
    cam_infos = list(sorted(cam_infos, key=lambda x: x.id))
    
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    print(f"Scene: {scene_dir }")
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    # ---- Compute PSNR* masks ----
    source_path = cfg.source_path
    use_tracker = cfg.data.get('use_tracker', False)
    expand_factor = 1.5  # Paper: expand bounding box by 1.5x in length and width

    print("Computing PSNR* masks from 3D tracked bounding boxes (1.5x expanded) ...")
    psnr_star_masks = compute_psnr_star_masks(
        source_path=source_path,
        cam_infos=cam_infos,
        use_tracker=use_tracker,
        expand_factor=expand_factor,
        speed_threshold=1.0,  # Only moving objects (speed >= 1.0 m/s)
    )
    print(f"PSNR* masks computed for {len(psnr_star_masks)} views.")
    # ---- End PSNR* mask computation ----
    
    for method in os.listdir(test_dir):
        print("Method:", method)
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}    


        renders = []
        gts = []
        image_names = []

        for cam_info in tqdm(cam_infos, desc="Reading image progress"):
            image_name = cam_info.image_name
            render_path = test_dir / method / f'{image_name}_rgb.png'
            gt_path = test_dir / method / f'{image_name}_gt.png'
            
            render = Image.open(render_path)
            gt = Image.open(gt_path)
            renders.append(tf.to_tensor(render)[:3, :, :])
            gts.append(tf.to_tensor(gt)[:3, :, :])
            image_names.append(image_name)

        psnrs = []
        ssims = []
        lpipss = []
        psnr_stars = []
        psnr_star_names = []  # Track which images actually have moving objects

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render = renders[idx].cuda()
            gt = gts[idx].cuda()
            ssims.append(ssim(render, gt))
            psnrs.append(psnr(render, gt))
            lpipss.append(lpips(render, gt, net_type='alex'))

            # Compute PSNR*
            img_name = image_names[idx]
            if img_name in psnr_star_masks:
                mask = psnr_star_masks[img_name]
                ps = psnr_star(render, gt, mask)
                if ps is not None:
                    psnr_stars.append(ps)
                    psnr_star_names.append(img_name)
        
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

        if len(psnr_stars) > 0:
            psnr_star_mean = torch.tensor(psnr_stars).mean().item()
            print("  PSNR*: {:>12.7f}".format(psnr_star_mean, ".5"))
            print(f"  (PSNR* computed over {len(psnr_stars)}/{len(renders)} views with moving objects)")
        else:
            psnr_star_mean = None
            print("  PSNR*: N/A (no moving objects detected in any test view)")
        print("")
        
        result_dict = {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }
        if psnr_star_mean is not None:
            result_dict["PSNR*"] = psnr_star_mean

        full_dict[scene_dir][method].update(result_dict)

        per_view_result = {
            "SSIM": {name: ssim_val for ssim_val, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: psnr_val for psnr_val, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
        }
        if len(psnr_stars) > 0:
            per_view_result["PSNR*"] = {
                name: ps_val for ps_val, name in zip(torch.tensor(psnr_stars).tolist(), psnr_star_names)
            }
        per_view_dict[scene_dir][method].update(per_view_result)

    with open(scene_dir + f"/results_{split}.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + f"/per_view_{split}.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    if cfg.eval.eval_train:
        evaluate(split='train')
    if cfg.eval.eval_test:
        evaluate(split='test')
