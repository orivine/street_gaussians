import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim, weighted_l1_loss
from lib.utils.priority_utils import (
    build_priority_mask,
    build_camera_priority_scores,
    build_residual_priority_mask,
    build_priority_weight_map,
    build_priority_sampling_probs,
    compute_priority_warmup,
    get_camera_valid_mask,
    sample_view_index,
    summarize_priority_mask,
)
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False


def _cfg_to_plain_dict(node):
    if isinstance(node, dict):
        return {key: _cfg_to_plain_dict(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_cfg_to_plain_dict(value) for value in node]
    return node


def prepare_wandb():
    wandb_cfg = cfg.logging.wandb
    if not wandb_cfg.enabled:
        return None
    if not WANDB_FOUND:
        print("wandb not available: not logging to wandb")
        return None

    run_name = wandb_cfg.name if len(wandb_cfg.name) > 0 else f"{cfg.task}-{cfg.exp_name}"
    entity = wandb_cfg.entity if len(wandb_cfg.entity) > 0 else None
    tags = list(wandb_cfg.tags)

    try:
        return wandb.init(
            project=wandb_cfg.project,
            entity=entity,
            name=run_name,
            tags=tags,
            mode=wandb_cfg.mode,
            dir=cfg.record_dir,
            save_code=wandb_cfg.save_code,
            sync_tensorboard=wandb_cfg.sync_tensorboard,
            config=_cfg_to_plain_dict(cfg),
        )
    except Exception as exc:
        print(f"Failed to initialize wandb: {exc}")
        return None

def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data
    method_args = cfg.method
    if method_args.priority.source == 'box_residual':
        if method_args.priority.residual_ema:
            raise NotImplementedError('Priority residual EMA is not implemented yet')
        if method_args.priority.residual_norm != 'percentile':
            raise NotImplementedError(f'Residual norm "{method_args.priority.residual_norm}" is not implemented yet')

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)

    gaussians.training_setup()
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    train_cameras = scene.getTrainCameras()
    priority_sampler = None
    if method_args.sampler.mode == 'priority_mix':
        priority_sampler = build_camera_priority_scores(
            cameras=train_cameras,
            priority_source=method_args.priority.source,
            priority_dilate=method_args.priority.dilate,
            score_type=method_args.sampler.score_type,
            score_source=method_args.sampler.score_source,
        )
        priority_sampler['probs'] = build_priority_sampling_probs(
            scores=priority_sampler['scores'],
            eta=method_args.sampler.eta,
        )
    elif method_args.sampler.mode != 'uniform':
        raise NotImplementedError(f'Sampler mode "{method_args.sampler.mode}" is not implemented yet')

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        selected_view_score = None
        selected_view_priority_ratio = None
        if method_args.sampler.mode == 'uniform':
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        else:
            selected_view_idx = sample_view_index(priority_sampler['probs'])
            viewpoint_cam: Camera = train_cameras[selected_view_idx]
            selected_view_score = float(priority_sampler['scores'][selected_view_idx].item())
            selected_view_priority_ratio = float(priority_sampler['priority_ratios'][selected_view_idx].item())
        
        gt_image = viewpoint_cam.original_image
        mask = get_camera_valid_mask(viewpoint_cam)
        obj_bound = viewpoint_cam.guidance.get('obj_bound')
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
        mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
        if 'lidar_depth' in viewpoint_cam.guidance:
            lidar_depth = viewpoint_cam.guidance['lidar_depth']
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
        if 'sky_mask' in viewpoint_cam.guidance:
            sky_mask = viewpoint_cam.guidance['sky_mask']
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
        if obj_bound is not None:
            obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound

        priority_mask = None
        box_priority_mask = None
        if method_args.priority.source != 'none' or method_args.photo_loss.mode == 'priority_l1':
            priority_source = method_args.priority.source
            if priority_source == 'box_residual':
                priority_source = 'box'
            priority_mask = build_priority_mask(
                obj_bound=obj_bound,
                valid_mask=mask,
                source=priority_source,
                dilate=method_args.priority.dilate,
            )
            if method_args.priority.source == 'box_residual':
                box_priority_mask = priority_mask
        
            
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]

        residual_priority_stats = None
        if method_args.priority.source == 'box_residual' and method_args.photo_loss.mode == 'priority_l1':
            priority_mask, residual_priority_stats = build_residual_priority_mask(
                image=image,
                gt_image=gt_image,
                valid_mask=mask,
                box_mask=box_priority_mask,
                residual_scope=method_args.priority.residual_scope,
                residual_lambda=method_args.priority.residual_lambda,
                residual_blend=method_args.priority.residual_blend,
                residual_norm=method_args.priority.residual_norm,
                residual_percentile=method_args.priority.residual_percentile,
                return_stats=True,
            )

        scalar_dict = dict()
        if priority_mask is not None:
            scalar_dict.update(summarize_priority_mask(priority_mask, mask))
            scalar_dict['priority_source_box'] = 1.0 if method_args.priority.source == 'box' else 0.0
            scalar_dict['priority_source_box_residual'] = 1.0 if method_args.priority.source == 'box_residual' else 0.0
            scalar_dict['priority_dilate'] = float(method_args.priority.dilate)
        if residual_priority_stats is not None:
            scalar_dict.update(residual_priority_stats)
            scalar_dict['residual_lambda'] = float(method_args.priority.residual_lambda)
            scalar_dict['residual_blend'] = float(method_args.priority.residual_blend)
            scalar_dict['residual_percentile'] = float(method_args.priority.residual_percentile)
            scalar_dict['residual_scope_box'] = 1.0 if method_args.priority.residual_scope == 'box' else 0.0
            scalar_dict['residual_scope_global'] = 1.0 if method_args.priority.residual_scope == 'global' else 0.0
        if method_args.photo_loss.mode == 'priority_l1':
            scalar_dict['photo_loss_priority_l1'] = 1.0
        if method_args.sampler.mode == 'priority_mix':
            scalar_dict['sampler_priority_mix'] = 1.0
        if selected_view_score is not None:
            scalar_dict['sampler_eta'] = method_args.sampler.eta
            scalar_dict['selected_view_score'] = selected_view_score
            scalar_dict['selected_view_priority_ratio'] = selected_view_priority_ratio
        
        # rgb loss
        if method_args.photo_loss.mode == 'baseline':
            Ll1 = l1_loss(image, gt_image, mask)
            scalar_dict['l1_loss'] = Ll1.item()
        elif method_args.photo_loss.mode == 'priority_l1':
            priority_warmup = compute_priority_warmup(
                iteration=iteration,
                warmup_start=method_args.photo_loss.warmup_start,
                warmup_end=method_args.photo_loss.warmup_end,
            )
            weight_map = build_priority_weight_map(
                valid_mask=mask,
                priority_mask=priority_mask,
                lambda_p=method_args.photo_loss.lambda_p,
                warmup=priority_warmup,
            )
            Ll1 = weighted_l1_loss(image, gt_image, weight_map)
            scalar_dict['l1_loss'] = Ll1.item()
            scalar_dict['weighted_l1_loss'] = Ll1.item()
            scalar_dict['priority_lambda'] = method_args.photo_loss.lambda_p
            scalar_dict['priority_warmup'] = priority_warmup
        else:
            raise NotImplementedError(f'Photometric loss mode "{method_args.photo_loss.mode}" is not implemented yet')

        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
    
        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss
        
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss

        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:            
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            expected_depth = depth / (render_pkg['acc'] + 1e-10)  
            depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
            depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
            lidar_depth_loss = depth_error.mean()
            scalar_dict['lidar_depth_loss'] = lidar_depth_loss
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
                    
        scalar_dict['loss'] = loss.item()
        
        loss.backward()
        
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 1000 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
        with torch.no_grad():
            
            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:                    
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})
            progress_bar.update(1)
            # if iteration == training_args.iterations:
            #     progress_bar.close()

            # Save ply
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)



def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    prepare_wandb()

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    if WANDB_FOUND and wandb.run is not None:
        wandb.finish()

    # All done
    print("\nTraining complete.")
