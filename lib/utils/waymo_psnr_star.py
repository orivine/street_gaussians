"""
Helper module for computing PSNR* metric as described in the Street Gaussians paper.

PSNR* projects 3D tracked bounding boxes to the 2D image plane (expanded 1.5x in
length and width) and computes PSNR only on pixels inside the projected mask.
"""

import os
import math
import numpy as np
import torch

from lib.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
from lib.utils.general_utils import quaternion_to_matrix_numpy


# Waymo camera image sizes (indexed by camera label: 0=FRONT, 1=FRONT_LEFT, ...)
IMAGE_HEIGHTS = [1280, 1280, 1280, 886, 886]
IMAGE_WIDTHS = [1920, 1920, 1920, 1920, 1920]

# Mapping from track_info.txt class names to labels
WAYMO_TRACK2LABEL = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}


def load_track_info(datadir, use_tracker=False):
    """
    Load tracking data from track_info.txt (or track_info_castrack.txt).
    
    Returns:
        tracklets: list of dicts, each with keys:
            frame_id, track_id, object_class, height, width, length,
            center_x, center_y, center_z, heading, speed
    """
    # if use_tracker:
    #     track_path = os.path.join(datadir, 'track', 'track_info_castrack.txt')
    # else:
    #     track_path = os.path.join(datadir, 'track', 'track_info.txt')
    track_path = os.path.join(datadir, 'track', 'track_info.txt')

    tracklets = []
    with open(track_path, 'r') as f:
        lines = f.read().splitlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 11:
            continue

        frame_id = int(parts[0])
        track_id = int(parts[1])
        object_class = parts[2]

        # Skip signs and misc objects
        if object_class in ['sign', 'misc']:
            continue

        tracklet = {
            'frame_id': frame_id,
            'track_id': track_id,
            'object_class': object_class,
            'height': float(parts[4]),
            'width': float(parts[5]),
            'length': float(parts[6]),
            'center_x': float(parts[7]),
            'center_y': float(parts[8]),
            'center_z': float(parts[9]),
            'heading': float(parts[10]),
        }

        # Speed field may or may not exist (castrack format doesn't have it)
        if len(parts) > 11:
            tracklet['speed'] = float(parts[11])
        else:
            tracklet['speed'] = None

        tracklets.append(tracklet)

    return tracklets


def compute_psnr_star_mask_for_frame(
    tracklets_in_frame,
    cam_id,
    intrinsic,
    extrinsic,
    ego_pose,
    height,
    width,
    expand_factor=1.5,
    speed_threshold=1.0,
):
    """
    Compute the PSNR* binary mask for a single image by projecting 3D bounding boxes
    (expanded by expand_factor in length and width) to the 2D image plane.

    Args:
        tracklets_in_frame: list of tracklet dicts for this frame
        cam_id: integer camera ID (0=FRONT, 1=FRONT_LEFT, etc.)
        intrinsic: [3, 3] camera intrinsic matrix
        extrinsic: [4, 4] camera-to-ego (cam_to_vehicle) extrinsic matrix
        ego_pose: [4, 4] ego vehicle pose (vehicle-to-world, but we use vehicle frame)
        height: image height in pixels
        width: image width in pixels
        expand_factor: factor to expand length and width (default 1.5 per paper)
        speed_threshold: minimum speed (m/s) to consider an object as "moving".
            Set to None to include all objects (not just moving ones).
            For PSNR*, we want all tracked objects, so default is 1.0 to match
            the dynamic mask logic; set to 0.0 if you want all objects.
    Returns:
        mask: [H, W] boolean numpy array, True for pixels inside projected boxes
    """
    mask = np.zeros((height, width), dtype=np.bool_)

    w2c = np.linalg.inv(extrinsic)  # vehicle-to-camera

    for tracklet in tracklets_in_frame:
        # Optional speed filtering (for PSNR* we typically include all moving objects)
        if speed_threshold is not None and tracklet.get('speed') is not None:
            if tracklet['speed'] < speed_threshold:
                continue

        # Build 3D bounding box dimension with 1.5x expansion in length and width
        length = tracklet['length'] * expand_factor
        width_box = tracklet['width'] * expand_factor
        height_box = tracklet['height']  # height NOT expanded per paper

        # Build 3D bounding box corners
        bbox = np.array([
            [-length, -width_box, -height_box],
            [length, width_box, height_box]
        ]) * 0.5
        corners_local = bbox_to_corner3d(bbox)

        # Build object pose in vehicle frame
        tx = tracklet['center_x']
        ty = tracklet['center_y']
        tz = tracklet['center_z']
        heading = tracklet['heading']
        c = math.cos(heading)
        s = math.sin(heading)
        rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        obj_pose_vehicle = np.eye(4)
        obj_pose_vehicle[:3, :3] = rotz_matrix
        obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

        # Transform corners from local object frame to vehicle frame
        corners_homo = np.concatenate(
            [corners_local, np.ones_like(corners_local[..., :1])], axis=-1
        )
        corners_vehicle = corners_homo @ obj_pose_vehicle.T  # [8, 4]

        # Check if any corner is visible in this camera using projection
        corners_cam = corners_vehicle[..., :3] @ w2c[:3, :3].T + w2c[:3, 3:].T
        # At least one corner should be in front of camera
        if not np.any(corners_cam[:, 2] > 0):
            continue

        # Generate 2D mask using existing utility
        obj_mask = get_bound_2d_mask(
            corners_3d=corners_vehicle[..., :3],
            K=intrinsic,
            pose=w2c,
            H=height,
            W=width,
        )
        mask = np.logical_or(mask, obj_mask)

    return mask


def compute_psnr_star_masks(
    source_path,
    cam_infos,
    use_tracker=False,
    expand_factor=1.5,
    speed_threshold=1.0,
):
    """
    Compute PSNR* masks for all test camera views.

    Args:
        source_path: path to the data directory (e.g., data/waymo/validation/006)
        cam_infos: list of Camera objects (from dataset.test_cameras[1])
        use_tracker: whether to use castrack tracking data
        expand_factor: factor to expand length and width (default 1.5 per paper)
        speed_threshold: minimum speed to consider an object as moving

    Returns:
        masks: dict mapping image_name -> [H, W] boolean numpy mask
    """
    # Load intrinsics and extrinsics from disk
    intrinsics_dir = os.path.join(source_path, 'intrinsics')
    extrinsics_dir = os.path.join(source_path, 'extrinsics')

    intrinsics = {}
    extrinsics = {}
    for i in range(5):
        intr_raw = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intr_raw[0], intr_raw[1], intr_raw[2], intr_raw[3]
        intrinsics[i] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        extrinsics[i] = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))

    # Load ego poses
    ego_pose_dir = os.path.join(source_path, 'ego_pose')

    # Load tracking data
    tracklets = load_track_info(source_path, use_tracker=use_tracker)

    # Group tracklets by frame_id
    tracklets_by_frame = {}
    for t in tracklets:
        fid = t['frame_id']
        if fid not in tracklets_by_frame:
            tracklets_by_frame[fid] = []
        tracklets_by_frame[fid].append(t)

    # Compute masks for each camera view
    masks = {}
    for cam_info in cam_infos:
        image_name = cam_info.image_name  # e.g., "000004_0"

        # Parse frame and camera from image name
        parts = image_name.split('_')
        frame_id = int(parts[0])
        cam_id = int(parts[1])

        h = cam_info.image_height
        w = cam_info.image_width

        # Get tracklets for this frame
        frame_tracklets = tracklets_by_frame.get(frame_id, [])

        # Camera intrinsic and extrinsic
        intr = intrinsics[cam_id]
        ext = extrinsics[cam_id]  # cam_to_vehicle

        # Ego pose for this frame (not strictly needed since tracklets are in vehicle frame)
        ego_pose_path = os.path.join(ego_pose_dir, f"{frame_id:06d}.txt")
        if os.path.exists(ego_pose_path):
            ego_pose = np.loadtxt(ego_pose_path)
        else:
            ego_pose = np.eye(4)

        # Scale intrinsic to match the actual rendered image resolution
        # The original images are IMAGE_HEIGHTS[cam_id] x IMAGE_WIDTHS[cam_id]
        orig_h = IMAGE_HEIGHTS[cam_id]
        orig_w = IMAGE_WIDTHS[cam_id]
        scale_h = h / orig_h
        scale_w = w / orig_w
        scale = min(scale_h, scale_w)  # They should be the same

        scaled_intr = intr.copy()
        scaled_intr[0, :] *= scale  # fx, cx
        scaled_intr[1, :] *= scale  # fy, cy

        mask = compute_psnr_star_mask_for_frame(
            tracklets_in_frame=frame_tracklets,
            cam_id=cam_id,
            intrinsic=scaled_intr,
            extrinsic=ext,
            ego_pose=ego_pose,
            height=h,
            width=w,
            expand_factor=expand_factor,
            speed_threshold=speed_threshold,
        )

        masks[image_name] = mask

    return masks


def psnr_star(render, gt, mask):
    """
    Compute PSNR* given rendered image, ground truth, and a 2D mask.

    Args:
        render: [C, H, W] tensor, rendered image
        gt: [C, H, W] tensor, ground truth image
        mask: [H, W] boolean numpy array or tensor

    Returns:
        psnr_value: scalar tensor
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(render.device)

    # If mask has no True pixels, return 0 (no moving objects in this frame)
    if not mask.any():
        return None

    # Extract pixels inside the mask
    # render, gt: [C, H, W] -> [H, W, C]
    render_hw = render.permute(1, 2, 0)  # [H, W, C]
    gt_hw = gt.permute(1, 2, 0)  # [H, W, C]

    render_masked = render_hw[mask]  # [N, C]
    gt_masked = gt_hw[mask]  # [N, C]

    mse = torch.mean((render_masked - gt_masked) ** 2)
    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_val
