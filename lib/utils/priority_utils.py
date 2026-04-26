import torch
import torch.nn.functional as F


def _ensure_mask_shape(mask):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f'Expected mask with shape [1, H, W] or [H, W], got {tuple(mask.shape)}')
    return mask.bool()


def _ensure_map_shape(value, name):
    if value is None:
        return None
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f'Expected {name} with shape [1, H, W] or [H, W], got {tuple(value.shape)}')
    return value


def _ensure_image_shape(image, name):
    if image.ndim != 3:
        raise ValueError(f'Expected {name} with shape [C, H, W], got {tuple(image.shape)}')
    return image


def get_camera_valid_mask(camera):
    valid_mask = camera.guidance.get('mask')
    if valid_mask is None:
        valid_mask = torch.ones_like(camera.original_image[0:1], dtype=torch.bool)
    return _ensure_mask_shape(valid_mask)


def dilate_binary_mask(mask, radius):
    mask = _ensure_mask_shape(mask)
    if radius < 0:
        raise ValueError(f'dilate radius must be non-negative, got {radius}')
    if radius == 0:
        return mask

    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(0).bool()


def build_priority_mask(obj_bound, valid_mask, source='none', dilate=0):
    valid_mask = _ensure_mask_shape(valid_mask)

    if source == 'none':
        return torch.zeros_like(valid_mask, dtype=torch.bool)

    if source != 'box':
        raise NotImplementedError(f'Priority source "{source}" is not implemented yet')

    obj_bound = _ensure_mask_shape(obj_bound)
    if obj_bound is None:
        raise ValueError('Priority source "box" requires obj_bound guidance')
    if obj_bound.shape != valid_mask.shape:
        raise ValueError(
            f'obj_bound shape {tuple(obj_bound.shape)} does not match valid_mask shape {tuple(valid_mask.shape)}'
        )

    priority_mask = dilate_binary_mask(obj_bound, int(dilate))
    priority_mask = torch.logical_and(priority_mask, valid_mask)
    return priority_mask


def compute_priority_warmup(iteration, warmup_start, warmup_end):
    if iteration < warmup_start:
        return 0.0
    if warmup_end <= warmup_start:
        return 1.0
    if iteration >= warmup_end:
        return 1.0
    return float(iteration - warmup_start) / float(warmup_end - warmup_start)


def build_priority_weight_map(valid_mask, priority_mask, lambda_p, warmup):
    valid_mask = _ensure_mask_shape(valid_mask)
    priority_mask = _ensure_map_shape(priority_mask, 'priority_mask')

    if priority_mask.shape != valid_mask.shape:
        raise ValueError(
            f'priority_mask shape {tuple(priority_mask.shape)} does not match valid_mask shape {tuple(valid_mask.shape)}'
        )
    if not torch.isfinite(priority_mask.float()).all():
        raise ValueError('priority_mask contains non-finite values')

    weight_map = valid_mask.float()
    weight_map = weight_map * (1.0 + float(lambda_p) * float(warmup) * priority_mask.float())
    return weight_map


def build_residual_priority_mask(
    image,
    gt_image,
    valid_mask,
    box_mask,
    residual_scope='box',
    residual_lambda=1.0,
    residual_blend=0.5,
    residual_norm='percentile',
    residual_percentile=0.95,
    eps=1e-6,
    return_stats=False,
):
    image = _ensure_image_shape(image, 'image')
    gt_image = _ensure_image_shape(gt_image, 'gt_image')
    valid_mask = _ensure_mask_shape(valid_mask)

    if image.shape != gt_image.shape:
        raise ValueError(f'image shape {tuple(image.shape)} does not match gt_image shape {tuple(gt_image.shape)}')
    if image.shape[-2:] != valid_mask.shape[-2:]:
        raise ValueError(f'image spatial shape {tuple(image.shape[-2:])} does not match valid_mask shape {tuple(valid_mask.shape[-2:])}')
    if residual_norm != 'percentile':
        raise NotImplementedError(f'Residual norm "{residual_norm}" is not implemented yet')
    if not 0.0 <= float(residual_percentile) <= 1.0:
        raise ValueError(f'residual_percentile must be in [0, 1], got {residual_percentile}')
    if float(residual_lambda) < 0.0:
        raise ValueError(f'residual_lambda must be non-negative, got {residual_lambda}')
    if not 0.0 <= float(residual_blend) <= 1.0:
        raise ValueError(f'residual_blend must be in [0, 1], got {residual_blend}')

    if residual_scope == 'box':
        box_mask = _ensure_mask_shape(box_mask)
        if box_mask is None:
            raise ValueError('residual_scope "box" requires box_mask')
        if box_mask.shape != valid_mask.shape:
            raise ValueError(
                f'box_mask shape {tuple(box_mask.shape)} does not match valid_mask shape {tuple(valid_mask.shape)}'
            )
        support_mask = torch.logical_and(box_mask, valid_mask)
    elif residual_scope == 'global':
        support_mask = valid_mask
    else:
        raise NotImplementedError(f'Residual scope "{residual_scope}" is not implemented yet')

    residual = torch.abs(image - gt_image).mean(dim=0, keepdim=True).detach()
    priority = torch.zeros_like(residual)
    quantile_value = residual.new_tensor(0.0)

    if support_mask.any().item():
        support_residual = residual[support_mask]
        quantile_value = torch.quantile(support_residual, float(residual_percentile))
        residual_normed = torch.clamp(residual / (quantile_value + float(eps)), 0.0, 1.0)
        residual_scaled = torch.clamp(float(residual_lambda) * residual_normed, 0.0, 1.0)
        priority = support_mask.float() * ((1.0 - float(residual_blend)) + float(residual_blend) * residual_scaled)

    priority = torch.clamp(priority.detach(), 0.0, 1.0)

    if not return_stats:
        return priority

    support_area = float(support_mask.sum().item())
    stats = {
        'residual_support_area': support_area,
        'residual_support_ratio': support_area / float(valid_mask.sum().item()) if valid_mask.sum().item() > 0 else 0.0,
        'residual_quantile': float(quantile_value.detach().item()),
        'residual_priority_mean': float(priority[support_mask].mean().item()) if support_area > 0 else 0.0,
        'residual_priority_max': float(priority.max().item()) if priority.numel() > 0 else 0.0,
    }
    return priority, stats


def compute_priority_score(priority_mask, score_type='box_area'):
    priority_mask = _ensure_mask_shape(priority_mask)

    if score_type in ['box_area', 'box_mass']:
        # In the current binary-mask stage, box_mass is identical to box_area.
        return float(priority_mask.sum().item())

    raise NotImplementedError(f'Sampler score type "{score_type}" is not implemented yet')


def build_camera_priority_scores(cameras, priority_source=None, priority_dilate=0, score_type='box_area', score_source=None):
    if score_source is None:
        score_source = priority_source if priority_source is not None else 'box'
    if score_source == 'box':
        score_mask_source = 'box'
    elif score_source == 'none':
        score_mask_source = 'none'
    else:
        raise NotImplementedError(f'Sampler score_source "{score_source}" is not implemented yet')

    scores = []
    priority_ratios = []

    for camera in cameras:
        valid_mask = get_camera_valid_mask(camera)

        priority_mask = build_priority_mask(
            obj_bound=camera.guidance.get('obj_bound'),
            valid_mask=valid_mask,
            source=score_mask_source,
            dilate=priority_dilate,
        )
        priority_summary = summarize_priority_mask(priority_mask, valid_mask)

        scores.append(compute_priority_score(priority_mask, score_type=score_type))
        priority_ratios.append(priority_summary['priority_ratio'])

    return {
        'scores': torch.tensor(scores, dtype=torch.float32),
        'priority_ratios': torch.tensor(priority_ratios, dtype=torch.float32),
    }


def build_priority_sampling_probs(scores, eta):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(f'Expected non-empty 1D scores tensor, got shape {tuple(scores.shape)}')
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError(f'Sampler eta must be in [0, 1], got {eta}')

    num_scores = scores.numel()
    uniform_probs = torch.full((num_scores,), 1.0 / num_scores, dtype=scores.dtype)
    score_sum = scores.sum()
    if score_sum > 0:
        priority_probs = scores / score_sum
    else:
        priority_probs = uniform_probs

    probs = (1.0 - float(eta)) * uniform_probs + float(eta) * priority_probs
    probs = probs / probs.sum()
    return probs


def sample_view_index(probs):
    probs = torch.as_tensor(probs, dtype=torch.float32)
    if probs.ndim != 1 or probs.numel() == 0:
        raise ValueError(f'Expected non-empty 1D probability tensor, got shape {tuple(probs.shape)}')
    return int(torch.multinomial(probs, 1).item())


def summarize_priority_mask(priority_mask, valid_mask):
    priority_mask = _ensure_map_shape(priority_mask, 'priority_mask')
    valid_mask = _ensure_mask_shape(valid_mask)
    if priority_mask.shape != valid_mask.shape:
        raise ValueError(
            f'priority_mask shape {tuple(priority_mask.shape)} does not match valid_mask shape {tuple(valid_mask.shape)}'
        )
    if not torch.isfinite(priority_mask.float()).all():
        raise ValueError('priority_mask contains non-finite values')

    priority_area = float(priority_mask.float().sum().item())
    valid_area = float(valid_mask.sum().item())
    priority_ratio = priority_area / valid_area if valid_area > 0 else 0.0

    return {
        'priority_area': priority_area,
        'priority_ratio': priority_ratio,
    }
