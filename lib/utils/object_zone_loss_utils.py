import torch


def _ensure_image_shape(image, name):
    if image.ndim != 3:
        raise ValueError(f'Expected {name} with shape [C, H, W], got {tuple(image.shape)}')
    return image


def _ensure_mask_shape(mask, name):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f'Expected {name} with shape [1, H, W] or [H, W], got {tuple(mask.shape)}')
    return mask.bool()


def _ensure_weight_shape(weight, name):
    if weight is None:
        return None
    if weight.ndim == 2:
        weight = weight.unsqueeze(0)
    if weight.ndim != 3:
        raise ValueError(f'Expected {name} with shape [1, H, W] or [H, W], got {tuple(weight.shape)}')
    return weight.float()


def compute_object_zone_warmup(iteration, warmup_start, warmup_end):
    if iteration < warmup_start:
        return 0.0
    if warmup_end <= warmup_start:
        return 1.0
    if iteration >= warmup_end:
        return 1.0
    return float(iteration - warmup_start) / float(warmup_end - warmup_start)


def build_object_zone_mask(obj_bound, valid_mask, min_area=32):
    valid_mask = _ensure_mask_shape(valid_mask, 'valid_mask')
    if valid_mask is None:
        raise ValueError('Object-zone mask requires valid_mask')
    if int(min_area) < 0:
        raise ValueError(f'min_area must be non-negative, got {min_area}')

    if obj_bound is None:
        return torch.zeros_like(valid_mask, dtype=torch.bool)

    obj_bound = _ensure_mask_shape(obj_bound, 'obj_bound')
    if obj_bound.shape != valid_mask.shape:
        raise ValueError(
            f'obj_bound shape {tuple(obj_bound.shape)} does not match valid_mask shape {tuple(valid_mask.shape)}'
        )

    return torch.logical_and(obj_bound, valid_mask)


def object_zone_l1_loss(image, gt_image, zone_mask, zone_weight=None, detach_zone_weight=True, eps=1e-6):
    image = _ensure_image_shape(image, 'image')
    gt_image = _ensure_image_shape(gt_image, 'gt_image')
    zone_mask = _ensure_mask_shape(zone_mask, 'zone_mask')

    if image.shape != gt_image.shape:
        raise ValueError(f'image shape {tuple(image.shape)} does not match gt_image shape {tuple(gt_image.shape)}')
    if image.shape[-2:] != zone_mask.shape[-2:]:
        raise ValueError(
            f'image spatial shape {tuple(image.shape[-2:])} does not match zone_mask shape {tuple(zone_mask.shape[-2:])}'
        )

    if zone_weight is None:
        l1_weight = zone_mask.float()
    else:
        zone_weight = _ensure_weight_shape(zone_weight, 'zone_weight')
        if zone_weight.shape != zone_mask.shape:
            raise ValueError(
                f'zone_weight shape {tuple(zone_weight.shape)} does not match zone_mask shape {tuple(zone_mask.shape)}'
            )
        if not torch.isfinite(zone_weight).all():
            raise ValueError('zone_weight contains non-finite values')
        if (zone_weight < 0.0).any().item():
            raise ValueError('zone_weight must be non-negative')
        l1_weight = zone_weight * zone_mask.float()

    if detach_zone_weight:
        l1_weight = l1_weight.detach()

    zone_mass = l1_weight.sum()
    if zone_mass.item() <= 0.0:
        return image.new_zeros(())

    pixel_l1 = torch.abs(image - gt_image).mean(dim=0, keepdim=True)
    return (pixel_l1 * l1_weight).sum() / (zone_mass + float(eps))


def _zero_object_zone_stats(
    image,
    warmup,
    zone_area,
    valid_area,
    lambda_zone,
    source_priority=False,
    priority_area=0.0,
    priority_mass=0.0,
    l1_uses_priority=False,
):
    zone_area_value = float(zone_area)
    valid_area_value = float(valid_area)
    return image.new_zeros(()), {
        'object_zone_loss': 0.0,
        'object_zone_l1_loss': 0.0,
        'object_zone_dssim_loss': 0.0,
        'object_zone_warmup': float(warmup),
        'object_zone_area': zone_area_value,
        'object_zone_ratio': zone_area_value / valid_area_value if valid_area_value > 0.0 else 0.0,
        'object_zone_lambda': float(lambda_zone),
        'object_zone_enabled': 1.0,
        'object_zone_source_priority': 1.0 if source_priority else 0.0,
        'object_zone_priority_area': float(priority_area),
        'object_zone_priority_mass': float(priority_mass),
        'object_zone_l1_uses_priority': 1.0 if l1_uses_priority else 0.0,
    }


def compute_object_zone_loss(
    image,
    gt_image,
    obj_bound,
    valid_mask,
    iteration,
    source='box',
    zone_weight=None,
    lambda_zone=0.05,
    lambda_l1=1.0,
    lambda_dssim=0.2,
    warmup_start=7000,
    warmup_end=12000,
    min_area=32,
    detach_zone_weight=True,
    use_crop_ssim=False,
    ssim_fn=None,
    eps=1e-6,
    return_stats=True,
):
    image = _ensure_image_shape(image, 'image')
    gt_image = _ensure_image_shape(gt_image, 'gt_image')
    valid_mask = _ensure_mask_shape(valid_mask, 'valid_mask')

    if image.shape != gt_image.shape:
        raise ValueError(f'image shape {tuple(image.shape)} does not match gt_image shape {tuple(gt_image.shape)}')
    if valid_mask is None:
        raise ValueError('Object-zone loss requires valid_mask')
    if image.shape[-2:] != valid_mask.shape[-2:]:
        raise ValueError(
            f'image spatial shape {tuple(image.shape[-2:])} does not match valid_mask shape {tuple(valid_mask.shape[-2:])}'
        )
    if source not in ['box', 'priority']:
        raise NotImplementedError(f'Object-zone source "{source}" is not implemented')
    if source == 'priority' and zone_weight is None:
        raise ValueError('Object-zone source "priority" requires zone_weight for L1 weighting')
    if use_crop_ssim:
        raise NotImplementedError('Object-zone crop SSIM is not implemented in Feature H v1')
    if float(lambda_zone) < 0.0:
        raise ValueError(f'lambda_zone must be non-negative, got {lambda_zone}')
    if float(lambda_l1) < 0.0:
        raise ValueError(f'lambda_l1 must be non-negative, got {lambda_l1}')
    if float(lambda_dssim) < 0.0:
        raise ValueError(f'lambda_dssim must be non-negative, got {lambda_dssim}')

    warmup = compute_object_zone_warmup(iteration, warmup_start, warmup_end)
    zone_mask = build_object_zone_mask(obj_bound=obj_bound, valid_mask=valid_mask, min_area=min_area)
    zone_area = float(zone_mask.sum().item())
    valid_area = float(valid_mask.sum().item())
    source_priority = source == 'priority'
    l1_weight = None
    priority_area = 0.0
    priority_mass = 0.0

    if source_priority:
        l1_weight = _ensure_weight_shape(zone_weight, 'zone_weight')
        if l1_weight.shape != zone_mask.shape:
            raise ValueError(
                f'zone_weight shape {tuple(l1_weight.shape)} does not match object-zone shape {tuple(zone_mask.shape)}'
            )
        if not torch.isfinite(l1_weight).all():
            raise ValueError('zone_weight contains non-finite values')
        if (l1_weight < 0.0).any().item():
            raise ValueError('zone_weight must be non-negative')
        l1_weight = l1_weight * zone_mask.float()
        if detach_zone_weight:
            l1_weight = l1_weight.detach()
        priority_area = float((l1_weight > float(eps)).sum().item())
        priority_mass = float(l1_weight.sum().detach().item())

    if zone_area <= 0.0 or zone_area < float(min_area):
        if return_stats:
            return _zero_object_zone_stats(
                image,
                warmup,
                zone_area,
                valid_area,
                lambda_zone,
                source_priority=source_priority,
                priority_area=priority_area,
                priority_mass=priority_mass,
                l1_uses_priority=source_priority,
            )
        return image.new_zeros(())

    if source_priority and priority_mass <= float(eps):
        if return_stats:
            return _zero_object_zone_stats(
                image,
                warmup,
                zone_area,
                valid_area,
                lambda_zone,
                source_priority=True,
                priority_area=priority_area,
                priority_mass=priority_mass,
                l1_uses_priority=True,
            )
        return image.new_zeros(())

    l1_loss = image.new_zeros(())
    if float(lambda_l1) > 0.0:
        l1_loss = object_zone_l1_loss(
            image=image,
            gt_image=gt_image,
            zone_mask=zone_mask,
            zone_weight=l1_weight,
            detach_zone_weight=detach_zone_weight,
            eps=eps,
        )

    dssim_loss = image.new_zeros(())
    if float(lambda_dssim) > 0.0:
        if ssim_fn is None:
            raise ValueError('Object-zone DSSIM requires ssim_fn')
        ssim_value = ssim_fn(image, gt_image, mask=zone_mask)
        if not torch.is_tensor(ssim_value):
            ssim_value = image.new_tensor(float(ssim_value))
        dssim_loss = 1.0 - ssim_value

    loss = float(lambda_zone) * float(warmup) * (
        float(lambda_l1) * l1_loss + float(lambda_dssim) * dssim_loss
    )

    if not return_stats:
        return loss

    return loss, {
        'object_zone_loss': float(loss.detach().item()),
        'object_zone_l1_loss': float(l1_loss.detach().item()),
        'object_zone_dssim_loss': float(dssim_loss.detach().item()),
        'object_zone_warmup': float(warmup),
        'object_zone_area': zone_area,
        'object_zone_ratio': zone_area / valid_area if valid_area > 0.0 else 0.0,
        'object_zone_lambda': float(lambda_zone),
        'object_zone_enabled': 1.0,
        'object_zone_source_priority': 1.0 if source_priority else 0.0,
        'object_zone_priority_area': priority_area,
        'object_zone_priority_mass': priority_mass,
        'object_zone_l1_uses_priority': 1.0 if source_priority else 0.0,
    }
