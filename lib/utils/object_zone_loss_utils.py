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


def object_zone_l1_loss(image, gt_image, zone_mask, detach_zone_weight=True, eps=1e-6):
    image = _ensure_image_shape(image, 'image')
    gt_image = _ensure_image_shape(gt_image, 'gt_image')
    zone_mask = _ensure_mask_shape(zone_mask, 'zone_mask')

    if image.shape != gt_image.shape:
        raise ValueError(f'image shape {tuple(image.shape)} does not match gt_image shape {tuple(gt_image.shape)}')
    if image.shape[-2:] != zone_mask.shape[-2:]:
        raise ValueError(
            f'image spatial shape {tuple(image.shape[-2:])} does not match zone_mask shape {tuple(zone_mask.shape[-2:])}'
        )

    zone_weight = zone_mask.float()
    if detach_zone_weight:
        zone_weight = zone_weight.detach()

    zone_area = zone_weight.sum()
    if zone_area.item() <= 0.0:
        return image.new_zeros(())

    pixel_l1 = torch.abs(image - gt_image).mean(dim=0, keepdim=True)
    return (pixel_l1 * zone_weight).sum() / (zone_area + float(eps))


def _zero_object_zone_stats(image, warmup, zone_area, valid_area, lambda_zone):
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
    }


def compute_object_zone_loss(
    image,
    gt_image,
    obj_bound,
    valid_mask,
    iteration,
    source='box',
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
    if source != 'box':
        raise NotImplementedError(f'Object-zone source "{source}" is not implemented in Feature H v1')
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

    if zone_area <= 0.0 or zone_area < float(min_area):
        if return_stats:
            return _zero_object_zone_stats(image, warmup, zone_area, valid_area, lambda_zone)
        return image.new_zeros(())

    l1_loss = image.new_zeros(())
    if float(lambda_l1) > 0.0:
        l1_loss = object_zone_l1_loss(
            image=image,
            gt_image=gt_image,
            zone_mask=zone_mask,
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
    }
