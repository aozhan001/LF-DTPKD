import random

import torch
import torch.nn.functional as F


DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _normalize_stats(values):
    values = [float(v) for v in values]
    if max(abs(v) for v in values) > 1.5:
        values = [v / 255.0 for v in values]
    return values


def _stats_to_tensor(values, ref_tensor):
    values = _normalize_stats(values)
    tensor = torch.tensor(values, device=ref_tensor.device, dtype=torch.float32)
    return tensor.view(1, len(values), 1, 1)


def _resolve_input_stats(cfg=None):
    if cfg is not None:
        mean = getattr(cfg.INPUT, "PIXEL_MEAN", None)
        std = getattr(cfg.INPUT, "PIXEL_STD", None)
        if mean is not None and std is not None:
            return list(mean), list(std)
    return CLIP_MEAN, CLIP_STD


def _resolve_pixel_batch(image, cfg=None):
    if image.dim() != 4:
        raise ValueError("Expected a batch tensor of shape [B, C, H, W]")

    input_mean, input_std = _resolve_input_stats(cfg)
    image_float = image.float()
    src_mean = _stats_to_tensor(input_mean, image_float)
    src_std = _stats_to_tensor(input_std, image_float)
    pixel = image_float * src_std + src_mean
    return pixel.clamp_(0.0, 1.0)


def _pixel_to_dino(pixel):
    dino_mean = _stats_to_tensor(DINO_MEAN, pixel)
    dino_std = _stats_to_tensor(DINO_STD, pixel)
    return (pixel.float() - dino_mean) / dino_std


def renormalize_for_dino(image, cfg=None):
    """Convert a CLIP-normalized batch into DINO/ImageNet normalization."""
    pixel = _resolve_pixel_batch(image, cfg=cfg)
    dino_image = _pixel_to_dino(pixel)
    return dino_image.to(dtype=image.dtype)


def _random_resized_crop(image, size=224, scale=(0.5, 1.0)):
    batch, _, height, width = image.shape
    outputs = []

    for idx in range(batch):
        sample = image[idx : idx + 1]
        area = float(height * width)
        target_area = random.uniform(scale[0], scale[1]) * area
        crop_size = int(round(target_area ** 0.5))
        crop_size = max(1, min(crop_size, height, width))

        top = random.randint(0, max(0, height - crop_size))
        left = random.randint(0, max(0, width - crop_size))

        sample = sample[:, :, top : top + crop_size, left : left + crop_size]
        sample = F.interpolate(sample, size=(size, size), mode="bicubic", align_corners=False)

        if random.random() < 0.5:
            sample = torch.flip(sample, dims=[3])

        outputs.append(sample)

    return torch.cat(outputs, dim=0)


def _adjust_brightness(sample, strength=0.4):
    factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
    return (sample * factor).clamp_(0.0, 1.0)


def _adjust_contrast(sample, strength=0.4):
    factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
    mean = sample.mean(dim=(2, 3), keepdim=True)
    return ((sample - mean) * factor + mean).clamp_(0.0, 1.0)


def _adjust_saturation(sample, strength=0.2):
    factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
    gray = sample.mean(dim=1, keepdim=True)
    return ((sample - gray) * factor + gray).clamp_(0.0, 1.0)


def _maybe_grayscale(sample, p=0.2):
    if random.random() >= p:
        return sample
    gray = sample.mean(dim=1, keepdim=True)
    return gray.repeat(1, sample.size(1), 1, 1)


def _maybe_blur(sample, p=0.5):
    if random.random() >= p:
        return sample
    return F.avg_pool2d(sample, kernel_size=3, stride=1, padding=1)


def _augment_pixel_batch(pixel, crop_size=224, crop_scale=(0.5, 1.0)):
    pixel = _random_resized_crop(pixel, size=crop_size, scale=crop_scale)
    outputs = []

    for idx in range(pixel.size(0)):
        sample = pixel[idx : idx + 1]
        if random.random() < 0.8:
            sample = _adjust_brightness(sample, strength=0.4)
            sample = _adjust_contrast(sample, strength=0.4)
            sample = _adjust_saturation(sample, strength=0.2)
        sample = _maybe_grayscale(sample, p=0.2)
        sample = _maybe_blur(sample, p=0.5)
        outputs.append(sample.clamp_(0.0, 1.0))

    return torch.cat(outputs, dim=0)


def make_two_dino_views(image, cfg=None, crop_size=224):
    """Create two stronger DINO-style tensor views from a model-ready batch."""
    pixel = _resolve_pixel_batch(image, cfg=cfg)
    view1 = _augment_pixel_batch(pixel, crop_size=crop_size, crop_scale=(0.5, 1.0))
    view2 = _augment_pixel_batch(pixel, crop_size=crop_size, crop_scale=(0.3, 0.9))
    view1 = _pixel_to_dino(view1).to(dtype=image.dtype)
    view2 = _pixel_to_dino(view2).to(dtype=image.dtype)
    return view1, view2
