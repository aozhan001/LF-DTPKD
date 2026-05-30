import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dino_transforms import renormalize_for_dino


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_dinov2_repo_or_dir(repo_or_dir=""):
    repo_or_dir = os.path.expanduser(str(repo_or_dir).strip())
    if repo_or_dir:
        return repo_or_dir

    default_repo_or_dir = os.path.join(_PROJECT_ROOT, "dinov2")
    if os.path.isdir(default_repo_or_dir):
        return default_repo_or_dir

    return ""


def resolve_dinov2_checkpoint_path(ckpt_path="", model_name="dinov2_vitl14"):
    ckpt_path = os.path.expanduser(str(ckpt_path).strip())
    if ckpt_path:
        return ckpt_path

    default_ckpt_path = os.path.join(_PROJECT_ROOT, "clip", f"{model_name}_pretrain.pth")
    if os.path.isfile(default_ckpt_path):
        return default_ckpt_path

    return ""


def build_dinov2_model(model_name="dinov2_vitl14", repo_or_dir="", pretrained=True):
    """Build a DINOv2 model, preferring a local repo when provided."""
    errors = []
    repo_or_dir = resolve_dinov2_repo_or_dir(repo_or_dir)

    if repo_or_dir:
        if os.path.isdir(repo_or_dir):
            try:
                return torch.hub.load(repo_or_dir, model_name, source="local", pretrained=pretrained)
            except Exception as exc:
                errors.append(
                    "Failed to load DINOv2 from local repo '{}': {}".format(repo_or_dir, exc)
                )
        else:
            errors.append("DINO_REPO_OR_DIR does not exist: {}".format(repo_or_dir))

    try:
        return torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
    except Exception as exc:
        errors.append(
            "Failed to load DINOv2 model '{}' from torch.hub: {}".format(model_name, exc)
        )

    raise RuntimeError(
        "\n".join(
            [
                "Unable to build DINOv2 model '{}'.".format(model_name),
                "If this machine has no network access, please set TRAINER.PROMPTKD_DINO.DINO_REPO_OR_DIR or TRAINER.DINOV2_PRETRAIN.DINO_REPO_OR_DIR to a local DINOv2 repository path.",
            ]
            + errors
        )
    )


def _strip_known_prefixes(state_dict):
    prefixes = [
        "module.",
        "backbone.",
        "teacher.",
        "teacher.backbone.",
        "student.",
        "student.backbone.",
    ]
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value

    return cleaned


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model", "teacher", "student"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported DINOv2 checkpoint format: {}".format(type(checkpoint)))


def load_dinov2_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_known_prefixes(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    print(
        "Loaded DINOv2 checkpoint from {} (missing keys: {}, unexpected keys: {})".format(
            ckpt_path, len(missing_keys), len(unexpected_keys)
        )
    )
    if missing_keys:
        print("First missing keys: {}".format(missing_keys[:10]))
    if unexpected_keys:
        print("First unexpected keys: {}".format(unexpected_keys[:10]))

    return incompatible


def extract_dino_features(model, image):
    if hasattr(model, "forward_features"):
        features = model.forward_features(image)
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if "x_prenorm" in features:
                return features["x_prenorm"][:, 0]
        elif torch.is_tensor(features):
            return features

    output = model(image)
    if isinstance(output, dict):
        if "x_norm_clstoken" in output:
            return output["x_norm_clstoken"]
        if "x_prenorm" in output:
            return output["x_prenorm"][:, 0]
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output


class DINOv2Teacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dino_cfg = cfg.TRAINER.PROMPTKD_DINO
        self.cfg = cfg
        ckpt_path = resolve_dinov2_checkpoint_path(
            ckpt_path=dino_cfg.DINO_CKPT,
            model_name=dino_cfg.DINO_MODEL_NAME,
        )
        self.model = build_dinov2_model(
            model_name=dino_cfg.DINO_MODEL_NAME,
            repo_or_dir=dino_cfg.DINO_REPO_OR_DIR,
            pretrained=not bool(ckpt_path),
        )

        if ckpt_path:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError("DINO checkpoint not found: {}".format(ckpt_path))
            load_dinov2_checkpoint(self.model, ckpt_path)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image):
        dino_image = renormalize_for_dino(image, cfg=self.cfg)
        feature = extract_dino_features(self.model, dino_image)
        feature = F.normalize(feature.float(), dim=-1)
        return feature
