import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer

from trainers.dinov2_teacher import (
    build_dinov2_model,
    extract_dino_features,
    load_dinov2_checkpoint,
    resolve_dinov2_checkpoint_path,
)
from utils.dino_transforms import make_two_dino_views


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim=1024, out_dim=65536, bottleneck_dim=256):
        super().__init__()
        hidden_dim = max(in_dim, 2048)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_norm = nn.LayerNorm(bottleneck_dim)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x.float())
        x = self.last_norm(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class DINOv2SelfDistillModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pre_cfg = cfg.TRAINER.DINOV2_PRETRAIN
        init_ckpt = resolve_dinov2_checkpoint_path(
            ckpt_path=pre_cfg.INIT_CKPT,
            model_name=pre_cfg.MODEL_NAME,
        )
        self.student_backbone = build_dinov2_model(
            model_name=pre_cfg.MODEL_NAME,
            repo_or_dir=pre_cfg.DINO_REPO_OR_DIR,
            pretrained=not bool(init_ckpt),
        )
        self.teacher_backbone = copy.deepcopy(self.student_backbone)

        if init_ckpt:
            if not os.path.isfile(init_ckpt):
                raise FileNotFoundError("DINOv2 init checkpoint not found: {}".format(init_ckpt))
            load_dinov2_checkpoint(self.student_backbone, init_ckpt)
            load_dinov2_checkpoint(self.teacher_backbone, init_ckpt)

        self.student_head = DINOProjectionHead(pre_cfg.FEATURE_DIM, pre_cfg.OUT_DIM)
        self.teacher_head = copy.deepcopy(self.student_head)

        for param in self.teacher_backbone.parameters():
            param.requires_grad_(False)
        for param in self.teacher_head.parameters():
            param.requires_grad_(False)

    def student_forward(self, image):
        features = extract_dino_features(self.student_backbone, image)
        return self.student_head(features)

    @torch.no_grad()
    def teacher_forward(self, image):
        features = extract_dino_features(self.teacher_backbone, image)
        return self.teacher_head(features)

    @torch.no_grad()
    def momentum_update_teacher(self, momentum):
        for param_s, param_t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1.0 - momentum)
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1.0 - momentum)


@TRAINER_REGISTRY.register()
class DINOv2Pretrain(TrainerX):
    def build_model(self):
        cfg = self.cfg
        print("Building DINOv2 self-distillation model: {}".format(cfg.TRAINER.DINOV2_PRETRAIN.MODEL_NAME))
        self.model = DINOv2SelfDistillModel(cfg)
        self.model.to(self.device)

        self.trainable = nn.ModuleList([self.model.student_backbone, self.model.student_head])
        self.optim = build_optimizer(self.trainable, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("DINOv2Pretrain", self.model, self.optim, self.sched)

        out_dim = cfg.TRAINER.DINOV2_PRETRAIN.OUT_DIM
        self.center = torch.zeros(1, out_dim, device=self.device)
        prec = getattr(cfg.TRAINER.DINOV2_PRETRAIN, "PREC", "amp")
        self.scaler = GradScaler() if prec == "amp" else None
        print("DINOv2Pretrain ignores labels and trains only with images.")

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device)

    def _teacher_momentum(self):
        pre_cfg = self.cfg.TRAINER.DINOV2_PRETRAIN
        total_steps = max(1, self.cfg.OPTIM.MAX_EPOCH * self.num_batches)
        current_step = self.epoch * self.num_batches + self.batch_idx
        progress = min(1.0, float(current_step) / float(total_steps))
        return 1.0 - 0.5 * (1.0 - pre_cfg.MOMENTUM) * (1.0 + math.cos(math.pi * progress))

    def _teacher_temperature(self):
        pre_cfg = self.cfg.TRAINER.DINOV2_PRETRAIN
        base_temp = max(float(pre_cfg.TEMPERATURE), 1e-4)
        warmup_epochs = max(1, int(getattr(pre_cfg, "TEACHER_TEMP_WARMUP", 10)))
        warmup_start = 0.04
        progress = min(1.0, float(self.epoch + 1) / float(warmup_epochs))
        return warmup_start + progress * (base_temp - warmup_start)

    def _student_temperature(self):
        return max(float(getattr(self.cfg.TRAINER.DINOV2_PRETRAIN, "STUDENT_TEMPERATURE", 0.1)), 1e-4)

    def _dino_loss(self, student_out1, student_out2, teacher_out1, teacher_out2):
        student_temp = self._student_temperature()
        teacher_temp = self._teacher_temperature()

        student_logp1 = F.log_softmax(student_out1 / student_temp, dim=-1)
        student_logp2 = F.log_softmax(student_out2 / student_temp, dim=-1)
        teacher_prob1 = F.softmax((teacher_out1 - self.center) / teacher_temp, dim=-1).detach()
        teacher_prob2 = F.softmax((teacher_out2 - self.center) / teacher_temp, dim=-1).detach()

        loss12 = -(teacher_prob1 * student_logp2).sum(dim=-1).mean()
        loss21 = -(teacher_prob2 * student_logp1).sum(dim=-1).mean()
        return 0.5 * (loss12 + loss21)

    @torch.no_grad()
    def _update_center(self, teacher_out1, teacher_out2):
        momentum = self.cfg.TRAINER.DINOV2_PRETRAIN.CENTER_MOMENTUM
        teacher_logits = torch.cat([teacher_out1, teacher_out2], dim=0)
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)

    def forward_backward(self, batch):
        image = self.parse_batch_train(batch)
        view1, view2 = make_two_dino_views(image, cfg=self.cfg)
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)

        use_amp = self.scaler is not None
        if use_amp:
            with autocast():
                student_out1 = self.model.student_forward(view1)
                student_out2 = self.model.student_forward(view2)
                with torch.no_grad():
                    teacher_out1 = self.model.teacher_forward(view1)
                    teacher_out2 = self.model.teacher_forward(view2)
                loss = self._dino_loss(student_out1, student_out2, teacher_out1, teacher_out2)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            student_out1 = self.model.student_forward(view1)
            student_out2 = self.model.student_forward(view2)
            with torch.no_grad():
                teacher_out1 = self.model.teacher_forward(view1)
                teacher_out2 = self.model.teacher_forward(view2)
            loss = self._dino_loss(student_out1, student_out2, teacher_out1, teacher_out2)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        momentum = self._teacher_momentum()
        self.model.momentum_update_teacher(momentum)
        self._update_center(teacher_out1, teacher_out2)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {
            "loss": loss.item(),
            "teacher_momentum": momentum,
            "teacher_temp": self._teacher_temperature(),
        }

    def after_epoch(self):
        super().after_epoch()
        if (self.epoch + 1) == self.max_epoch:
            self._save_teacher_checkpoint()

    @torch.no_grad()
    def _save_teacher_checkpoint(self):
        pre_cfg = self.cfg.TRAINER.DINOV2_PRETRAIN
        save_dir = os.path.expanduser(pre_cfg.SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, pre_cfg.SAVE_NAME)
        checkpoint = {
            "teacher": self.model.teacher_backbone.state_dict(),
            "student": self.model.student_backbone.state_dict(),
            "teacher_head": self.model.teacher_head.state_dict(),
            "student_head": self.model.student_head.state_dict(),
            "center": self.center.detach().cpu(),
            "epoch": self.epoch + 1,
            "config": self.cfg.dump(),
        }
        torch.save(checkpoint, save_path)
        print("Saved adapted DINOv2 teacher checkpoint to {}".format(save_path))
