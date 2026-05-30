import math
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.data import DatasetWrapper
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from clip import clip
from clip.model import convert_weights
from datasets.oxford_pets import OxfordPets
from trainers.dinov2_teacher import DINOv2Teacher


class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        return final_feat.squeeze(-1).squeeze(-1)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        prompt_cfg = cfg.TRAINER.PROMPTKD_DINO
        n_cls = len(classnames)
        assert prompt_cfg.PROMPT_DEPTH_TEXT >= 1

        n_ctx = prompt_cfg.N_CTX_TEXT
        ctx_init = prompt_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize

        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        if self.train_modal == "base2novel":
            split_idx = math.ceil(self.n_cls / 2)
            self.register_buffer("token_prefix", embedding[:split_idx, :1, :])
            self.register_buffer("token_suffix", embedding[:split_idx, 1 + n_ctx :, :])
            self.register_buffer("token_prefix2", embedding[split_idx:, :1, :])
            self.register_buffer("token_suffix2", embedding[split_idx:, 1 + n_ctx :, :])
        else:
            self.register_buffer("token_prefix", embedding[:, :1, :])
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
            self.register_buffer("token_prefix2", embedding[:, :1, :])
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx :, :])

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.train_modal == "base2novel":
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        return self.construct_prompts(ctx, prefix, suffix)


def _clip_model_path(backbone_name):
    if backbone_name == "ViT-B/16":
        return "./clip/ViT-B-16.pt"
    if backbone_name == "ViT-L/14":
        return "./clip/ViT-L-14.pt"
    if backbone_name == "ViT-B/32":
        return "./clip/ViT-B-32.pt"
    raise ValueError("Unsupported CLIP backbone: {}".format(backbone_name))


def _load_clip_state_dict(model_path):
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        model = None
        state_dict = torch.load(model_path, map_location="cpu")
    return model, state_dict


def load_clip_to_cpu_student(cfg):
    model_path = _clip_model_path(cfg.MODEL.BACKBONE.NAME)
    model, state_dict = _load_clip_state_dict(model_path)
    design_details = {
        "trainer": "IVLP",
        "vision_depth": cfg.TRAINER.PROMPTKD_DINO.PROMPT_DEPTH_VISION,
        "language_depth": cfg.TRAINER.PROMPTKD_DINO.PROMPT_DEPTH_TEXT,
        "vision_ctx": cfg.TRAINER.PROMPTKD_DINO.N_CTX_VISION,
        "language_ctx": cfg.TRAINER.PROMPTKD_DINO.N_CTX_TEXT,
    }
    return clip.build_model(state_dict or model.state_dict(), design_details)


def load_clip_to_cpu_teacher(cfg):
    model_path = _clip_model_path(cfg.TRAINER.PROMPTKD_DINO.CLIP_TEACHER_NAME)
    model, state_dict = _load_clip_state_dict(model_path)
    design_details = {
        "trainer": "IVLP",
        "vision_depth": cfg.TRAINER.PROMPTKD_DINO.PROMPT_DEPTH_VISION,
        "language_depth": cfg.TRAINER.PROMPTKD_DINO.PROMPT_DEPTH_TEXT,
        "vision_ctx": cfg.TRAINER.PROMPTKD_DINO.N_CTX_VISION,
        "language_ctx": cfg.TRAINER.PROMPTKD_DINO.N_CTX_TEXT,
    }
    return clip.build_model(state_dict or model.state_dict(), design_details)


class CustomCLIPStudent(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
        convert_weights(self.VPT_image_trans)

        dino_dim = cfg.TRAINER.PROMPTKD_DINO.DINO_FEATURE_DIM
        self.dino_projector = nn.Sequential(
            nn.Linear(768, dino_dim),
            nn.LayerNorm(dino_dim),
            nn.GELU(),
            nn.Linear(dino_dim, dino_dim),
        )

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        dino_student_features = self.dino_projector(image_features.float())
        dino_student_features = F.normalize(dino_student_features, dim=-1)
        logit_scale = self.logit_scale.exp()
        return image_features.float(), logit_scale, dino_student_features


class PromptKDCLIPTeacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image=None, label=None):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if image is None:
            return None, text_features.float(), None

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return image_features.float(), text_features.float(), logits.float()


@TRAINER_REGISTRY.register()
class PromptKDDINO(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD_DINO.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        super().build_data_loader()
        self.base_test_loader = None
        self.final_test_loader = None

        if self.cfg.TRAINER.MODAL == "base2novel":
            # For base2new distillation, validation/evaluation uses test_base, not the original val set.
            # For base2new final testing, evaluation uses test_val.
            tfm_test = build_transform(self.cfg, is_train=False)
            sampler = self.cfg.DATALOADER.TEST.SAMPLER
            batch_size = self.cfg.DATALOADER.TEST.BATCH_SIZE
            dataset = self.dm.dataset

            full_test = getattr(dataset, "_test", None) or dataset.test
            if not full_test:
                raise RuntimeError("Cannot find the full test split required for PromptKDDINO base2new evaluation")

            train_x = dataset.train_x
            if self.cfg.DATASET.NAME.lower() == "imagenet":
                _, test_base = OxfordPets.subsample_classes(train_x, full_test, subsample="base")
                _, test_val = OxfordPets.subsample_classes(train_x, full_test, subsample="new")
            else:
                val_source = getattr(dataset, "_val", None) or full_test
                _, _, test_base = OxfordPets.subsample_classes(train_x, val_source, full_test, subsample="base")
                _, _, test_val = OxfordPets.subsample_classes(train_x, val_source, full_test, subsample="new")

            self.base_test_loader = build_data_loader(
                self.cfg,
                sampler_type=sampler,
                data_source=test_base,
                batch_size=batch_size,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=DatasetWrapper,
            )
            self.final_test_loader = build_data_loader(
                self.cfg,
                sampler_type=sampler,
                data_source=test_val,
                batch_size=batch_size,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=DatasetWrapper,
            )
            self.val_loader = self.base_test_loader
            self.test_loader = self.final_test_loader

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        self.train_modal = cfg.TRAINER.MODAL
        self.base_split = math.ceil(self.n_cls / 2)

        print("Loading CLIP student (backbone: {})".format(cfg.MODEL.BACKBONE.NAME))
        clip_model = load_clip_to_cpu_student(cfg)
        clip_teacher_model = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PROMPTKD_DINO.PREC in ["fp32", "amp"]:
            clip_model.float()
            clip_teacher_model.float()

        print("Building PromptKDDINO student")
        self.model = CustomCLIPStudent(cfg, classnames, clip_model)

        print("Building PromptKD-style CLIP teacher without loading a teacher checkpoint")
        self.model_teacher = PromptKDCLIPTeacher(cfg, classnames, clip_teacher_model)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()
        for param in self.model_teacher.parameters():
            param.requires_grad_(False)

        self.dino_teacher = DINOv2Teacher(cfg)
        self.dino_teacher.to(self.device)
        self.dino_teacher.eval()
        for param in self.dino_teacher.parameters():
            param.requires_grad_(False)

        print("Turning off gradients in both the image and text encoders except VPT and dino_projector")
        for name, param in self.model.named_parameters():
            if "VPT" in name or "dino_projector" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print("Parameters to be updated: {}".format(enabled))
        print("Parameters count: {}".format(len(enabled)))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.trainable_list = nn.ModuleList([self.model])
        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD_DINO.PREC == "amp" else None
        self.temperature = cfg.TRAINER.PROMPTKD_DINO.TEMPERATURE

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print("Multiple GPUs detected (n_gpus={}), use all of them!".format(device_count))
            self.model = nn.DataParallel(self.model)

    def _compute_dino_rel_loss(self, dino_student_features, dino_teacher_features):
        cfg = self.cfg.TRAINER.PROMPTKD_DINO
        if not cfg.DINO_USE_RELATIONAL_KD or dino_student_features.size(0) <= 1:
            return dino_student_features.new_tensor(0.0)

        sim_student = dino_student_features @ dino_student_features.t()
        sim_teacher = dino_teacher_features @ dino_teacher_features.t()

        mask = ~torch.eye(sim_student.size(0), device=sim_student.device, dtype=torch.bool)
        sim_student = sim_student.masked_fill(~mask, -1e4)
        sim_teacher = sim_teacher.masked_fill(~mask, -1e4)

        tau = cfg.DINO_TEMPERATURE
        return F.kl_div(
            F.log_softmax(sim_student / tau, dim=1),
            F.softmax(sim_teacher / tau, dim=1),
            reduction="batchmean",
        ) * (tau * tau)

    def _current_dino_weight(self):
        cfg = self.cfg.TRAINER.PROMPTKD_DINO
        if cfg.DINO_WARMUP_EPOCH > 0:
            warmup_factor = min(1.0, float(self.epoch + 1) / float(cfg.DINO_WARMUP_EPOCH))
        else:
            warmup_factor = 1.0
        return cfg.DINO_WEIGHT * warmup_factor

    def _compute_loss(self, image_ft, logit_scale, dino_student_features, teacher_text_features, teacher_logits, dino_teacher_features):
        stu_logits = logit_scale * image_ft @ teacher_text_features.t().detach()
        temperature = self.temperature
        loss_clip_kd = F.kl_div(
            F.log_softmax(stu_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="sum",
        ) * (temperature * temperature) / stu_logits.numel()

        loss_dino_feat = torch.mean(1.0 - F.cosine_similarity(dino_student_features, dino_teacher_features, dim=-1))
        loss_dino_rel = self._compute_dino_rel_loss(dino_student_features, dino_teacher_features)

        cfg = self.cfg.TRAINER.PROMPTKD_DINO
        loss_dino = cfg.DINO_FEAT_WEIGHT * loss_dino_feat + cfg.DINO_REL_WEIGHT * loss_dino_rel
        current_dino_weight = self._current_dino_weight()
        loss = cfg.KD_WEIGHT * loss_clip_kd + current_dino_weight * loss_dino
        return loss, loss_clip_kd, loss_dino_feat, loss_dino_rel, current_dino_weight

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        with torch.no_grad():
            _, teacher_text_features, teacher_logits = self.model_teacher(image)
            dino_teacher_features = self.dino_teacher(image)

        model = self.model
        optim = self.optim
        prec = self.cfg.TRAINER.PROMPTKD_DINO.PREC

        if prec == "amp":
            with autocast():
                image_ft, logit_scale, dino_student_features = model(image, label)
                loss, loss_clip_kd, loss_dino_feat, loss_dino_rel, current_dino_weight = self._compute_loss(
                    image_ft,
                    logit_scale,
                    dino_student_features,
                    teacher_text_features,
                    teacher_logits,
                    dino_teacher_features,
                )
            optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            self.scaler.update()
        else:
            image_ft, logit_scale, dino_student_features = model(image, label)
            loss, loss_clip_kd, loss_dino_feat, loss_dino_rel, current_dino_weight = self._compute_loss(
                image_ft,
                logit_scale,
                dino_student_features,
                teacher_text_features,
                teacher_logits,
                dino_teacher_features,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "loss_clip_kd": loss_clip_kd.item(),
            "loss_dino_feat": loss_dino_feat.item(),
            "loss_dino_rel": loss_dino_rel.item(),
            "dino_weight": current_dino_weight,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            print("Loading weights to {} from \"{}\" (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar",
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if self.train_modal == "base2novel":
            if split == "val":
                data_loader = self.base_test_loader
                text_slice = slice(0, self.base_split)
            else:
                split = "test"
                data_loader = self.final_test_loader
                text_slice = slice(self.base_split, self.n_cls)
        else:
            text_slice = slice(0, self.n_cls)
            if split == "val" and self.val_loader is not None:
                data_loader = self.val_loader
            else:
                split = "test"
                data_loader = self.test_loader

        print("Evaluate on the *{}* set".format(split))

        for batch_idx, batch in enumerate(data_loader):
            image, label = self.parse_batch_test(batch)
            _, teacher_text_features, _ = self.model_teacher(image, label)
            teacher_text_features = teacher_text_features[text_slice]
            image_ft, logit_scale, _ = self.model(image, label)
            output = logit_scale * image_ft @ teacher_text_features.t()
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for key, value in results.items():
            tag = "{}/{}".format(split, key)
            self.write_scalar(tag, value, self.epoch)

        return list(results.values())[0]
