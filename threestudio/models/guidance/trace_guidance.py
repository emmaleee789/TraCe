import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

import numpy as np
import random
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *
from threestudio.utils.sch_bridge.utils import make_beta_schedule
from threestudio.utils.sch_bridge.utils import SchBridgeDiffusion
from threestudio.utils.dreamtime import precompute_prior, time_prioritize, precompute_prior_bridge


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)
    

@threestudio.register("trace-guidance")
class TraceGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-1"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.
        guidance_scale_lora: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        camera_condition_type: str = "extrinsics"
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        stage_one_weight: float = 1.
        stage_two_weight: float = 100.
        stage_two_start_step: int = 20000
        max_steps: int = 100000

        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        use_bridge: bool = False
        use_sch_bridge: bool = True

        use_dreamtime: bool = False
        use_annealing: bool = False
        use_src_tgt_noise: bool = False
        evenodd: bool = False
        pure_src_uncond: bool = False

        var_red: bool = True
        weighting_strategy: str = "sch_bridge"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        # LoRA configuration
        lora_learning_rate: float = 0.001

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            pipe_lora = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path_lora,
                **pipe_lora_kwargs,
            ).to(self.device)
            del pipe_lora.vae
            cleanup()
            pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        ).to(self.device)
        self.unet_lora.class_embedding = self.camera_embedding

        ############################################################

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
            self.device
        )
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        # Set LoRA layers to be trainable
        for p in self.lora_layers.parameters():
            p.requires_grad_(True)

        ############################################################

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )
        
        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.scheduler_lora.config.prediction_type = "v_prediction"

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        # Ensure all scheduler tensors are on the correct device
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler_lora.alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(self.device)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.betas = make_beta_schedule(n_timestep=1000, linear_end=0.3/1000)
        self.betas = np.concatenate([self.betas[:1000//2], np.flip(self.betas[:1000//2])])
        self.SchBridgeDiffusion = SchBridgeDiffusion(self.betas, self.device)
        self.x0 = torch.randn(1) # dummy x0

        self.grad_clip_val: Optional[float] = None

        self.phase_id = 1

        self.use_dreamtime = self.cfg.use_dreamtime
        self.use_annealing = self.cfg.use_annealing
        self.use_src_tgt_noise = self.cfg.use_src_tgt_noise
        self.evenodd = self.cfg.evenodd
        self.pure_src_uncond = self.cfg.pure_src_uncond

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(1000 * min_step_percent) # self.num_train_timesteps
        self.max_step = int(1000 * max_step_percent) # self.num_train_timesteps

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @property
    def lora_parameters(self):
        return self.lora_layers.parameters()

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            # Make sure scheduler tensors are on the same device as inputs
            sample_scheduler.alphas_cumprod = sample_scheduler.alphas_cumprod.to(t.device)
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images
    
    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    def sample_lora(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        B = elevation.shape[0]
        camera_condition_cfg = torch.cat(
            [
                camera_condition.view(B, -1),
                torch.zeros_like(camera_condition.view(B, -1)),
            ],
            dim=0,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale_lora,
            class_labels=camera_condition_cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        )



    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    # def compute_noise(self, step, x0, xt):
    #     """ Eq 12 """
    #     std_fwd = self.SchBridgeDiffusion.get_std_fwd(step, xdim=x0.shape[1:])
    #     noise = (xt - x0) / std_fwd
    #     return noise.detach()

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding
    
    def compute_grad_sds_sch_bridge(
        self,
        true_global_step: int,
        latents: Float[Tensor, "B 4 64 64"],
        latents_pred_denoised_src: Float[Tensor, "B 4 64 64"],
        latents_pred_denoised_tgt: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        elevation: Float[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        
        batch_size = latents.shape[0]

        with torch.no_grad():

            # step = torch.randint(20, 600, (batch_size,), dtype=torch.long, device=self.device)
            
            # compute step using annealing or not
            # if self.phase_id == 1: # since beta scheduling is a /\ shape (shown as figure 12 in paper)
            #     if self.use_annealing:
            #         steps = torch.arange(2, 450, device=self.device)
            #         m1 = 0  # Center of Gaussian for descending weights
            #         s1 = 150  # Standard deviation controlling the spread
            #         weights = torch.exp(-((steps - m1) ** 2) / (2 * s1 * s1))
            #         weights = weights / weights.sum()  # Normalize to probability distribution
            #         step = torch.multinomial(weights, batch_size, replacement=True)
            #     else:
            #         step = torch.randint(20, 980, (batch_size,), dtype=torch.long, device=self.device) # opt.interval = 1000 # TODO: use dreamtime/dreamflow time scheduling
            # elif self.phase_id == 2:  
            #     if self.use_annealing:
            #         steps = torch.arange(550, 1000, device=self.device)
            #         m2 = 1000  # Center of Gaussian for ascending weights
            #         s2 = 150  # Standard deviation controlling the spread
            #         weights = torch.exp(-((steps - m2) ** 2) / (2 * s2 * s2))
            #         weights = weights / weights.sum()  # Normalize to probability distribution
            #         step = torch.multinomial(weights, batch_size, replacement=True)
            #     else:
            #         step = torch.randint(20, 500, (batch_size,), dtype=torch.long, device=self.device) # opt.interval = 1000 # TODO: use dreamtime/dreamflow time scheduling
            
            # ===============================
            step = None
            min_t_value = 20
            max_t_value = 500

            # step = torch.randint(20, 500, (batch_size,), dtype=torch.long, device=self.device)
            
            # if self.phase_id == 1:
            #     step = torch.randint(20, 500, (batch_size,), dtype=torch.long, device=self.device)
            # elif self.phase_id == 2:
            #     step = torch.randint(20, 980, (batch_size,), dtype=torch.long, device=self.device)
            # elif self.phase_id == 3:
            #     step = torch.randint(20, 980, (batch_size,), dtype=torch.long, device=self.device)

            # if self.phase_id == 1:
            #     max_t_value = 980
            # elif self.phase_id == 2:
            #     max_t_value = 500
            # else:
            #     raise ValueError(f"Unknown phase_id {self.phase_id}")
            time_prior, _ = precompute_prior_bridge(min_t=min_t_value, max_t=max_t_value, phase_id=self.phase_id)
            step_ratio = (true_global_step - 1) / self.cfg.max_steps
            step = time_prioritize(step_ratio, time_prior, min_t=min_t_value)
            step = torch.full([batch_size], int(step), dtype=torch.long, device=self.device)

            # ===============================

            # step = torch.randint(2, 998, (batch_size,), dtype=torch.long, device=self.device) # opt.interval = 1000 # TODO: use dreamtime/dreamflow time scheduling
            
            # compute xt and bridge noise using src/tgt noise or not
            if self.use_src_tgt_noise:
                if self.evenodd: 
                    # allow "uncond + w(cond-uncond)" to be learned in two steps, an odd step and a even step
                    # less unexpected colors, e.g. green and blue
                    if true_global_step % 2 == 1: # odd
                        xt = self.SchBridgeDiffusion.q_sample(step, x0=latents_pred_denoised_src, x1=latents, ot_ode=False) # latents_noisy
                        bridge_noise = self.SchBridgeDiffusion.bridge_noise(step, xt, x0=latents_pred_denoised_src, x1=latents) # which is the target noise
                    else: # even
                        xt = self.SchBridgeDiffusion.q_sample(step, x0=latents_pred_denoised_tgt, x1=latents_pred_denoised_src, ot_ode=False) # latents_noisy
                        bridge_noise = self.SchBridgeDiffusion.bridge_noise(step, xt, x0=latents_pred_denoised_tgt, x1=latents_pred_denoised_src) # which is the target noise
                elif self.pure_src_uncond:
                        # this will only learn uncond to cond, but not learn noise to uncond
                        # cause extreme shiny cloud and over-saturation
                        xt = self.SchBridgeDiffusion.q_sample(step, x0=latents_pred_denoised_tgt, x1=latents_pred_denoised_src, ot_ode=False) # latents_noisy
                        bridge_noise = self.SchBridgeDiffusion.bridge_noise(step, xt, x0=latents_pred_denoised_tgt, x1=latents_pred_denoised_src) # which is the target noise
                else:
                    if self.phase_id == 1:
                        xt = self.SchBridgeDiffusion.q_sample(step, x0=latents_pred_denoised_tgt, x1=latents, ot_ode=False) # latents_noisy
                        bridge_noise = self.SchBridgeDiffusion.bridge_noise(step, xt, x0=latents_pred_denoised_tgt, x1=latents) # which is the target noise
                    elif self.phase_id == 2:
                        xt = self.SchBridgeDiffusion.q_sample(step, x0=latents_pred_denoised_tgt, x1=latents, ot_ode=False) # latents_noisy
                        bridge_noise = self.SchBridgeDiffusion.bridge_noise(step, xt, x0=latents_pred_denoised_tgt, x1=latents) # which is the target noise
            else:
                xt = self.SchBridgeDiffusion.q_sample(step, x0=latents_pred_denoised_tgt, x1=latents, ot_ode=False) # latents_noisy
                bridge_noise = self.SchBridgeDiffusion.bridge_noise(step, xt, x0=latents_pred_denoised_tgt, x1=latents) # which is the target noise
            
            # add noise -- xt is already a noised version
            # noise = torch.randn_like(latents)
            # latents_noisy = self.scheduler.add_noise(latents, noise, step)
            latent_model_input = torch.cat([xt] * 2, dim=0)

            # use view-independent text embeddings in LoRA
            text_embeddings_cond, _ = text_embeddings.chunk(2)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([step] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(batch_size, -1),
                        torch.zeros_like(camera_condition.view(batch_size, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            self.alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=xt.device, dtype=xt.dtype
            )
            alpha_t = self.alphas_cumprod[step] ** 0.5
            sigma_t = (1 - self.alphas_cumprod[step]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        elif self.scheduler_lora.config.prediction_type == "epsilon":
            self.alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=xt.device, dtype=xt.dtype
            )

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        w =  1 # self.SchBridgeDiffusion.mu_x0[step].view(-1, 1, 1, 1) # *((1 - self.alphas_cumprod[step])/self.alphas_cumprod[step]) # should be alphas_cumprod not alphas? yes

        grad = w * (bridge_noise - noise_pred_est)

        alpha = (self.alphas_cumprod[step] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas_cumprod[step]) ** 0.5).view(-1, 1, 1, 1)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            latents_denoised_pretrain = (
                xt - sigma * bridge_noise
            ) / alpha
            latents_denoised_est = (xt - sigma * noise_pred_est) / alpha
            image_denoised_pretrain = self.decode_latents(latents_denoised_pretrain)
            image_denoised_est = self.decode_latents(latents_denoised_est)
            grad_img = (
                w * (image_denoised_est - image_denoised_pretrain) * alpha / sigma
            )
        else:
            grad_img = None

        return grad, grad_img, step, xt, bridge_noise, noise_pred_est, w
    
    
    def train_lora(
        self,
        step: Int[Tensor, "B"],
        xt: Float[Tensor, "B 4 64 64"],
        bridge_noise: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = xt.shape[0]
        xt = xt.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        bridge_noise = bridge_noise.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = step.detach().repeat(self.cfg.lora_n_timestamp_samples)

        # t = torch.randint(
        #     int(self.num_train_timesteps * self.cfg.min_step_percent), # should be 200?
        #     int(self.num_train_timesteps * self.cfg.max_step_percent), # should be 800?
        #     [B * self.cfg.lora_n_timestamp_samples],
        #     dtype=torch.long,
        #     device=self.device,
        # ) # 1st

        # noise = torch.randn_like(xt) # should that be adding noise to xt?
        # noisy_xt = self.scheduler_lora.add_noise(xt, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = bridge_noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(xt, bridge_noise, t) # should that be xt, bridge_noise, t? yes
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.unet_lora,
            xt,
            t, 
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    

    def __call__(
        self,
        true_global_step: int,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)


        t = None
        if self.use_dreamtime:
            time_prior, _ = precompute_prior(max_t=int(0.8 * 1000))
            # if self.phase_id == 1:
            #     step_ratio = (true_global_step - 1) / self.cfg.stage_two_start_step
            # else:
            #     step_ratio = (true_global_step - 1) / (self.cfg.max_steps - self.cfg.stage_two_start_step)
            step_ratio = (true_global_step - 1) / self.cfg.max_steps
            t = time_prioritize(step_ratio, time_prior)
            t = torch.full([batch_size], int(t), dtype=torch.long, device=self.device)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            # Keep this!!!!!!!
            if self.phase_id == 1:
                t = torch.randint(
                    self.min_step,
                    700,
                    [batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                t = torch.randint(
                    self.min_step,
                    500,
                    [batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
        latents_pred_denoised_tgt, latents_pred_denoised_src, text_embeddings = self.predicted_image_denoised(
            latents, t, prompt_utils, elevation, azimuth, camera_distances
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        if self.cfg.use_sch_bridge:
            grad, grad_img, step, xt, bridge_noise, noise_pred_est, w = self.compute_grad_sds_sch_bridge(
                true_global_step, latents, latents_pred_denoised_src, latents_pred_denoised_tgt, text_embeddings, elevation, prompt_utils, camera_condition
            )
        

        grad = torch.nan_to_num(grad)
        
        # # Print gradient statistics for debugging
        # with torch.no_grad():
        #     grad_mean = grad.mean().item()
        #     grad_std = grad.std().item()
        #     grad_min = grad.min().item()
        #     grad_max = grad.max().item()
        #     grad_norm = grad.norm().item()
            
        #     print(f"Gradient stats: mean={grad_mean:.6f}, std={grad_std:.6f}, min={grad_min:.6f}, max={grad_max:.6f}, norm={grad_norm:.6f}")
            
        #     # Print shape information
        #     print(f"Gradient shape: {grad.shape}")
            
        #     # Check for NaN or Inf values (should be handled by torch.nan_to_num above, but double-checking)
        #     nan_count = torch.isnan(grad).sum().item()
        #     inf_count = torch.isinf(grad).sum().item()
        #     if nan_count > 0 or inf_count > 0:
        #         print(f"Warning: Gradient contains {nan_count} NaN values and {inf_count} Inf values")

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach() # should that be xt - grad? no
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size # should that be mse between xt and target? np
        loss_lora = self.train_lora(step, xt, bridge_noise, text_embeddings, camera_condition) # should that be xt, text_embeddings, camera_condition? yes

        guidance_out = {
            "loss_sds": loss_sds,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
            "w": w,
            # "noise_pred_est": noise_pred_est,
            # "bridge_noise": bridge_noise,
        }

        # if guidance_eval:
        #     guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
        #     texts = []
        #     for n, e, a, c in zip(
        #         guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
        #     ):
        #         texts.append(
        #             f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
        #         )
        #     guidance_eval_out.update({"texts": texts})
        #     guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    def __del__(self):
        # Ensure all CUDA tensors are properly deleted before the object is garbage collected
        if hasattr(self, 'SchBridgeDiffusion'):
            # Clear all properties that might hold CUDA tensors
            if hasattr(self.SchBridgeDiffusion, 'betas'):
                self.SchBridgeDiffusion.betas = None
            if hasattr(self.SchBridgeDiffusion, 'std_fwd'):
                self.SchBridgeDiffusion.std_fwd = None
            if hasattr(self.SchBridgeDiffusion, 'std_bwd'):
                self.SchBridgeDiffusion.std_bwd = None
            if hasattr(self.SchBridgeDiffusion, 'std_sb'):
                self.SchBridgeDiffusion.std_sb = None
            if hasattr(self.SchBridgeDiffusion, 'mu_x0'):
                self.SchBridgeDiffusion.mu_x0 = None
            if hasattr(self.SchBridgeDiffusion, 'mu_x1'):
                self.SchBridgeDiffusion.mu_x1 = None
        
        # Force garbage collection and CUDA memory cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    
    # def target_score(self, t, predicted_x0, text_embeddings, elevation, prompt_utils):
    #     noise_pred, latents_noisy = self.get_noise_pred(predicted_x0, t, text_embeddings, use_perp_neg=prompt_utils.use_perp_neg)
    #     alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
    #     sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
    #     target_score = - noise_pred / sigma
    #     return target_score
    
    # def target_noise(self, t, predicted_x0, text_embeddings, elevation, prompt_utils):
    #     noise_pred, latents_noisy = self.get_noise_pred(predicted_x0, t, text_embeddings, use_perp_neg=prompt_utils.use_perp_neg)
    #     alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
    #     sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
    #     target_noise = - noise_pred
    #     return target_noise


    @torch.cuda.amp.autocast(enabled=False)
    def predicted_image_denoised(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        use_guidance_scale: bool = True,
    ):
        """
        Predict the denoised latents (x0) from noisy latents using the UNet model.
        
        Args:
            latents_noisy: Noisy latents (xt)
            t: Timesteps
            prompt_utils: Prompt processor output
            elevation, azimuth, camera_distances: Camera parameters
            use_guidance_scale: Whether to apply classifier-free guidance scale
            
        Returns:
            Predicted denoised latents (x0)
        """
        batch_size = latents.shape[0]
        device = latents.device
        
        # Get text embeddings based on whether we use perpendicular negative or not
        # Standard text embeddings
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        
        # Predict noise
        noise_pred_text, noise_pred_uncond, latents_noisy = self.get_noise_pred_bridge(latents, t, text_embeddings, use_perp_neg=prompt_utils.use_perp_neg)
        
        # Calculate predicted x0 from the noisy latent and the predicted noise
        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)

        if self.phase_id == 1:
            noise_pred_tgt = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond) # noise_pred_uncond
            noise_pred_src = noise_pred_uncond # self.cfg.guidance_scale

            # Get the previous latents for each noise prediction
            # Make sure scheduler tensors are on the same device as inputs
            device = t.device
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            
            step_output_tgt = self.scheduler.step(noise_pred_tgt, t, latents_noisy)
            latents_pred_tgt = step_output_tgt.pred_original_sample

            latents_pred_src = None # self.scheduler.step(noise_pred_src, t, latents_noisy)
            # latents_pred_src = step_output_src.pred_original_sample

            # latents_denoised_tgt = (latents_noisy - sigma * noise_pred_tgt) / alpha
            # latents_denoised_src = (latents_noisy - sigma * noise_pred_src) / alpha
        elif self.phase_id == 2:
            noise_pred_tgt = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred_src = noise_pred_uncond # self.cfg.guidance_scale

            # Get the previous latents for each noise prediction
            # Make sure scheduler tensors are on the same device as inputs
            device = t.device
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            
            step_output_tgt = self.scheduler.step(noise_pred_tgt, t, latents_noisy)
            latents_pred_tgt = step_output_tgt.pred_original_sample

            latents_pred_src = None # self.scheduler.step(noise_pred_src, t, latents_noisy)
            # latents_pred_src = step_output_src.pred_original_sample

        # ================================================

        # if self.phase_id == 1:
        #     noise_pred_tgt = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond) # noise_pred_uncond
        #     noise_pred_src = noise_pred_uncond # self.cfg.guidance_scale

        #     # Get the previous latents for each noise prediction
        #     # Make sure scheduler tensors are on the same device as inputs
        #     device = t.device
        #     self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        #     self.scheduler.timesteps = self.scheduler.timesteps.to(device) # Ensure timesteps are on the correct device

        #     step_output_tgt = self.scheduler.step(noise_pred_tgt, t, latents_noisy)
        #     # latents_pred_tgt = step_output_tgt.pred_original_sample # Original single-step prediction

        #     # --- Full Denoising Loop ---
        #     current_latents_tgt = step_output_tgt.prev_sample
        #     timesteps = self.scheduler.timesteps
        #     current_t_index_tensor = (timesteps == t.squeeze()).nonzero() # Find index of current timestep t

        #     if current_t_index_tensor.numel() > 0:
        #         current_t_index = current_t_index_tensor.item()

        #         # Loop through the remaining timesteps (from t-1 down to 0)
        #         for i in range(current_t_index + 1, min(current_t_index + 100, len(timesteps))):
        #             loop_t = timesteps[i].repeat(batch_size) # Ensure loop_t has batch dimension

        #             # Scale model input
        #             latent_model_input = self.scheduler.scale_model_input(current_latents_tgt, loop_t)

        #             # Predict noise using UNet
        #             # We need text and uncond inputs for forward_unet structure, even if only using text part for scheduler
        #             latent_model_input_unet = torch.cat([latent_model_input] * 2, dim=0)
        #             loop_t_unet = torch.cat([loop_t] * 2) # Duplicate timestep for UNet
        #             with self.disable_unet_class_embedding(self.unet) as unet:
        #                 cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        #                 noise_pred_unet = self.forward_unet(
        #                     unet,
        #                     latent_model_input_unet,
        #                     loop_t_unet,
        #                     encoder_hidden_states=text_embeddings, # Reuse embeddings calculated earlier
        #                     cross_attention_kwargs=cross_attention_kwargs,
        #                 )

        #             # Perform guidance (same as outside the loop)
        #             noise_pred_text_loop, noise_pred_uncond_loop = noise_pred_unet.chunk(2)
        #             noise_pred_loop = noise_pred_uncond_loop + self.cfg.guidance_scale * (noise_pred_text_loop - noise_pred_uncond_loop)

        #             # Scheduler step
        #             step_output_loop = self.scheduler.step(noise_pred_loop, loop_t, current_latents_tgt)

        #             # Update latents for the next iteration
        #             current_latents_tgt = step_output_loop.prev_sample

        #         # Final denoised latents
        #         latents_denoised_full_tgt = current_latents_tgt
        #     else:
        #         # Handle case where t is not found in timesteps (should not happen ideally)
        #         threestudio.warn(f"Timestep {t.item()} not found in scheduler timesteps, returning single-step prediction.")
        #         latents_denoised_full_tgt = step_output_tgt.pred_original_sample # Fallback to original single-step prediction

        #     latents_pred_tgt = latents_denoised_full_tgt # Assign fully denoised result
        #     # --- End Full Denoising Loop ---


        #     latents_pred_src = torch.zeros_like(latents_pred_tgt)# self.scheduler.step(noise_pred_src, t, latents_noisy)

        # ================================================

        return latents_pred_tgt, latents_pred_src, text_embeddings
    
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred_bridge(
        self,
        latents,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents.shape[0]

        # add noise
        noise = torch.randn_like(latents)  # TODO: use torch generator
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        with self.disable_unet_class_embedding(self.unet) as unet:
            cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
            noise_pred = self.forward_unet(
                unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)


        return noise_pred_text, noise_pred_uncond, latents_noisy # noise_pred, latents_noisy
    
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred( # not used in this version
        self,
        latents,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents.shape[0]

        # add noise
        noise = torch.randn_like(latents)  # TODO: use torch generator
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        noise_pred = self.forward_unet(
            latent_model_input,
            torch.cat([t] * 2),
            encoder_hidden_states=text_embeddings,
        )
        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)

        # Apply classifier-free guidance
        if self.phase_id == 1:
            noise_pred = noise_pred_uncond
        elif self.phase_id == 2:
            noise_pred = noise_pred_text

        return noise_pred, latents_noisy

    # @torch.cuda.amp.autocast(enabled=False)
    # @torch.no_grad()
    # def guidance_eval(
    #     self,
    #     t_orig,
    #     text_embeddings,
    #     latents_noisy,
    #     noise_pred,
    #     use_perp_neg=False,
    #     neg_guidance_weights=None,
    # ):
    #     # use only 50 timesteps, and find nearest of those to t
    #     self.scheduler.set_timesteps(50)
    #     self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
    #     bs = (
    #         min(self.cfg.max_items_eval, latents_noisy.shape[0])
    #         if self.cfg.max_items_eval > 0
    #         else latents_noisy.shape[0]
    #     )  # batch size
    #     large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
    #         :bs
    #     ].unsqueeze(
    #         -1
    #     )  # sized [bs,50] > [bs,1]
    #     idxs = torch.min(large_enough_idxs, dim=1)[1]
    #     t = self.scheduler.timesteps_gpu[idxs]

    #     fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
    #     imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

    #     # get prev latent
    #     latents_1step = []
    #     pred_1orig = []
    #     for b in range(bs):
    #         step_output = self.scheduler.step(
    #             noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
    #         )
    #         latents_1step.append(step_output["prev_sample"])
    #         pred_1orig.append(step_output["pred_original_sample"])
    #     latents_1step = torch.cat(latents_1step)
    #     pred_1orig = torch.cat(pred_1orig)
    #     imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
    #     imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

    #     latents_final = []
    #     for b, i in enumerate(idxs):
    #         latents = latents_1step[b : b + 1]
    #         text_emb = (
    #             text_embeddings[
    #                 [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
    #             ]
    #             if use_perp_neg
    #             else text_embeddings[[b, b + len(idxs)], ...]
    #         )
    #         neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
    #         for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
    #             # pred noise
    #             noise_pred = self.get_noise_pred(
    #                 latents, t, text_emb, use_perp_neg, neg_guid
    #             )
    #             # get prev latent
    #             latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
    #                 "prev_sample"
    #             ]
    #         latents_final.append(latents)

    #     latents_final = torch.cat(latents_final)
    #     imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

    #     return {
    #         "bs": bs,
    #         "noise_levels": fracs,
    #         "imgs_noisy": imgs_noisy,
    #         "imgs_1step": imgs_1step,
    #         "imgs_1orig": imgs_1orig,
    #         "imgs_final": imgs_final,
    #     }

    # def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
    #     # clip grad for stable training as demonstrated in
    #     # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
    #     # http://arxiv.org/abs/2303.15413
    #     if self.cfg.grad_clip is not None:
    #         self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

    #     self.set_min_max_steps(
    #         min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
    #         max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
    #     )

    def cleanup_resources(self):
        """Explicitly release resources to prevent memory leaks and segfaults"""
        if hasattr(self, 'SchBridgeDiffusion'):
            try:
                if hasattr(self.SchBridgeDiffusion, 'cleanup'):
                    self.SchBridgeDiffusion.cleanup()
                self.SchBridgeDiffusion = None
            except Exception as e:
                threestudio.warn(f"Error cleaning up SchBridgeDiffusion: {e}")
        
        # Clean up CUDA tensors
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'vae'):
            del self.vae
        if hasattr(self, 'unet'):
            del self.unet
        if hasattr(self, 'scheduler'):
            del self.scheduler
            
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()