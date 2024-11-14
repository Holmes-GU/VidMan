import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


from opensora.registry import MODELS, build_module
import clip


@MODELS.register_module()
class OXE_DiffusionPolicy():
    def __init__(self,
                 net,
                 act_dim,
                 sequence_length,
                 chunk_size,
                 num_train_steps,
                 num_infer_steps,
                 device,
                 action_loss_ratio=100,
                 ):
        self.net = build_module(net, MODELS, act_dim=act_dim, chunk_size=chunk_size,
                                sequence_length=sequence_length,
                                device=device).to(device)
        self.ema = EMAModel(
            parameters=self.net.parameters(),
            power=0.75)
        self.ema_net = copy.deepcopy(self.net)

        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,  # TODO
            prediction_type='sample',
            set_alpha_to_one=True,
            steps_offset=0,
        )
        self.noise_scheduler.set_timesteps(num_infer_steps)
        self.device = device
        self.action_loss_ratio = action_loss_ratio

    def infer(self, rgb, hand_rgb, state, language, attention_mask):
        B, T, _, _, _ = rgb.shape
        encoded_feat = self.ema_net(
            mode='encode',
            rgb=rgb,
            hand_rgb=hand_rgb,
            state=state,
            language=language,
        )
        noisy_actions = torch.randn((B, T, self.chunk_size, self.act_dim), device=self.device)
        timesteps = self.noise_scheduler.timesteps
        noisy_actions = self.noise_scheduler.scale_model_input(noisy_actions, timesteps)
        if encoded_feat['obs_targets'] is not None:
            noisy_obs = torch.randn(encoded_feat['obs_targets'].shape, device=self.device)
            noisy_hand_obs = torch.randn(encoded_feat['obs_hand_targets'].shape, device=self.device)
            noisy_obs = self.noise_scheduler.scale_model_input(noisy_obs, timesteps)
            noisy_hand_obs = self.noise_scheduler.scale_model_input(noisy_hand_obs, timesteps)
        else:
            noisy_hand_obs = torch.randn([B, T, 28, 28, 4], device=self.device) #hard code
            noisy_obs = torch.randn([B, T, 28, 28, 4], device=self.device)
            noisy_obs = self.noise_scheduler.scale_model_input(noisy_obs, timesteps)
            noisy_hand_obs = self.noise_scheduler.scale_model_input(noisy_hand_obs, timesteps)
        for k in self.noise_scheduler.timesteps:
            timesteps = torch.ones((B,), dtype=torch.long, device=self.device) * k
            pred = self.ema_net(
                mode='predict',
                stacked_inputs=encoded_feat['stacked_inputs'],
                noisy_obs=noisy_obs,
                noisy_hand_obs=noisy_hand_obs,
                noisy_actions=noisy_actions,
                diffusion_step=timesteps,
                attention_mask=attention_mask,
            )
            pred_actions = torch.cat((pred['arm_action_preds'], pred['gripper_action_preds']), dim=-1)
            noisy_actions = self.noise_scheduler.step(
                model_output=pred_actions,
                timestep=k,
                sample=noisy_actions,
            ).prev_sample
        return noisy_actions