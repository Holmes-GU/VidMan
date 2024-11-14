import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opensora.registry import MODELS, build_module
import clip
from einops import rearrange



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

@MODELS.register_module()
class OXE_Diffusion_KL_Sora(nn.Module):
    def __init__(
            self,
            text_encoder,
            image_encoder,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,

            n_layer,
            sora,
            without_norm_pixel_loss=False,
            wo_state=True, 
            device='cuda',
            skip_frame=2,
            condition_cat_zero=False, 
    ):
        super().__init__()
        self.act_pred = True
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.wo_state = wo_state
        self.hidden_size = hidden_size
        if text_encoder.type == 'clip':
            self.text_encoder, _ = clip.load(text_encoder['clip_backbone'], device=device)
        else:
            self.text_encoder = build_module(text_encoder, MODELS)

        for _, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        self.image_encoder = build_module(image_encoder, MODELS)
        for _, param in self.image_encoder.named_parameters():
            param.requires_grad = False

        input_size = (sequence_length*2, *sora.rgb_shape) 
        sora.pop("rgb_shape")
        latent_size = self.image_encoder.get_latent_size(input_size)

        _sora_config = dict(
            type="STDiT_Transformer",
            depth=n_layer,
            hidden_size=hidden_size,
            caption_channels=512,
            class_dropout_prob=0.,  # uncond drop #设置为0
            model_max_length=1,
            # not_load_key=["y_embedder", "final_layer", "x_embedder"],
            not_load_key=["y_embedder", "final_layer"],
            # space_scale=0.5,
            # time_scale=1.0,
            from_pretrained="/cache/pretrained_models/Open-Sora/OpenSora-v1-HQ-16x256x256.pth",
            # from_pretrained=None,
            enable_flashattn=False,
            enable_layernorm_kernel=False,  # need apex
            # patchemb3d_in_channel=8,
            patchemb3d_in_channel=4,  # 
            pred_sigma=False,
            pad_emb3d=False  # 
        )
        _sora_config.update(sora)

        self.transformer = build_module(_sora_config, MODELS, input_size=latent_size,
                                        in_channels=self.image_encoder.out_channels)
        self.skip_frame = skip_frame
        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224


        self.without_norm_pixel_loss = without_norm_pixel_loss

        if not self.wo_state:
            self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
            self.embed_gripper_state = torch.nn.Linear(2, hidden_size)  # one-hot gripper state
            self.embed_state = torch.nn.Linear(2 * hidden_size, hidden_size)

        self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)
        self.action_queries = nn.Embedding(chunk_size, hidden_size) 

        self.embed_noisy_actions = nn.Linear(self.act_dim, hidden_size)
        self.embed_noisy_actions.weight.data.fill_(0)

        self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(2, hidden_size)
        self.embed_state = torch.nn.Linear(2 * hidden_size, hidden_size)

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.Mish(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 2)])
        self.pred_arm_act = nn.Linear(hidden_size // 2, self.act_dim - 1)  # arm action
        self.pred_gripper_act = nn.Linear(hidden_size // 2, 1)  # gripper action (binary)
        self.condition_cat_zero = condition_cat_zero

    def _encode(self,
                            rgb=None,
                            hand_rgb=None,
                            state=None,
                            language=None, ):
        obs_targets = None
        obs_hand_targets = None
        batch_size, sequence_length, c, h, w = rgb.shape

        if not self.wo_state:
            arm_state = state['arm']
            gripper_state = state['gripper']
            arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.state_dim - 1))
            gripper_state_embeddings = self.embed_gripper_state(gripper_state)
            state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
            state_embeddings = self.embed_state(state_embeddings)  # (b, t, h)

        with torch.no_grad():
            lang_embeddings = self.text_encoder.encode_text(language)
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6)  # normalization

        with torch.no_grad():
            patch_embeddings = self.image_encoder.encode(rearrange(rgb, "B T C H W -> B C T H W"))

        with torch.no_grad():
            hand_patch_embeddings = self.image_encoder.encode(rearrange(hand_rgb, "B T C H W -> B C T H W"))

        lang_embeddings = lang_embeddings.view(batch_size, 1, -1)

        if not self.wo_state:
            state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = (lang_embeddings, state_embeddings, patch_embeddings, hand_patch_embeddings)
        else:
            stacked_inputs = (lang_embeddings, patch_embeddings, hand_patch_embeddings)
        prediction = {
            'obs_targets': obs_targets,
            'obs_hand_targets': obs_hand_targets,
            'stacked_inputs': stacked_inputs,
        }
        return prediction

    def _predict(self,
                             stacked_inputs=None,
                             noisy_obs=None,
                             noisy_hand_obs=None,
                             noisy_actions=None,
                             diffusion_step=None,
                             attention_mask=None, ):

        arm_action_preds = None
        gripper_action_preds = None
        if not self.wo_state:
            lang_embeddings, state_embeddings, patch_embeddings, hand_patch_embeddings = stacked_inputs
        else:
            lang_embeddings, patch_embeddings, hand_patch_embeddings = stacked_inputs
            state_embeddings = None


        batch_size, C, sequence_length, H, W = patch_embeddings.shape
        diffusion_step_for_video = diffusion_step
        diffusion_step = self.diffusion_step_encoder(diffusion_step)  # (b, h)
        diffusion_step = diffusion_step.view(batch_size, 1, 1, self.hidden_size).repeat(1, sequence_length, 1,
                                                                                        1)  # TODO: (b t 1 h)

        if self.act_pred:
            action_queries = self.action_queries.weight  # (chunk_size, h)
            action_queries = action_queries.view(1, 1, self.chunk_size, self.hidden_size).repeat(batch_size,
                                                                                                 sequence_length, 1,
                                                                                                 1)  # (b, t, chunk_size, h)
            action_queries += self.embed_noisy_actions(noisy_actions)  # (b, t, chunk_size, h)

        if self.condition_cat_zero:
            condition_obs = torch.cat([patch_embeddings[:, :, :self.skip_frame],
                                       torch.zeros_like(patch_embeddings[:, :, self.skip_frame:])], dim=2)
            condition_hand = torch.cat([hand_patch_embeddings[:, :, :self.skip_frame],
                                        torch.zeros_like(hand_patch_embeddings[:, :, self.skip_frame:])], dim=2)
            condition_x = torch.cat([condition_obs, condition_hand], dim=2)

        else:
            condition_x = torch.cat([patch_embeddings, hand_patch_embeddings], dim=2)

        patch_tokens = rearrange(torch.cat([noisy_obs, noisy_hand_obs], dim=1), "B T H W C -> B C T H W") #这里的是B T H W C所以要在

        time_embeddings = self.embed_timestep.weight 

        transformer_outputs = self.transformer(
            x=patch_tokens,
            timestep=diffusion_step_for_video,
            y=lang_embeddings,
            mask=None,
            condition_x=condition_x,
            action_queries=action_queries,
            x_self_mask=attention_mask,
            state=state_embeddings,
            sequence_time_embedding=time_embeddings,
            action_diffusion_t_emb=diffusion_step,
        )

        obs_preds, obs_hand_preds = transformer_outputs['x'][:, :, :sequence_length], transformer_outputs['x'][:, :, sequence_length:]
        action_embedding = transformer_outputs['action_queries']

        if self.act_pred:

            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, t, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, t, 1)
        obs_preds = rearrange(obs_preds, "B C T H W -> B T H W C")
        obs_hand_preds = rearrange(obs_hand_preds, "B C T H W -> B T H W C")
        arm_action_preds = arm_action_preds.view(batch_size, sequence_length, self.chunk_size, self.act_dim-1)
        gripper_action_preds = gripper_action_preds.view(batch_size, sequence_length, self.chunk_size, 1)


        prediction = {
            'obs_preds': obs_preds,
            'obs_hand_preds': obs_hand_preds,
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds,
        }
        return prediction

    def forward(self,
                mode,
                rgb=None,
                hand_rgb=None,
                state=None,
                language=None,
                stacked_inputs=None,
                noisy_obs=None,
                noisy_hand_obs=None,
                noisy_actions=None,
                diffusion_step=None,
                attention_mask=None,
                ):

        if mode == 'encode':  
            return self._encode(rgb=rgb, hand_rgb=hand_rgb, state=state, language=language)
        elif mode == 'predict':
            return self._predict(stacked_inputs=stacked_inputs, noisy_obs=noisy_obs,
                                             noisy_hand_obs=noisy_hand_obs,
                                             noisy_actions=noisy_actions, diffusion_step=diffusion_step,
                                             attention_mask=attention_mask)