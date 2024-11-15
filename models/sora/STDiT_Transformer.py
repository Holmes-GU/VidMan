from opensora.registry import MODELS, build_module
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (    
PatchEmbed3D,
)
from opensora.registry import MODELS
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from opensora.models.stdit import STDiT

class Attention_B(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0, ):
        super().__init__()
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((1024, 1024), dtype=torch.bool)).view(
                1, 1, 1024, 1024
            ),
            persistent=False,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads

        self.attn = Conv1D(3 * self.embed_dim, self.embed_dim)

        self.proj = Conv1D(self.embed_dim, self.embed_dim)


        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    def forward(self, x, mask=None):
        device = x.device
        q, k, v = self.attn(x).split(self.embed_dim, dim=2)
        q = self._split_heads(q, self.num_heads, self.head_dim)
        k = self._split_heads(k, self.num_heads, self.head_dim)
        v = self._split_heads(v, self.num_heads, self.head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / torch.full(
            [], v.size(-1) ** 0.5, dtype=attn.dtype, device=device
        )
        causal_mask = self.bias[:, :, k.shape[-2] - q.shape[-2]: k.shape[-2], :k.shape[-2]]
        mask_value = torch.full([], torch.finfo(attn.dtype).min, dtype=attn.dtype).to(device)
        attn = torch.where(causal_mask, attn.to(attn.dtype), mask_value)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn.type(v.dtype))
        x = torch.matmul(attn, v)
        x = self._merge_heads(x, self.num_heads, self.head_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, mult=4, drop_rate=0.1):
        super().__init__()
        self.fc = Conv1D(embed_dim * mult, embed_dim)
        self.proj = Conv1D(embed_dim, embed_dim * mult)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

@MODELS.register_module()
class GatedSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=12, ff_mult=4, gate_hidden=True):
        super().__init__()
        self.attn_gate = nn.Parameter(torch.tensor([6.0]))
        self.ff_gate = nn.Parameter(torch.tensor([6.]))

        self.action_layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn_action = Attention_B(
            hidden_size,
            num_heads=num_heads,
            attn_drop=0.1,
            proj_drop=0.1,
        )
        self.action_layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.action_ff = MLP(embed_dim=hidden_size, mult=ff_mult, drop_rate=0.1)  # 默认是hidden_size*4

        self.hidden_states_gate = nn.Parameter(torch.tensor([0.0]))
        self.gate_hidden = gate_hidden

    def forward(self, x=None, y=None, mask=None, t=None, state=None, language=None, hidden_states=None, 
                x_from_sora=None, input_x=None, x_from_sora_states=None, input_mask=None):

        sequence_length = 4
        chunk_size = 10
        if hidden_states is None:

            input_x = self.attn_action(self.action_layer_norm1(input_x), input_mask) * self.attn_gate.tanh() + input_x
            input_x = self.action_ff(self.action_layer_norm2(input_x)) * self.ff_gate.tanh() + input_x
        else:

            input_x = hidden_states + x_from_sora_states * self.hidden_states_gate.tanh()
            input_x = self.attn_action(self.action_layer_norm1(input_x), input_mask) * self.attn_gate.tanh() + input_x
            input_x = self.action_ff(self.action_layer_norm2(input_x)) * self.ff_gate.tanh() + input_x

        return input_x[:, - sequence_length * chunk_size:], input_x

@MODELS.register_module()
class STDiT_Transformer(STDiT):
    def __init__(
            self,

            read_out_num_heads=1,
            hidden_size=1152,
            depth=28,
            freeze_not_attn=False,
            perseiver_resampler=None,
            action_attn_type=None,  
            action_attn_pos="after",
            pad_mask_in_cross_attn=False,
            sequence_length=4,  
            **kwargs,
    ):
        super().__init__(hidden_size=hidden_size, depth=depth, **kwargs)
        _action_attn_type = dict(type="GatedSlefAttentionBlock", hidden_size=hidden_size, num_heads=16, ff_mult=1)
        _action_attn_type.update(action_attn_type)
        self.action_attn_pos = action_attn_pos
        self.action_attn_type = _action_attn_type

        self.read_out_attn_blocks = nn.ModuleList(
            [
                build_module(
                    _action_attn_type, MODELS
                )
                for i in range(depth)
            ]
        )
        self.initialize_weights()
        self.read_out_attn_mask = None
        if self.from_pretrained is not None:
            self.load_pretrained(self.from_pretrained, self.not_load_key, self.pad_emb3d)
        if freeze_not_attn:
            self.freeze_not_attn()

        self.y_embedder = torch.nn.Linear(self.caption_channels, hidden_size)
        self.pad_mask_in_cross_attn = pad_mask_in_cross_attn
        self.sequence_length = sequence_length
        self.cond_x_proj = PatchEmbed3D(self.patch_size, int(self.patchemb3d_in_channel / 2),
                                        self.hidden_size)  
        self.action_block_position_emb = nn.Embedding(1024, self.hidden_size)
        self.action_pre_ln_norm = nn.LayerNorm(hidden_size)
        self.action_pre_drop = nn.Dropout(0.1)
        self.action_ln_norm = nn.LayerNorm(hidden_size)
        self.cond_x_gate = nn.Parameter(torch.tensor([0.0]))
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def freeze_not_attn(self):
        for n, p in self.named_parameters():
            if "read_out_self_attn_blocks" not in n:
                p.requires_grad = False

    def load_pretrained(self, from_pretrained, not_load_key, pad_emb3d=False):
        from opensora.utils.ckpt_utils import find_model
        state_dict = find_model(from_pretrained)
        if not_load_key is not None:
            for key in not_load_key:
                for ll in list(state_dict.keys()):
                    if key in ll:
                        state_dict.pop(ll)

        if pad_emb3d:
            state_dict['x_embedder.proj.weight'] = torch.cat(
                [state_dict["x_embedder.proj.weight"], torch.zeros_like(state_dict["x_embedder.proj.weight"])], dim=1)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if dist.get_rank() == 0:
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")


    def process_input_for_adaptor(self, x, y, mask=None, t=None, state=None, language=None, hidden_states=None,
                                  x_from_sora=None):
        if hidden_states is None:

            B = x.shape[0]
            hidden_size = x.shape[-1]
            sequence_length = 4
            chunk_size = 10
            x = x.view(B, sequence_length, -1, hidden_size)
            # t = t.view(B, 1, 1, -1).repeat(1, sequence_length, 1, 1)
            zero_masks = torch.zeros(
                (B, sequence_length, chunk_size + t.shape[2]), dtype=torch.long,
                device=x.device)

            y = y.view(B, sequence_length, -1, hidden_size)
            mask = mask.view(B, sequence_length, 1)
            mask = mask.repeat(1, 1, 1 + 1 + y.shape[2])

            input_x = torch.cat([language, state, y, t, x], dim=2).reshape(B, -1, hidden_size)
            input_mask = torch.cat([mask, zero_masks], dim=2).reshape(B, -1)
            input_mask = input_mask[:, None, None, :]
            pos = torch.arange(0, input_x.shape[1], dtype=torch.long, device=x.device)
            pos = pos.unsqueeze(0)
            input_x = self.action_pre_ln_norm(input_x)
            pos_emb = self.action_block_position_emb(pos)
            input_x = input_x + pos_emb
            input_x = self.action_pre_drop(input_x)
            x_from_sora_states = None

        else:

            B = x.shape[0]
            hidden_size = x.shape[-1]
            sequence_length = 4
            chunk_size = 10
            t_tokens = 1

            x = x.view(B, sequence_length, -1, hidden_size)

            zero_masks = torch.zeros(
                (B, sequence_length, chunk_size + t_tokens), dtype=torch.long,
                device=x.device)

            x_from_sora = x_from_sora.view(B, sequence_length * 2, -1, hidden_size)  
            x_from_sora = torch.cat([x_from_sora[:, :sequence_length], x_from_sora[:, sequence_length:]], dim=2)
            mask = mask.view(B, sequence_length, 1)
            mask = mask.repeat(1, 1, 1 + 1 + int(y.shape[1]/sequence_length) )
            input_mask = torch.cat([mask, zero_masks], dim=2).reshape(B, -1)
            input_mask = input_mask[:, None, None, :]
            x_from_sora_states = torch.cat([language, state, x_from_sora, t, x], dim=2).reshape(B, -1, hidden_size)
            input_x = None
        return dict(input_x=input_x, x_from_sora_states=x_from_sora_states,
                    hidden_states=hidden_states, input_mask=input_mask)

    def forward(self, x, timestep, y, mask=None, condition_x=None, action_queries=None, x_self_mask=None,
                sequence_time_embedding=None, cat_t_for_action=False, state=None, action_diffusion_t_emb=None):

        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # embedding
        if condition_x is not None:

            x = torch.cat([x, condition_x], dim=1)
            condx_shape2 = condition_x.shape[2]
            patch, hand = condition_x[:, :, :int(condx_shape2 / 2), :], condition_x[:, :, int(condx_shape2 / 2):, :]
            patch = self.cond_x_proj(patch)
            hand = self.cond_x_proj(hand)
            patch = rearrange(patch, "B (T S) C -> B T S C", T=int(self.num_temporal/2), S=self.num_spatial)
            hand = rearrange(hand, "B (T S) C -> B T S C", T=int(self.num_temporal/2), S=self.num_spatial)
            patch = patch + sequence_time_embedding.unsqueeze(1)
            hand = hand + sequence_time_embedding.unsqueeze(1)
            cond_x = torch.cat([patch, hand], dim=2)

        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial)

        if sequence_time_embedding is not None:
            x = x + sequence_time_embedding.view(-1, 1, self.hidden_size).repeat(2, 1, 1)

        B, T, S, C = x.shape
        x = x + self.pos_embed

        x = rearrange(x, "B T S C -> B (T S) C")
        cond_x = rearrange(cond_x, "B T S C -> B (T S) C")

        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="down")

        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t0 = self.t_block(t)  # [B, C]
        y = self.y_embedder(y).view(B, 1, -1, C)  # 

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        if sequence_time_embedding is not None:
            state = state + sequence_time_embedding[:,None,:]
            language_for_act = y.view(B,-1, 1, x.shape[-1]).repeat(1,4,1,1) + sequence_time_embedding[:,None,:]

        input_x_for_action = cond_x + x*self.cond_x_gate.tanh()
        hidden_states = None
        for i, (block, read_out_attn) in enumerate(zip(self.blocks, self.read_out_attn_blocks)):

            action_queries = action_queries.view(B, -1,
                                                 self.hidden_size)

            if self.action_attn_pos == 'before_skip':
                action_queries, hidden_states = (action_queries + read_out_attn(
                    **self.process_input_for_adaptor(action_queries,
                                                    input_x_for_action, x_self_mask,
                                                    t=action_diffusion_t_emb, state=state, language=language_for_act,
                                                    hidden_states=hidden_states,
                                                    x_from_sora=x)
                )) / 2.0
            elif self.action_attn_pos == 'before':
                action_queries, hidden_states = read_out_attn(
                    **self.process_input_for_adaptor(action_queries, input_x_for_action, x_self_mask,
                                                    t=action_diffusion_t_emb, state=state, language=language_for_act,
                                                    hidden_states=hidden_states, x_from_sora=x))

            if i == 0:
                if self.enable_sequence_parallelism:
                    tpe = torch.chunk(
                        self.pos_embed_temporal, dist.get_world_size(get_sequence_parallel_group()), dim=1
                    )[self.sp_rank].contiguous()
                else:
                    tpe = self.pos_embed_temporal
            else:
                tpe = None
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, tpe, x_self_mask=x_self_mask.repeat(1, 2))
            if self.action_attn_pos == 'after_skip':
                action_queries, hidden_states = (action_queries + read_out_attn(
                    **self.process_input_for_adaptor(action_queries,
                                                    input_x_for_action, x_self_mask,
                                                    t=action_diffusion_t_emb, state=None, language=y,
                                                    hidden_states=hidden_states,
                                                    x_from_sora=x))) / 2.0
            elif self.action_attn_pos == 'after':
                action_queries, hidden_states = read_out_attn(
                    **self.process_input_for_adaptor(action_queries, input_x_for_action, x_self_mask,
                                                    t=action_diffusion_t_emb, state=None, language=y,
                                                    hidden_states=hidden_states, x_from_sora=x))

        if self.enable_sequence_parallelism:
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="up")

        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)

        if action_queries is not None:
            hidden_states = self.action_ln_norm(hidden_states)
            hidden_states = hidden_states.reshape(B, self.sequence_length, -1, self.hidden_size)
            action_queries = hidden_states[:, :, -10:]

        return dict(x=x, action_queries=action_queries)