import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    MultiHeadCrossAttention,

    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)

class STDiTBlock_ForAction(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            d_s=None,
            d_t=None,
            mlp_ratio=4.0,
            drop_path=0.0,
            enable_flashattn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False,
            stdit_blocks_for_action=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self._enable_sequence_parallelism = enable_sequence_parallelism
        self.stdit_blocks_for_action = stdit_blocks_for_action #space_attn, temp_attn

        if enable_sequence_parallelism:
            self.attn_cls = SeqParallelAttention
            self.mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            self.attn_cls = Attention
            self.mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        if self._enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            # make sure d_t is divisible by sp_size
            assert d_t % sp_size == 0
            self.d_t = d_t // sp_size

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
        )
        if stdit_blocks_for_action.action_attn:
            self.attn_action = self.attn_cls(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                enable_flashattn=self.enable_flashattn,
            )
            self.action_layer_norm = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)

    def forward(self, x, y, t, mask=None, tpe=None, x_self_mask=None, action_queries=None):
        #x_mask used to mask padding images
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1) #这里用t
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)

        x_s = self.attn(x_s)

        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        if x_self_mask is not None:
            B, sequence = x_self_mask.shape #(b, sequence)
            x_self_mask = x_self_mask.repeat(self.d_s,1).view(-1, 1, 1, sequence) #[3332,1,1,8]
        # if self.stdit_blocks_for_action.temp_attn:
        #     action_queries = action_queries.view(B*self.d_t, -1, self.hidden_size)#[17,4,chunk,1152]->[17*4,chunk,1152]
        #     x_s = torch.cat(x_s, action_queries)
        x_t = self.attn_temp(x_t, mask=x_self_mask)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        if self.stdit_blocks_for_action.action_attn:
            input = torch.cat([x, y, t, action_queries])
            input_mask = mask.view(B, -1)
            input_mask = input_mask[:, None, None, :]
            self.action_layer_norm(input, mask=input_mask)
            self.attn_action()

        return x