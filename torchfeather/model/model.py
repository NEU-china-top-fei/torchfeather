import math

import torch
from torch import nn

from torchfeather.model.attention import (
    ScaledDotProductAttentionWrapper,
)
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.q_lora_rank = (
            args.q_lora_rank
        )  # the process of compress and up project on token sequence is just like the lora tech
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_rope_head_dim + args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            # no down projection on q
            self.wq = nn.Linear(self.dim, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.wqa = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank)
            self.wqb = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
            # up project:in fact need twice to get the no-rope and rope
            # but we can do one mm and split to reduce kernel launch overhead


class TransformerBlock(nn.Module):
    def __int__(self, layer_id: int, args: DeepSeekV3ModelArgs):
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        self.is_moe = self.layer_id >= args.n_dense_layers
        if self.is_moe:
            self.moe = MoE(args.moe_args, dim=args.dim, hidden_dim=args.inter_dim)
        else:
            self.ffn = FFN(args.dim, args.inter_dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

    def foward(self, x: torch.tensor, freq_cis: torch.tensor):
        atten_process = x + self.attention(self.attention_norm(x), freq_cis)
        if self.is_moe:
            out = self.moe(self.ffn_norm(atten_process))
        else:
            out = self.ffn(self.ffn_norm(atten_process))
        return out

    def init_weights(
        self, init_std: float | None = None, buffer_device: torch.device | None = None
    ):
        if buffer_device is None:
            raise ValueError(
                "buffer device must be assigned when transformer block initialization"
            )
        for n in (self.attention_norm, self.ffn_norm):
            n.reset_parameters()
        self.attention.init_weight(self.weight_init_std)
        if self.moe:
            self.moe.init_weight(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.ffn.init_weight(self.weight_init_std)


class DeepSeekV3Model(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.register_buffer("freq_cis", precompute_freqs_cis(args), True)
        self.transformerlayer = nn.ModuleDict()

        for layer_id in range(args.n_layers):
            self.transformerlayer[layer_id] = TransformerBlock(layer_id, args)

        self.norm = nn.RMSNorm(args.dim)
        self.output = nn.Linear(
            args.dim, args.vocab_size, bias=False, dtype=torch.get_default_dtype()
        )

    def init_weights(
        self, std: float | None = None, buffer_device: torch.device | None = None
    ):
        buffer_device = buffer_device or self.freq_cis.device
        with torch.device(
            buffer_device
        ):  # tensor created within the scope are defaultly on the device
            self.freq_cis = precompute_freqs_cis(self.args)
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=std, buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        out_std = self.args.dim ** (-0.5)
        factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=out_std,
                a=-1 * factor * out_std,
                b=factor * out_std,
            )

    def forward(self, x):
        new_x = self.embedding(x) if self.embedding is not None else x

        for layer in self.layer.values():
            new_x = layer(new_x, self.freq_cis)

        norm_x = self.norm(new_x) if self.norm is not None else new_x
        output = self.output(norm_x) if self.output is not None else norm_x

        return output
