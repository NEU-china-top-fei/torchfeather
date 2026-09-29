import math

import torch
from torch import nn

from torchfeather.model.attention import (
    ScaledDotProductAttentionWrapper,
)
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class DeepSeekV3Model(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.register_buffer("freq_cis", precompute_freqs_cis(args), True)
        self.transformerlayer = nn.ModuleDict()

        for layer_id in range(args.n_layers):
            self.transformerlayer[layer_id] = Transformer(layer_id, args)

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
