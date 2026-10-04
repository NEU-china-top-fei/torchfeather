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
        self.args = args

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
        self.wkv_a = nn.Linear(
            self.dim,
            (self.kv_lora_rank + self.qk_rope_head_dim),
            bias=False,
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=args.norm_eps)

        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(self.num_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** (-0.5)
        self.inner_attention = ScaledDotProductAttentionWrapper()
        if args.max_seq_len > args.original_seq_len:
            scale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * scale * scale

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        freq = precompute_freqs_cis(args=self.args)

        # process q
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wqb(self.q_norm(self.wqa(x)))

        headed_q = q.view(batch_size, seq_len, self.num_heads, -1)
        q_nope, q_rope = torch.split(
            headed_q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_rope = apply_rotary_emb(q_rope, freq)
        final_q = torch.concat([q_nope, q_rope], dim=-1)

        # process k

        # k->[batchsize,seqlen,kv_lora_rank+qk_rope_dim]
        rawkv = self.wkv_a(x)
        kv, k_rope = torch.split(
            rawkv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freq)  # [b,s,d]->[b,s,1,d]
        k, final_v = torch.split(
            self.wqb(self.kv_norm(kv)).view(batch_size, seq_len, self.num_heads, -1),
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # we need to expand  because it is shared for every K heads. This basically adds a new dimension with 0 stride.
        final_k = torch.concat([k, k_rope.expand(-1, -1, self.num_heads, -1)], dim=-1)
        # [b,s,h,d]->[b,h,s,d]
        output = self.inner_attention(
            final_q.transpose(1, 2),
            final_k.transpose(1, 2),
            final_v.transpose(1, 2),
            scale=self.softmax_scale,
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()

    @torch.no_grad()
    def absorb_mla_weights(self) -> None:
        device = self.wq.weight.device
        dtype = self.wq.weight.dtype
        if self.q_lora_rank == 0:
            raise NotImplementedError()
        wq_nope, wq_rope = torch.split(
            self.wq.weight.view(self.num_heads, -1, self.dim),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=1,
        )

        w_uk, w_uv = torch.split(
            self.wkv_b.weight.view(self.num_heads, -1, self.kv_lora_rank),
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=1,
        )

        wq_abs = torch.concat(
            [torch.bmm(wq_nope.transpose(1, 2), w_uk), wq_rope], dim=1
        ).reshape(
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim), self.dim
        )

        self.wq_abs = nn.Linear(
            self.num_heads * (self.kv_lora_rank + self.self.qk_rope_head_dim),
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wq_abs.weight.copy_(wq_abs)
        self.wq_abs.requires_grad_(False)

        wo_head_abs = torch.bmm(
            self.wo.view(self.dim, self.num_heads, -1).permute(1, 0, 2), w_uv
        )
        wo_abs = wo_head_abs.permute(1, 0, 2).reshape(
            self.dim, self.num_heads, self.kv_lora_rank
        )
        self.wo_abs = nn.Linear(
            self.num_heads * self.kv_lora_rank,
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wo_abs.weight.copy_(wo_abs)
        self.wo_abs.requires_grad_(False)

    def foward_absorbed(self, x: torch.tensor, freq_cis: torch.tensor) -> torch.tensor:
        assert self.wq_abs is not None
        assert self.wo_abs is not None
        batch_size, seq_len, _ = x.shape

        q_nope, q_rope = torch.split(
            self.wq_abs(x).view(batch_size, seq_len, self.num_heads, -1),
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_emb(q_rope, freq_cis)
        q = torch.concat([q_nope, q_rope], dim=-1).transpose(1, 2)

        latent, k_rope = torch.split(
            self.wkv_a(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        cache = torch.concat(
            [self.kv_norm(latent).unsqueeze(2), apply_rotary_emb(k_rope)], dim=-1
        ).transpose(1, 2)

        pesudo_k = cache
        pesudo_v = cache[..., : self.kv_lora_rank]

        output = (
            self.inner_attention(q, pesudo_k, pesudo_v, scale=self.softmax_scale)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.kv_lora_rank)
        )
        return self.wo_abs(output)


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
