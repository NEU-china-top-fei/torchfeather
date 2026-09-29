from dataclasses import dataclass, field
from loguru import logger

from torch import nn
from torchfeather.model.moe import MoEArgs


@dataclass  # used for reduce redundant code
class DeepSeekV3ModelArgs:
    max_seq_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1488
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5
    # used for RMSnorm

    # MoE args
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    # field to assign complex initial value

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # compute the flops and parameters
    def get_numparams_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        num_params_embedding = 0
        num_params_router = 0
        num_params_shared_experts = 0
        num_params_experts = 0
        num_params_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                num_params_embedding += p.numel()
                num_params_dense += p.numel()
            elif "moe.shared_experts" in name:
                num_params_shared_experts += p.numel()
            elif "moe.router" in name:
                num_params_router += p.numel()
            elif "moe.expert" in name:
                num_params_experts += p.numel()
            else:
                num_params_dense += p.numel()

        num_params_sparse = (
            num_params_router + num_params_shared_experts + num_params_experts
        )
        num_params = num_params_dense + num_params_sparse
        num_sparse_active = (
            num_params_router
            + num_params_shared_experts
            + num_params_experts * self.moe_args.top_k // self.moe_args.num_experts
        )

        logger.info(
            f"Total parameter count: dense {num_params_dense:,}, "
            f"sparse {num_params_sparse :,},activa{num_params_dense+num_sparse_active:,}"
        )

        n_layers = self.n_layers
        n_heads = self.n_heads
        head_dims = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim
        # embedding didn't count because it's just look up
        num_flops_per_token = (
            6 * (num_params_dense - num_params_embedding + num_sparse_active)
            + 6 * n_layers * n_heads * head_dims * seq_len
        )  # the latter complement the attention cost without adding additional parameters(such as QK.T)

        return num_params, num_flops_per_token
