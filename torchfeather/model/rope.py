import math
import torch

from torchfeather.model.model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.tensor:
    """
    take in the config and compute the rope position embedding matrix

    Returns:
        Of shape [seqlen,d/2],contains each vector of position
        (in polar coordinate)
    """
    theta = 1 / (
        args.rope_theta
        ** (
            torch.arange(0, args.qk_rope_head_dim, 2, dtype=torch.float32)
            / args.qk_rope_head_dim
        )
    )
    position = torch.arange(0, args.max_seq_len, dtype=torch.float32)

    intermediate = torch.outer(position, theta)

    return torch.polar(torch.ones_like(intermediate), intermediate)


def apply_rope_emb(x: torch.tensor, cis: torch.tensor) -> torch.tensor:
    """
    apply RoPE rotation to the input

    Args:
        x:[B,S,H,D]
        cis:[S,D/2]
    """

    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))

    # transform between real and image both change dim
    cis_ready = cis.view(1, cis.shape[0], 1, cis.shape[1])

    return torch.view_as_real(x_complex * cis_ready).flatten(3).to(x.dtype)
