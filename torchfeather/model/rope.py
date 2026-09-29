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

    def rotation_to_dim(dim: int, l: int, r: int, base: float) -> float:
        """
        given the num of rotation,find the corresponding dim in input
        ps:obvious the cos(wx) has freq w,cos(mtheta) has freq theta
        Args:
            dim: dimention of input
            l:origin max_seq_len

        """
        return dim * math.log(l / (math.pi * 2 * r)) / (2 * math.log(base))

    def find_dim_range(
        r_high: float, r_low: float, dim: int, l: int, base: float
    ) -> tuple[int, int]:
        """
        given the threshold,calculate the dim divide the region whether we should operate
        """
        high = math.ceil(
            rotation_to_dim(dim, l, r_high, base)
        )  # reaching r_high is considered as fast enough
        low = math.floor(
            rotation_to_dim(dim, l, r_low, base)
        )  # too low so we need to change

        return max(low, 0), min(high, dim - 1)

    def r_value(alpha: float, beta: float, dim: int) -> torch.tensor:
        """
        Return:
            the value tensor of r in formular
        """
        return torch.clamp(
            (torch.arange(dim, dtype=torch.float32) - alpha) / (beta - alpha), 0, 1
        )

    theta = 1 / (
        args.rope_theta
        ** (
            torch.arange(0, args.qk_rope_head_dim, 2, dtype=torch.float32)
            / args.qk_rope_head_dim
        )
    )

    # apply yarn
    if args.max_seq_len > args.original_seq_len:
        alpha, beta = find_dim_range(
            args.beta_fast,
            args.beta_slow,
            args.dim,
            args.original_seq_len,
            args.rope_theta,
        )

        r = r_value(alpha, beta, args.dim // 2)
        theta = (1 - r) * theta / args.rope_factor + r * theta

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
