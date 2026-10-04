import contextlib
import os
import signal
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import timedelta
from typing import Any, cast

import torch
from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

from torchfeather.components.checkpoint import CheckpointManager
from torchfeather.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchfeather.components.loss import (
    IGNORE_INDEX,
    LossFunction,
    build_cross_entropy_loss,
)
from torchfeather.components.lr_scheduler import (
    LRSchedulersContainer,
    build_lr_schedulers,
)
from torchfeather.components.metrics import (
    MetricsProcessor,
    collect_parameter_norm_metrics,
)
from torchfeather.components.optimizer import (
    OptimizersContainer,
    build_optimizers_with_moe_load_balancing,
)
from torchfeather.components.tokenizer import (
    DeepSeekV3Tokenizer,
)
from torchfeather.config import TORCH_DTYPE_MAP, JobConfig
from torchfeather.config.default_configs import (
    get_config,
)
from torchfeather.config.job_config import Parallelism
from torchfeather.datasets.hf_datasets import build_hf_dataloader
from torchfeather.distributed import ParallelDims
from torchfeather.distributed import utils as dist_utils
from torchfeather.distributed.pipeline_parallel import pipeline_llm
from torchfeather.model.model import DeepSeekV3Model
from torchfeather.model.parallelize import parallelize_deepseekv3
from torchfeather.tools import device_utils, utils
from torchfeather.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(Stateful):
    job_config: JobConfig
    parallel_dims: ParallelDims

    tokenizer: DeepSeekV3Tokenizer
    dataloader: BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Callable[..., contextlib.AbstractContextManager]
    maybe_enable_amp: contextlib.AbstractContextManager
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    step: int
    ntokens_seen: int

    @record
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        device_module, device_type = (
            device_utils.device_module,
            device_utils.device_type,
        )
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=job_config.comm.init_timeout_seconds),
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = self._create_parallel_dims(parallelism_config, world_size)

        _ = self.parallel_dims.world_mesh
        # setup dp config
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        # take control of gc,every n steps do once
        self.gc_handler = utils.GarbageCollenction(gc_freq=job_config.training.gc_freq)
        # when use same seed , output same
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.tokenizer = DeepSeekV3Tokenizer(job_config.model.hf_assets_path)
        self.dataloader = build_hf_dataloader(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            job_config=job_config,
        )
        model_args = job_config.model.args
        model_args.max_seq_len = job_config.training.seq_len
        # record ,do true init after ensuring slice
        with (
            torch.device("meta"),
            device_utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = DeepSeekV3Model(model_args)

        self.metrics_processor = MetricsProcessor(job_config, self.parallel_dims)
        model_param_count, self.metrics_processor.num_flops_per_token = (
            model_args.get_numparams_flops(model, job_config.training.seq_len)
        )
        logger.info(f"model total parameters: {model_param_count:,}")
        self.loss_fn = build_cross_entropy_loss(job_config)

        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            global_batch_size = job_config.training.local_batch_size * dp_degree
        # assert global_batch_size > 0, global_batch_size
        # assert global_batch_size%(job_config.training.local_batch_size*dp_degree)==0,global_batch_size,job_config.training.local_batch_size*dp_degree
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
