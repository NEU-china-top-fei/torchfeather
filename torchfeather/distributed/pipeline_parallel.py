import copy
import math
from collections.abc import Callable

import torch
from loguru import logger
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    get_schedule_class,
)

from torchfeather.components.loss import LossFunction
from torchfeather.config import JobConfig
from torchfeather.distributed import ParallelDims

__all__ = [
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_llm",
    "pipeline_module_split",
]


def pipeline_llm(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    num_layers: int,
    parallelize_fn: Callable[[nn.Module, ParallelDims, JobConfig], nn.Module],
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )

    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    # some method like 1f1b interleaved will put layer0,8 on rank 0,this is called non single stage
    layer_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage
    input_weight = job_config.parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = job_config.parallelism.pipeline_parallel_last_stage_less_layers

    if layer_per_stage is not None:
        # the embeding and outprojection are not the full layer,so  we should see them virtual layers to see how many stages
        num_virtual_stages = (
            math.ceil(num_layers + input_weight + output_weight) / layer_per_stage
        )
        assert num_virtual_stages % parallel_dims.pp == 0, (
            num_virtual_stages,
            parallel_dims.pp,
        )
        stage_per_rank = num_virtual_stages // parallel_dims.pp

        assert not is_single_stage_schedule or stage_per_rank == 1, (
            is_single_stage_schedule,
            stage_per_rank,
        )
        assert is_single_stage_schedule or stage_per_rank >= 2, (
            is_single_stage_schedule,
            stage_per_rank,
        )
    else:
        stage_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stage_per_rank

    module_name_per_stage = job_config.parallelism.module_fqns_per_model_part
    # something like layer1.ffn_norm.weight
    if module_name_per_stage is None:
        module_name_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_name_per_stage):
        logger.debug(f"stage {i}:{stage_ms}")

    stages, model_parts = pipeline_module_spilt(
        model,
        pp_mesh,
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_name_per_stage,
    )

    for i, m in enumerate(model_parts):
        m = parallelize_fn(m, parallel_dims, job_config)
        # apply other types of parallel and optimizer, checkpointing
        model_parts[i] = m
        # in case that the model being changed by something like torch.compile

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    has_first = False  # indicate whether this part contains the first embedding
    has_last = False
    for stage in stages:
        if stage.is_first:
            has_first = True
        if stage.is_last:
            has_last = True

    return pp_schedule, model_parts, has_first, has_last
