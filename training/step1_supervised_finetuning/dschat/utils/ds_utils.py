# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        gradient_accumulation_steps=1,
                        learning_rate=0.01,
                        weight_decay=3e-07,
                        lr_scheduler_type="WarmupLR",
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
        "allgather_partitions":False,
        "reduce_bucket_size": 5e4,
        "allgather_bucket_size": 5e4
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
        '''
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }
    '''
    return {
        "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": learning_rate,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-08,
            "weight_decay": weight_decay
        }
    },
    "scheduler": {
        "type": lr_scheduler_type,
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    },
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False,
        "synchronize_checkpoint_boundary": False,
        "profile": True
    },
    "fp16": {
        "enabled": True,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "consecutive_hysteresis": False,
        "min_loss_scale": 1
    },
    "logging": {
        "log_level": "DEBUG",
        "log_gradients": True,
        "log_gradient_norm": True,
        "log_batch_step_time": True
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "zero_optimization": zero_opt_dict,
    "nccl_options": {
        "env": {
            "NCCL_SOCKET_IFNAME": "eno1"
        }
    },
    "flops_profiler": {
        "enabled": True,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": "./output_step1_llama2_7b_lora/flops_profile.log"
        }
    }


def get_eval_ds_config(offload, dtype, stage=0):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {
            "enabled": True,
        }
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
