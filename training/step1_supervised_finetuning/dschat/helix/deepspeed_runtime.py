"""Runtime fixes layered on the custom DeepSpeed build in ``ours_math``.

Keeping these fixes in the project makes the exact paper implementation
auditable.  The installed DeepSpeed still supplies the variable-shape engine
API; this module corrects its 1-D slicing, colored group order, and unsafe
checkpoint gather without replacing unrelated DeepSpeed behavior.
"""

from __future__ import annotations

import inspect
from typing import Iterable, List, Mapping, Sequence, Tuple

import torch


def _rectangle_view(tensor: torch.Tensor, rectangle):
    if tensor.ndim == 1:
        return tensor[rectangle[0][1] : rectangle[1][1] + 1]
    if tensor.ndim == 2:
        return tensor[
            rectangle[0][0] : rectangle[1][0] + 1,
            rectangle[0][1] : rectangle[1][1] + 1,
        ]
    raise ValueError(f"Helix rectangle references unsupported tensor shape {tuple(tensor.shape)}")


def _mapping_entries(mapping: Mapping[str, Sequence]) -> List[Tuple[str, object]]:
    return [
        (name, rectangle)
        for name, rectangles in mapping.items()
        for rectangle in rectangles
    ]


def _chunks(entries: Sequence[Tuple[str, object]], state: Mapping[str, torch.Tensor], limit: int):
    chunk = []
    elements = 0
    for name, rectangle in entries:
        size = _rectangle_view(state[name], rectangle).numel()
        if chunk and elements + size > limit:
            yield chunk
            chunk = []
            elements = 0
        chunk.append((name, rectangle))
        elements += size
    if chunk:
        yield chunk


def _patched_reduce_non_expert_gradients(self, grads, elements_per_buffer):
    from deepspeed.runtime import engine as engine_module

    split_sparse, split_dense = engine_module.split_half_float_double_sparse(grads)
    if self.pipeline_parallelism:
        data_parallel_group = self.mpu.get_data_parallel_group()
        data_parallel_size = engine_module.dist.get_world_size(data_parallel_group)
    else:
        data_parallel_group = engine_module.groups._get_sequence_data_parallel_group()
        data_parallel_size = (
            engine_module.dist.get_world_size(data_parallel_group)
            / float(self.sequence_parallel_size)
        )

    for sparse_tuple in split_sparse:
        if sparse_tuple:
            _, sparse_bucket = sparse_tuple
            self.sparse_allreduce_no_retain(
                sparse_bucket,
                dp_group=data_parallel_group,
                dp_world_size=data_parallel_size,
            )

    clusters = getattr(self, "helix_overlap_group_clusters", None)
    if clusters is None:
        clusters = [[group] for group in self.groups_to_offset_rectangles]

    for dense_tuple in split_dense:
        if not dense_tuple:
            continue
        _, dense_bucket = dense_tuple
        for cluster in clusters:
            # A valid graph color contains at most one subgroup involving this
            # rank. Different ranks enter their disjoint blocking collectives
            # simultaneously, which realizes the paper's parallel cluster.
            local_groups = [
                group for group in cluster if group in self.groups_to_offset_rectangles
            ]
            if len(local_groups) > 1:
                raise RuntimeError(
                    f"invalid Helix color: rank participates in multiple groups {local_groups}"
                )
            if not local_groups:
                continue
            group = local_groups[0]
            process_group = self.all_overlap_groups[group]
            bucket = []
            for layer, rectangles in self.groups_to_offset_rectangles[group].items():
                layer_id = self.layer_to_id_ls[layer]
                gradient = dense_bucket[layer_id]
                bucket.extend(_rectangle_view(gradient, rectangle) for rectangle in rectangles)
            if bucket:
                self.allreduce_no_retain(
                    bucket,
                    dp_group=process_group,
                    numel_per_bucket=elements_per_buffer,
                    dp_world_size=engine_module.dist.get_world_size(process_group),
                )


def _patched_gather_full_model(self, numel_per_bucket=6_291_456):
    """Reconstruct the full model with waited, bounded point-to-point copies."""

    rank = torch.distributed.get_rank()
    root = 0
    local_state = self.module.state_dict()
    full_state = self.full_model.state_dict()
    global_by_rank = self.complementary_groups_ranks_to_rectangles
    local_by_rank = self.complementary_groups_to_offset_rectangles

    torch.distributed.barrier()
    if rank == root:
        # Copy regions already owned by rank 0 without a device transfer.
        for name, global_rectangles in global_by_rank.get(root, {}).items():
            local_rectangles = local_by_rank[root][name]
            for local_rectangle, global_rectangle in zip(local_rectangles, global_rectangles):
                with torch.no_grad():
                    _rectangle_view(full_state[name], global_rectangle).copy_(
                        _rectangle_view(local_state[name], local_rectangle).to(
                            device=full_state[name].device,
                            dtype=full_state[name].dtype,
                        )
                    )

        for source in self.complementary_rank_ls:
            if source == root:
                continue
            global_entries = _mapping_entries(global_by_rank.get(source, {}))
            for chunk in _chunks(global_entries, full_state, numel_per_bucket):
                total = sum(_rectangle_view(full_state[name], rectangle).numel() for name, rectangle in chunk)
                received = torch.empty(
                    total,
                    dtype=self.communication_data_type,
                    device=self.device,
                )
                torch.distributed.recv(received, src=source, group=self.complementary_group)
                offset = 0
                for name, rectangle in chunk:
                    destination = _rectangle_view(full_state[name], rectangle)
                    size = destination.numel()
                    with torch.no_grad():
                        destination.copy_(
                            received[offset : offset + size]
                            .reshape(destination.shape)
                            .to(device=destination.device, dtype=destination.dtype)
                        )
                    offset += size
    elif rank in self.complementary_rank_ls:
        local_entries = _mapping_entries(local_by_rank.get(rank, {}))
        for chunk in _chunks(local_entries, local_state, numel_per_bucket):
            tensors = [
                _rectangle_view(local_state[name], rectangle).contiguous().reshape(-1)
                for name, rectangle in chunk
            ]
            payload = torch.cat(tensors).to(dtype=self.communication_data_type)
            torch.distributed.send(payload, dst=root, group=self.complementary_group)
    torch.distributed.barrier()


def install_deepspeed_runtime_patch() -> None:
    """Install Helix fixes after validating the custom engine API is present."""

    from deepspeed.runtime.engine import DeepSpeedEngine

    initialize_parameters = inspect.signature(DeepSpeedEngine.__init__).parameters
    required = {
        "groups_to_offset_rectangles",
        "all_overlap_groups_ls",
        "complementary_groups_ranks_to_rectangles",
        "complementary_groups_to_offset_rectangles",
    }
    missing = sorted(required.difference(initialize_parameters))
    if missing:
        raise RuntimeError(
            "The active DeepSpeed is not the custom Helix build; missing engine arguments: "
            + ", ".join(missing)
        )
    if getattr(DeepSpeedEngine, "_helix_paper_patch_installed", False):
        return
    DeepSpeedEngine._reduce_non_expert_gradients = _patched_reduce_non_expert_gradients
    DeepSpeedEngine.gather_full_model = _patched_gather_full_model
    DeepSpeedEngine._helix_paper_patch_installed = True
