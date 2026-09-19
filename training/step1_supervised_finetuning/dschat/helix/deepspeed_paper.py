"""Paper-safe runtime layer for the custom ``ours_math`` DeepSpeed build."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple

import torch

from .communication import greedy_color_groups
from .deepspeed_runtime import _rectangle_view, install_deepspeed_runtime_patch
from .reconstruction import validate_reconstruction_metadata


def _bounded_views(tensor: torch.Tensor, limit: int) -> Iterator[torch.Tensor]:
    """Tile a 1-D/2-D view without any tile exceeding ``limit`` elements."""

    if limit < 1:
        raise ValueError("Helix communication bucket size must be positive")
    if tensor.ndim == 1:
        for start in range(0, tensor.numel(), limit):
            yield tensor[start : start + limit]
        return
    if tensor.ndim != 2:
        raise ValueError(f"unsupported Helix communication tensor: {tuple(tensor.shape)}")
    rows, columns = tensor.shape
    column_width = min(columns, limit)
    for column_start in range(0, columns, column_width):
        column_end = min(columns, column_start + column_width)
        rows_per_tile = max(1, limit // (column_end - column_start))
        for row_start in range(0, rows, rows_per_tile):
            yield tensor[
                row_start : min(rows, row_start + rows_per_tile),
                column_start:column_end,
            ]


def _bounded_buckets(views: Iterable[torch.Tensor], limit: int) -> Iterator[List[torch.Tensor]]:
    bucket: List[torch.Tensor] = []
    elements = 0
    for view in views:
        for tile in _bounded_views(view, limit):
            size = tile.numel()
            if bucket and elements + size > limit:
                yield bucket
                bucket = []
                elements = 0
            bucket.append(tile)
            elements += size
    if bucket:
        yield bucket


def _paper_reduce_non_expert_gradients(self, grads, elements_per_buffer):
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

    if not hasattr(self, "helix_overlap_group_clusters"):
        self.helix_overlap_group_clusters = greedy_color_groups(self.all_overlap_groups_ls)
    for dense_tuple in split_dense:
        if not dense_tuple:
            continue
        _, dense_bucket = dense_tuple
        for cluster in self.helix_overlap_group_clusters:
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
            views = (
                _rectangle_view(dense_bucket[self.layer_to_id_ls[layer]], rectangle)
                for layer, rectangles in self.groups_to_offset_rectangles[group].items()
                for rectangle in rectangles
            )
            for bucket in _bounded_buckets(views, elements_per_buffer):
                # Passing a pre-bounded bucket prevents the custom engine's
                # append-before-flush logic from flattening a giant embedding
                # rectangle into one hundreds-of-MiB temporary tensor.
                self.allreduce_no_retain(
                    bucket,
                    dp_group=process_group,
                    numel_per_bucket=elements_per_buffer,
                    dp_world_size=engine_module.dist.get_world_size(process_group),
                )


def _paired_tiles(
    local_tensor: torch.Tensor,
    local_rectangle,
    global_tensor: torch.Tensor,
    global_rectangle,
    limit: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    local_view = _rectangle_view(local_tensor, local_rectangle)
    global_view = _rectangle_view(global_tensor, global_rectangle)
    if local_view.shape != global_view.shape:
        raise RuntimeError(
            f"reconstruction shape mismatch: local={tuple(local_view.shape)}, "
            f"global={tuple(global_view.shape)}"
        )
    local_tiles = _bounded_views(local_view, limit)
    global_tiles = _bounded_views(global_view, limit)
    for local_tile, global_tile in zip(local_tiles, global_tiles):
        if local_tile.shape != global_tile.shape:
            raise RuntimeError("Helix reconstruction tiling diverged")
        yield local_tile, global_tile


def _reconstruction_tiles(self, source: int, limit: int):
    local_state = self.module.state_dict()
    full_state = self.full_model.state_dict()
    global_layers = self.complementary_groups_ranks_to_rectangles.get(source, {})
    local_layers = self.complementary_groups_to_offset_rectangles.get(source, {})
    for name, global_rectangles in global_layers.items():
        local_rectangles = local_layers.get(name, [])
        if len(local_rectangles) != len(global_rectangles):
            raise RuntimeError(f"reconstruction metadata mismatch for {name}/rank{source}")
        for local_rectangle, global_rectangle in zip(local_rectangles, global_rectangles):
            yield from _paired_tiles(
                local_state[name],
                local_rectangle,
                full_state[name],
                global_rectangle,
                limit,
            )


def _collective_validate_reconstruction(self, sources: Sequence[int]) -> None:
    """Make every world rank fail together before reconstruction P2P begins."""

    rank = torch.distributed.get_rank()
    local_error = None
    try:
        if 0 not in set(self.complementary_rank_ls):
            raise ValueError("reconstruction gather group does not include rank 0")
        full_state = self.full_model.state_dict()
        required_names = [name for name, _ in self.full_model.named_parameters()]
        local_states = (
            {rank: self.module.state_dict()}
            if rank in self.complementary_groups_ranks_to_rectangles
            else None
        )
        validate_reconstruction_metadata(
            full_state,
            self.complementary_groups_ranks_to_rectangles,
            self.complementary_groups_to_offset_rectangles,
            required_names=required_names,
            allowed_sources=sources,
            local_states_by_rank=local_states,
        )
    except Exception as error:  # all ranks must still enter the consensus collective
        local_error = f"rank {rank}: {type(error).__name__}: {error}"

    errors: List[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, local_error)
    failures = [str(error) for error in errors if error is not None]
    if failures:
        raise RuntimeError(
            "Helix reconstruction metadata validation failed collectively: "
            + "; ".join(failures)
        )


def _paper_gather_full_model(self, numel_per_bucket=6_291_456):
    """Reconstruct on rank zero with bounded, waited point-to-point messages."""

    rank = torch.distributed.get_rank()
    root = 0
    group = self.complementary_group
    sources = sorted(set(self.complementary_rank_ls).union({root}))
    _collective_validate_reconstruction(self, sources)
    if rank not in sources:
        return
    torch.distributed.barrier(group=group)
    for source in sources:
        if source == root:
            if rank == root:
                for local_tile, global_tile in _reconstruction_tiles(self, source, numel_per_bucket):
                    with torch.no_grad():
                        global_tile.copy_(
                            local_tile.to(device=global_tile.device, dtype=global_tile.dtype)
                        )
        elif rank == root:
            # Root metadata references its full model for shape/tiling.  The
            # local view itself belongs to ``source`` and is never read here.
            global_state = self.full_model.state_dict()
            global_layers = self.complementary_groups_ranks_to_rectangles.get(source, {})
            local_layers = self.complementary_groups_to_offset_rectangles.get(source, {})
            for name, global_rectangles in global_layers.items():
                for local_rectangle, global_rectangle in zip(local_layers[name], global_rectangles):
                    global_view = _rectangle_view(global_state[name], global_rectangle)
                    # Shape is encoded by the local rectangle, while tiling is
                    # identical because local/global rectangles have equal extents.
                    for destination in _bounded_views(global_view, numel_per_bucket):
                        received = torch.empty(
                            destination.numel(),
                            dtype=self.communication_data_type,
                            device=self.device,
                        )
                        torch.distributed.recv(received, src=source, group=group)
                        with torch.no_grad():
                            destination.copy_(
                                received.reshape(destination.shape).to(
                                    device=destination.device,
                                    dtype=destination.dtype,
                                )
                            )
        elif rank == source:
            for local_tile, _ in _reconstruction_tiles(self, source, numel_per_bucket):
                payload = local_tile.contiguous().reshape(-1).to(
                    dtype=self.communication_data_type
                )
                torch.distributed.send(payload, dst=root, group=group)
        torch.distributed.barrier(group=group)


def _install_global_overflow_consensus() -> None:
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.utils import CheckOverflow

    if getattr(CheckOverflow, "_helix_world_consensus_installed", False):
        return
    original = CheckOverflow.has_overflow

    def has_overflow_world(self, params, has_moe_params=None):
        overflow = original(self, params, has_moe_params=has_moe_params)
        if self.deepspeed is not None and getattr(
            self.deepspeed.__class__, "_helix_paper_runtime_installed", False
        ):
            flag = get_accelerator().ByteTensor([overflow])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=dist.get_world_group())
            overflow = bool(flag[0].item())
        return overflow

    CheckOverflow.has_overflow = has_overflow_world
    CheckOverflow._helix_world_consensus_installed = True


def install_paper_deepspeed_runtime() -> None:
    """Validate and patch the custom engine before any engine is created."""

    install_deepspeed_runtime_patch()
    from deepspeed.runtime.engine import DeepSpeedEngine

    if not getattr(DeepSpeedEngine, "_helix_paper_runtime_installed", False):
        DeepSpeedEngine._reduce_non_expert_gradients = _paper_reduce_non_expert_gradients
        DeepSpeedEngine.gather_full_model = _paper_gather_full_model
        DeepSpeedEngine._helix_paper_runtime_installed = True
    _install_global_overflow_consensus()
