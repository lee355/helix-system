import os
import tempfile
import unittest

import torch
import torch.multiprocessing as mp

from dschat.helix.dynamic_safety import _collective_error, _destroy_custom_groups


def _safety_worker(rank, world_size, init_file):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        subgroup = torch.distributed.new_group([0, 1])
        caught = False
        try:
            _collective_error(
                "preflight",
                ValueError("rank-local failure") if rank == 1 else None,
            )
        except RuntimeError as error:
            caught = "rank 1" in str(error) and "rank-local failure" in str(error)
        assert caught
        _destroy_custom_groups([subgroup])
        value = torch.tensor([rank + 1.0])
        torch.distributed.all_reduce(value)
        assert float(value.item()) == 3.0
    finally:
        torch.distributed.destroy_process_group()


class HelixDynamicSafetyTest(unittest.TestCase):
    def test_rank_error_consensus_and_subgroup_destroy_keep_world_alive(self):
        if os.name == "nt":
            self.skipTest("file:// Gloo spawn test is Linux-only")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _safety_worker,
                args=(2, os.path.join(directory, "group")),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
