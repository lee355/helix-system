import unittest
from collections import OrderedDict

import torch

from dschat.helix.adam_migration import (
    NamedAdamParameterState,
    mask_linear_indices,
    mask_local_shape,
    migrate_named_adam_states,
)


def _slice(tensor, mask):
    return tensor.reshape(-1).index_select(
        0, mask_linear_indices(mask, tensor.shape)
    ).reshape(mask_local_shape(mask, tensor.shape))


def _states(global_state, masks):
    result = []
    for rank_mask in masks:
        local = OrderedDict()
        for name, state in global_state.items():
            local[name] = NamedAdamParameterState(
                parameter=_slice(state.parameter, rank_mask[name]),
                master_parameter=_slice(state.master_parameter, rank_mask[name]),
                exp_avg=_slice(state.exp_avg, rank_mask[name]),
                exp_avg_sq=_slice(state.exp_avg_sq, rank_mask[name]),
                step=state.step.clone(),
            )
        result.append(local)
    return result


class HelixAdamTransitionCoverageTest(unittest.TestCase):
    def test_migration_rejects_a_gap_on_either_side_of_transition(self):
        reference = OrderedDict(vector=torch.zeros(4))
        complete = [
            OrderedDict(vector=torch.tensor([0, 1])),
            OrderedDict(vector=torch.tensor([2, 3])),
        ]
        incomplete = [
            OrderedDict(vector=torch.tensor([0, 1])),
            OrderedDict(vector=torch.tensor([2])),
        ]
        values = torch.arange(4, dtype=torch.float32)
        global_state = OrderedDict(
            vector=NamedAdamParameterState(
                parameter=values,
                master_parameter=values + 10,
                exp_avg=values + 20,
                exp_avg_sq=values + 30,
                step=torch.tensor(5.0),
            )
        )

        with self.assertRaisesRegex(ValueError, "old: structured masks do not cover"):
            migrate_named_adam_states(
                reference,
                incomplete,
                complete,
                _states(global_state, incomplete),
            )
        with self.assertRaisesRegex(ValueError, "new: structured masks do not cover"):
            migrate_named_adam_states(
                reference,
                complete,
                incomplete,
                _states(global_state, complete),
            )


if __name__ == "__main__":
    unittest.main()
