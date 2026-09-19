import unittest

from test_helix_masking import _config, _toy_state

from dschat.helix.communication import build_communication_plan
from dschat.helix.cost import estimate_cluster_reduce_seconds
from dschat.helix.masking import build_structured_masks


class HelixCommunicationCostTest(unittest.TestCase):
    def test_colored_ring_estimate_is_positive_and_requires_all_links(self):
        state = _toy_state()
        masks, _ = build_structured_masks(state, [0.5, 0.5, 0.5, 0.5], config=_config())
        names = [name for name in state if "embed_tokens" not in name and "lm_head" not in name]
        plan = build_communication_plan(state, masks, names)
        bandwidths = {
            (left, right): 1_000_000_000.0
            for left in range(4)
            for right in range(left + 1, 4)
        }
        self.assertGreater(
            estimate_cluster_reduce_seconds(plan, state, bandwidths),
            0.0,
        )
        bandwidths.pop((0, 1))
        with self.assertRaisesRegex(ValueError, "Missing bandwidth"):
            estimate_cluster_reduce_seconds(plan, state, bandwidths)


if __name__ == "__main__":
    unittest.main()
