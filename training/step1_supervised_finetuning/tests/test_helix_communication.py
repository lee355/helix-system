import unittest

from test_helix_masking import _config, _toy_state

from dschat.helix.communication import build_communication_plan, greedy_color_groups
from dschat.helix.masking import build_structured_masks


class HelixCommunicationTest(unittest.TestCase):
    def test_coloring_places_disjoint_subgroups_in_same_cluster(self):
        clusters = greedy_color_groups([(0, 2), (1, 3), (0, 1, 2, 3)])
        color = {group: index for index, cluster in enumerate(clusters) for group in cluster}
        self.assertEqual(color[(0, 2)], color[(1, 3)])
        self.assertNotEqual(color[(0, 2)], color[(0, 1, 2, 3)])

        for cluster in clusters:
            occupied = set()
            for group in cluster:
                self.assertTrue(occupied.isdisjoint(group))
                occupied.update(group)

    def test_exact_plan_has_no_proxy_grid_or_manual_group_filter(self):
        state = _toy_state()
        masks, _ = build_structured_masks(state, [0.5, 0.5, 0.5, 0.5], config=_config())
        parameter_names = [
            name
            for name in state
            if "embed_tokens" not in name and "lm_head" not in name
        ]
        plan = build_communication_plan(state, masks, parameter_names)

        self.assertIn((0, 2), plan.overlap_groups)
        self.assertIn((1, 3), plan.overlap_groups)
        self.assertIn((0, 1, 2, 3), plan.overlap_groups)

        color = {
            group: index
            for index, cluster in enumerate(plan.clusters)
            for group in cluster
        }
        self.assertEqual(color[(0, 2)], color[(1, 3)])
        self.assertNotEqual(color[(0, 2)], color[(0, 1, 2, 3)])

        for rank, local_plan in enumerate(plan.local_groups_to_rectangles):
            self.assertTrue(all(rank in group for group in local_plan))
        self.assertEqual(plan.gather_ranks[0], 0)

    def test_uncovered_matrix_region_is_rejected(self):
        state = _toy_state()
        masks, _ = build_structured_masks(state, [0.5], config=_config())
        with self.assertRaisesRegex(ValueError, "do not cover|uncovered"):
            build_communication_plan(
                state,
                masks,
                ["model.layers.0.self_attn.q_proj.weight"],
            )


if __name__ == "__main__":
    unittest.main()
