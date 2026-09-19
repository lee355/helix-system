import unittest
from unittest import mock

import torch

from dschat.helix.deepspeed_paper import _bounded_views
from dschat.helix import deepspeed_paper_v2
from dschat.helix.model_paper_sdpa import shape_safe_rate
from dschat.helix.paper_semantics import (
    apply_paper_deepspeed_config,
    paper_quantized_count,
)


class HelixPaperSemanticsTest(unittest.TestCase):
    def test_section_35_exact_allocations_use_nearest_integer(self):
        self.assertEqual(paper_quantized_count(0.77, 24), 18)
        self.assertEqual(paper_quantized_count(0.45, 24), 11)
        self.assertEqual(paper_quantized_count(0.61, 24), 15)
        self.assertEqual(paper_quantized_count(0.77, 8192), 6308)
        self.assertEqual(paper_quantized_count(0.45, 8192), 3686)
        self.assertEqual(paper_quantized_count(0.61, 8192), 4997)

    def test_optional_alignment_never_enlarges_allocation(self):
        self.assertEqual(paper_quantized_count(0.45, 8192, alignment=128), 3584)

    def test_shape_safe_rate_survives_integer_reconstruction(self):
        retained = 4997
        width = 11008
        rate = shape_safe_rate(retained / width)
        self.assertEqual(int(width * rate), retained)

    def test_paper_config_explicitly_disables_local_model_clipping(self):
        config = apply_paper_deepspeed_config({"gradient_clipping": 1.0})
        self.assertEqual(config["gradient_clipping"], 0.0)


class HelixBoundedRuntimeTest(unittest.TestCase):
    def test_large_matrix_is_tiled_below_limit(self):
        matrix = torch.arange(70).reshape(10, 7)
        tiles = list(_bounded_views(matrix, limit=12))
        self.assertTrue(tiles)
        self.assertTrue(all(tile.numel() <= 12 for tile in tiles))
        self.assertEqual(sum(tile.numel() for tile in tiles), matrix.numel())

    def test_runtime_caps_custom_engine_500m_default(self):
        with mock.patch.object(
            deepspeed_paper_v2,
            "_paper_reduce_non_expert_gradients",
        ) as reduce_impl:
            deepspeed_paper_v2._bounded_reduce(object(), [], 500_000_000)
        self.assertEqual(
            reduce_impl.call_args.args[2],
            deepspeed_paper_v2.PAPER_COMMUNICATION_BUCKET_ELEMENTS,
        )


if __name__ == "__main__":
    unittest.main()
