import unittest

import torch

from dschat.helix.deepspeed_runtime import _rectangle_view


class HelixDeepSpeedRuntimeTest(unittest.TestCase):
    def test_rectangle_view_uses_tensor_dimension_not_parameter_name(self):
        vector = torch.arange(6)
        matrix = torch.arange(20).reshape(4, 5)
        torch.testing.assert_close(
            _rectangle_view(vector, ([0, 2], [0, 4])),
            torch.tensor([2, 3, 4]),
        )
        torch.testing.assert_close(
            _rectangle_view(matrix, ([1, 2], [3, 4])),
            matrix[1:4, 2:5],
        )


if __name__ == "__main__":
    unittest.main()
