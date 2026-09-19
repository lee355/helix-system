import unittest

from dschat.helix.planner_bounded import _profile_bounded_candidates
from dschat.helix.planner_final import fast_search
from dschat.helix.profiling import BilinearModel, DeviceProfile, ProfileSample


def _profile(rank):
    samples = [
        ProfileSample(1, 0.25, 0.20, 120.0),
        ProfileSample(1, 0.75, 0.30, 170.0),
        ProfileSample(3, 0.25, 0.30, 140.0),
        ProfileSample(3, 0.75, 0.45, 190.0),
    ]
    return DeviceProfile(
        rank=rank,
        device_name=f"device-{rank}",
        memory_budget_bytes=400.0,
        compute=BilinearModel(0.10, 0.05, 0.10, 0.02),
        memory=BilinearModel(80.0, 10.0, 100.0, 0.0),
        samples=samples,
    )


class HelixFinalPlannerTest(unittest.TestCase):
    def test_candidates_do_not_extrapolate_past_successful_batch_or_size(self):
        candidates = _profile_bounded_candidates(
            _profile(0),
            max_micro_batch_size=8,
            minimum_submodel_size=0.1,
            memory_slack_bytes=0.0,
        )
        self.assertLessEqual(max(item.micro_batch_size for item in candidates), 3)
        self.assertLessEqual(max(item.submodel_size for item in candidates), 0.75)

    def test_quantized_coverage_predicate_is_a_hard_constraint(self):
        calls = []

        def reject_every_plan(sizes):
            calls.append(tuple(sizes))
            return False

        with self.assertRaisesRegex(ValueError, "quantized"):
            fast_search(
                [_profile(0), _profile(1)],
                dataset_size=100,
                max_micro_batch_size=3,
                minimum_submodel_size=0.25,
                coverage_predicate=reject_every_plan,
            )
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
