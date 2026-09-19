import tempfile
import unittest
from pathlib import Path

from dschat.helix.planner import (
    adjust_rank_to_memory_change,
    fast_search,
    load_plan,
    save_plan,
)
from dschat.helix.profiling import (
    ProfileSample,
    build_device_profile,
    fit_bilinear_model,
    load_profiles,
    save_profiles,
)


def _value(coefficients, batch_size, submodel_size):
    c0, c_batch, c_size, c_cross = coefficients
    return c0 + c_batch * batch_size + c_size * submodel_size + c_cross * batch_size * submodel_size


def _samples(compute_coefficients, memory_coefficients):
    result = []
    for batch_size, submodel_size in ((1, 0.25), (1, 0.75), (3, 0.25), (3, 0.75), (2, 0.5)):
        result.append(
            ProfileSample(
                micro_batch_size=batch_size,
                submodel_size=submodel_size,
                compute_seconds=_value(compute_coefficients, batch_size, submodel_size),
                memory_bytes=_value(memory_coefficients, batch_size, submodel_size),
            )
        )
    return result


def _profiles():
    memory = (100.0, 20.0, 100.0, 10.0)
    return [
        build_device_profile(
            rank=0,
            device_name="fast",
            memory_budget_bytes=350.0,
            samples=_samples((0.10, 0.020, 0.030, 0.010), memory),
        ),
        build_device_profile(
            rank=1,
            device_name="slow",
            memory_budget_bytes=350.0,
            samples=_samples((0.10, 0.040, 0.050, 0.020), memory),
        ),
    ]


class HelixPlannerTest(unittest.TestCase):
    def test_bilinear_fit_recovers_synthetic_coefficients(self):
        expected = (2.0, 3.0, 5.0, 7.0)
        samples = _samples(expected, (10.0, 2.0, 4.0, 6.0))
        fitted = fit_bilinear_model(samples, "compute_seconds")
        actual = (fitted.c0, fitted.c_batch, fitted.c_size, fitted.c_batch_size)
        for observed, target in zip(actual, expected):
            self.assertAlmostEqual(observed, target, places=9)
        self.assertLess(fitted.mape, 1e-12)

    def test_fast_search_respects_memory_coverage_and_batch_sum(self):
        communication_calls = []

        def communication_time(sizes):
            communication_calls.append(tuple(sizes))
            return 0.01 * max(sizes)

        plan = fast_search(
            _profiles(),
            dataset_size=1000,
            communication_time=communication_time,
            max_micro_batch_size=6,
            minimum_submodel_size=0.1,
        )
        self.assertEqual(sum(plan.batch_sizes), plan.total_batch_size)
        self.assertGreaterEqual(sum(plan.submodel_sizes), 1.0)
        self.assertTrue(communication_calls)
        for profile, allocation in zip(_profiles(), plan.allocations):
            self.assertLessEqual(allocation.estimated_memory_bytes, profile.memory_budget_bytes + 1e-8)

    def test_dynamic_adjustment_keeps_compute_near_target_under_new_budget(self):
        profile = _profiles()[1]
        allocation = adjust_rank_to_memory_change(
            profile,
            previous_compute_seconds=0.25,
            new_memory_budget_bytes=280.0,
            max_micro_batch_size=6,
            minimum_submodel_size=0.1,
        )
        self.assertLessEqual(allocation.estimated_memory_bytes, 280.0 + 1e-8)
        self.assertGreaterEqual(allocation.submodel_size, 0.1)

    def test_profile_and_plan_json_round_trip(self):
        profiles = _profiles()
        plan = fast_search(profiles, dataset_size=100, max_micro_batch_size=3)
        with tempfile.TemporaryDirectory() as directory:
            profile_path = str(Path(directory) / "profiles.json")
            plan_path = str(Path(directory) / "plan.json")
            save_profiles(profile_path, profiles)
            save_plan(plan_path, plan)
            self.assertEqual([item.rank for item in load_profiles(profile_path)], [0, 1])
            self.assertEqual(load_plan(plan_path).batch_sizes, plan.batch_sizes)


if __name__ == "__main__":
    unittest.main()
