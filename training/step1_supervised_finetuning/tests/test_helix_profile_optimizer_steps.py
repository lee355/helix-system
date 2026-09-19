"""A profile must include real Adam updates after FP16 loss-scale warmup."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import helix_profile_ds as profiler


class HelixProfileOptimizerStepsTest(unittest.TestCase):
    def _run_profile(self, applied_steps, expected_error=None):
        outcomes = iter(applied_steps)
        engine = Mock(device="cpu")
        state = SimpleNamespace(calls=0, updates=0)
        engine.was_step_applied.return_value = False

        def train_step(*_args):
            state.calls += 1
            applied = next(outcomes)
            engine.was_step_applied.return_value = applied
            state.updates += int(applied)

        def reset_memory(_device):
            # The first optimizer update has materialized Adam moments before
            # the measurement window, even when all requested warmups skipped.
            self.assertGreater(state.updates, 0)

        event = Mock()
        event.elapsed_time.return_value = 3000.0
        with ExitStack() as stack:
            stack.enter_context(patch.object(profiler, "_train_step", side_effect=train_step))
            stack.enter_context(patch.object(profiler.torch.cuda, "synchronize"))
            reset = stack.enter_context(
                patch.object(profiler.torch.cuda, "reset_peak_memory_stats", side_effect=reset_memory)
            )
            events = stack.enter_context(patch.object(profiler.torch.cuda, "Event", return_value=event))
            peak = stack.enter_context(
                patch.object(profiler.torch.cuda, "max_memory_allocated", return_value=4096)
            )
            if expected_error is None:
                result = profiler._profile_engine(engine, 1, 4, 16, 2, 3)
                self.assertEqual(result, (1.0, 4096.0))
                event.synchronize.assert_called_once()
            else:
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    profiler._profile_engine(engine, 1, 4, 16, 2, 3)
                peak.assert_not_called()
        return state, reset, events

    def test_stable_steps_keep_requested_window(self):
        state, reset, _ = self._run_profile([True] * 5)
        self.assertEqual((state.calls, state.updates), (5, 5))
        reset.assert_called_once_with("cpu")

    def test_warmup_waits_for_first_real_adam_update(self):
        # Initial warmup=2 and a further 3 overflows precede lazy Adam init.
        state, reset, _ = self._run_profile([False] * 5 + [True] * 4)
        self.assertEqual((state.calls, state.updates), (9, 4))
        reset.assert_called_once_with("cpu")

    def test_warmup_retries_are_bounded_and_never_produce_sample(self):
        state, reset, events = self._run_profile(
            [False] * 34, "warmup produced no optimizer update after 34 attempts"
        )
        self.assertEqual(state.calls, 34)
        reset.assert_not_called()
        events.assert_not_called()

    def test_measurement_overflow_discards_whole_sample(self):
        state, reset, _ = self._run_profile(
            [True, True, True, False], "measurement skipped an optimizer update at step 2/3"
        )
        self.assertEqual((state.calls, state.updates), (4, 3))
        reset.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()
