import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from wengua_recovery import v59_watchdog as recovery


class RecoveryTests(unittest.TestCase):
    def test_worker_policy_digest_is_independent_of_json_key_order(self):
        self.assertEqual(recovery.canonical_sha({'b': 1, 'a': {'y': 2, 'x': 3}}),
                         recovery.canonical_sha({'a': {'x': 3, 'y': 2}, 'b': 1}))

    def test_future_status_mtime_does_not_hide_new_heartbeat(self):
        with TemporaryDirectory() as td:
            heart, status = Path(td)/'heart', Path(td)/'status'
            heart.write_text('one'); status.write_text('status')
            os.utime(status, ns=(10**18, 10**18)); os.utime(heart, ns=(10**9, 10**9))
            clock = recovery.ProgressClock(0)
            self.assertEqual(clock.observe([heart, status], 10), 0)
            self.assertEqual(clock.observe([heart, status], 20), 10)
            heart.write_text('two'); os.utime(heart, ns=(2*10**9, 2*10**9))
            self.assertEqual(clock.observe([heart, status], 21), 0)

    def test_monotonic_quiet_time_not_reset_by_identical_files(self):
        clock = recovery.ProgressClock(0)
        self.assertEqual(clock.observe([], 1), 0)
        self.assertEqual(clock.observe([], 601), 600)
        self.assertLess(recovery.POLICY['warn_seconds'], recovery.POLICY['hard_stall_seconds'])
        self.assertGreater(recovery.POLICY['hard_stall_seconds'], 600)
        self.assertFalse(recovery.POLICY['process_liveness_is_training_progress'])

    def test_stall_stack_is_captured_without_fake_heartbeat_and_can_be_exported(self):
        from wengua_recovery.v59_worker import diagnostics
        from wengua_course.runner import export_review
        import zipfile
        with TemporaryDirectory() as td:
            root = Path(td)/'run'; calls = root/'trials/mixed_pilot/worker_calls'
            calls.mkdir(parents=True); request = calls/'one.json'
            request.write_text('{}'); request.with_suffix('.dump_requested').touch()
            stopped = Mock(); stopped.wait.side_effect = [False, True]
            diagnostics(request, stopped)
            self.assertIn('Recovery diagnostic', request.with_suffix('.stack.log').read_text())
            self.assertFalse(request.with_suffix('.heartbeat.json').exists())
            self.assertFalse(request.with_suffix('.dump_requested').exists())
            with zipfile.ZipFile(export_review(root, Path(td)/'review.zip')) as archive:
                self.assertIn('trials/mixed_pilot/worker_calls/one.stack.log', archive.namelist())

    def test_observed_unused_9gib_cache_is_released(self):
        cuda = SimpleNamespace(memory_reserved=Mock(side_effect=[int(15.1171875*2**30), 8*2**30]),
                               memory_allocated=Mock(return_value=int(5.904*2**30)), empty_cache=Mock())
        result = recovery.release_cache(SimpleNamespace(cuda=cuda))
        cuda.empty_cache.assert_called_once_with()
        self.assertGreater(result['before_reserved_gib'], result['after_reserved_gib'])
        self.assertFalse(result['tensor_precision_changed'])

    def test_live_tensor_pressure_is_not_treated_as_releasable_cache(self):
        cuda = SimpleNamespace(memory_reserved=Mock(return_value=11*2**30),
                               memory_allocated=Mock(return_value=10*2**30), empty_cache=Mock())
        self.assertIsNone(recovery.release_cache(SimpleNamespace(cuda=cuda)))
        cuda.empty_cache.assert_not_called()

    def test_healthy_cache_is_kept(self):
        cuda = SimpleNamespace(memory_reserved=Mock(return_value=9*2**30),
                               memory_allocated=Mock(return_value=6*2**30), empty_cache=Mock())
        self.assertIsNone(recovery.release_cache(SimpleNamespace(cuda=cuda)))
        cuda.empty_cache.assert_not_called()

    def test_limit_is_applied_before_model_load_without_dtype_mutation(self):
        cuda = SimpleNamespace(current_device=Mock(return_value=0), set_per_process_memory_fraction=Mock(),
                               get_device_properties=Mock(return_value=SimpleNamespace(total_memory=16*2**30)))
        result = recovery.allocator_limit(SimpleNamespace(cuda=cuda))
        cuda.set_per_process_memory_fraction.assert_called_once_with(0.80, 0)
        self.assertEqual(result['allocator_limit_gib'], 12.8)

    def test_cpu_fixture_does_not_claim_cuda_policy_was_applied(self):
        torch = SimpleNamespace(cuda=Mock())
        self.assertFalse(recovery.allocator_limit(torch, testing=True)['allocator_policy_applied'])
        self.assertEqual(torch.cuda.mock_calls, [])

    def test_manifest_pins_original_v59_files_and_existing_dataset(self):
        manifest = json.loads(recovery.MANIFEST.read_text())
        self.assertEqual(manifest['policy'], recovery.POLICY)
        self.assertEqual(manifest['frozen_commit'], '277047d8489edf8c3db48f95eb9cf7f5cd5b376a')
        for name, sha in manifest['frozen_v59_code_sha256'].items():
            if name.startswith('wengua_course_v59/') or name in ['configs/course_v59.json', 'train_wengua_v59.cmd', 'run_course_v59_worker.py']:
                self.assertEqual(recovery.digest((recovery.ROOT/name).read_text()), sha, name)
        self.assertFalse(recovery.POLICY['training_config_changed'])


if __name__ == '__main__':
    unittest.main()
