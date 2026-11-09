"""Resume an original V59 checkpoint under recovery transport; tiny CPU only."""
from pathlib import Path
from unittest.mock import patch
import json
import os
import subprocess
import sys


def main():
    from wengua_course_v59 import runtime
    runtime.configure()
    from wengua_course_v59 import process, check_training
    from . import v59_watchdog as recovery
    before = runtime.hashes()
    recovery.verify_frozen()
    original = process.launch
    calls = {'original_interrupted': 0, 'recovery_resumed': 0}
    with recovery.installed():
        recovered = process.launch
        def choose(*args, **kwargs):
            if kwargs.get('stop_step'):
                calls['original_interrupted'] += 1
                return original(*args, **kwargs)
            if not kwargs.get('probe'):
                calls['recovery_resumed'] += 1
            return recovered(*args, **kwargs)
        with patch.object(process, 'launch', choose):
            check_training.main()
    assert calls['original_interrupted'] == 1 and calls['recovery_resumed'] > 1
    assert runtime.hashes() == before
    windows = {'native_windows': False}
    if os.name == 'nt':
        env = os.environ.copy(); env['WENGUA_PY'] = sys.executable
        result = subprocess.run(['cmd.exe', '/d', '/c', str(recovery.ROOT/'resume_wengua_v59.cmd'), '--help'],
                                env=env, cwd=recovery.ROOT, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0 and 'check' in result.stdout
        windows = {'native_windows': True, 'actual_recovery_cmd_help_verified': True}
    proof = {'software_fixture_only': True, 'original_v59_checkpoint_resumed_by_recovery': True,
             'same_final_adapter_tensors_verified_by_original_fixture': True,
             'frozen_v59_hashes_unchanged': True, 'training_kernel_and_data_changed': False,
             'gpu_cache_policy_verified_on_real_gpu': False, 'windows': windows, 'calls': calls}
    target = recovery.ROOT/'local/course_v59_build/recovery_cpu_verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()
