"""Resume frozen V5.9 with bounded allocator cache and diagnostic stall monitoring."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/knowledge/course_v59/recovery_v1.json'
FILES = ('wengua_recovery/__init__.py', 'wengua_recovery/v59_watchdog.py',
         'wengua_recovery/v59_worker.py', 'resume_wengua_v59.cmd')
POLICY = dict(version='v59_memory_recovery_v1', allocator_fraction=0.80,
              release_reserved_gib=10.0, release_inactive_gib=2.0,
              warn_seconds=180, diagnostic_seconds=300, hard_stall_seconds=1800,
              process_liveness_is_training_progress=False,
              training_config_changed=False, original_pressure_is_new_policy_verification=False)


def digest(raw):
    return hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


def canonical_sha(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def recipe():
    return {'policy': POLICY, 'code_sha256': {n: digest((ROOT/n).read_text(encoding='utf8')) for n in FILES}}


def verify_frozen():
    from wengua_course_v59 import runtime
    expected = json.loads(MANIFEST.read_text(encoding='utf8'))
    if runtime.hashes() != expected['frozen_v59_code_sha256']:
        raise ValueError('Frozen V5.9 code or release differs; existing training output preserved')
    if POLICY != expected['policy']:
        raise ValueError('Unreviewed recovery policy')
    return expected


class ProgressClock:
    """Compare each file independently; a future mtime cannot mask heartbeats."""
    def __init__(self, now):
        self.last = now
        self.seen = None

    def observe(self, paths, now):
        stamps = []
        for path in paths:
            try:
                s = path.stat()
                stamps.append((str(path), s.st_mtime_ns, s.st_size))
            except FileNotFoundError:
                pass
        stamps = tuple(sorted(stamps))
        if stamps != self.seen:
            self.seen = stamps
            self.last = now
        return max(0, now - self.last)


def allocator_limit(torch, *, testing=False):
    if testing:
        return {'software_fixture': True, 'allocator_policy_applied': False}
    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(POLICY['allocator_fraction'], device)
    return {'allocator_policy_applied': True, 'device': device,
            'fraction': POLICY['allocator_fraction'],
            'allocator_limit_gib': torch.cuda.get_device_properties(device).total_memory * POLICY['allocator_fraction']/2**30,
            'not_a_reservation_against_other_processes': True}


def release_cache(torch):
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    if reserved < POLICY['release_reserved_gib']*2**30 or reserved-allocated < POLICY['release_inactive_gib']*2**30:
        return None
    torch.cuda.empty_cache()
    return {'before_reserved_gib': reserved/2**30, 'allocated_gib': allocated/2**30,
            'after_reserved_gib': torch.cuda.memory_reserved()/2**30,
            'tensor_precision_changed': False}


def gpu_snapshot():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return {'returncode': r.returncode, 'csv': r.stdout.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'unavailable': type(e).__name__}


def launch(p, job, c, deadline, operation, *, probe=False, checkpoint=None, best=None, stop_step=None):
    from wengua_course_v59 import process as original
    b = p.book
    job = b.private_output(job)
    root = job.parent.parent
    verify_frozen()
    original.execution(root, p)
    proof = recipe() | {'frozen_code_manifest_sha256': digest(MANIFEST.read_bytes()),
                        'software_fixture': p.testing}
    folder = root/'recovery_receipts'
    folder.mkdir(exist_ok=True)
    receipt = folder/(canonical_sha(proof)+'.json')
    if receipt.exists() and b.read_json(receipt) != proof:
        raise ValueError('Recovery receipt differs')
    if not receipt.exists():
        b.atomic_json(receipt, proof)
    if probe:
        # This original smoke is immutable and is NOT advertised as testing the new cap.
        return original_launch(p, job, c, deadline, operation, probe=True,
                               checkpoint=checkpoint, best=best, stop_step=stop_step)
    calls = job/'worker_calls'
    calls.mkdir(exist_ok=True)
    nonce = uuid.uuid4().hex
    request = calls/(str(time.time_ns())+'-'+nonce+'.json')
    out = request.with_suffix('.result.json')
    b.atomic_json(request, {'schema': original.SCHEMA, 'job': str(job), 'config': c,
                           'deadline': deadline, 'operation': operation, 'probe': False,
                           'checkpoint': str(checkpoint) if checkpoint else None, 'best': best,
                           'nonce': nonce, 'software_fixture': p.testing, 'fixture_stop_step': stop_step,
                           'recovery': proof})
    request_sha = b.file_sha(request)
    old_failure = job/'cycle_01/failure.json'
    if old_failure.exists():
        request.with_suffix('.previous_failure.json').write_bytes(old_failure.read_bytes())
    child = subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'wengua_recovery.v59_worker', str(request)],
                             cwd=b.ROOT, env=os.environ.copy(),
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    b.progress('v59_recovery_worker_started', trial=job.name, operation=operation, launcher_pid=child.pid)
    clock = ProgressClock(time.monotonic())
    warned = diagnosed = False
    try:
        while child.poll() is None:
            paths = [request.with_suffix('.heartbeat.json'), job/'cycle_01/stage_status.json',
                     request.with_suffix('.main_progress.json')]
            paths += list((job/'generations').glob('*.jsonl'))
            quiet = clock.observe(paths, time.monotonic())
            if quiet < POLICY['warn_seconds']:
                warned = diagnosed = False
            if quiet >= POLICY['warn_seconds'] and not warned:
                warned = True
                b.progress('v59_recovery_waiting_for_main_progress', quiet_seconds=round(quiet),
                           liveness_is_not_training_progress=True)
            if quiet >= POLICY['diagnostic_seconds'] and not diagnosed:
                diagnosed = True
                request.with_suffix('.dump_requested').touch()
                b.atomic_json(request.with_suffix('.stall_diagnostic.json'),
                              {'quiet_seconds': quiet, 'gpu': {} if p.testing else gpu_snapshot(),
                               'original_checkpoint_preserved': True})
            reason = 'stall' if quiet >= POLICY['hard_stall_seconds'] else 'deadline' if time.time() > deadline+60 else None
            if reason:
                original.stop_child(child)
                b.atomic_json(request.with_suffix('.monitor_stop.json'),
                              {'reason': reason, 'quiet_seconds': quiet, 'checkpoint_preserved': True})
                raise TimeoutError('V5.9 recovery paused after '+reason+'; checkpoints and stack diagnostics preserved: '+str(request))
            time.sleep(0.5)
        code = child.wait()
    except BaseException:
        original.stop_child(child)
        raise
    if not out.exists():
        b.atomic_json(request.with_suffix('.process_exit.json'),
                      {'returncode': code, 'windows_status_hex': hex(code & 0xffffffff), 'request_sha256': request_sha})
        raise RuntimeError('Recovery worker exited without receipt; outputs preserved: '+str(request))
    reply = b.read_json(out)
    if (reply.get('schema') != original.SCHEMA or reply.get('nonce') != nonce
            or reply.get('request_sha256') != request_sha or b.file_sha(request) != request_sha
            or type(reply.get('pid')) is not int or reply['pid'] <= 0
            or reply.get('recovery_sha256') != canonical_sha(proof)):
        raise ValueError('Recovery worker/request provenance differs')
    b.progress('v59_recovery_worker_exited', launcher_pid=child.pid, worker_pid=reply['pid'], exit_code=code)
    if code or not reply['ok']:
        error = reply['error']
        if error['type'] == 'KeyboardInterrupt':
            raise KeyboardInterrupt()
        raise RuntimeError(('CUDA out of memory: ' if error.get('cuda_oom') else 'Recovery worker failed: ')+error['message'])
    return reply['result']


original_launch = None


@contextmanager
def installed():
    global original_launch
    from unittest.mock import patch
    from wengua_course_v59 import process
    if original_launch is not None:
        raise RuntimeError('Nested recovery installation refused')
    original_launch = process.launch
    try:
        with patch.object(process, 'launch', launch):
            yield
    finally:
        original_launch = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run'], nargs='?', default='run')
    parser.add_argument('--output', type=Path, default=ROOT/'local/course_v59/run1')
    parser.add_argument('--app-root', type=Path)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    args = parser.parse_args(argv)
    from wengua_course_v59 import runtime, runner
    runtime.configure(args.app_root)
    verify_frozen()
    if args.command == 'check':
        runtime.verify_data()
        print('Frozen V5.9 source, data and training contracts verified; allocator cap 80%; resume original intact checkpoint.')
        return 0
    forwarded = ['run', '--resume', '--output', str(args.output), '--hours', str(args.hours)]
    if args.app_root:
        forwarded += ['--app-root', str(args.app_root)]
    if args.resume_from_checkpoint:
        forwarded += ['--resume-from-checkpoint', str(args.resume_from_checkpoint)]
    with installed():
        return runner.main(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
