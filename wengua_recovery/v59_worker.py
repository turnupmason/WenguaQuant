"""Recovery instrumentation around the unchanged, receipt-bound V5.9 worker."""
from pathlib import Path
import faulthandler
import os
import sys
import threading
import time
import traceback

from .v59_watchdog import MANIFEST, allocator_limit, canonical_sha, digest, recipe, release_cache, verify_frozen


def diagnostics(request, stopped):
    """No CUDA calls, tensor reads, RNG use or pretend training heartbeats."""
    trigger = request.with_suffix('.dump_requested')
    try:
        with request.with_suffix('.stack.log').open('a', encoding='utf8') as out:
            while not stopped.wait(2):
                if trigger.exists():
                    out.write('\nRecovery diagnostic at '+str(time.time())+'\n')
                    out.flush()
                    faulthandler.dump_traceback(file=out, all_threads=True)
                    trigger.unlink(missing_ok=True)
    except BaseException:
        request.with_suffix('.diagnostic_error.log').write_text(traceback.format_exc(), encoding='utf8')


def main():
    request = Path(sys.argv[1]).resolve()
    from wengua_course_v59 import runtime
    from wengua_course import data as d
    runtime.configure()
    payload = d.read(request)
    code = 1
    stopped = threading.Event()
    thread = None
    try:
        if not request.is_relative_to((runtime.ROOT/'local').resolve()) or request.parent.name != 'worker_calls':
            raise ValueError('Recovery request outside private worker calls')
        verify_frozen()
        expected = recipe() | {'frozen_code_manifest_sha256': digest(MANIFEST.read_bytes()),
                               'software_fixture': payload.get('software_fixture') is True}
        if payload.get('recovery') != expected:
            raise ValueError('Recovery policy or transport code changed')
        from wengua_course_v59 import process
        import torch
        from transformers import TrainerCallback
        from unittest.mock import patch
        applied = allocator_limit(torch, testing=payload.get('software_fixture') is True)
        d.write(request.with_suffix('.allocator_policy.json'), applied)
        thread = threading.Thread(target=diagnostics, args=(request, stopped), daemon=True)
        thread.start()
        actual_profile = process.profile

        def profile(**kwargs):
            p = actual_profile(**kwargs)
            fit = p.engine.fit

            class CacheGuard(TrainerCallback):
                def pulse(self, phase, state):
                    d.write(request.with_suffix('.main_progress.json'),
                            {'phase': phase, 'step': state.global_step, 'time': time.time()})
                    if not p.testing:
                        released = release_cache(torch)
                        if released:
                            # Append only bounded operational facts, never training targets.
                            with request.with_suffix('.cache_releases.jsonl').open('a', encoding='utf8') as f:
                                f.write(d.dumps(released | {'step': state.global_step, 'time': time.time()})+'\n')

                def on_substep_end(self, args, state, control, **kw):
                    self.pulse('microbatch_completed', state)

                def on_step_end(self, args, state, control, **kw):
                    self.pulse('optimizer_step_completed', state)

                def on_prediction_step(self, args, state, control, **kw):
                    self.pulse('validation_batch_completed', state)

            def instrumented(*a, extra_callbacks=(), **kw):
                return fit(*a, extra_callbacks=(CacheGuard(), *extra_callbacks), **kw)

            p.engine.fit = instrumented
            return p

        with patch.object(process, 'profile', profile):
            code = process.worker(request)
    except BaseException as error:
        traceback.print_exc()
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        if request.is_relative_to((runtime.ROOT/'local').resolve()) and request.parent.name == 'worker_calls':
            d.write(request.with_suffix('.result.json'),
                    {'schema': 'wengua_source_course_v59_process', 'nonce': payload.get('nonce'),
                     'request_sha256': digest(request.read_bytes()), 'pid': os.getpid(), 'ok': False,
                     'error': {'type': type(error).__name__, 'message': str(error), 'cuda_oom': False}})
    finally:
        stopped.set()
        if thread is not None:
            thread.join(timeout=3)
        out = request.with_suffix('.result.json')
        if out.exists():
            reply = d.read(out)
            reply['recovery_sha256'] = canonical_sha(payload.get('recovery'))
            d.write(out, reply)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
