"""Pinned learned V66 ordinary step77; do not inherit the old optimizer."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v53 import initialization as kernel
from wengua_course_v61.initialization import snapshot as snapshot_kernel
from . import runtime
ESSENTIAL=kernel.ESSENTIAL
_globals=dict(globals(),runtime=runtime.before)
_inspect=FunctionType(kernel.inspect.__code__,_globals,'_inspect',kernel.inspect.__defaults__);_inspect.__kwdefaults__=kernel.inspect.__kwdefaults__


def inspect(path,*,testing=False):return _inspect(path,testing=testing)


def verify_source_run(root):
    root=Path(root);pins=d.read(runtime.ROOT/'data/reviews/v67_parent_sources.json')
    for name,sha in pins['files'].items():
        if d.digest((root/name).read_bytes())!=sha:raise ValueError('Completed V66 source changed: '+name)
    run=d.read(root/'sweep_contract.json')
    if run['code_sha256']!=runtime.before.hashes():raise ValueError('Original V66 runtime changed')
    return root


def default_source():
    root=verify_source_run(runtime.ROOT/'local/course_v66/run1');selected=d.read(runtime.RELEASE)['parent']
    path=runtime.ROOT/selected['relative_path'];proof=inspect(path)
    if proof['files']['adapter_model.safetensors']!=selected['sha256']:raise ValueError('Different V66 ordinary learned adapter')
    return path


snapshot=FunctionType(snapshot_kernel.__code__,globals(),'snapshot',snapshot_kernel.__defaults__)
snapshot.__kwdefaults__=snapshot_kernel.__kwdefaults__
