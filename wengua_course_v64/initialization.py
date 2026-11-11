"""Preserve and inherit the selected nonzero V63 adapter; never a fresh fallback."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v53 import initialization as kernel
from wengua_course_v61.initialization import snapshot as inherited_snapshot
from wengua_course_v63 import initialization as original
from . import runtime
ESSENTIAL=kernel.ESSENTIAL
_inspect=FunctionType(kernel.inspect.__code__,globals(),'_inspect',kernel.inspect.__defaults__)
_inspect.__kwdefaults__=kernel.inspect.__kwdefaults__


def inspect(path,*,testing=False):
    schema=d.read(Path(path)/'book_adapter_receipt.json')['schema']
    if schema==runtime.SCHEMA:return _inspect(path,testing=testing)
    if schema==original.runtime.SCHEMA:return original.inspect(path,testing=testing)
    raise ValueError('V64 requires its own formal adapter or the selected V63 source')


def default_source():
    selected=d.read(runtime.RELEASE)['reviewed_parent'];path=runtime.ROOT/selected['relative_path']
    proof=inspect(path)
    if proof['files']['adapter_model.safetensors']!=selected['adapter_sha256']:
        raise ValueError('Selected V63 step371 weights changed; old files preserved')
    return path


snapshot=FunctionType(inherited_snapshot.__code__,globals(),'snapshot',inherited_snapshot.__defaults__)
snapshot.__kwdefaults__=inherited_snapshot.__kwdefaults__
