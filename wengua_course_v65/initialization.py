"""Compare two pinned starts without replacing any existing adapter."""
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
    raise ValueError('Only V65 formal adapters or the two pinned V63 source adapters are allowed')


def default_source(variant):
    expected=d.read(runtime.RELEASE)['sources'][variant];path=runtime.ROOT/expected['relative_path'];proof=inspect(path)
    if proof['files']['adapter_model.safetensors']!=expected['sha256']:raise ValueError('Pinned source weights changed')
    if variant=='official_zero':original.verify_zero(path)
    elif proof['parent_config'].get('initialization_only'):raise ValueError('Trained branch cannot adopt a zero initializer')
    return path


snapshot=FunctionType(inherited_snapshot.__code__,globals(),'snapshot',inherited_snapshot.__defaults__)
snapshot.__kwdefaults__=inherited_snapshot.__kwdefaults__
