"""One pinned learned E2 parent; no fresh-base fallback or optimizer import."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v65 import initialization as old
from wengua_course_v61.initialization import snapshot as kernel
from . import runtime
ESSENTIAL=old.ESSENTIAL


def inspect(path,*,testing=False):return old.inspect(path,testing=testing)


def default_source():
    from research_audits.v65_learning_probe import selected_sources
    source=selected_sources(runtime.ROOT/'local/course_v65/run1')['v65_epoch2'];expected=d.read(runtime.RELEASE)['parent']
    if source['adapter_sha256']!=expected['sha256']:raise ValueError('V65 E2 differs from reviewed inherited parent')
    return Path(source['adapter'])


snapshot=FunctionType(kernel.__code__,globals(),'snapshot',kernel.__defaults__)
snapshot.__kwdefaults__=kernel.__kwdefaults__
