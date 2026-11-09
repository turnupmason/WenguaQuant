"""Inherit frozen V58 epoch1 weights. No fresh official fallback and no old optimizer."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v53 import initialization as kernel
from wengua_course_v58 import initialization as previous
from . import runtime
ESSENTIAL=previous.ESSENTIAL
_inspect=FunctionType(kernel.inspect.__code__,globals(),'_inspect',kernel.inspect.__defaults__);_inspect.__kwdefaults__=kernel.inspect.__kwdefaults__

def inspect(path,*,testing=False):
    schema=d.read(Path(path)/'book_adapter_receipt.json')['schema']
    return _inspect(path,testing=testing) if schema==runtime.SCHEMA else previous.inspect(path,testing=testing)

def snapshot(root,path,*,testing=False):
    root=Path(root);proof=inspect(path,testing=testing);dest=root/'parent_adapter';receipt=root/'parent_provenance.json';root.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if d.read(receipt)!=proof:raise ValueError('Inherited source changed; existing weights preserved')
    else:
        pending=root/'parent_adapter.pending';pending.mkdir()
        for name in ESSENTIAL:shutil.copy2(Path(path)/name,pending/name)
        pending.rename(dest);d.write(receipt,proof)
    for name,sha in proof['files'].items():
        if d.digest((dest/name).read_bytes())!=sha:raise ValueError('Inherited snapshot bytes differ')
    return dest

def default_source():
    entry=d.read(runtime.RELEASE)['reviewed_parent'];path=runtime.ROOT/entry['path'];proof=inspect(path)
    stage=d.read(path.parent.parent/'stage_status.json')
    if proof['schema']!=previous.runtime.SCHEMA or stage['state']!='complete' or not any(e['steps']==97 for e in stage['epochs']):raise ValueError('Completed V58 epoch1 step97 required')
    if proof['files']['adapter_model.safetensors']!=entry['adapter_sha256']:raise ValueError('Reviewed V58 epoch1 bytes differ')
    c=proof['parent_config'];runtime.verify_snapshot(c)
    receipt=Path(c['book_parent_snapshot']).parent/'parent_provenance.json'
    if d.digest(receipt.read_bytes())!=c['book_parent_provenance_sha256']:raise ValueError('V58 ancestry receipt changed')
    if d.read(receipt)['files']['adapter_model.safetensors']!=entry['ancestor_v57_sha256']:raise ValueError('V58 ancestor differs')
    return path
