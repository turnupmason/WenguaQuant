"""Continue the reviewed V54 weights; a new curriculum gets a fresh optimizer."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v53 import initialization as kernel
from wengua_course_v55 import initialization as old
from . import runtime
ESSENTIAL=old.ESSENTIAL
_inspect=FunctionType(kernel.inspect.__code__,globals(),'_inspect',kernel.inspect.__defaults__);_inspect.__kwdefaults__=kernel.inspect.__kwdefaults__

def inspect(path,*,testing=False):
 schema=d.read(Path(path)/'book_adapter_receipt.json')['schema']
 if schema!=runtime.SCHEMA:return old.inspect(path,testing=testing)
 return _inspect(path,testing=testing)

def snapshot(root,path,*,testing=False):
 root=Path(root);proof=inspect(path,testing=testing);dest=root/'parent_adapter';receipt=root/'parent_provenance.json';root.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if d.read(receipt)!=proof:raise ValueError('Source adapter changed; existing weights retained')
 else:
  pending=root/'parent_adapter.pending';pending.mkdir()
  for name in ESSENTIAL:shutil.copy2(Path(path)/name,pending/name)
  pending.rename(dest);d.write(receipt,proof)
 for name,sha in proof['files'].items():
  if d.digest((dest/name).read_bytes())!=sha:raise ValueError('Continuation snapshot differs')
 return dest

def default_sources():
 release=d.read(runtime.RELEASE);result={}
 for label,entry in release['reviewed_parents'].items():
  path=runtime.ROOT/entry['path'];proof=inspect(path)
  if proof['files']['adapter_model.safetensors']!=entry['adapter_sha256']:raise ValueError('Reviewed inherited adapter differs: '+label)
  state=d.read(path.parents[4]/'run_status.json')
  if state.get('state')!='complete_review_required':raise ValueError('Reviewed source run is incomplete: '+label)
  result[label]=path
 return result
