"""Continue sealed V53 weights; a new curriculum gets a fresh optimizer."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v53 import initialization as old
from . import runtime
ESSENTIAL=old.ESSENTIAL
_inspect=FunctionType(old.inspect.__code__,globals(),'_inspect',old.inspect.__defaults__);_inspect.__kwdefaults__=old.inspect.__kwdefaults__

def inspect(path,*,testing=False):
 schema=d.read(Path(path)/'book_adapter_receipt.json')['schema']
 if schema in ('wengua_source_grounded_course_v53','wengua_two_book_conditional_course_v52'):return old.inspect(path,testing=testing)
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

def default_source(source_run):
 state=d.read(Path(source_run)/'run_status.json')
 if state.get('state')!='needs_course_review':raise ValueError('This repair entry expects the preserved V53 course-review stop')
 entry=state['foundation']['v52_comparison'];path=Path(entry['adapter']);inspect(path)
 if entry['steps']!=256:raise ValueError('Expected the reviewed V53 step256; choose --parent explicitly for another sealed adapter')
 return path
