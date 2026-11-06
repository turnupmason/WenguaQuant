"""Continue the reviewed V54 weights; a new curriculum gets a fresh optimizer."""
from pathlib import Path
from types import FunctionType
import shutil
from wengua_course import data as d
from wengua_course_v53 import initialization as kernel
from wengua_course_v54 import initialization as old
from . import runtime
ESSENTIAL=old.ESSENTIAL
_inspect=FunctionType(kernel.inspect.__code__,globals(),'_inspect',kernel.inspect.__defaults__);_inspect.__kwdefaults__=kernel.inspect.__kwdefaults__

def inspect(path,*,testing=False):
 schema=d.read(Path(path)/'book_adapter_receipt.json')['schema']
 if schema in ('wengua_source_grounded_course_v53','wengua_two_book_conditional_course_v52','wengua_actual_rag_worked_source_course_v54'):return old.inspect(path,testing=testing)
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
 root=Path(source_run);state=d.read(root/'run_status.json')
 if state.get('state')!='complete_review_required' or not state.get('all_planned_groups_finished'):raise ValueError('Preserve the completed reviewed V54 run')
 epochs=state['trials']['source_lr_2em05']['epochs'];entry=next(e for e in epochs if e['steps']==134 and e['epoch']==1.0)
 path=root/'trials/source_lr_2em05/cycle_01/adapters/step-134';proof=inspect(path)
 expected=d.read(runtime.RELEASE)['reviewed_parent_adapter_sha256']
 if proof['files']['adapter_model.safetensors']!=expected:raise ValueError('V54 step134 differs from the reviewed candidate')
 return path
