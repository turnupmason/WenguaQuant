"""Only the pinned official base's sealed, zero-update LoRA can seed V63."""
import shutil
from pathlib import Path
from types import FunctionType
from wengua_course import data as d
from wengua_course_v53 import initialization as kernel
from wengua_course_v61.initialization import snapshot as old_snapshot
from . import runtime
ESSENTIAL=kernel.ESSENTIAL
_inspect=FunctionType(kernel.inspect.__code__,globals(),'_inspect',kernel.inspect.__defaults__)
_inspect.__kwdefaults__=kernel.inspect.__kwdefaults__
from .numerics import forward_parity
_create=FunctionType(kernel.create.__code__,globals(),'_create',kernel.create.__defaults__)
_create.__kwdefaults__=kernel.create.__kwdefaults__

def inspect(path,*,testing=False):
    path=Path(path)
    if d.read(path/'book_adapter_receipt.json')['schema']!=runtime.SCHEMA:raise ValueError('V63 never accepts earlier trained adapters')
    return _inspect(path,testing=testing)

def verify_zero(path,*,testing=False):
    from safetensors.torch import load_file
    path=Path(path);proof=inspect(path,testing=testing)
    contract=d.read(path.parent.parent/'stage_contract.json')
    if path.name!='step-0' or not contract.get('initialization_only') or contract.get('updates')!=0:raise ValueError('Only fresh step0 may initialize V63')
    weights=load_file(str(path/'adapter_model.safetensors'),device='cpu')
    matrices=[v for k,v in weights.items() if 'lora_B' in k]
    if not matrices or any(bool(v.count_nonzero()) for v in matrices):raise ValueError('Initializer is trained/nonzero; previous weights refused')
    return proof|{'official_fresh_lora':True,'weight_updates':0,'all_lora_B_zero':True}

def create(p,job,c,*,factory=None):
    if c.get('initialization_only') is not True:raise ValueError('Initialization-only configuration required')
    result=_create(p,job,c,factory=factory)
    result['zero_update_proof']=verify_zero(result['adapter'],testing=p.testing)
    d.write(Path(job)/'zero_update_proof.json',result['zero_update_proof'])
    return result

_snapshot=FunctionType(old_snapshot.__code__,globals(),'_snapshot',old_snapshot.__defaults__)
_snapshot.__kwdefaults__=old_snapshot.__kwdefaults__
def snapshot(root,path,*,testing=False):
    verify_zero(path,testing=testing)
    return _snapshot(root,path,testing=testing)
