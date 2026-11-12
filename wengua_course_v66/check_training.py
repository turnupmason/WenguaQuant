"""Real tiny CPU optimizer tests, not real 8B quality or CUDA verification."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from copy import deepcopy
import os,tempfile,time,sys,subprocess
from wengua_course import data as d
from . import runtime,initialization,runner,evaluation,loss


def mathematical_checks():
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen_finetune import prepare_frozen_base
    torch.set_num_threads(2);tok,base,_=fixture_factory();base=prepare_frozen_base(base,gradient_checkpointing=False)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj']))
    ids=torch.tensor([[4,6,10,2,5,6,9,7,8,2]]);labels=ids.clone();labels[:,:6]=-100
    inputs={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels,'decision_mask':torch.tensor([[0,0,0,0,0,0,0,1,0,0]])}
    model.train();head=base.lm_head
    for factor in (1,4):
        model.zero_grad();reference=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,:-1,:].float()
        ce=F.cross_entropy(reference.reshape(-1,reference.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none')
        weights=(1+(factor-1)*inputs['decision_mask'][:,1:].float())*labels[:,1:].ne(-100);weights*=4/weights.sum()
        expected=(ce*weights.reshape(-1)).sum()/4;expected.backward()
        gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad();audit={'max_projected_positions':0};actual=loss.supervised_loss(model,inputs,factor=factor,chunk_tokens=2,audit=audit);actual.backward()
        assert torch.allclose(actual,expected,atol=1e-6) and audit['max_projected_positions']<=2
        assert all(torch.allclose(p.grad,gradients[n],atol=1e-6) for n,p in model.named_parameters() if p.grad is not None)
        # Same accumulation-group normalizer as Trainer: no extra 1/GAS.
        normalized=loss.supervised_loss(model,inputs,factor=factor,chunk_tokens=2,num_items_in_batch=12)
        assert torch.allclose(normalized,expected/3,atol=1e-6)
    # Mask corruption must fail rather than silently weight a prompt.
    broken={**inputs,'decision_mask':torch.ones_like(ids)}
    try:loss.supervised_loss(model,broken,factor=4)
    except ValueError:pass
    else:raise AssertionError('Invalid prompt mask accepted')
    from transformers.trainer import Trainer
    old_method=Trainer.compute_loss
    with loss.memory_trainer(128,factor=4):assert Trainer.compute_loss is not old_method
    assert Trainer.compute_loss is old_method


def software_pack():
    from wengua_course_v65.check_training import software_pack as old_pack
    pack=old_pack();pack['schema']=runtime.SCHEMA
    for r in pack['records']:
        r['pilot_selected']=r['split']=='train';r['provenance']={'decision_spans':[{'start':7,'end':10,'text':'one'}]}
        r['messages'][-1]['content']='answer one two';r['answer']='answer one two'
    pack['summary']={'software_fixture_only':True}
    first=deepcopy(pack['records'][0]);first['provenance']={'support':[]};first['components']={'automatic_professional_pass':False};first['teacher_forcing']=True
    first['provenance']['decision_spans']=[{'start':7,'end':10,'text':'one'}]
    episode={'id':'fixture','system':'question','source_ids':first['source_ids'],'support':[],
       'steps':[{'id':'fixture:first','question':'question one','answer':'answer one','lesson_contract':{}},
                {'id':'fixture:next','question':'question two','answer':'answer two','lesson_contract':{}}]}
    pack['fixture_evaluation_pack']={'schema':runtime.SCHEMA,'records':[first],'episodes':[episode],
                                    'summary':{'per_model_answers':3}}
    return pack


def make_v65_parent(root):
    from wengua_course_v64.check_training import make_parent
    from wengua_course_v65 import runtime as rt,runner as run,initialization as start
    from wengua_course_v65.check_training import software_pack as old_pack
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    source,small=make_parent(root);parent=start.snapshot(root/'v65_parent',source,testing=True);tok,_,_=fixture_factory()
    c=run.config(parent,root,'v63_trained')|small;p=rt.profile(testing=True);job=p.engine.open_job(root/'v65/trials/learn',c,resume=True)
    rt.prepare(job,c,tokenizer=tok,pack=old_pack(),testing=True)
    with attention_route(testing=True),memory_trainer(128):result=p.engine.fit(job,c,time.time()+3600,cpu_factory=fixture_factory)
    return Path(result['last_adapter']),small


def main():
    runtime.configure();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');mathematical_checks()
    import torch
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from peft.utils.save_and_load import get_peft_model_state_dict
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    from .process import launch
    (runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v66_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);source,small=make_v65_parent(root);small|={'num_train_epochs':1,'generation_epochs':[1]}
        before=d.digest((source/'adapter_model.safetensors').read_bytes());weights=load_file(str(source/'adapter_model.safetensors'))
        assert any(bool(v.count_nonzero()) for k,v in weights.items() if 'lora_B' in k)
        parent=initialization.snapshot(root/'parent',source,testing=True);tok,_,_=fixture_factory();pack=software_pack();p=runtime.profile(testing=True)
        original_config=runner.config
        def short(parent,root,variant):return original_config(parent,root,variant)|small
        c=short(parent,root,'decision_ce')
        class Inherited(TrainerCallback):
            seen=False
            def on_train_begin(self,args,state,control,**kw):
                got=get_peft_model_state_dict(kw['model']);assert all(torch.equal(weights[k],got[k].detach().cpu()) for k in weights)
                assert not kw['optimizer'].state;self.seen=True
        job=p.engine.open_job(root/'direct/trials/full',c,resume=True);runtime.prepare(job,c,tokenizer=tok,pack=pack,testing=True);check=Inherited()
        with attention_route(testing=True),loss.memory_trainer(128,factor=4) as audit:result=p.engine.fit(job,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
        assert check.seen and result['state']=='complete' and result['steps']==3 and audit['weighted_training_calls']>0
        assert audit['evaluation_factor']==1 and audit['eval_loss_calls']>0
        final=load_file(str(Path(result['last_adapter'])/'adapter_model.safetensors'))
        assert any(not torch.equal(weights[k],final[k]) for k in weights)
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        stopped=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert stopped['state']=='paused_training'
        restored=launch(p,resumed,c,time.time()+3600,'fit');actual=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'))
        assert all(torch.equal(final[k],actual[k]) for k in final)
        real_profile=runtime.profile;real_prepare=runtime.prepare
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(initialization,'default_source',lambda:source),patch.object(runner,'config',short):
            args=Namespace(app_root=runtime.app_root(),output=root/'loop',hours=1,resume=True,resume_from_checkpoint=None)
            assert runner.run(args)==0;state=d.read(args.output/'run_status.json');assert state['state']=='complete_review_required'
            assert set(state['trials'])==set(runner.VARIANTS)
            metas=[d.read(args.output/'trials'/v/'prepared/receipt.json') for v in runner.VARIANTS]
            assert metas[0]['training_tensor_sha256']==metas[1]['training_tensor_sha256'] and metas[0]['decision_tokens']>0
            matrices=[]
            for v in runner.VARIANTS:
                t=state['trials'][v];assert t['fit']['parent_weights_reused'] and t['fit']['steps']==3
                matrices.append(load_file(str(Path(t['fit']['last_adapter'])/'adapter_model.safetensors')))
                answers=evaluation.read_cache(args.output/'trials'/v/'generations'/(v+'.jsonl'));assert len(answers)==3
                token_records=evaluation.read_cache(args.output/'trials'/v/'generations'/(v+'.tokens.jsonl'))
                assert len(token_records)==1 and token_records[0]['critical_tokens']>0 and token_records[0]['critical_mean_nll'] is not None
                assert answers[-1]['question'][2]['content']==answers[-2]['answer'];assert not answers[-1]['provenance']['gold_prior_answers_injected']
                for f in (args.output/'trials'/v/'worker_calls').glob('*.loss_path.json'):
                    path=d.read(f);assert path['route']=='v66_normalized_decision_ce_v1'
            assert any(not torch.equal(matrices[0][k],matrices[1][k]) for k in matrices[0])
            old_files={str(f):d.digest(f.read_bytes()) for f in args.output.glob('trials/*/generations/*.jsonl')}
            assert runner.run(args)==0;assert all(d.digest(Path(f).read_bytes())==h for f,h in old_files.items())
            from wengua_course.runner import export_review
            assert export_review(args.output,root/'review.zip').is_file()
        assert d.digest((source/'adapter_model.safetensors').read_bytes())==before
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            child=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v66.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert child.returncode==0 and 'check' in child.stdout
        proof={'software_fixture_only':True,'unweighted_and_weighted_native_loss_and_gradients_equal':True,
             'same_tensor_data_order_and_parent':True,'weighted_and_ordinary_updates_differ':True,'actual_weighted_training_route_verified':True,
             'evaluation_is_unweighted':True,'inherited_nonzero_weights_unchanged':True,'new_optimizer_empty':True,
             'partial_accumulation_tail_exact_resume':True,'actual_model_own_history_and_cache_resume':True,'native_windows_cmd_help':os.name=='nt',
             'actual_8b_cuda_improvement_verified':False}
        d.write(runtime.ROOT/'local/course_v66_cpu_verification.json',proof);print(d.dumps(proof))


if __name__=='__main__':main()
