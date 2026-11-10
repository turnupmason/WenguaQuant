"""Real tiny CPU three-variant inference, cache, export and native CMD fixture."""
from pathlib import Path
import tempfile,os,sys,subprocess
from wengua_course import data as d
from wengua_course_v62 import runtime
from . import v62_probe as probe

def normalization_fixture():
    import torch
    from transformers import Trainer,TrainingArguments,TrainerCallback
    from peft import get_peft_model,LoraConfig
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen_finetune import prepare_frozen_base,collator
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    from src.wengua.qwen8b_attention_v1 import attention_route
    samples=[]
    for n in range(4,20):
     ids=[4,6]+[7,8]*(n//2)+[2]
     samples.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100,-100]+ids[2:]})
    allgrads=[]
    for gas,batch in [(1,16),(16,1)]:
     tok,base,_=fixture_factory();base=prepare_frozen_base(base,gradient_checkpointing=False)
     model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
     grads=[]
     class Audit(TrainerCallback):
      def on_pre_optimizer_step(self,args,state,control,**kw):grads.append(torch.cat([p.grad.detach().reshape(-1).cpu() for p in kw['model'].parameters() if p.requires_grad and p.grad is not None]))
     with tempfile.TemporaryDirectory() as tmp:
      args=TrainingArguments(output_dir=tmp,use_cpu=True,max_steps=1,per_device_train_batch_size=batch,gradient_accumulation_steps=gas,max_grad_norm=0,save_strategy='no',report_to='none',seed=3407,remove_unused_columns=False)
      trainer=Trainer(model=model,args=args,train_dataset=samples,data_collator=collator(tok.pad_token_id),processing_class=tok,callbacks=[Audit()])
      print('gas',gas,'accepts_loss_kwargs',trainer.model_accepts_loss_kwargs,'accelerator_gas',trainer.accelerator.gradient_accumulation_steps)
      with attention_route(testing=True),memory_trainer(128):trainer.train()
     allgrads.append(grads[0])
    a,b=allgrads
    relative=float((a-b).norm()/a.norm())
    if relative>=1e-5:raise AssertionError('Accumulated supervised gradient differs from full batch')
    return {'gas':16,'unequal_length_samples':16,'relative_gradient_error':relative,'software_fixture_only':True}


def main():
    runtime.configure()
    normalization=normalization_fixture()
    from wengua_course.fixtures import fixture_factory
    from wengua_course_v53 import runtime as old,initialization as oi
    tok,_,_=fixture_factory();ROOT=runtime.ROOT;(ROOT/'local').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v62_probe_cpu_',dir=ROOT/'local') as tmp:
        root=Path(tmp);p=old.profile(testing=True);c=d.read(old.CONFIG)|{'initialization_only':True}
        job=p.engine.open_job(root/'seed/trials/init',c,resume=True);created=oi.create(p,job,c,factory=fixture_factory);source=Path(created['adapter']);sha=d.digest((source/'adapter_model.safetensors').read_bytes());from safetensors.torch import load_file,save_file
        import torch
        weights=load_file(str(source/'adapter_model.safetensors'))
        torch.manual_seed(3407)
        for name,weight in weights.items():
            if 'lora_B' in name:weight.copy_(torch.randn_like(weight)*.01)
        save_file(weights,str(source/'adapter_model.safetensors'))
        sha=d.digest((source/'adapter_model.safetensors').read_bytes())
        original={p.name:d.digest(p.read_bytes()) for p in source.iterdir() if p.is_file()}
        rows=[]
        for i in range(3):
            rows.append({'id':'fixture:'+str(i),'question':'question one '+str(i),'answer':'answer two','messages':[{'role':'user','content':'question one '+str(i)},{'role':'assistant','content':'answer two'}],'source_ids':[],'format':'normal_dialogue_diagnostic','task':'fixture','group':'fixture','source_mode':'software_fixture_only'})
        pack={'records':rows,'official_rag_rows':[rows[0]|{'id':'fixture:rag'}],'software_fixture_only':True};sources={k:{'adapter':None if k=='official_base' else str(source),'adapter_sha256':None if k=='official_base' else sha,'variant':k,'software_fixture_only':True} for k in probe.LABELS}
        config=d.read(runtime.CONFIG)|{'max_seq_length':512,'inference_max_context':512}
        out=root/'probe'
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=config)==0
        state=d.read(out/'run_status.json');assert state['generated_answers']==10 and state['weight_updates']==0
        assert {p.name:d.digest(p.read_bytes()) for p in source.iterdir() if p.is_file()}==original
        first={p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}
        with (out/'answers/official_base.jsonl').open('ab') as f:f.write(b'{"interrupted":"\xff')
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=config)==0
        assert {p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}==first
        assert len(list((out/'answers').glob('*.partial-*')))==1
        assert all(m['cache_hit_without_model_load'] for m in d.read(out/'run_status.json')['models'].values())
        from wengua_course.runner import export_review
        assert export_review(out,root/'probe.zip').is_file()
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            r=subprocess.run(['cmd.exe','/d','/c',str(ROOT/'check_wengua_v62_learning.cmd'),'--help'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert r.returncode==0 and 'export' in r.stdout and 'run' in r.stdout
        for label in probe.LABELS[1:]:
            activation=d.read(out/'answers'/(label+'.activation.json'));assert activation['nonzero_lora_B_matrices']>0 and activation['max_last_position_logit_delta']>0 and activation['weight_updates']==0 and activation['parameters_frozen']
        result={'normalization_fixture':normalization,'software_fixture_only':True,'three_variants_actual_tiny_inference':True,'all_source_adapter_bytes_unchanged':True,'cache_rerun_without_model_load':True,'export_complete':True,'weight_updates':0,'actual_8b_cuda_quality_claimed':False}
        d.write(ROOT/'local/v62_probe_cpu_verification.json',result);print(d.dumps(result))
if __name__=='__main__':main()
