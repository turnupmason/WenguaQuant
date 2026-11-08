"""V5.8 versioned continuation fit; skip redundant official-answer generation.

Based on the continuous Trainer protocol, with explicit parent initialization,
fresh phase optimizer and parent baseline. Its helper globals are bound to the
new isolated runtime by course_training_v51.profile.
"""
from pathlib import Path
import math,time,shutil,gc

def fit(job,c,deadline,*,probe=False,cpu_factory=None,extra_callbacks=(),resume_from_checkpoint=None):
    """One Trainer and scheduler for all epochs; every optimizer restore is sealed."""
    import torch
    from transformers import AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,TrainerState,set_seed
    from peft import LoraConfig,PeftModel,get_peft_model
    from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
    from src.wengua.qwen_finetune import collator,prepare_frozen_base
    job=Path(job);stage=job/('smoke' if probe else 'cycle_01');data=job/'prepared'
    rows=book.read_rows(data/('smoke.jsonl.gz' if probe else 'cycle_01.jsonl.gz'))
    validation=book.read_rows(data/'validation_loss.jsonl.gz');formats=sorted({r['format'] for r in validation})
    datasets={fmt:[r['encoded'] for r in validation if r['format']==fmt] for fmt in formats}
    steps_per_epoch=math.ceil(len(rows)/c['gradient_accumulation_steps']);planned=c['smoke_steps'] if probe else steps_per_epoch*c['num_train_epochs']
    contract={'schema':SCHEMA,'job_contract_sha256':book.file_sha(job/'job_contract.json'),'unit':stage.name,'config':c,
        'data_sha256':book.file_sha(data/('smoke.jsonl.gz' if probe else 'cycle_01.jsonl.gz')),
        'validation_sha256':book.file_sha(data/'validation_loss.jsonl.gz'),'planned_updates':planned,
        'parent_provenance_sha256':c['book_parent_provenance_sha256'],'new_phase_optimizer':True,'stack':book.stack_versions(),'fresh_official_base':False,'parent_weights_reused':True,'optimizer_reset_between_epochs':False}
    if stage.exists():
        if not (stage/'stage_contract.json').exists() or book.read_json(stage/'stage_contract.json')!=contract:raise ValueError('Continuous stage contract changed')
    else:stage.mkdir();book.atomic_json(stage/'stage_contract.json',contract)
    statusfile=stage/'stage_status.json';prior=book.read_json(statusfile) if statusfile.exists() else {}
    cp=explicit_checkpoint(job,resume_from_checkpoint) if resume_from_checkpoint is not None else latest(stage)
    if prior.get('state')=='complete':book.checked_adapter(prior['adapter']);return prior
    if time.time()>deadline-240:return prior|{'state':'paused_before_model_load'}
    state=book.read_json(cp/'continuous_state.json') if cp else {'history':[],'early_stopped':False}
    if prior.get('steps',0)>0 and cp is None:raise ValueError('No intact continuous optimizer checkpoint; refusing fresh restart')
    pending=prior.get('state') in ('trained_pending_validation','trained_pending_generation')
    if (stage/'epoch_history.json').exists() and not pending:
        old=book.read_json(stage/'epoch_history.json')
        if old!=state['history']:book.atomic_json(stage/('epoch_history.before_restore-'+str(time.time_ns())+'.json'),old)
    book.atomic_json(stage/'epoch_history.json',prior.get('epochs',[]) if pending else state['history'])
    model=trainer=base=None;started=time.time();set_seed(c['seed'])
    try:
        if cpu_factory:tokenizer,base,compute=cpu_factory()
        else:
            tokenizer=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
            base,compute=book.load_book_cuda(c,training=True)
        tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
        if pending:model=PeftModel.from_pretrained(base,prior['last_adapter'],is_trainable=False)
        else:
            base=prepare_frozen_base(base,gradient_checkpointing=c['gradient_checkpointing'])
            from wengua_course.runtime import verify_snapshot
            parent=verify_snapshot(c)
            model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
            book.progress('v51_parent_adapter_loaded',parent=str(parent),optimizer_inherited=False,
                resume_checkpoint=str(cp) if cp else None)

        class EpochAudit(TrainerCallback):
            active=False;early=state['early_stopped'];history=prior.get('epochs',[]) if pending else state['history'];partial={}
            def on_train_begin(self,args,state,control,**kw):self.active=True;return control
            def on_train_end(self,args,state,control,**kw):self.active=False;return control
            def on_step_end(self,args,state,control,**kw):
                if time.time()>=deadline-180:control.should_save=True;control.should_training_stop=True
                return control
            def on_epoch_end(self,args,state,control,**kw):
                if state.epoch is None or abs(state.epoch-round(state.epoch))>1e-6:control.should_evaluate=False
                return control
            def on_evaluate(self,args,state,control,metrics=None,**kw):
                if not self.active or probe:return control
                self.partial.update({k:v for k,v in (metrics or {}).items() if k.startswith('eval_') and k.endswith('_loss')})
                if not all('eval_'+fmt+'_loss' in self.partial for fmt in formats):return control
                values=macro_losses(self.partial,formats);self.partial={}
                adapter=book.save_adapter(kw['model'],tokenizer,stage,state.global_step)
                row=dict(values,cycle=round(state.epoch),epoch=state.epoch,steps=state.global_step,
                    state='complete',adapter=str(adapter),optimizer_reset_between_epochs=False)
                self.history,self.early=update_history(self.history,row,c)
                book.atomic_json(stage/'epoch_history.json',self.history)
                control.should_save=True
                if self.early:control.should_training_stop=True
                return control
            def on_save(self,args,state,control,**kw):seal(stage,Path(args.output_dir)/f'checkpoint-{state.global_step}',self.history,self.early);return control
        audit=EpochAudit();cpu=cpu_factory is not None
        args=TrainingArguments(output_dir=str(stage/'trainer'),num_train_epochs=c['num_train_epochs'],max_steps=c['smoke_steps'] if probe else -1,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=c['gradient_accumulation_steps'],
            learning_rate=c['learning_rate'],warmup_ratio=c['warmup_ratio'],lr_scheduler_type='cosine',logging_steps=1,
            save_strategy='steps',save_steps=c['save_steps'],save_total_limit=c['save_total_limit'],eval_strategy='no' if probe else 'epoch',
            gradient_checkpointing=c['gradient_checkpointing'] and not pending,gradient_checkpointing_kwargs={'use_reentrant':False},
            optim='adamw_torch',report_to='none',seed=c['seed'],data_seed=c['seed'],dataloader_num_workers=0,
            remove_unused_columns=False,use_cpu=cpu,dataloader_pin_memory=not cpu,
            bf16=str(compute).endswith('bfloat16'),fp16=str(compute).endswith('float16') and not str(compute).endswith('bfloat16'))
        trainer=Trainer(model=model,args=args,train_dataset=[r['encoded'] for r in rows],eval_dataset=datasets,
            processing_class=tokenizer,data_collator=collator(tokenizer.pad_token_id),callbacks=[audit,*extra_callbacks])
        if not cpu:torch.cuda.reset_peak_memory_stats()
        baseline_path=job/'base_validation_loss.json'
        if not baseline_path.exists():
            with model.disable_adapter():values=trainer.evaluate()
            book.atomic_json(baseline_path,macro_losses(values,formats))
        baseline=book.read_json(baseline_path)
        parent_loss_path=job/'parent_validation_loss.json'
        if not parent_loss_path.exists():
            if cp or pending:
                # Never call an already trained checkpoint the parent baseline.
                raise ValueError('Parent baseline missing after training began')
            book.atomic_json(parent_loss_path,macro_losses(trainer.evaluate(),formats))
        if not probe and not cp and not pending:
            gen=job/'generations';gen.mkdir(exist_ok=True)
            dev=book.read_rows(data/'validation_prompts.jsonl.gz')
            result=book.generate_rows(model,tokenizer,dev,gen/'validation_parent.jsonl',c,deadline,'parent')
            if result['state']!='complete':return {'state':'paused_parent_generation','steps':0}
            seen=book.read_rows(data/'seen_diagnostic_prompts.jsonl.gz')
            diagnostic=book.generate_rows(model,tokenizer,seen,gen/'parent_seen_diagnostics.jsonl',c,deadline,'parent_seen_diagnostics')
            if diagnostic['state']!='complete':return {'state':'paused_parent_generation','steps':0}

        if not pending:
            # Parent generation enables KV caching. Restore training policy.
            model.config.use_cache=False
            book.atomic_json(statusfile,{'state':'training','steps':int(cp.name[11:]) if cp else 0,'planned_steps':planned,'resume_checkpoint':str(cp) if cp else None})
            if cp and not probe and int(cp.name[11:])%steps_per_epoch==0 and (not audit.history or audit.history[-1]['steps']!=int(cp.name[11:])):
                # The step checkpoint can precede epoch validation. Recover that
                # evaluation before Trainer skips to the next epoch on resume.
                set_peft_model_state_dict(model,load_peft_weights(str(cp),device='cpu'),adapter_name='default')
                trainer.state=TrainerState.load_from_json(str(cp/'trainer_state.json'));trainer._load_rng_state(str(cp))
                audit.active=True;trainer.evaluate();audit.active=False
                for name in ('continuous_state.json','trainer_state.json','rng_state.pth'):
                    shutil.copy2(cp/name,cp/(name+'.before_boundary_validation'))
                trainer.state.save_to_json(str(cp/'trainer_state.json'));trainer._save_rng_state(str(cp))
                seal(stage,cp,audit.history,audit.early)
            completed=cp and (int(cp.name[11:])>=planned or audit.early)
            if completed:
                set_peft_model_state_dict(model,load_peft_weights(str(cp),device='cpu'),adapter_name='default')
                trainer.state=TrainerState.load_from_json(str(cp/'trainer_state.json'));metrics={'finalized_without_extra_update':True}
            else:
                # A cached baseline skips evaluation dataloader RNG draws. Fresh
                # training must start identically whether that cache existed;
                # Trainer subsequently restores checkpoint RNG on real resume.
                set_seed(c['seed'])
                metrics=trainer.train(resume_from_checkpoint=str(cp) if cp else None).metrics
            count=trainer.state.global_step;finished=count>=planned or audit.early
            if finished and not probe and (not audit.history or audit.history[-1]['steps']!=count):
                audit.active=True;trainer.evaluate();audit.active=False
            checkpoint=stage/'trainer'/f'checkpoint-{count}'
            if count and not checkpoint.exists():trainer._save_checkpoint(trainer.model,trial=None);seal(stage,checkpoint,audit.history,audit.early)
            last=next((r['adapter'] for r in audit.history if r['steps']==count),None)
            last=last or str(book.save_adapter(model,tokenizer,stage,count))
            status={'state':'trained_pending_generation' if finished else 'paused_training','steps':count,'planned_steps':planned,
                'epochs':audit.history,'early_stopped':audit.early,'last_adapter':last,'last_optimizer_checkpoint':str(latest(stage)),
                'resume_checkpoint':str(cp) if cp else None,'optimizer_reset_between_epochs':False,'fresh_official_base':False,'parent_weights_reused':True,
                'runtime_seconds':time.time()-started+prior.get('runtime_seconds',0),'training_metrics':metrics,
                'parent_validation_loss':book.read_json(parent_loss_path)['validation_loss'],'base_validation_loss':baseline['validation_loss'],'base_per_format_loss':baseline['per_format_loss'],
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if not cpu else None,
                'professional_correctness_verified':False,'test_used_for_selection':False}
            book.atomic_json(statusfile,status)
        else:status=dict(prior)
        if status['state']=='paused_training':return status
        if probe:
            # Also exercise longest held-out forward and an actual bounded generation.
            longest=max(validation,key=lambda r:r['tokens']);trainer.evaluate(eval_dataset=[longest['encoded']])
            sample=book.read_rows(data/'validation_prompts.jsonl.gz')[:1]
            inference=c
            if (data/'extended_validation_plan.json').exists():
                extension=book.read_json(data/'extended_validation_plan.json')
                if extension['rows']:
                    sample=book.read_rows(data/'extended_validation_prompts.jsonl.gz')[:1]
                    inference=dict(c,eval_max_new_tokens=extension['max_new_tokens'])
            trainer=None;gc.collect()
            if not cpu:torch.cuda.empty_cache()
            sample_result=book.generate_rows(model,tokenizer,sample,job/'smoke_generation.jsonl',inference,deadline,'smoke')
            if not cpu:status['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            status['state']='complete' if sample_result['state']=='complete' else 'trained_pending_generation'
            status['adapter']=status['last_adapter'];status['smoke_adapter_not_used_for_formal_training']=True
            book.atomic_json(statusfile,status);return status
        if not status['epochs']:raise ValueError('No complete format validation to select an adapter')
        best=min(status['epochs'],key=lambda r:(r['validation_loss'],r['steps']));book.checked_adapter(best['adapter'])
        status.update(adapter=best['adapter'],selected_epoch=best['cycle'],validation_loss=best['validation_loss'],per_format_loss=best['per_format_loss'],selection_status='language_loss_nominee_only_runner_must_review_declared_constraints',final_test_run=False)
        # Release Adam before inference; parameter shape stays constant within this trial.
        trainer=None;gc.collect()
        if not cpu:torch.cuda.empty_cache()
        gen=job/'generations';gen.mkdir(exist_ok=True);dev=book.read_rows(data/'validation_prompts.jsonl.gz')
        complete=True
        if c.get('generate_official_baseline',False):
            complete=book.generate_rows(model,tokenizer,dev,gen/'validation_base.jsonl',c,deadline,'base',disable_adapter=True)['state']=='complete'
        for epoch in status['epochs']:
            set_peft_model_state_dict(model,load_peft_weights(epoch['adapter'],device='cpu'),adapter_name='default')
            result=book.generate_rows(model,tokenizer,dev,gen/f'epoch_{epoch["cycle"]:02d}_validation.jsonl',c,deadline,f'epoch_{epoch["cycle"]:02d}')
            complete=complete and result['state']=='complete'
            if result['state']!='complete':break
            seen=book.read_rows(data/'seen_diagnostic_prompts.jsonl.gz')
            diagnostic=book.generate_rows(model,tokenizer,seen,gen/f'epoch_{epoch["cycle"]:02d}_seen_diagnostics.jsonl',c,deadline,'seen_diagnostics')
            complete=complete and diagnostic['state']=='complete'
            if diagnostic['state']!='complete':break
        if complete and (data/'extended_validation_plan.json').exists():
            extension=book.read_json(data/'extended_validation_plan.json');status['extended_validation']=extension
            if extension['rows']:
                longdev=book.read_rows(data/'extended_validation_prompts.jsonl.gz');extended=dict(c,eval_max_new_tokens=extension['max_new_tokens'])
                set_peft_model_state_dict(model,load_peft_weights(best['adapter'],device='cpu'),adapter_name='default')
                a=book.generate_rows(model,tokenizer,longdev,gen/'extended_validation_base.jsonl',extended,deadline,'extended_validation_base',disable_adapter=True)
                b=book.generate_rows(model,tokenizer,longdev,gen/'extended_validation_selected.jsonl',extended,deadline,'extended_validation_selected')
                complete=a['state']==b['state']=='complete'
        status['state']='complete' if complete else 'trained_pending_generation'
        book.atomic_json(statusfile,status);return status
    except BaseException as error:
        book.atomic_json(stage/'failure.json',{'type':type(error).__name__,'message':str(error)[:2000],
            'last_complete_checkpoint':str(latest(stage)),'all_outputs_preserved':True});raise
    finally:
        trainer=model=base=None;gc.collect()
        if cpu_factory is None:torch.cuda.empty_cache()
