import io,json,tempfile,unittest,zipfile
from pathlib import Path
from copy import deepcopy
from wengua_course import data as d
from research_audits import v67_complete_review as r


class V67CompletedActualReviewBoundaries(unittest.TestCase):
    def zip(self,values):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            for name,value in values.items():z.writestr(name,value)
        buffer.seek(0);return zipfile.ZipFile(buffer)

    def test_recovery_may_append_answers_and_report_but_not_rewrite_training(self):
        old={'run_status.json':b'failed','results.html':b'old','trials/object_condition_repair/generations/epoch_01.jsonl':b'original\r\n','training.json':b'weights preserved'}
        new=old|{'run_status.json':b'complete','results.html':b'new','trials/object_condition_repair/generations/epoch_01.jsonl':b'original\r\nnext\n',
                 'evaluation_recovery/original_run_status.json':b'failed'}
        with self.zip(old) as a,self.zip(new) as b:self.assertTrue(r.verify_original_members(a,b)['old_parent_and_epoch1_answer_bytes_preserved'])
        for change in [{'training.json':b'changed'},{'trials/object_condition_repair/generations/epoch_01.jsonl':b'rewritten\n'}]:
            with self.subTest(change=change),self.zip(old) as a,self.zip(new|change) as b,self.assertRaises(ValueError):r.verify_original_members(a,b)

    def test_completed_seal_binds_raw_answers_variant_config_and_adapter(self):
        raw=b'{"answer":"actual"}\n';c={'fixed':True};stem='answers/epoch_01'
        receipt={'config':c,'pack_sha256':'pack','record_variant':'epoch_01','professional_approved':False,
                 'history_source':'model_own_previous_answers','adapter_files':{'weights':'source'}}
        payload=d.dumps(receipt).encode();complete={'contract_sha256':d.digest(payload),'answers_sha256':d.digest(raw),'answers':1,'adapter_files':receipt['adapter_files']}
        files={stem+'.receipt.json':payload,stem+'.complete.json':d.dumps(complete),stem+'.jsonl':raw}
        with self.zip(files) as z:r.verify_seal(z,stem,c,'pack','epoch_01',1)
        with self.zip(files|{stem+'.jsonl':b'changed'}) as z,self.assertRaises(ValueError):r.verify_seal(z,stem,c,'pack','epoch_01',1)

    def test_input_format_is_read_from_actual_current_prompt_not_position(self):
        row={'components':{'source_contract':{'kind':'typed_objects','goal':'palaces'}},'question':[{'content':'program card in an earlier context'},{'content':'下面是程序核定的九宫星位摘录。\n乾6宫：天心星'}]}
        self.assertEqual(r.format_kind(row),'raw_full_names')
        row['question'][-1]['content']='程序位置：天心星在乾宫，天芮星在坤宫。';self.assertEqual(r.format_kind(row),'fact_card')
        row['components']['source_contract']['goal']='stars';self.assertIsNone(r.format_kind(row))

    def test_seen_teacher_and_verdict_balance_are_observed_not_assumed(self):
        def row(split,answer,skill):return {'split':split,'answer':answer,'skill':skill,'messages':[{'role':'user','content':'question'},{'role':'assistant','content':answer}]}
        course={'records':[row('train','可以保留','star_claim'),row('train','需要改正','star_claim'),row('validation','可以保留','star_claim'),row('validation','unseen answer','star_claim')]}
        x=r.curriculum_diagnostics(course);self.assertEqual(x['development_teacher_texts_also_in_training'],1);self.assertEqual(x['star_claim_verdicts'],{'retain':1,'correct':1})
        self.assertFalse(x['full_194_case_training_claimed'])

    def test_worker_memory_and_prefix_are_bound_to_actual_new_answers(self):
        name='calls/123-nonce.result.json';c={'fixed':True};recipe={'fixed_policy':True};rows=b''.join((d.dumps({'id':f'q{i}'})+'\n').encode() for i in range(24))
        q={'schema':r.recovery.SCHEMA,'operation':'evaluate','config':c,'recipe':recipe,'limit':24,'software_fixture':False,'nonce':'nonce','best':{'tag':'epoch_02'}}
        request=d.dumps(q).encode();res={'completed':24,'new_answers':24,'requested':161,'training_updates':0,'professional_approved':False,
            'state':'chunk_complete','previous_prefix_sha256':d.digest(b''),'previous_bytes_preserved':True,'source_files_unchanged':True}
        reply={'schema':r.recovery.SCHEMA,'software_fixture':False,'ok':True,'nonce':'nonce','request_sha256':d.digest(request),
               'recipe_sha256':d.digest(d.dumps(recipe)),'training_updates':0,'pid':123,'result':res,'elapsed_seconds':1}
        base={'pid':123,'tag':'epoch_02','free_gib':8.,'reserved_gib':6.,'peak_allocated_gib':7.,'peak_reserved_gib':12.,'allocator_ooms':0}
        mem=[base|{'phase':'before_model_load'},base|{'phase':'model_loaded'}]
        for i in range(24):mem += [base|{'phase':'before_answer','completed':i,'answer_id':f'q{i}'},base|{'phase':'answer_saved_cache_released','completed':i+1,'answer_id':f'q{i}'}]
        att={'software_fixture':False,'cuda_calls':1,'math_fallback_allowed':False,'selected_backend':'EFFICIENT_ATTENTION'}
        files={name:d.dumps(reply),name.replace('.result.json','.json'):request,name.replace('.result.json','.memory.json'):d.dumps(mem),
               name.replace('.result.json','.attention_path.json'):d.dumps(att),name.replace('.result.json','.allocator_policy.json'):d.dumps({'allocator_policy_applied':True,'fraction':.8})}
        with self.zip(files) as z:self.assertEqual(r.verify_worker(z,name,c,recipe,rows,0)['new_answers'],24)
        changed=rows.replace(b'q10',b'other_id')
        with self.zip(files) as z,self.assertRaises(ValueError):r.verify_worker(z,name,c,recipe,changed,0)
        with self.zip(files|{name.replace('.result.json','.memory.json'):d.dumps(mem[:-1])}) as z,self.assertRaises(ValueError):r.verify_worker(z,name,c,recipe,rows,0)

    def test_another_upload_is_rejected_before_opening_prior_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'other.zip';p.write_bytes(b'different upload')
            with self.assertRaises(ValueError):r.audit(p,Path(tmp)/'absent.zip',Path(tmp)/'out')


if __name__=='__main__':unittest.main()
