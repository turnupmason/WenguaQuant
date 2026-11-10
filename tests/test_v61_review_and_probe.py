import unittest,json
from copy import deepcopy
from research_audits.v61_probe import verify_cached,verify_reply,SCHEMA,result_contract
from research_audits.v61_review import UPLOAD_SHA
from research_audits.v60_review import prompt_sha
from wengua_course import data as d

class ReadonlyProbeTests(unittest.TestCase):
    def row(self):
        gold={'id':'test','messages':[{'role':'system','content':'按问题回答。'},{'role':'user','content':'A和B的关系？'},{'role':'assistant','content':'A生B。'}],'answer':'A生B。'}
        row={'id':'test','question':gold['messages'][:-1],'answer':'A生B。','expected':gold['answer'],'prompt_sha256':prompt_sha(gold['messages'][:-1]),'variant':'official_base'}
        return row,gold
    def test_exact_cached_prompt_and_target_required(self):
        row,gold=self.row();self.assertTrue(verify_cached([row],[gold],'official_base'))
        for field,value in [('expected','A克B。'),('question',[]),('prompt_sha256','0'*64),('variant','retained_v58')]:
            wrong=deepcopy(row);wrong[field]=value
            with self.assertRaises(ValueError):verify_cached([wrong],[gold],'official_base')
    def test_duplicate_or_foreign_question_rejected(self):
        row,gold=self.row()
        with self.assertRaises(ValueError):verify_cached([row,row],[gold])
        row['id']='another'
        with self.assertRaises(ValueError):verify_cached([row],[gold])
        self.assertFalse(verify_cached([],[gold]))
    def test_receipt_must_prove_no_weight_updates(self):
        req={'nonce':'nonce','software_fixture':True};raw=d.dumps(req).encode();reply={'schema':SCHEMA,'nonce':'nonce','request_sha256':d.digest(raw),'software_fixture':True,'pid':12,'ok':True,'result':{'weight_updates':0,'state':'complete'}}
        self.assertEqual(verify_reply(reply,req,raw)['weight_updates'],0)
        for field,value in [('nonce','changed'),('request_sha256','0'*64),('software_fixture',False),('pid',0)]:
            bad=deepcopy(reply);bad[field]=value
            with self.assertRaises(ValueError):verify_reply(bad,req,raw)
        bad=deepcopy(reply);bad['result']['weight_updates']=1
        with self.assertRaises(ValueError):verify_reply(bad,req,raw)
    def test_model_budget_and_full_messages_are_in_the_cache_contract(self):
        _,row=self.row();c={'model_id':'Qwen/Qwen3-8B','revision':'fixed','eval_max_new_tokens':384,'inference_max_context':4096};s={'adapter_sha256':None,'variant':'official_pinned_base'};a=result_contract(c,s,[row])
        self.assertNotEqual(a,result_contract(c|{'eval_max_new_tokens':256},s,[row]))
        self.assertNotEqual(a,result_contract(c,s|{'adapter_sha256':'f'*64},[row]))
        changed=deepcopy(row);changed['messages'][1]['content']='换了问题'
        self.assertNotEqual(a,result_contract(c,s,[changed]));self.assertEqual(len(UPLOAD_SHA),64)
if __name__=='__main__':unittest.main()
