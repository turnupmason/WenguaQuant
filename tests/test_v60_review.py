import copy
import json
import unittest
from research_audits.v60_review import verify_reply, verify_generation, prompt_sha
from wengua_course import data as d


class Review60Tests(unittest.TestCase):
    def worker_pair(self):
        request={'nonce':'synthetic-only','software_fixture':False}
        raw=json.dumps(request).encode()
        reply={'request_sha256':d.digest(raw),'nonce':request['nonce'],'pid':8,
               'software_fixture':False,'ok':True,'result':{'state':'complete'}}
        return request,raw,reply

    def test_worker_requires_bound_real_completed_receipt(self):
        request,raw,reply=self.worker_pair();verify_reply(reply,request,raw)
        for field,value in [('nonce','changed'),('request_sha256','0'*64),('pid',True),
                            ('pid',0),('software_fixture',True),('ok',False)]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):verify_reply(dict(reply,**{field:value}),request,raw)

    def test_unfinished_result_and_fixture_request_stay_distinct(self):
        request,raw,reply=self.worker_pair()
        with self.assertRaises(ValueError):verify_reply(dict(reply,result={'state':'paused'}),request,raw)
        request['software_fixture']=True
        with self.assertRaises(ValueError):verify_reply(reply,request,raw)

    def fixture(self):
        messages=[{'content':'synthetic rule','role':'system'},{'content':'synthetic question','role':'user'},
                  {'content':'synthetic answer','role':'assistant'}]
        gold=[{'id':'synthetic','messages':messages,'answer':'synthetic answer',
               'skill':'open','provenance':{}}]
        rows=[{'id':'synthetic','question':messages[:-1],'expected':'synthetic answer',
               'answer':'synthetic answer','prompt_sha256':prompt_sha(messages[:-1])}]
        proof={'adapter_sha256':'a'*64,'max_new_tokens':512,'eval_stage':'control',
               'ids':['synthetic'],'prompt_sha256':d.digest(d.dumps([messages]))}
        return rows,gold,proof

    def test_generation_checks_exact_input_weight_and_budget(self):
        rows,gold,proof=self.fixture()
        result=verify_generation(rows,gold,proof,adapter_sha='a'*64,token_budget=512,stage='control')
        self.assertTrue(result['complete']);self.assertFalse(result['professional_approval'])
        for field,value in [('adapter_sha256','b'*64),('max_new_tokens',768),('ids',[]),
                            ('eval_stage','final'),('prompt_sha256','0'*64)]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    verify_generation(rows,gold,dict(proof,**{field:value}),adapter_sha='a'*64,token_budget=512,stage='control')

    def test_tampered_question_or_target_and_missing_answers_rejected(self):
        rows,gold,proof=self.fixture()
        for field,value in [('prompt_sha256','0'*64),('expected','changed')]:
            changed=copy.deepcopy(rows);changed[0][field]=value
            with self.assertRaises(ValueError):verify_generation(changed,gold,proof,adapter_sha='a'*64,token_budget=512,stage='control')
        self.assertFalse(verify_generation([],gold,proof,adapter_sha='a'*64,token_budget=512,stage='control')['complete'])

    def test_prompt_digest_uses_actual_v60_message_serialization(self):
        message=[{'content':'synthetic','role':'user'}]
        self.assertEqual(prompt_sha(message),d.digest(json.dumps(message,ensure_ascii=False,separators=(',',':'))))
        self.assertNotEqual(prompt_sha(message),prompt_sha([{'role':'user','content':'synthetic'}]))


if __name__=='__main__':unittest.main()
