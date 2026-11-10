import unittest
from copy import deepcopy
from wengua_course import data as d
from research_audits import v62_probe_data as data,v62_probe as probe,v62_review as review

class ReadonlyInputControls(unittest.TestCase):
    def pack(self):
        rows=[]
        for skill in ['identity','direction','door_relation','branch_conditions','symbol_types','source_dialogue']:
            rows.append({'id':'fixture:'+skill,'skill':skill,'split':'train','question':'fixture '+skill,'answer':'fixture source answer','messages':[{'role':'system','content':'fixture_course_system'},{'role':'user','content':'fixture '+skill},{'role':'assistant','content':'fixture source answer'}],'source_ids':['s'],'provenance':{'support':[{'source_id':'s','quote':'literal fixture'}]}})
        rag=[{'id':'rag_fixture:'+str(i),'messages':[{'role':'user','content':'fixture RAG '+str(i)},{'role':'assistant','content':'fixture answer'}]} for i in range(18)]
        return {'records':rows,'actual_rag_rows':rag,'units':{'s':{'original_quote':'literal fixture','book_id':'shenqi'}},'software_fixture_only':True}
    def test_exact_six_training_prompts_are_not_rewritten(self):
        p=self.pack();result=data.build(p);self.assertEqual(len(result['records']),30)
        originals={r['id']:r for r in p['records']}
        for r in result['records'][24:]:
            old=originals[r['id'].removeprefix('v62_exact_teacher:')]
            self.assertEqual(r['messages'],old['messages']);self.assertEqual(r['answer'],old['answer'])
            self.assertEqual(r['split'],'diagnostic');self.assertTrue(r['not_training_or_rag_teacher'])
    def test_system_factor_preserves_the_complete_user_content(self):
        rs={r['id']:r for r in data.build(self.pack())['records']}
        for task in ['self','father','water_wood','marriage']:
            for fmt in ['plain','structured','source']:
                a=rs[f'v62_fit_probe:{task}:neutral:{fmt}'];b=rs[f'v62_fit_probe:{task}:course:{fmt}']
                self.assertEqual(a['question'],b['question']);self.assertEqual(a['answer'],b['answer'])
                self.assertNotEqual(a['messages'][0],b['messages'][0])
    def test_representation_preserves_the_question_and_target(self):
        rs={r['id']:r for r in data.build(self.pack())['records']}
        for task in ['self','father','water_wood','marriage']:
            for system in ['neutral','course']:
                a=rs[f'v62_fit_probe:{task}:{system}:plain'];b=rs[f'v62_fit_probe:{task}:{system}:structured']
                self.assertEqual(a['answer'],b['answer']);self.assertEqual(a['question'].split('所问：')[-1],b['question'].split('所问：')[-1])
                self.assertEqual(a['support'],b['support'])
    def test_existing_rag_input_is_exact_and_probe_is_readonly(self):
        p=self.pack();result=data.build(p)
        self.assertEqual(result['official_rag_rows'],p['actual_rag_rows']);self.assertEqual(result['weight_updates'],0)
        self.assertTrue(result['inference_only']);self.assertFalse(result['unseen_case_or_prediction_benchmark'])
        p['units']['s']['original_quote']='another source'
        with self.assertRaises(ValueError):data.build(p)
    def test_review_rejects_changed_question_target_or_sha(self):
        g={'messages':[{'role':'user','content':'question'},{'role':'assistant','content':'answer'}],'answer':'answer'}
        r={'question':g['messages'][:-1],'expected':'answer','answer':'actual','empty':False,'prompt_sha256':review.prompt_sha(g['messages'][:-1])}
        review.same_prompt(r,g)
        for key,value in [('expected','different'),('prompt_sha256','0'*64),('question',[])]:
            bad=deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):review.same_prompt(bad,g)
    def test_default_export_has_learning_probe_name(self):
        from unittest.mock import patch
        with patch('wengua_course.runner.export_review',return_value='fixture.zip') as export,patch('builtins.print'):
            self.assertEqual(probe.main(['export']),0)
        self.assertTrue(export.call_args.args[1].name.startswith('Wengua_V62_learning_probe_'))

if __name__=='__main__':unittest.main()
