import unittest
from copy import deepcopy
from wengua_course_v61 import dataset as ds
from wengua_course_v61.runner import RATES
from wengua_course import data as d

class DirectDialogueContracts(unittest.TestCase):
    def pack(self):
        quote='日干为本人；代问父亲以年干为用，兼看乾宫。'
        rows=[]
        for split in ['train','validation']:
            q='本人问健康先取谁？' if split=='train' else '代问父亲先取谁？'
            a='本人求测先取日干，不能凭出生年补出日干。' if split=='train' else '子女代问父亲先取年干，兼看乾宫。'
            rows.append({'id':split,'answer':a,'messages':[{'role':'system','content':ds.SYSTEM},{'role':'user','content':q},{'role':'assistant','content':a}],'split':split,'source_ids':['s'],'case_ids':[],'group':'knowledge:s','provenance':{'support':[{'source_id':'s','quote':quote}],'source_was_train':split=='train','teacher_is_model_output':False}})
        return {'schema':ds.SCHEMA,'source_books':['kaiwu','shenqi'],'units':{'s':{'book_id':'shenqi','original_quote':quote}},'records':rows,'case_splits':{}}
    def test_direct_source_teacher_and_distinct_development(self):
        self.assertEqual(ds.validate_pack(self.pack()),{'train':1,'validation':1})
    def test_model_output_never_promoted_as_teacher(self):
        p=self.pack();p['records'][0]['provenance']['teacher_is_model_output']=True
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_original_validation_source_cannot_become_train(self):
        p=self.pack();p['records'][0]['provenance']['source_was_train']=False
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_literal_source_mismatch_rejected(self):
        p=self.pack();p['records'][0]['provenance']['support'][0]['quote']='日干为父亲。'
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_case_boundary_is_never_overridden(self):
        for phase in ['validation','sealed_test']:
            p=self.pack();p['records'][0]['case_ids']=['case'];p['case_splits']['case']=phase
            with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_bare_token_training_target_rejected(self):
        p=self.pack();r=p['records'][0];r['answer']=r['messages'][-1]['content']='日干'
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_duplicate_question_and_answer_mismatch_rejected(self):
        p=self.pack();p['records'][1]['messages']=deepcopy(p['records'][0]['messages']);p['records'][1]['answer']=p['records'][0]['answer']
        with self.assertRaises(ValueError):ds.validate_pack(p)
        p=self.pack();p['records'][0]['answer']='不相同'
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_finite_loop_and_no_full_training_claim(self):
        self.assertEqual(RATES,(3e-6,1e-5))

if __name__=='__main__':unittest.main()
