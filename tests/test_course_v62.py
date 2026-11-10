import unittest
from unittest.mock import patch
from copy import deepcopy
from wengua_course_v62 import dataset as ds
from wengua_course_v62.runner import RATES
from tests import test_course_v61 as original
from research_audits import v61_dialogue_review as review

class FreshSourceContracts(unittest.TestCase):
    def pack(self):
        p=original.DirectDialogueContracts().pack();p['schema']=ds.SCHEMA
        p['fresh_official_lora_required']=True;p['known_regression_rows']=[]
        for r in p['records']:r['source_mode']='two_book_source_application'
        return p
    def test_source_validation_is_retained(self):
        self.assertEqual(ds.validate_pack(self.pack()),{'train':1,'validation':1})
        p=self.pack();p['records'][0]['provenance']['support'][0]['quote']='虚构另一条规则'
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_old_initialization_and_model_teachers_rejected(self):
        p=self.pack();p['fresh_official_lora_required']=False
        with self.assertRaises(ValueError):ds.validate_pack(p)
        p=self.pack();p['records'][0]['provenance']['teacher_is_model_output']=True
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_known_generic_evaluation_cannot_enter_sft(self):
        p=self.pack();p['records'][0]['source_mode']='eval_only_no_training'
        with self.assertRaises(ValueError):ds.validate_pack(p)
        p=self.pack();p['known_regression_rows']=[deepcopy(p['records'][0])|{'not_training_or_rag_teacher':True}]
        with self.assertRaises(ValueError):ds.validate_pack(p)
    def test_auxiliary_condition_cannot_replace_branch_requirement(self):
        p=self.pack();p['records'][0]['provenance']['rule_structure']={'本条可成分支':['R1','R2'],'辅助观察':['R2']}
        with self.assertRaises(ValueError):ds.validate_pack(p)
        p['records'][0]['provenance']['rule_structure']['辅助观察']=['R3']
        ds.validate_pack(p)
    def test_two_rates_and_all_manual_reviews_have_one_note(self):
        self.assertEqual(RATES,(1e-5,2e-5))
        self.assertEqual(set(review.CORRECT),set(review.NOTES))
        for label in review.CORRECT:
            self.assertEqual(len(review.NOTES[label]),24)
            self.assertFalse(set(review.CORRECT[label]) & set(review.PARTIAL[label]))
            self.assertTrue(all(0<=i<24 for i in review.CORRECT[label]+review.PARTIAL[label]))
    def test_default_result_export_has_current_version_name(self):
        from wengua_course_v62.cli import main
        with patch('wengua_course.runner.export_review',return_value='fixture_archive') as export,patch('builtins.print'):
            self.assertEqual(main(['export']),0)
        self.assertTrue(export.call_args.args[1].name.startswith('Wengua_V62_review_'))

if __name__=='__main__':unittest.main()
