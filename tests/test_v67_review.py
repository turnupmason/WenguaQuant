import tempfile,unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from research_audits import v67_review as r


class V67PartialActualReviewBoundaries(unittest.TestCase):
    def stage(self):
        fit={'state':'complete','steps':226,'planned_steps':226,'resume_checkpoint':None,'parent_weights_reused':True,
             'fresh_official_base':False,'source_variant':'v66_ordinary','optimizer_reset_between_epochs':False,
             'epochs':[{'steps':113,'epoch':1.},{'steps':226,'epoch':2.}]}
        state={'trial':{'fit':fit,'epoch_evaluations':{}},'state':'failed','active_epoch':1,'error':{'message':'CUDA error: out of memory'}}
        return state,fit

    def test_completed_training_is_distinct_from_incomplete_evaluation(self):
        state,fit=self.stage();r.verify_fit(state,fit)
        for change in [{'state':'complete_review_required'},{'active_epoch':2},{'error':{'message':'other failure'}}]:
            with self.subTest(change=change),self.assertRaises(ValueError):r.verify_fit(state|change,fit)

    def test_missing_or_different_epoch_cannot_be_hidden_by_low_loss(self):
        state,fit=self.stage()
        for change in [{'steps':113},{'epochs':[{'steps':113,'epoch':1.}]},{'optimizer_reset_between_epochs':True},{'fresh_official_base':True}]:
            wrong=fit|change;state['trial']['fit']=wrong
            with self.subTest(change=change),self.assertRaises(ValueError):r.verify_fit(state,wrong)

    def test_missing_future_answers_are_not_counted_as_correct_or_incorrect(self):
        row={'id':'q'};note={'whole_answer_supported':True}
        # observations deliberately summarizes only the actually present rows.
        rows=[{'id':str(i)} for i in range(124)];notes={'partial':{str(i):dict(note) for i in range(124)}}
        result=r.observations({'partial':rows},notes)['partial']
        self.assertFalse(result['all_requested_model_answers_present']);self.assertEqual(result['groups']['new_actual_history']['available'],0)
        self.assertEqual(result['groups']['new_objects']['available'],12);self.assertFalse(result['is_blind_case_accuracy'])

    def test_a_different_upload_is_rejected_before_source_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'other.zip';p.write_bytes(b'not this run')
            with self.assertRaises(ValueError):r.audit(p,Path(tmp)/'review')


if __name__=='__main__':unittest.main()
