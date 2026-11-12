import io,json,tempfile,unittest,zipfile
from copy import deepcopy
from pathlib import Path
from wengua_course import data as d
from research_audits import v66_review as review
from tests import test_v65_learning_review as previous_fixture


class V66ActualReviewBoundaries(unittest.TestCase):
    def fixture(self):
        target,encoded,row=previous_fixture.V65LearningActualReviewBoundaries().fixture()
        spans=[{'start':0,'end':1,'text':'克'}];target['provenance']={'decision_spans':spans};encoded['answer_start_character']=10
        for i,t in enumerate(row['tokens']):t['in_v66_critical_span']=i==0
        row.update(critical_spans=spans,critical_tokens=1,critical_mean_nll=.1,critical_gold_prefix_miss_indices=[],conditional_score_is_not_free_answer_accuracy=True)
        class Tok:
            def decode(self,ids):return 'software token'
        return target,encoded,row,Tok()

    def test_critical_masks_spans_and_means_are_independently_recomputed(self):
        target,encoded,row,tok=self.fixture();review.verify_critical(row,target,encoded,tok)
        for change in [{'critical_tokens':2},{'critical_mean_nll':.01},{'critical_gold_prefix_miss_indices':[0]},{'critical_spans':[]},{'conditional_score_is_not_free_answer_accuracy':False}]:
            with self.subTest(change=change),self.assertRaises(ValueError):review.verify_critical(row|change,target,encoded,tok)
        bad=deepcopy(row);bad['tokens'][1]['in_v66_critical_span']=True
        with self.assertRaises(ValueError):review.verify_critical(bad,target,encoded,tok)

    def test_gold_prefix_direction_score_is_not_free_answer_accuracy(self):
        target,encoded,row,_=self.fixture();row['id']='software:original'
        s=review.token_summary([row],{}, {row['id']:encoded})['all']
        self.assertEqual(s['direction_predicate_gold_prefix_misses'],0);self.assertFalse(s['is_free_answer_accuracy'])
        self.assertEqual(s['gold_prefix_top1_misses'],1)
        # A low direction-word NLL cannot certify another generated prefix.
        self.assertEqual(s['direction_predicate_mean_nll'],.1)

    def test_only_the_predeclared_training_loss_factor_may_differ(self):
        c={'learning_rate':1e-5,'decision_loss_factor':1,'parent':'same'}
        pair={'ordinary_ce':c,'decision_ce':c|{'decision_loss_factor':4}};review.matched_configs(pair)
        for key,value in [('learning_rate',3e-5),('parent','other'),('decision_loss_factor',8)]:
            bad=deepcopy(pair);bad['decision_ce'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):review.matched_configs(bad)

    def test_completed_raw_answers_and_token_bytes_are_sealed(self):
        answer=b'{"answer":"actual"}\n';tokens=b'{"score":0.1}\n';c={'fixed':True};stem='trials/baseline/generations/parent'
        receipt={'pack_sha256':'pack','config':c,'history_source':'model_own_previous_answers','professional_approved':False,'adapter_files':{'weights':'source'}}
        raw=d.dumps(receipt).encode();complete={'contract_sha256':d.digest(raw),'answer_file_sha256':d.digest(answer),'token_file_sha256':d.digest(tokens),
                    'answers':103,'critical_token_targets':38,'adapter_files':receipt['adapter_files']}
        def archive(tamper):
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w') as z:
                for name,payload in [('.receipt.json',raw),('.complete.json',d.dumps(complete)),('.jsonl',answer),('.tokens.jsonl',tokens if not tamper else b'changed')]:z.writestr(stem+name,payload)
            buf.seek(0);return zipfile.ZipFile(buf)
        with archive(False) as z:review.verify_completion(z,'baseline','parent','pack',c)
        with archive(True) as z,self.assertRaises(ValueError):review.verify_completion(z,'baseline','parent','pack',c)

    def test_condition_coverage_is_read_from_visible_lesson_contracts(self):
        def row(rel):return {'pilot_selected':True,'skill':'marriage','split':'train','messages':[{'role':'user'},{'role':'assistant'}],
                            'provenance':{'lesson_contract':{'palace_relation':rel,'state':'已具备'}}}
        pack={'records':[row('相生'),row('比和')],'units':{},'case_splits':{}}
        found=review.source_inventory(pack);self.assertTrue(found['first_condition_always_known_satisfied']);self.assertEqual(found['marriage_views'],2)
        pack['records'][0]=row('未知')
        self.assertFalse(review.source_inventory(pack)['no_first_condition_false_or_unknown_contrast'])

    def test_another_upload_is_rejected_before_source_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'other.zip';p.write_bytes(b'other run')
            with self.assertRaises(ValueError):review.audit(p,Path(tmp)/'audit')


if __name__=='__main__':unittest.main()
