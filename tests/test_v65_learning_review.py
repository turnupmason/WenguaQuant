import tempfile,unittest
from copy import deepcopy
from pathlib import Path
from wengua_course import data as d
from research_audits import v65_learning_review as review


class V65LearningActualReviewBoundaries(unittest.TestCase):
    def fixture(self):
        target={'id':'software','answer':'克。因。','messages':[{'role':'user','content':'q'},{'role':'assistant','content':'克。因。'}]}
        segments=[{'name':'decision','start':0,'end':2,'text':'克。'},{'name':'explanation','start':2,'end':4,'text':'因。'}]
        encoded={'full':[1,2,3,4,5],'prefix':[1,2],'decision_mask':[True,False,False],
            'token_segment_indices':[0,1,None],'target_offsets':[(10,12),(12,14),(14,19)],'target_segments':segments}
        records=[{'target_id':t,'target_text':'software token','nll':loss,'gold_prefix_top1_id':pred,
             'gold_prefix_top1_equals_target':t==pred,'in_decision_sentence':m,'teacher_segment_index':j,'character_offsets':list(offset)}
             for t,loss,pred,m,j,offset in zip([3,4,5],[.1,2.,.1],[3,9,5],encoded['decision_mask'],encoded['token_segment_indices'],encoded['target_offsets'])]
        row={'id':target['id'],'target_sha256':d.digest(target['answer']),'prompt_sha256':d.digest(d.dumps(target['messages'][:-1])),
            'weight_updates':0,'conditional_on_gold_prefix':True,'free_generation_correctness_verified':False,'professional_probability':False,
            'segment_boundary_tokens_assigned_by_start_offset':True,'supervised_tokens':3,'decision_tokens':1,'tokens':records,
            'all_supervised_mean_nll':2.2/3,'decision_sentence_mean_nll':.1,'remaining_target_mean_nll':1.05,
            'teacher_segments':[segments[0]|{'supervised_tokens':1,'mean_nll':.1},segments[1]|{'supervised_tokens':1,'mean_nll':2.}]}
        return target,encoded,row

    def test_other_upload_rejected_before_loading_book_data_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'other.zip';path.write_bytes(b'another run')
            with self.assertRaises(ValueError):review.audit(path,Path(tmp)/'review')

    def test_token_labels_masks_offsets_and_means_are_recomputed(self):
        target,encoded,row=self.fixture();review.verify_token_row(row,target,encoded)
        for change in [{'all_supervised_mean_nll':.01},{'professional_probability':True},{'decision_tokens':2},{'weight_updates':1}]:
            with self.subTest(change=change),self.assertRaises(ValueError):review.verify_token_row(row|change,target,encoded)
        for change in [{'target_id':8},{'character_offsets':[0,1]},{'teacher_segment_index':None},{'in_decision_sentence':True},{'nll':float('nan')}]:
            r=deepcopy(row);r['tokens'][1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):review.verify_token_row(r,target,encoded)

    def test_top1_miss_cannot_have_target_probability_above_one_half(self):
        target,encoded,row=self.fixture();row['tokens'][1]['nll']=.1
        with self.assertRaises(ValueError):review.verify_token_row(row,target,encoded)

    def test_low_average_token_loss_is_not_declared_whole_answer_correctness(self):
        _,_,row=self.fixture();pack={'pairs':[{'original':'software','paraphrase':'software'}]}
        summary=review.token_summary([row],pack)['original']
        self.assertEqual(summary['gold_prefix_top1_errors'],1)
        self.assertEqual(summary['targets_with_any_gold_prefix_top1_error'],1)
        self.assertTrue(summary['conditional_on_gold_prefix_not_free_generation_accuracy'])

    def test_hidden_domain_or_future_teacher_word_does_not_supply_prompt_context(self):
        implicit={'id':'implicit','split':'train','skill':'marriage','domain':'marriage',
            'messages':[{'role':'user','content':'本条能成吗？'},{'role':'assistant','content':'婚姻分支成立。'}]}
        explicit=deepcopy(implicit);explicit.update(id='explicit');explicit['messages'][0]['content']='问婚姻，本条能成吗？'
        rows=[implicit,explicit];probe={'records':[{'id':'probe','messages':implicit['messages'],'provenance':{'lesson_contract':{'kind':'marriage'}}}]}
        found=review.source_scope_inventory({'records':rows},probe)
        self.assertEqual(found['marriage_supervised_views_without_explicit_visible_domain'],[{'id':'implicit','assistant_turn_index':1}])
        self.assertEqual(found['static_probe_marriage_questions_without_explicit_visible_domain'],['probe'])
        self.assertTrue(found['flags_are_review_candidates_not_silent_target_relabels'])


if __name__=='__main__':unittest.main()
