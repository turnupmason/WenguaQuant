import tempfile,unittest
from copy import deepcopy
from pathlib import Path
from wengua_course import data as d
from research_audits import v65_review as review


class V65ActualReviewBoundaries(unittest.TestCase):
    def pair(self,key):
        gold={'id':key,'messages':[{'role':'user','content':'known question '+key},{'role':'assistant','content':'source answer'}],'answer':'source answer'}
        row={'id':key,'question':gold['messages'][:-1],'expected':'source answer','answer':'right beginning; wrong ending',
             'prompt_sha256':review.prompt_sha(gold['messages'][:-1]),'empty':False}
        return gold,row

    def test_other_archive_is_rejected_before_private_sources_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'other.zip';path.write_bytes(b'other upload')
            with self.assertRaises(ValueError):review.audit(path,Path(tmp)/'review')

    def test_reused_baseline_can_precede_new_answers_without_reordering_content(self):
        a,ar=self.pair('first');b,br=self.pair('second')
        self.assertEqual(review.align_answers([br,ar],[a,b]),[ar,br])
        for rows in [[ar,ar],[ar],[ar,br|{'id':'foreign'}],[ar,br|{'expected':'changed'}],[ar,br|{'question':[]}]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):review.align_answers(rows,[a,b])

    def test_whole_answer_observation_cannot_be_reused_for_an_edited_tail(self):
        g,r=self.pair('known');note={'answer_sha256':d.digest(r['answer']),'verdict':'contradictory_whole_answer',
            'bounded_manual_observation_only':True,'not_a_training_teacher':True}
        review.verified_notes({'T2':{'known':note}},'T2',[r])
        for bad in [r|{'answer':'right beginning'},r|{'id':'new question'}]:
            with self.assertRaises(ValueError):review.verified_notes({'T2':{'known':note}},'T2',[bad])

    def test_reading_inventory_exposes_uncovered_board_and_multiturn_checks(self):
        g,_=self.pair('source');g.update(split='train',skill='objects',answer='这里比较甲乙。')
        v=deepcopy(g);v.update(id='development',split='validation',skill='identity')
        pack={'records':[g,v],'units':{'source':{}},'summary':{'used_source_units':1,'case_linked_cores':0}}
        inventory=review.source_inventory(pack)
        self.assertEqual(inventory['comparison_explanation_phrase_count'],1)
        self.assertEqual(inventory['new_development_multiturn'],0)
        self.assertEqual(inventory['review_board_or_location_questions'],0)
        self.assertFalse(inventory['full_two_book_case_training'])
        self.assertTrue(inventory['template_frequency_does_not_prove_a_causal_training_defect'])


if __name__=='__main__':unittest.main()
