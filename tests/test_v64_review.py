import tempfile,unittest
from pathlib import Path
from copy import deepcopy
from research_audits import v64_review as review
from wengua_course import data as d


class V64ActualReviewBoundaries(unittest.TestCase):
    def pair(self):
        g={'id':'source-question','messages':[{'role':'user','content':'which object?'},{'role':'assistant','content':'source answer'}],'answer':'source answer'}
        r={'id':g['id'],'question':g['messages'][:-1],'expected':g['answer'],'answer':'right start; wrong ending',
           'prompt_sha256':review.prompt_sha(g['messages'][:-1]),'empty':False}
        return g,r

    def test_reject_other_upload_before_reading_private_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'other.zip';p.write_bytes(b'different run')
            with self.assertRaises(ValueError):review.audit(p,Path(tmp)/'audit')

    def test_whole_questions_teachers_ids_and_order_must_match(self):
        g,r=self.pair();review.verify_answers([r],[g])
        for change in [{'expected':'rewritten source'},{'id':'other'},{'question':[]},{'prompt_sha256':'changed'},{'empty':True},{'answer':''}]:
            with self.subTest(change=change),self.assertRaises(ValueError):review.verify_answers([r|change],[g])
        with self.assertRaises(ValueError):review.verify_answers([],[g])

    def test_notes_bind_the_complete_answer_including_wrong_tail(self):
        g,r=self.pair();n={'answer_sha256':d.digest(r['answer']),'verdict':'contradictory_whole_answer',
                         'bounded_manual_observation_only':True,'not_a_training_teacher':True}
        notes={'P':{g['id']:n}};review.verified_notes(notes,'P',[r])
        with self.assertRaises(ValueError):review.verified_notes(notes,'P',[r|{'answer':'right start'}])
        for bad in [{'verdict':'professional_pass'},{'not_a_training_teacher':False},{'bounded_manual_observation_only':False}]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):review.verified_notes({'P':{g['id']:n|bad}},'P',[r])
        with self.assertRaises(ValueError):review.verified_notes({},'P',[r])

    def test_actor_ambiguity_is_a_data_finding_not_a_new_patient_label(self):
        base={'split':'train','answer':'source','skill':'identity','messages':[{'role':'system','content':''},{'role':'user','content':'这次关系是父亲，先取谁？'}],
              'provenance':{'initial_contract':{'kind':'identity','role':'父亲'}}}
        records=[base|{'id':'ambiguous'},base|{'id':'explicit','messages':[{'role':'system','content':''},{'role':'user','content':'我代父亲问病'}]},
                 base|{'id':'self','provenance':{'initial_contract':{'kind':'identity','role':'本人'}}},
                 base|{'id':'no-contract','provenance':{'initial_contract':None}}]
        pack={'records':records,'units':{'source':{}},'summary':{'training_source_unit_count':1,'case_linked_training_cores':0}}
        inventory=review.source_inventory(pack)
        self.assertEqual(inventory['family_label_without_explicit_proxy_train_ids'],['ambiguous'])
        self.assertTrue(inventory['ambiguity_requires_clarification_not_posthoc_teacher_relabel'])
        self.assertFalse(inventory['full_two_book_case_training'])


if __name__=='__main__':unittest.main()
