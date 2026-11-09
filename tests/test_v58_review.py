"""Post-run observations must stay bounded and preserve the frozen scores."""
import unittest
from wengua_course import data as d
from research_audits.v58_review import observation,validate_note


class Review58Tests(unittest.TestCase):
    def row(self,text):
        return {'id':'synthetic-note','answer':text,'question':[{'role':'user','content':'比较前者土与后者金。'}]}

    def test_correction_wording_records_conflict_without_regrading(self):
        r=self.row('应改记受生。土生金，前者生后者。')
        v=observation(r,{'checks':{'answer':'生'},'answer':'生。'})
        self.assertEqual(v['correction_head_conflicts'],['受生'])
        self.assertEqual(v['literal_law_conflicts_for_review'],[])
        self.assertTrue(v['posthoc_not_new_score'])
        self.assertFalse(v['professional_approval'])

    def test_literal_scan_excludes_idiom_and_keeps_wrong_law(self):
        r=self.row('土生土长。水生土。土生金。')
        v=observation(r,{'answer':'','checks':None})
        self.assertEqual(v['literal_law_conflicts_for_review'],['水生土'])

    def test_placeholder_is_observation_not_verdict(self):
        v=observation(self.row('孩子安全。〔待补局部判断〕'),{'answer':'安全。'})
        self.assertTrue(v['placeholder_retained'])
        self.assertNotIn('verdict',v)

    def test_unprovided_date_is_flagged_for_review_not_called_false(self):
        r=self.row('2009年8月24日。')
        self.assertEqual(observation(r,{'answer':''})['unprovided_absolute_dates_for_review'],['2009年8月24日'])
        r['question'][0]['content']+='已给日期2009年8月24日。'
        self.assertEqual(observation(r,{'answer':''})['unprovided_absolute_dates_for_review'],[])

    def test_manual_annotation_is_bound_to_exact_original(self):
        r=self.row('生。');note={'id':r['id'],'answer_sha256':d.digest(r['answer']),'verdict':'correct','reason':'限定主要关系，非专家认证。'}
        self.assertEqual(validate_note(note,r),note)
        for changed in [note|{'id':'other'},note|{'answer_sha256':'0'*64},note|{'reason':''}]:
            with self.assertRaises(ValueError):validate_note(changed,r)


if __name__=='__main__':unittest.main()
