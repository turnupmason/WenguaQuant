"""Source/data contracts only; real tensor tests run in check_training."""
import unittest,tempfile
from pathlib import Path
from copy import deepcopy
from wengua_course_v65 import dataset as old
from wengua_course_v66 import dataset as ds,evaluation
from wengua_course_v66.runner import checkpoint_scope
from tests.test_course_v65 import source_fixture


class V66SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source,boards=source_fixture();cls.source=old.build(source,boards);cls.pack=ds.build(cls.source)

    def test_source_registry_and_every_old_core_are_preserved(self):
        self.assertEqual(self.source['units'],self.pack['units']);self.assertEqual(self.source['case_splits'],self.pack['case_splits'])
        self.assertEqual(len(self.source['records']),len(self.pack['records']))
        self.assertTrue(all(r['split']=='train' for r in self.pack['records'] if r['pilot_selected']))
        self.assertLess(len(self.pack['pilot_ids']),self.pack['summary']['preserved_train_cores'])

    def test_all_marriage_supervised_turns_make_the_domain_and_rule_visible(self):
        count=0
        for r in self.pack['records']:
            if r['skill']!='marriage':continue
            for i,m in enumerate(r['messages']):
                if m['role']=='assistant':
                    visible=''.join(a['content'] for a in r['messages'][:i] if a['role']=='user')
                    self.assertIn('婚姻',visible);self.assertIn('并且逢吉门吉格',visible);count+=1
        self.assertGreater(count,40)

    def test_all_25_directed_laws_have_correct_fact_first_conclusions(self):
        generates={'木':'火','火':'土','土':'金','金':'水','水':'木'}
        controls={'木':'土','土':'水','水':'火','火':'金','金':'木'}
        for x in generates:
            for y in generates:
                text,clause=ds.direction('A','B',x,y)
                rel='比和' if x==y else '生' if generates[x]==y else '克' if controls[x]==y else '受生' if generates[y]==x else '受克'
                expected={'生':'A生B','克':'A克B','受生':'A受B所生','受克':'A受B所克','比和':'A与B比和'}[rel]
                self.assertEqual(clause,expected);self.assertIn('A属'+x,text);self.assertIn('B属'+y,text)

    def test_decisive_spans_are_literal_and_not_prompt_tokens(self):
        for r in self.pack['records']:
            spans=r['provenance'].get('decision_spans',[])
            for s in spans:self.assertEqual(r['answer'][s['start']:s['end']],s['text'])
        bad=deepcopy(self.pack);r=next(r for r in bad['records'] if r['skill']=='objects')
        r['messages'][-1]['content']='艮受乾所生。'
        with self.assertRaises(ValueError):ds.validate(bad,self.source)

    def test_critical_relation_and_condition_buckets_are_present_in_pilot(self):
        rows=[r for r in self.pack['records'] if r['pilot_selected']]
        relations=[r['provenance']['lesson_contract'] for r in rows if r['skill']=='relation']
        self.assertEqual(len({(c['x'],c['y']) for c in relations}),25)
        states={r['provenance']['lesson_contract']['state'] for r in rows if r['skill']=='marriage'}
        self.assertEqual(states,{'已具备','未提供','明确不具备'})

    def test_eval_has_explicit_scope_and_model_own_history_without_gold_injection(self):
        pack=evaluation.build(self.source);rows=evaluation.sequence(pack,[])
        self.assertEqual(len(rows),len(pack['records'])+4)
        first=next(r for r in rows if r['task']=='interactive_condition_update')
        answer={'id':first['id'],'answer':'软件夹具实际先前答复'}
        nextrow=next(r for r in evaluation.sequence(pack,[answer]) if r['id'].endswith('patient_takeover:1'))
        self.assertEqual(nextrow['messages'][2]['content'],answer['answer']);self.assertFalse(nextrow['provenance']['gold_prior_answers_injected'])
        for r in pack['records']:
            if r['provenance'].get('lesson_contract',{}).get('kind')=='marriage':self.assertIn('婚姻',r['messages'][-2]['content'])

    def test_source_target_and_split_mutations_are_rejected(self):
        for target in ['units','split','mask','pilot']:
            pack=deepcopy(self.pack)
            if target=='units':next(iter(pack['units'].values()))['original_quote']='changed'
            elif target=='split':pack['records'][0]['split']='test'
            elif target=='mask':pack['records'][0]['provenance']['decision_spans']=[{'start':0,'end':1,'text':'假'}]
            else:pack['pilot_ids']=[]
            with self.assertRaises(ValueError):ds.validate(pack,self.source)

    def test_resume_cannot_cross_version_or_adopt_smoke_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'trials/decision_ce/cycle_01/trainer/checkpoint-20';p.mkdir(parents=True)
            self.assertEqual(checkpoint_scope(root,p),p.resolve())
            wrong=root/'trials/decision_ce/smoke/trainer/checkpoint-2';wrong.mkdir(parents=True)
            with self.assertRaises(ValueError):checkpoint_scope(root,wrong)


if __name__=='__main__':unittest.main()
