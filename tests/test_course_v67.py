import tempfile,unittest
from copy import deepcopy
from pathlib import Path
from wengua_course import data as d
from wengua_course_v65 import dataset as first
from wengua_course_v66 import dataset as second
from wengua_course_v67 import dataset as ds,evaluation
from wengua_course_v67.runner import checkpoint_scope,TRIAL
from tests.test_course_v65 import source_fixture


class V67IndependentSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source,boards=source_fixture();source['units']['base']['original_quote']+=' 死门、天芮俱是土。'
        for r in source['records']:
            for s in r['provenance']['support']:
                if s['source_id']=='base':s['quote']=source['units']['base']['original_quote']
        cls.first=first.build(source,boards);cls.source=second.build(cls.first);cls.pack=ds.build(cls.source)

    def test_all_two_premise_combinations_keep_false_and_unknown_distinct(self):
        found=set()
        for r in self.pack['records']:
            c=r['provenance'].get('lesson_contract',{})
            if r['skill']!='independent_conditions' or r['split']!='train':continue
            state=ds.condition_state(c);a=c['palace_relation'];b=c['gate_state']
            found.add((state['palace_requirement'],state['auspicious_requirement']))
            expected=False if a=='相克' or b=='明确不具备' else None if a=='未提供' or b=='未提供' else True
            self.assertIs(state['joint'],expected)
            if a=='相生':self.assertIn('两宫相生',r['answer']);self.assertNotIn('两宫相生没有成立',r['answer'])
        self.assertEqual(len(found),9)
        with self.assertRaises(ValueError):ds.condition_state({'palace_relation':'typo','gate_state':'已具备'})

    def test_fixed_star_properties_are_separate_from_palace_properties(self):
        for board in self.pack['physical_star_boards']:
            facts=ds.object_facts(board);self.assertEqual(facts['symbols']['天心星']['intrinsic_element'],'金')
            self.assertEqual(facts['symbols']['天芮星']['intrinsic_element'],'土')
            for goal in ds.GOALS:
                c={'kind':'typed_objects','board_id':board['base_id'],'goal':goal,'facts':facts}
                text=ds.answer(c,{board['base_id']:board});a,b,x,y=ds.objects(board,goal)
                self.assertIn(a+'属'+x,text);self.assertIn(b+'属'+y,text)

    def test_every_supervised_turn_has_visible_current_states_or_object_selection(self):
        for r in self.pack['records']:
            if r['provenance'].get('unchanged_v66_source_replay'):continue
            p=r['provenance'];users=[m for m in r['messages'] if m['role']=='user']
            self.assertEqual(len(users),len(p['turn_contracts']))
            for m,c in zip(users,p['turn_contracts'],strict=True):
                self.assertEqual(ds.visible_question(c,m['content']),m['content'])
                if c['kind'] in ['independent_conditions','condition_claim']:
                    self.assertIn('两宫关系='+c['palace_relation'],m['content']);self.assertIn('吉门吉格='+c['gate_state'],m['content'])

    def test_old_sources_records_case_splits_and_replays_remain_exact(self):
        self.assertEqual(self.pack['units'],self.source['units']);self.assertEqual(self.pack['case_splits'],self.source['case_splits'])
        self.assertEqual(self.pack['prior_records'],self.source['records'])
        old={r['id']:r for r in self.source['records']}
        for r in self.pack['records']:
            name=r['provenance'].get('unchanged_v66_source_replay')
            if name:self.assertEqual(r['messages'],old[name]['messages'])

    def test_wrong_facts_transcription_hidden_state_and_future_teacher_are_rejected(self):
        for kind in ['typed','visible','history','source','board']:
            p=deepcopy(self.pack)
            if kind=='typed':
                r=next(r for r in p['records'] if r['skill']=='typed_objects');r['provenance']['lesson_contract']['facts']['symbols']['天心星']['intrinsic_element']='木'
            elif kind=='visible':
                r=next(r for r in p['records'] if r['skill']=='independent_conditions');r['messages'][-2]['content']='只问能成吗？'
                r['provenance']['visible_user_sha256']=d.digest(d.dumps([m['content'] for m in r['messages'] if m['role']=='user']))
            elif kind=='history':
                r=next(r for r in p['records'] if r['provenance'].get('initial_chain'));r['messages'][2]['content']='book never said this'
            elif kind=='source':next(iter(p['units'].values()))['original_quote']='changed'
            else:p['physical_star_boards'][0]['stars']['天心星']='乾' if p['physical_star_boards'][0]['stars']['天心星']!='乾' else '坎'
            with self.subTest(kind=kind),self.assertRaises(ValueError):ds.validate(p,self.source)

    def test_single_slot_updates_preserve_other_known_slot_and_both_claim_classes(self):
        updates=[r for r in self.pack['records'] if r['skill']=='independent_conditions' and r['provenance'].get('initial_contract')]
        self.assertEqual(len(updates),36)
        for r in updates:
            a=r['provenance']['initial_contract'];b=r['provenance']['lesson_contract']
            self.assertEqual(sum(a[k]!=b[k] for k in ['palace_relation','gate_state']),1)
        claims=[r['provenance']['lesson_contract'] for r in self.pack['records'] if r['skill']=='condition_claim']
        self.assertEqual(len(claims),24)

    def test_parent_retention_precedes_new_questions_and_new_turns_use_own_answer(self):
        pack=evaluation.build(self.pack,self.first)
        static=pack['retention']['records'];answers=[]
        count=pack['summary']['retention_answers']
        while len(answers)<count:
            row=evaluation.sequence(pack,answers)[len(answers)];answers.append({'id':row['id'],'answer':'actual previous answer'})
        rows=evaluation.sequence(pack,answers);self.assertEqual(rows[count]['id'],pack['new']['records'][0]['id'])
        answers += [{'id':r['id'],'answer':'actual static answer'} for r in pack['new']['records']]
        row=evaluation.sequence(pack,answers)[len(answers)];answers.append({'id':row['id'],'answer':'actual wrong initial answer'})
        nextrow=evaluation.sequence(pack,answers)[len(answers)]
        self.assertEqual(nextrow['messages'][2]['content'],'actual wrong initial answer');self.assertFalse(nextrow['provenance']['gold_prior_answers_injected'])

    def test_development_compares_same_palaces_in_raw_and_fact_card_inputs(self):
        grouped={}
        for r in self.pack['records']:
            c=r['provenance'].get('lesson_contract',{})
            if r['split']=='validation' and c.get('kind')=='typed_objects' and c['goal']=='palaces':
                grouped.setdefault(c['board_id'],[]).append(r)
        self.assertEqual(len(grouped),8)
        for rows in grouped.values():
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['answer'],rows[1]['answer'])
            self.assertEqual(sum('程序位置：' in r['messages'][-2]['content'] for r in rows),1)

    def test_missing_current_chart_or_month_is_not_filled_from_a_general_rule(self):
        c={'kind':'source_scope','symbol':'时干','role':'丢失物'}
        text=ds.answer(c,{})
        self.assertIn('时干代表丢失物',text);self.assertIn('不能据此确认当前落宫',text)
        bad=deepcopy(self.pack);bad['new_books_used']=True
        with self.assertRaises(ValueError):ds.validate(bad,self.source)

    def test_resume_does_not_adopt_previous_version_or_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'trials'/TRIAL/'cycle_01/trainer/checkpoint-20';p.mkdir(parents=True)
            self.assertEqual(checkpoint_scope(root,p),p.resolve())
            p=root/'trials'/TRIAL/'smoke/trainer/checkpoint-2';p.mkdir(parents=True)
            with self.assertRaises(ValueError):checkpoint_scope(root,p)


if __name__=='__main__':unittest.main()
