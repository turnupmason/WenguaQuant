import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from wengua_course import data as d
from wengua_course_v65 import dataset as ds,protocol,runtime
from wengua_course_v65.runner import checkpoint_scope


def source_fixture():
    quotes={'base':'软件测试夹具，不是真实案例。天心星属金。[图7 后天九宫的五行] 乾金 坎水 坤土 巽木 离火 兑金 艮土 震木 中五土。',
            'ww':'戊/戊，甲甲比肩，名谓“伏吟”。','dr':'丁/壬，天盘丁，地盘壬，名“奇仪相合”格。','gw':'庚/戊，故名“值符伏宫”。'}
    units={key:{'id':key,'book_id':'shenqi','chapter':'software fixture','lines':[1,1],'original_quote':q} for key,q in quotes.items()}
    kinds=['identity','direction','door','objects','marriage','loan','sale','retirement','types','liuhe','exam']
    rows=[{'id':'fixture:'+k,'skill':k,'split':'train','source_ids':['base'],'provenance':{'support':[{'source_id':'base','quote':quotes['base']}]}} for k in kinds]
    src={'schema':ds.before.SCHEMA,'records':rows,'units':units,'source_books':['shenqi','kaiwu'],'case_splits':{},'old_manual_scores_used':False}
    outer=['坎','艮','震','巽','离','坤','兑','乾'];stars=['天蓬星','天任星','天冲星','天辅星','天英星','天芮星','天柱星','天心星'];boards=[]
    for i in range(8):
        p={s:outer[(j+i)%8] for j,s in enumerate(stars)};p['天禽星']=p['天芮星'];boards.append({'base_id':'software:'+str(i),'stars':p})
    return src,boards


class V65BindingAndGoal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.src,cls.boards=source_fixture();cls.pack=ds.build(cls.src,cls.boards)

    def test_relations_use_both_bound_attributes_and_reverse_the_subject(self):
        self.assertTrue(ds.binding_answer('艮宫','乾宫','土','金').startswith('艮宫生乾宫'))
        self.assertTrue(ds.binding_answer('乾宫','艮宫','金','土').startswith('乾宫受艮宫所生'))
        self.assertIn('金克木',ds.binding_answer('天心星','震宫','金','木'))
        self.assertIn('比和',ds.binding_answer('A','B','火','火'))

    def test_claim_verification_is_balanced_and_does_not_copy_the_wrong_label(self):
        rows=[r for r in self.pack['records'] if r['split']=='train' and r['skill']=='verify']
        yes=[r for r in rows if r['provenance']['lesson_contract']['claim']==ds.relation(r['provenance']['lesson_contract']['x'],r['provenance']['lesson_contract']['y'])]
        self.assertEqual(len(rows),200);self.assertEqual(len(yes),100)
        for r in rows:
            c=r['provenance']['lesson_contract']
            self.assertEqual(r['answer'].startswith('这个标注成立'),c['claim']==ds.relation(c['x'],c['y']))

    def test_actual_caller_role_and_birth_are_independent(self):
        fields={'年干':'甲','月干':'乙','日干':'辛','时干':'丁'}
        for role,field in {'本人':'日干辛','父亲':'年干甲','母亲':'年干甲','同事':'月干乙','兄弟':'月干乙','儿子':'时干丁','女儿':'时干丁'}.items():
            text=ds.identity_answer({'role':role,'fields':fields,'birth':'癸'})
            self.assertIn(field,text);self.assertIn('出生年命癸',text)
        self.assertNotIn('具体字未提供',ds.identity_answer({'role':'本人','fields':fields,'birth':None}))
        self.assertIn('具体字未提供',ds.identity_answer({'role':'本人','fields':{'年干':'甲'},'birth':'癸'}))

    def test_missing_false_and_true_conditions_do_not_collapse(self):
        yes=ds.answer({'kind':'marriage','state':'已具备','goal':'can'})
        unknown=ds.answer({'kind':'marriage','state':'未提供','goal':'can'})
        no=ds.answer({'kind':'marriage','state':'明确不具备','goal':'can'})
        self.assertIn('分支成立',yes);self.assertIn('需要核实',unknown);self.assertIn('分支不成立',no)
        self.assertNotEqual(unknown,no)
        self.assertIn('对银行不利',ds.answer({'kind':'loan','relation':'值符生天乙'}))
        self.assertIn('对银行有利',ds.answer({'kind':'loan','relation':'天乙生值符'}))
        self.assertIn('不准退役',ds.answer({'kind':'retirement','relation':'杜门生日干'}))

    def test_explicit_object_source_and_stem_pair_mutations_are_rejected(self):
        for kind,mutate in [('objects',lambda c:c.update(x='火' if c['x']!='火' else '水')),
                            ('formation',lambda c:c.update(upper='乙'))]:
            pack=deepcopy(self.pack);row=next(r for r in pack['records'] if r['skill']==kind);c=row['provenance']['lesson_contract'];mutate(c)
            row['answer']=ds.answer(c);row['messages'][-1]['content']=row['answer']
            with self.assertRaises(ValueError):ds.validate(pack,self.src)
        pack=deepcopy(self.pack);pack['physical_star_boards'][0]['score']=99
        with self.assertRaises(ValueError):ds.validate(pack,self.src)

    def test_current_turn_revises_condition_and_field_without_future_leakage(self):
        multi=[r for r in self.pack['records'] if r['provenance'].get('initial_contract')]
        self.assertTrue(any(r['skill']=='marriage' for r in multi))
        self.assertTrue(any(r['skill']=='objects' for r in multi))
        for r in multi:self.assertEqual(r['messages'][2]['content'],ds.answer(r['provenance']['initial_contract']))

    def test_training_and_generation_prefix_masks_must_match(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
                if add_generation_prompt:return [1,2,3]
                return [1,2,3,4,5]
        row={'id':'software','messages':[{'role':'user','content':'question'},{'role':'assistant','content':'answer'}],
             'encoded':{'input_ids':[1,2,3,4,5],'labels':[-100,-100,-100,4,5]}}
        self.assertEqual(protocol.verify([row],Tokenizer())['supervised_tokens'],2)
        for bad in [{'labels':[-100,-100,3,4,5]},{'input_ids':[1,2,3,4]}]:
            r=deepcopy(row);r['encoded'].update(bad)
            with self.assertRaises(ValueError):protocol.verify([r],Tokenizer())

    def test_resume_scope_cannot_adopt_smoke_or_another_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);valid=root/'trials/v63_trained/cycle_01/trainer/checkpoint-20';valid.mkdir(parents=True)
            self.assertEqual(checkpoint_scope(root,valid),valid.resolve())
            wrong=root/'trials/official_zero/smoke/trainer/checkpoint-2';wrong.mkdir(parents=True)
            with self.assertRaises(ValueError):checkpoint_scope(root,wrong)


if __name__=='__main__':unittest.main()
