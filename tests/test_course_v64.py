import tempfile,unittest,json
from copy import deepcopy
from pathlib import Path
from wengua_course import data as d
from wengua_course_v64 import dataset as ds,runtime,review,baselines
from wengua_course_v64.runner import checkpoint_scope


class V64ConditionalTruth(unittest.TestCase):
    def test_birth_and_available_year_do_not_replace_actual_patient_role(self):
        expected={'本人':'日干','父亲':'年干','母亲':'年干','同事':'月干','兄弟':'月干','儿子':'时干','女儿':'时干'}
        for role,field in expected.items():
            c={'kind':'identity','role':role,'fields':{'年干':'甲','月干':'丙','日干':'丁','时干':'戊'},'birth':'辛'}
            self.assertIn('起局的'+field,ds.answer(c));self.assertIn('年命辛',ds.answer(c))
            c['fields'][field]=None;c['birth']=None
            self.assertIn('具体'+field+'还没提供',ds.answer(c));self.assertIn('年命先留空',ds.answer(c))
        c={'kind':'identity','role':'本人','fields':{'年干':'甲'},'birth':'辛','focus':True}
        self.assertIn('起局的日干',ds.answer(c));self.assertNotIn('起局的年干',ds.answer(c))

    def test_explicit_objects_use_palaces_instead_of_the_star_element(self):
        c={'kind':'objects','heart':'乾','illness':'坤','scope':'palaces'}
        self.assertEqual(ds.answer(c),'土生金，所以天心所在的乾宫受天芮所在的坤宫所生。')
        self.assertIn('同属金',ds.answer(c|{'scope':'heart_and_palace'}))
        self.assertEqual(ds.answer(c|{'scope':'locations'}),'天心星当前在乾宫，天芮星当前在坤宫。')

    def test_correct_claim_is_not_trained_as_an_error(self):
        c={'kind':'verify_direction','a':'甲','b':'乙','x':'土','y':'金','claim':'生'}
        self.assertTrue(ds.answer(c).startswith('这句判断对。土生金'))
        self.assertTrue(ds.answer(c|{'claim':'克'}).startswith('这句判断不对。土生金'))
        self.assertIn('比和',ds.answer(c|{'x':'火','y':'火','claim':'比和'}))

    def test_door_terminology_keeps_control_direction(self):
        self.assertIn('门克宫，称门迫',ds.answer({'kind':'door','door':'景','palace':'乾'}))
        self.assertIn('宫克门，称门受制',ds.answer({'kind':'door','door':'开','palace':'离'}))
        text=ds.answer({'kind':'door','door':'休','palace':'坎'})
        self.assertEqual(text.count('比和'),1);self.assertNotIn('这里只',text)

    def test_marriage_unknown_false_and_date_are_distinct(self):
        c={'kind':'marriage','condition':'未提供','goal':'能否'}
        self.assertIn('需要核实',ds.answer(c));self.assertIn('不等于婚事必败',ds.answer(c))
        self.assertIn('不满足本条',ds.answer(c|{'condition':'明确不具备'}))
        for state in ['未提供','明确不具备','已具备']:
            text=ds.answer(c|{'condition':state,'goal':'日期'})
            self.assertIn('日期',text);self.assertNotIn('日干代表结婚年',text)

    def test_loan_birth_and_control_relations_are_not_reversed(self):
        for rel in ['值符克天乙','天乙生值符']:self.assertIn('对银行有利',ds.answer({'kind':'loan','relation':rel}))
        for rel in ['天乙克值符','值符生天乙']:self.assertIn('对银行不利',ds.answer({'kind':'loan','relation':rel}))
        self.assertIn('借贷人生银行',ds.answer({'kind':'loan','relation':'天乙生值符'}))
        self.assertIn('货恋人',ds.answer({'kind':'sale','relation':'时干生日干'}))
        self.assertNotIn('货恋人',ds.answer({'kind':'sale','relation':'时干克日干'}))
        self.assertIn('不准退役',ds.answer({'kind':'retirement','relation':'杜门生日干'}))

    def test_cross_domain_roles_and_symbol_types_remain_separate(self):
        text=ds.answer({'kind':'liuhe','domains':['婚姻','诉讼','交易求财']})
        for word in ['媒人','证人','中介人']:self.assertIn(word,text)
        text=ds.answer({'kind':'types'})
        self.assertIn('天心是九星',text);self.assertIn('乙奇是天干',text)
        self.assertIn('地盘之星',ds.answer({'kind':'source_fact','quote_requested':False,'role':'被告'}))


class V64EvidenceAndResume(unittest.TestCase):
    def source_and_pack(self):
        support=[{'source_id':'software','quote':'software-only source fixture'}]
        parent={'id':'train-source','split':'train','provenance':{'support':support}}
        source={'units':{'software':{'book_id':'shenqi','original_quote':support[0]['quote']}},'case_splits':{},'records':[parent]}
        c={'kind':'direction','a':'甲','b':'乙','x':'土','y':'金'};answer=ds.answer(c)
        row={'id':'lesson','source_ids':['software'],'split':'validation','answer':answer,'messages':[{'role':'system','content':ds.SYSTEM},{'role':'user','content':'software question'},{'role':'assistant','content':answer}],
             'provenance':{'support':support,'source_train_ids':['train-source'],'lesson_contract':c},'case_ids':[]}
        pack=source|{'schema':ds.SCHEMA,'source_books':['shenqi','kaiwu'],'physical_contexts':[],'records':[row]}
        return source,pack

    def test_source_model_and_score_contamination_is_rejected(self):
        source,pack=self.source_and_pack();self.assertTrue(ds.validate(pack,source))
        for name in ['old_manual_scores_used','model_outputs_used_as_teachers','original_final_test_used','new_books_used']:
            with self.subTest(name=name),self.assertRaises(ValueError):ds.validate(pack|{name:True},source)
        bad=deepcopy(pack);bad['records'][0]['provenance']['teacher_is_model_output']=True
        with self.assertRaises(ValueError):ds.validate(bad,source)
        bad=deepcopy(pack);bad['records'][0]['answer']='金克土'
        with self.assertRaises(ValueError):ds.validate(bad,source)

    def test_undeclared_manual_fields_are_not_allowed_in_placements(self):
        source,pack=self.source_and_pack()
        pack['physical_contexts']=[{'base_id':'fixture','heart':'乾','illness':'坤','door_relation_200':'manual'}]
        with self.assertRaises(ValueError):ds.validate(pack,source)

    def test_literal_span_must_bind_this_role_and_cannot_deny_visible_content(self):
        source,pack=self.source_and_pack();quote='software-only fixture:以开门为法官，以惊门为律师。'
        source['units']['software']['original_quote']=quote
        support=[{'source_id':'software','quote':quote}];source['records'][0]['provenance']['support']=support
        pack['units']=deepcopy(source['units']);row=pack['records'][0];row['provenance']['support']=deepcopy(support)
        c={'kind':'source_fact','domain_name':'诉讼','role':'律师','symbol':'惊门','literal':'以惊门为律师','quote_requested':False}
        row['provenance']['lesson_contract']=c;row['answer']=ds.answer(c);row['messages'][-1]['content']=row['answer']
        self.assertTrue(ds.validate(pack,source))
        c.update(symbol='开门',literal='以开门为法官');row['answer']=ds.answer(c);row['messages'][-1]['content']=row['answer']
        with self.assertRaises(ValueError):ds.validate(pack,source)
        c={'kind':'source_absence','domain_name':'诉讼','role':'律师','missing_term':'律师','needed':'原文'}
        row['provenance']['lesson_contract']=c;row['answer']=ds.answer(c);row['messages'][-1]['content']=row['answer']
        with self.assertRaises(ValueError):ds.validate(pack,source)

    def test_resume_preserves_every_complete_answer_and_rejects_wrong_question(self):
        row={'id':'fixture','messages':[{'role':'user','content':'question'},{'role':'assistant','content':'gold'}],'answer':'gold'}
        reply={'id':row['id'],'question':row['messages'][:-1],'expected':'gold','answer':'original full answer',
               'prompt_sha256':d.digest(json.dumps(row['messages'][:-1],ensure_ascii=False,separators=(',',':')))}
        raw=(json.dumps(reply,ensure_ascii=False,separators=(',',':'))+'\n').encode()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'answers.jsonl';p.write_bytes(raw+b'{"cut":"\xff')
            runtime.resume_generation_file(p,[row]);self.assertEqual(p.read_bytes(),raw)
            self.assertEqual(len(list(Path(tmp).glob('*.partial-*'))),1)
            with self.assertRaises(ValueError):runtime.resume_generation_file(p,[row|{'answer':'changed teacher'}])

    def test_xlsx_preserves_full_text_and_never_writes_a_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.xlsx';review.xlsx(path,[('课程',[['=not_a_formula','完整原文\n第二行']])])
            import zipfile
            with zipfile.ZipFile(path) as z:
                data=z.read('xl/worksheets/sheet1.xml')
                self.assertNotIn(b'<f>',data);self.assertIn(b'inlineStr',data)
            with self.assertRaises(ValueError):review.xlsx(path,[('课程',[['x'*32768]])])

    def test_explicit_checkpoint_is_never_silently_ignored_from_another_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);valid=root/'trials/lr_1em05/cycle_01/trainer/checkpoint-10';valid.mkdir(parents=True)
            self.assertEqual(checkpoint_scope(root,valid),valid)
            for wrong in [root/'trials/unknown/cycle_01/trainer/checkpoint-10',root/'trials/lr_1em05/smoke/trainer/checkpoint-2']:
                wrong.mkdir(parents=True)
                with self.assertRaises(ValueError):checkpoint_scope(root,wrong)
            with self.assertRaises(ValueError):checkpoint_scope(root,valid.parent/'checkpoint-11')


if __name__=='__main__':unittest.main()
