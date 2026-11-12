import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from test_course_v65 import source_fixture
from wengua_course import data as d
from wengua_course_v65 import dataset as course
from research_audits import v65_learning_data as data,v65_learning_probe as probe,v65_token_diagnostics as token


class V65LearningProbeBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source,boards=source_fixture();cls.course=course.build(source,boards);cls.pack=data.build(cls.course)

    def test_originals_are_actual_training_messages_and_paraphrases_keep_targets(self):
        pack=self.pack;source={r['id']:r for r in self.course['records']};by={r['id']:r for r in pack['records']}
        self.assertEqual(pack['summary']['per_model_answers'],81);self.assertEqual(len(pack['pairs']),19)
        for pair in pack['pairs']:
            a,b=by[pair['original']],by[pair['paraphrase']]
            self.assertEqual(a['messages'],source[pair['anchor_training_id']]['messages'])
            self.assertEqual(a['answer'],b['answer']);self.assertNotEqual(a['messages'],b['messages'])
        self.assertFalse(pack['new_blind_benchmark']);self.assertFalse(pack['source_models_used_as_teachers'])

    def test_changed_teacher_pair_or_physical_star_data_is_rejected(self):
        changed=deepcopy(self.pack);row=changed['records'][1];row['provenance']['lesson_contract']['x']='火'
        with self.assertRaises(ValueError):data.validate(changed,self.course)
        changed=deepcopy(self.pack);row=next(r for r in changed['records'] if r['task']=='physical_star_reading');row['question']=row['question'].replace('天心星','天任星')
        with self.assertRaises(ValueError):data.validate(changed,self.course)

    def test_interactive_history_is_the_models_actual_reply_and_has_no_future_turn(self):
        pack=self.pack;static=[{'id':r['id'],'answer':'actual static reply'} for r in pack['records']]
        first=data.sequence(pack,static)[len(static)]
        self.assertEqual(len(first['messages']),3);self.assertNotIn('刚补充患者可靠出生年干癸',first['messages'][1]['content'])
        replies=static+[{'id':first['id'],'answer':'model actually gave a wrong first reply'}]
        second=data.sequence(pack,replies)[len(replies)]
        self.assertEqual(second['messages'][2]['content'],'model actually gave a wrong first reply')
        self.assertNotEqual(second['messages'][2]['content'],first['answer'])
        self.assertFalse(second['provenance']['gold_prior_answers_injected'])
        self.assertEqual(second['provenance']['previous_answers_sha256'],[d.digest(replies[-1]['answer'])])

    def test_cache_retains_complete_bytes_and_preserves_an_invalid_utf8_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'answers.jsonl';raw=b'{"id":"a","answer":"one"}\r\n';path.write_bytes(raw+b'{"bad":"\xff')
            self.assertEqual(probe.read_cache(path),[{'id':'a','answer':'one'}]);self.assertEqual(path.read_bytes(),raw)
            self.assertTrue(list(path.parent.glob('*.partial-*')))
            path.write_bytes(b'{not json}\n{"id":"a"}\n')
            with self.assertRaises(ValueError):probe.read_cache(path)

    def test_saved_prompt_target_and_own_history_cannot_change(self):
        target=self.pack['records'][0];answer='actual answer'
        row={'id':target['id'],'variant':'v65_epoch2','question':target['messages'][:-1],'expected':target['answer'],
             'answer':answer,'answer_sha256':d.digest(answer),'prompt_sha256':d.digest(d.dumps(target['messages'][:-1])),'provenance':target['provenance']}
        probe.validate_answers([row],self.pack,'v65_epoch2')
        for change in [{'expected':'wrong teacher'},{'answer':'different'},{'question':[]},{'variant':'official'}]:
            with self.subTest(change=change),self.assertRaises(ValueError):probe.validate_answers([row|change],self.pack,'v65_epoch2')

    def test_token_diagnostic_preserves_prefix_offsets_and_decision_boundary(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
                return ([1,2] if add_generation_prompt else [1,2,3,4,5,6,7]) if tokenize else ('assistant:' if add_generation_prompt else 'assistant:正。因。[EOS]')
            def __call__(self,text,**kw):return {'input_ids':[1,2,3,4,5,6,7],'offset_mapping':[(0,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,19)]}
        row={'messages':[{'role':'user','content':'q'},{'role':'assistant','content':'正。因。'}],'answer':'正。因。'}
        encoded=token.encoded_target(Tokenizer(),row,7)
        self.assertEqual(encoded['decision_mask'],[True,True,False,False,False])
        with self.assertRaises(ValueError):token.encoded_target(Tokenizer(),row,6)
        verify={'answer':'这个标注需要改正。甲方克乙方。依据水克火。','provenance':{'lesson_contract':{'kind':'verify'}}}
        segments=token.target_segments(verify)
        self.assertEqual(segments[1]['name'],'directed_relation');self.assertEqual(segments[1]['text'],'甲方克乙方。')

    def test_uploaded_source_receipt_mutation_cannot_silently_choose_another_model(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d.write(root/'run_status.json',{'state':'complete_review_required','production_enabled':False})
            d.write(root/'sweep_contract.json',{'schema':probe.runtime.SCHEMA,'code_sha256':{'code':'fixed'}})
            pins=root/'pins.json';d.write(pins,{'files':{n:d.digest((root/n).read_bytes()) for n in ['run_status.json','sweep_contract.json']}})
            with patch.object(probe,'private',lambda p:Path(p)),patch.object(probe,'PINS',pins),patch.object(probe.runtime,'hashes',lambda:{'code':'fixed'}):
                probe.verify_source_receipts(root)
                d.write(root/'run_status.json',{'state':'complete_review_required','production_enabled':True})
                with self.assertRaises(ValueError):probe.verify_source_receipts(root)

    def test_worker_receipt_cannot_claim_weight_updates_or_another_request(self):
        payload={'nonce':'nonce','software_fixture':True};raw=b'original request'
        reply={'schema':probe.SCHEMA,'nonce':'nonce','request_sha256':d.digest(raw),'software_fixture':True,'pid':12,'ok':True,'result':{'weight_updates':0}}
        probe.verify_reply(reply,payload,raw)
        for bad in [reply|{'nonce':'other'},reply|{'result':{'weight_updates':1}},reply|{'pid':False}]:
            with self.assertRaises(ValueError):probe.verify_reply(bad,payload,raw)


if __name__=='__main__':unittest.main()
