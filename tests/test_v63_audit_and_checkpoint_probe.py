import unittest, tempfile, json
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from wengua_course import data as d
from research_audits import v63_checkpoint_probe as probe, v63_review as review


class V63ReviewBoundary(unittest.TestCase):
    def row(self, name):
        return {'id': name, 'split': 'validation', 'answer': 'source fixture answer',
                'messages': [{'role': 'user', 'content': 'fixture source question'},
                             {'role': 'assistant', 'content': 'source fixture answer'}]}

    def reply(self, row):
        question = row['messages'][:-1]
        return {'id': row['id'], 'question': question, 'expected': row['answer'], 'answer': 'original answer',
                'prompt_sha256': d.digest(json.dumps(question, ensure_ascii=False, separators=(',', ':')))}

    def test_all_eight_saved_checkpoints_and_official_are_declared(self):
        self.assertEqual(len(probe.LABELS), 9)
        self.assertEqual(set(probe.SELECTION.values()),
                         {('lr_5em05', i) for i in [1, 2, 4, 8]} | {('lr_1em04', i) for i in [1, 2, 4, 7]})

    def test_development_and_rag_prompts_remain_exact(self):
        pack = {'records': [self.row('dev:' + str(i)) for i in range(20)],
                'actual_rag_rows': [self.row('rag:' + str(i)) for i in range(18)]}
        before = deepcopy(pack); output = probe.build_prompts(pack)
        self.assertEqual(output['records'], pack['actual_rag_rows'] + pack['records'])
        self.assertEqual(pack, before); self.assertEqual(output['weight_updates'], 0)
        self.assertTrue(output['inference_only']); self.assertFalse(output['unseen_case_or_prediction_benchmark'])
        pack['actual_rag_rows'][0]['id'] = 'dev:0'
        with self.assertRaises(ValueError): probe.build_prompts(pack)

    def test_cache_checks_prompt_teacher_and_serialization(self):
        gold = self.row('fixture'); reply = self.reply(gold)
        self.assertTrue(probe.verify_cached([reply], [gold]))
        bad = deepcopy(reply); bad['expected'] = 'a model invented teacher'
        with self.assertRaises(ValueError): probe.verify_cached([bad], [gold])
        bad = deepcopy(reply); bad['question'][0] = dict(reversed(list(bad['question'][0].items())))
        with self.assertRaises(ValueError): probe.verify_cached([bad], [gold])
        with self.assertRaises(ValueError): probe.verify_cached([reply, reply], [gold])

    def test_changed_source_reuse_bytes_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'original.jsonl'; path.write_text('changed source')
            source = {'reuse_file': str(path), 'reuse_sha256': '0'*64}
            with self.assertRaises(ValueError): probe.stage_reuse(root, 'fixture', source, [], {})
            self.assertFalse((root / 'answers').exists())

    def test_other_archive_is_not_accepted_as_the_actual_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'other.zip'; path.write_bytes(b'not the uploaded run')
            with self.assertRaises(ValueError): review.audit(path, Path(tmp) / 'review')

    def test_default_export_identifies_checkpoint_probe(self):
        with patch('wengua_course.runner.export_review', return_value='fixture.zip') as export, patch('builtins.print'):
            self.assertEqual(probe.main(['export']), 0)
        self.assertTrue(export.call_args.args[1].name.startswith('Wengua_V63_checkpoint_probe_'))

    def test_worker_cannot_return_weight_updates(self):
        request = {'nonce': 'fixture', 'software_fixture': True}
        raw = b'fixture request'
        reply = {'schema': probe.SCHEMA, 'nonce': 'fixture', 'request_sha256': d.digest(raw),
                 'pid': 1, 'software_fixture': True, 'ok': True, 'result': {'weight_updates': 1}}
        with self.assertRaises(ValueError): probe.verify_reply(reply, request, raw)


if __name__ == '__main__': unittest.main()
