"""Inference app contracts; synthetic fixtures are not Qimen quality evidence."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import ast, http.server, json, tempfile, threading, time, unittest
from wengua_assistant import prompt as p, inference as g, server, service


class CharacterTokenizer:
    """Cheap exact budgeting fixture with no model or book dependency."""
    def apply_chat_template(self, messages, **kw):
        return list(range(sum(len(m['content']) + len(m['role']) + 1 for m in messages)))


class AssistantTimeAndFacts(unittest.TestCase):
    def test_naive_beijing_time_becomes_aware_and_existing_offset_is_checked(self):
        self.assertEqual(p.aware_time('2009-07-18T09:45', 'Asia/Shanghai'), '2009-07-18T09:45:00+08:00')
        self.assertEqual(p.aware_time('2009-07-18T09:45+08:00', 'Asia/Shanghai'), '2009-07-18T09:45:00+08:00')
        with self.assertRaises(ValueError): p.aware_time('2009-07-18T09:45-04:00', 'Asia/Shanghai')

    def test_dst_gap_and_fold_are_rejected_but_explicit_fold_is_preserved(self):
        for t in ['2026-03-08T02:30', '2026-11-01T01:30']:
            with self.assertRaises(ValueError): p.aware_time(t, 'America/New_York')
        for offset in ['-04:00', '-05:00']:
            self.assertTrue(p.aware_time('2026-11-01T01:30' + offset, 'America/New_York').endswith(offset))

    def test_blank_birth_is_not_unknown_and_known_data_are_not_imputed(self):
        self.assertEqual(p.validate_facts({}), {})
        self.assertEqual(p.validate_facts({'birth_status': 'not_entered', 'birth_stem': ''}), {})
        self.assertEqual(p.validate_facts({'birth_status': 'unknown'}), {'birth_status': 'unknown'})
        self.assertEqual(p.validate_facts({'birth_status': 'known', 'birth_stem': '癸'}), {'birth_status': 'known', 'birth_stem': '癸'})
        for bad in ['甲子', '2009', '']:
            with self.assertRaises(ValueError): p.validate_facts({'birth_status': 'known', 'birth_stem': bad})
        with self.assertRaises(ValueError): p.validate_facts({'outcome': 'future'})

    def test_unknown_model_or_changed_adapter_cannot_silently_be_substituted(self):
        with self.assertRaises(ValueError): g.adapter_receipt('best')
        with tempfile.TemporaryDirectory() as temp, patch.object(g, 'ADAPTER', Path(temp)):
            for name in g.ADAPTER_SHA: (Path(temp) / name).write_bytes(b'changed')
            with self.assertRaises(ValueError): g.adapter_receipt('v67_e1')
        self.assertNotIn('adapter', g.adapter_receipt('official'))

    def test_inference_modules_do_not_invoke_training(self):
        for module in [g, service]:
            tree = ast.parse(Path(module.__file__).read_text(encoding='utf8'))
            calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            for name in ['train', 'fit', 'backward', 'training_step']: self.assertNotIn(name, calls)
        self.assertNotIn('snapshot_download', Path(g.__file__).read_text(encoding='utf8'))

    def test_windows_entry_uses_existing_environment_and_no_shell_profile_syntax(self):
        text = (g.ROOT / 'start_wengua_assistant.cmd').read_text(encoding='utf8')
        self.assertIn('cd /d "%~dp0"', text)
        self.assertIn('qwen_env\\Scripts\\python.exe', text)
        self.assertIn('-X utf8 -m wengua_assistant.server', text)
        self.assertNotIn('git reset', text)


class AssistantWebBoundaries(unittest.TestCase):
    def setUp(self):
        class FixtureService:
            def submit(self, value): return {'job': 'fixture', 'session': 'fixture'}
            def poll(self, jid): return {'state': 'complete'}
        self.http = http.server.ThreadingHTTPServer(('127.0.0.1', 0), server.handler(FixtureService(), 't'*64))
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True); self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.http.server_port)

    def tearDown(self): self.http.shutdown(); self.http.server_close(); self.thread.join()

    def test_static_page_and_only_same_origin_token_api(self):
        with urlopen(self.url) as response:
            self.assertIn(b't'*64, response.read())
            self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])
        for headers in [{}, {'X-Wengua-Token': 'wrong'}]:
            with self.assertRaises(HTTPError) as error: urlopen(Request(self.url + '/api/job?id=fixture', headers=headers))
            self.assertEqual(error.exception.code, 403)
        with urlopen(Request(self.url + '/api/job?id=fixture', headers={'X-Wengua-Token': 't'*64})) as response:
            self.assertEqual(json.load(response)['state'], 'complete')
        for origin in ['https://example.com', None]:
            headers = {'X-Wengua-Token': 't'*64, 'Content-Type': 'application/json'}
            if origin: headers['Origin'] = origin
            with self.assertRaises(HTTPError) as error: urlopen(Request(self.url + '/api/ask', data=b'{}', headers=headers))
            self.assertEqual(error.exception.code, 403)
        headers = {'X-Wengua-Token': 't'*64, 'Content-Type': 'application/json', 'Origin': self.url}
        with urlopen(Request(self.url + '/api/ask', data=b'{}', headers=headers)) as response:
            self.assertEqual(response.status, 202)

    def test_rebinding_host_and_path_traversal_are_refused(self):
        with self.assertRaises(HTTPError) as error: urlopen(Request(self.url, headers={'Host': 'attacker.invalid'}))
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error: urlopen(Request(self.url + '/../../AGENTS.md', headers={'X-Wengua-Token': 't'*64}))
        self.assertEqual(error.exception.code, 404)


class AssistantHistoryAndBudget(unittest.TestCase):
    def _old_prompt_module(self):
        # The frozen App source is an optional read-only dependency in CI.
        from wengua_course.runtime import configure
        try: configure()
        except ValueError: self.skipTest('Frozen App source not installed in this software-only job')
        from src.wengua import book_rag_query
        return book_rag_query

    def _found(self):
        source = {'book_title': 'synthetic', 'chapter': 'fixture', 'lines': [1, 1]}
        return {'mode': 'knowledge', 'documents': [{'id': 'synthetic:rule', 'kind': 'rule', 'phase': 'general_rule',
                'sources': [source], 'content': 'synthetic rule', 'title': 'fixture'}],
                'domain_candidates': [], 'workflow_candidates': [], 'all_workflow_rules_loaded': False}

    def test_own_history_citations_and_budget_preserve_question_literals(self):
        self._old_prompt_module()
        from wengua_assistant import sources
        history = [{'role': 'user', 'content': 'earlier source [S1]'}, {'role': 'assistant', 'content': 'actual reply [S1]'}]
        refs = [{'citation': 'S1', 'id': 'old'}]; tok = CharacterTokenizer()
        with patch.object(sources, 'retrieve', return_value=self._found()):
            result = p.build('用户原话 [S1] 不得改写', tok, history=history, previous_refs=refs, max_new_tokens=30)
        self.assertEqual(result['messages'][1:3], history)
        self.assertIn('用户原话 [S1] 不得改写', result['messages'][-1]['content'])
        self.assertIn('[S2] 《synthetic》', result['messages'][-1]['content'])
        self.assertEqual([x['citation'] for x in result['references']], ['S2'])
        self.assertEqual(result['previous_references_not_current_evidence'], refs)
        self.assertEqual(result['prompt_budget']['input_tokens'], len(tok.apply_chat_template(result['messages'])))
        self.assertEqual(refs, [{'citation': 'S1', 'id': 'old'}])

    def test_too_long_actual_history_is_not_silently_truncated(self):
        self._old_prompt_module()
        from wengua_assistant import sources
        history = [{'role': 'user', 'content': 'x'*4000}, {'role': 'assistant', 'content': 'actual reply'}]
        with patch.object(sources, 'retrieve', return_value=self._found()), self.assertRaises(ValueError):
            p.build('continue', CharacterTokenizer(), history=history, max_new_tokens=30)

    def test_source_content_is_never_rewritten_when_numbering_new_turn(self):
        self._old_prompt_module()
        from wengua_assistant import sources
        found = self._found(); quote = '[S1] 《synthetic》原文内部的标记\n不得改动这个片段。'
        found['documents'][0]['content'] = quote
        with patch.object(sources, 'retrieve', return_value=found):
            result = p.build('question', CharacterTokenizer(), previous_refs=[{'citation': 'S1', 'id': 'old'}], max_new_tokens=20)
        self.assertTrue(result['messages'][-1]['content'].endswith(quote))
        self.assertEqual(result['references'][0]['content'], quote)
        self.assertEqual(result['references'][0]['citation'], 'S2')

    def test_bad_role_or_teacher_history_order_is_rejected(self):
        self._old_prompt_module()
        with self.assertRaises(ValueError): p.build('continue', CharacterTokenizer(), history=[{'role': 'assistant', 'content': 'teacher'}])


class AssistantExports(unittest.TestCase):
    def test_export_contains_only_this_sessions_trace_and_no_weights(self):
        import io, zipfile
        with tempfile.TemporaryDirectory() as temp:
            app = service.Service(); app.root = Path(temp)
            app.sessions = {'one': {'busy': False}, 'two': {'busy': False}}
            app.jobs = {'first': {'session': 'one'}, 'second': {'session': 'two'}}
            for jid in ['first', 'second']:
                folder = app.root / 'jobs' / jid; folder.mkdir(parents=True)
                g.write(folder / 'trace.json', {'session': jid})
                (folder / 'adapter_model.safetensors').write_bytes(b'never export weights')
            with zipfile.ZipFile(io.BytesIO(app.export('one'))) as archive:
                self.assertEqual(set(archive.namelist()), {'manifest.json', 'first/trace.json'})
                self.assertFalse(json.loads(archive.read('manifest.json'))['weights_included'])
            with self.assertRaises(ValueError): app.export('../two')
            app.sessions['one']['busy'] = True
            with self.assertRaises(ValueError): app.export('one')
            app.close()


if __name__ == '__main__': unittest.main()
