"""Real tiny-Qwen/PEFT CPU generation through the app, not 8B validation."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json, os, subprocess, sys, tempfile, time
from . import inference as g, prompt, service, sources


def fixture_sources():
    return {'mode': 'knowledge', 'domain_candidates': [], 'workflow_candidates': [],
            'all_workflow_rules_loaded': False,
            'documents': [{'id': 'synthetic:book-rule', 'kind': 'rule', 'phase': 'general_rule',
                           'content': 'one two answer', 'title': 'synthetic source', 'retrieval_stage': 'fixture',
                           'sources': [{'book_title': 'synthetic fixture', 'chapter': 'software', 'lines': [1, 1],
                                        'method_id': 'wengua_shijia_zhuanpan'}]}]}


def cpu_reply(path):
    from wengua_course.runtime import configure
    configure()
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    from peft import get_peft_model, LoraConfig
    payload = json.loads(Path(path).read_text(encoding='utf8'))
    tok, model, _ = fixture_factory()
    if payload['variant'] == 'v67_e1':
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with attention_route(testing=True):
        result = g.decode(model, tok, payload['prompt'], testing=True)
    assert all((parameter == before[name]).all().item() for name, parameter in model.named_parameters())
    result['model'] = payload['model']; result['checks'] = g.assess(result['raw_answer'], payload['prompt'])
    g.write(Path(path).with_name('fixture_result.json'), result)
    return 0


def main():
    from wengua_course.runtime import configure
    configure()
    from wengua_course.fixtures import fixture_factory
    tok, _, _ = fixture_factory()
    if len(sys.argv) == 3 and sys.argv[1] == '--worker': return cpu_reply(sys.argv[2])
    (g.ROOT / 'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='assistant_cpu_', dir=g.ROOT / 'local') as temp:
        root = Path(temp); launches = []
        def launch(path):
            launches.append(path)
            subprocess.run([sys.executable, '-X', 'utf8', '-m', 'wengua_assistant.check_pipeline', '--worker', str(path)],
                           cwd=g.ROOT, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'}, check=True)
            return json.loads(path.with_name('fixture_result.json').read_text(encoding='utf8'))
        def fixture_receipt(variant):
            return {'variant': variant, 'software_fixture_only': True}
        original = prompt.build
        def short(*args, **kwargs): return original(*args, **kwargs, max_new_tokens=4)
        app = service.Service(tokenizer=tok, launch=launch); app.root = root / 'assistant'
        def retrieved(*args, **kwargs):
            found = deepcopy(fixture_sources())
            mode = args[2] if len(args) > 2 else kwargs.get('mode', 'knowledge')
            if mode == 'reading': found.update(mode='reading', domain_candidates=['health'])
            return found
        with patch.object(sources, 'retrieve', retrieved), \
             patch.object(g, 'adapter_receipt', fixture_receipt), patch.object(service, 'build', short):
            def ask(variant, session=None, reading=False):
                job = app.submit({'variant': variant, 'mode': 'reading' if reading else 'knowledge',
                                  'time': '2000-09-21T23:45' if reading else None,
                                  'zone': 'Asia/Shanghai' if reading else None, 'domain': 'health' if reading else None,
                                  'question': 'question one answer', 'session': session})
                for _ in range(600):
                    status = app.poll(job['job'])
                    if status['state'] in {'complete', 'failed'}: break
                    time.sleep(.1)
                assert status['state'] == 'complete', status
                return job, status
            first, a = ask('official'); second, b = ask('official', first['session'])
            third, c = ask('v67_e1')
            fourth, late = ask('official', reading=True)
            saved = json.loads(launches[1].read_text(encoding='utf8'))['prompt']
            assert saved['messages'][1]['content'] == 'question one answer'
            assert saved['messages'][2]['content'] == a['result']['raw_answer']
            assert saved['current_references'][0]['citation'] == 'S2'
            assert saved['previous_references_not_current_evidence'][0]['citation'] == 'S1'
            for answer in [a, b, c, late]:
                assert answer['result']['training_updates'] == 0 and answer['result']['all_parameters_frozen']
                trace = json.loads(Path(answer['trace']).read_text(encoding='utf8'))
                assert trace['added_to_training_or_rag'] is False
            assert first['session'] != third['session']
            final_prompt = json.loads(launches[-1].read_text(encoding='utf8'))['prompt']
            assert final_prompt['chart_receipt']['calendar']['four_pillars']['day'] == '癸未'
            assert final_prompt['chart_receipt']['time'] == '2000-09-21T23:45:00+08:00'
            assert '子初23点换日' in final_prompt['messages'][-1]['content']
        app.close()
    if os.name == 'nt':
        command = subprocess.run(['cmd.exe', '/d', '/c', str(g.ROOT / 'start_wengua_assistant.cmd'), '--help'],
                                 env=os.environ | {'WENGUA_PY': sys.executable}, capture_output=True,
                                 text=True, encoding='utf8', timeout=60)
        assert command.returncode == 0 and '--no-browser' in command.stdout, command.stdout + command.stderr
    proof = {'software_fixture_only': True, 'actual_tiny_qwen_generations': 4, 'separate_cpu_processes': 4,
             'actual_own_history': True, 'stable_source_identity': True, 'official_and_peft_paths': True,
             'actual_23h_chart_through_reading_prompt_and_generation': True,
             'parameters_unchanged': True, 'training_updates': 0, 'actual_8b_quality_verified': False,
             'native_windows_cmd_help': os.name == 'nt'}
    g.write(g.ROOT / 'local/assistant_cpu_verification.json', proof)
    print(json.dumps(proof)); return 0


if __name__ == '__main__': raise SystemExit(main())
