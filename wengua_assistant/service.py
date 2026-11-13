"""Serial local jobs, independent actual conversations, append-only traces."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io, json, os, subprocess, sys, threading, uuid, zipfile
from . import inference as g
from .prompt import build, cast, chart_fields, validate_facts


class Service:
    def __init__(self, tokenizer=None, launch=None):
        self.root = g.ROOT / 'local/assistant'
        self.tokenizer = tokenizer
        self.launch = launch or self._launch
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.sessions = {}; self.jobs = {}; self.lock = threading.Lock()
        self.stopping = threading.Event(); self.child = None

    def _tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(g.MODEL, revision=g.REVISION,
                                                          local_files_only=True, trust_remote_code=False)
        return self.tokenizer

    def submit(self, payload):
        if self.stopping.is_set(): raise ValueError('本地服务正在关闭')
        question = payload.get('question', '')
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError('请输入问题，最多4000字')
        facts = validate_facts(payload.get('facts', {}))
        config = {k: payload.get(k) for k in ('variant', 'mode', 'time', 'zone')}
        if config['variant'] not in {'official', 'v67_e1'} or config['mode'] not in {'knowledge', 'reading'}:
            raise ValueError('请选择模型与问答类型')
        domain = payload.get('domain') or None
        from .sources import corpus
        if domain is not None and domain not in corpus.r.DOMAINS: raise ValueError('领域不正确')
        with self.lock:
            sid = payload.get('session')
            if sid:
                if sid not in self.sessions: raise ValueError('对话不存在，请新开对话')
                session = self.sessions[sid]
                if session['config'] != config: raise ValueError('更换模型、时间或问答类型后请新开对话')
                if session.get('busy'): raise ValueError('上一条问题仍在处理')
            else:
                sid = uuid.uuid4().hex
                session = {'config': config, 'history': [], 'references': [], 'packet': None}
                self.sessions[sid] = session
            jid = uuid.uuid4().hex
            session['busy'] = True
            self.jobs[jid] = {'state': 'queued', 'session': sid}
            self.pool.submit(self._run, jid, sid, question.strip(), facts, domain)
            return {'job': jid, 'session': sid}

    def _run(self, jid, sid, question, facts, domain):
        session = self.sessions[sid]; folder = self.root / 'jobs' / jid
        folder.mkdir(parents=True)
        try:
            if self.stopping.is_set(): raise RuntimeError('本地服务已停止，原始记录保留')
            self.jobs[jid].update(state='running', phase='检索原书规则与领域流程')
            config = session['config']
            if session['packet'] is None and config['mode'] == 'reading':
                session['packet'] = cast(config['time'] or '', config['zone'] or '')
            receipt = g.adapter_receipt(config['variant'])
            prompt = build(question, self._tokenizer(), domain=domain, packet=session['packet'], facts=facts,
                           history=session['history'], previous_refs=session['references'])
            preview = {'chart': session['packet'], 'pillar_positions': chart_fields(session['packet']) if session['packet'] else None,
                       'references': prompt['current_references'], 'budget': prompt['prompt_budget'],
                       'retrieval_plan': prompt['retrieval_plan']}
            self.jobs[jid].update(preview=preview)
            if prompt['state'] != 'ready': raise ValueError('本轮未取到可完整放入上下文的来源，请缩小问题或新开对话')
            value = {'nonce': jid, 'variant': config['variant'], 'prompt': prompt, 'model': receipt,
                     'code_sha256': g.code_receipt()}
            path = folder / 'request.json'; g.write(path, value)
            result = self.launch(path)
            if result.get('model') != receipt: raise ValueError('实际模型与所选模型不一致')
            # Save the EXACT expanded prompt and actual reply for this model.
            actual_user = question + ('\n用户明确资料：' + json.dumps(facts, ensure_ascii=False) if facts else '')
            session['history'] += [{'role': 'user', 'content': actual_user},
                                   {'role': 'assistant', 'content': result['raw_answer']}]
            session['references'] += prompt['references']
            trace = {'question': question, 'session': sid, 'result': result, 'preview': preview,
                     'prompt_sha256': g.sha(path), 'training_updates': 0, 'added_to_training_or_rag': False}
            g.write(folder / 'trace.json', trace)
            self.jobs[jid].update(state='complete', result=result, trace=str(folder / 'trace.json'))
        except Exception as error:
            g.write(folder / 'error.json', {'error': str(error), 'type': type(error).__name__})
            self.jobs[jid].update(state='failed', error=str(error))
        finally:
            session['busy'] = False

    def _launch(self, path):
        folder = path.parent
        env = os.environ | {'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                            'PYTORCH_CUDA_ALLOC_CONF': 'backend:native,max_split_size_mb:128'}
        with (folder / 'worker.log').open('w', encoding='utf8') as log:
            with self.lock:
                if self.stopping.is_set(): raise RuntimeError('本地服务已停止')
                child = subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'wengua_assistant.inference', str(path)],
                                         cwd=g.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                self.child = child
            try:
                child.wait(timeout=1800)
            except BaseException:
                child.terminate()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                raise
            finally:
                with self.lock: self.child = None
        result_path = folder / 'result.json'
        if not result_path.exists(): raise RuntimeError('模型进程退出，查看 ' + str(folder / 'worker.log'))
        reply = json.loads(result_path.read_text(encoding='utf8'))
        if reply.get('nonce') != folder.name or reply.get('request_sha256') != g.sha(path):
            raise ValueError('模型回答与请求不一致')
        if child.returncode != 0 or not reply.get('ok'):
            raise RuntimeError(reply.get('error', '模型运行失败') + '；日志：' + str(folder / 'worker.log'))
        return reply['result']

    def poll(self, jid):
        with self.lock:
            if jid not in self.jobs: raise KeyError('任务不存在')
            result = dict(self.jobs[jid])
        progress = self.root / 'jobs' / jid / 'progress.json'
        if progress.exists(): result['progress'] = json.loads(progress.read_text(encoding='utf8'))
        return result

    def export(self, sid):
        with self.lock:
            if sid not in self.sessions: raise ValueError('对话不存在')
            if self.sessions[sid].get('busy'): raise ValueError('请等当前回答完成后再导出')
            jobs = [key for key, value in self.jobs.items() if value['session'] == sid]
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', json.dumps({'session': sid, 'jobs': jobs, 'training_updates': 0,
                              'weights_included': False, 'contains_private_question_and_selected_evidence': True}, ensure_ascii=False))
            for jid in jobs:
                for name in ('request.json', 'result.json', 'trace.json', 'error.json', 'worker.log', 'attention_probe.json', 'attention_path.json'):
                    path = self.root / 'jobs' / jid / name
                    if path.is_file(): archive.write(path, jid + '/' + name)
        return content.getvalue()

    def close(self):
        self.stopping.set()
        with self.lock: child = self.child
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        self.pool.shutdown(wait=True, cancel_futures=True)
