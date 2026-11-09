"""Windows/local entry for V6.0 source curriculum and bounded continuation."""
from argparse import ArgumentParser
from pathlib import Path
import re,time
from wengua_course import data as d
from . import runtime


def restore_all(key_file=None):
    from wengua_course_v59.runner import restore_all as restore_parent
    from wengua_corpus.__main__ import restore
    value=key_file.read_text(encoding='utf-8-sig').strip() if key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
    if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing 64-character delivery key')
    # Reuse the authenticated installer. A temporary key is removed even on
    # failure; it is never printed or added to an export.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='wengua_restore_') as tmp:
        key=Path(tmp)/'delivery.key';key.write_text(value,encoding='utf8');restore_parent(key)
    release=d.read(runtime.RELEASE)
    restore({'manifest':release['delivery_manifest'],'destination':runtime.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value))
    runtime.verify_data();print('V6.0课程、领域覆盖、SQLite与Excel已恢复并校验。')


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('command',choices=['restore','check','run','export','search','ask'])
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v60/run1');p.add_argument('--app-root',type=Path)
    p.add_argument('--hours',type=float,default=24);p.add_argument('--resume',action='store_true');p.add_argument('--resume-from-checkpoint',type=Path)
    p.add_argument('--key-file',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--question');p.add_argument('--domain')
    p.add_argument('--time');p.add_argument('--zone');p.add_argument('--chart',type=Path)
    a=p.parse_args(argv)
    if not 0<a.hours<=168:p.error('--hours must be in (0,168]')
    if a.command=='restore':restore_all(a.key_file);return 0
    if a.command=='export':
        from wengua_course.runner import export_review
        archive=a.archive or Path.home()/'Downloads'/('Wengua_V60_review_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
        print(export_review(a.output,archive));return 0
    runtime.configure(a.app_root)
    if a.command=='check':print(d.dumps(runtime.verify_data()['summary']));return 0
    if a.command=='search':
        from .corpus import retrieve
        if not a.question:p.error('--question is required')
        print(d.dumps(retrieve(a.question,domain=a.domain)));return 0
    if a.command=='ask':
        if not a.question:p.error('--question is required')
        from . import initialization,corpus
        from .runner import config
        from .process import launch
        from transformers import AutoTokenizer
        selected=d.read(a.output/'retained_best.json');adapter=Path(selected['adapter']);initialization.inspect(adapter)
        if d.digest((adapter/'adapter_model.safetensors').read_bytes())!=selected['adapter_sha256']:raise ValueError('Retained weights changed')
        profile=runtime.profile();snapshot=initialization.snapshot(a.output/'parents/interactive',adapter)
        c=config(snapshot,a.output,'pilot');job=profile.engine.open_job(a.output/'trials/interactive',c,resume=True)
        tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
        packet=corpus.prompt(a.question,tok,domain=a.domain,time_text=a.time,zone=a.zone,chart_file=a.chart,max_new_tokens=512)
        if packet['state']!='ready':print('检索到的原文不足以完整回答本题。');return 2
        target=a.output/'rag_prompts'/(str(time.time_ns())+'.json');d.write(target,packet)
        result=launch(profile,job,c,time.time()+3600,'ask',best={'prompt':str(target),'adapter':str(adapter)})
        print(result.get('answer',d.dumps(result)))
        if result.get('trace_path'):print('记录：'+result['trace_path'])
        return 0
    from .loop import run
    return run(a)
