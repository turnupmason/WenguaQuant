"""Launch and export V6.2 on the existing Windows 8B environment."""
from argparse import ArgumentParser
from pathlib import Path
import re,time,os
from wengua_course import data as d
from . import runtime

def restore(key_file=None):
    from wengua_corpus.__main__ import restore as install
    value=key_file.read_text(encoding='utf-8-sig').strip() if key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
    if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing 64-character textbook delivery key')
    release=d.read(runtime.RELEASE)
    install({'manifest':release['delivery_manifest'],'destination':runtime.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value))
    runtime.verify_data();print('V6.2官方新LoRA条件应用教材已恢复并校验。')

def main(argv=None):
    parser=ArgumentParser(description=__doc__);parser.add_argument('command',choices=['restore','check','run','export'])
    parser.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v62/run1');parser.add_argument('--app-root',type=Path)
    parser.add_argument('--hours',type=float,default=12);parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-from-checkpoint',type=Path)
    parser.add_argument('--key-file',type=Path);parser.add_argument('--archive',type=Path)
    a=parser.parse_args(argv)
    if not 0<a.hours<=168:parser.error('--hours must be in (0,168]')
    if a.command=='restore':restore(a.key_file);return 0
    if a.command=='export':
        from wengua_course.runner import export_review
        path=a.archive or Path.home()/'Downloads'/('Wengua_V62_review_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
        print(export_review(a.output,path));return 0
    runtime.configure(a.app_root)
    if a.command=='check':print(d.dumps(runtime.verify_data()['summary']));return 0
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']=runtime.ALLOCATOR;os.environ['PYTORCH_ALLOC_CONF']=runtime.ALLOCATOR
    from .runner import run
    return run(a)
