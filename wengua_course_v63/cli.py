"""Launch and export V6.3 on the existing Windows 8B environment."""
from argparse import ArgumentParser
from pathlib import Path
import re,time,os
from wengua_course import data as d
from . import runtime

def restore(key_file=None):
    from wengua_course_v62.cli import restore as restore_source
    restore_source(key_file)
    print(d.dumps(runtime.verify_data()['summary']))

def main(argv=None):
    parser=ArgumentParser(description=__doc__);parser.add_argument('command',choices=['restore','check','run','export'])
    parser.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v63/run1');parser.add_argument('--app-root',type=Path)
    parser.add_argument('--hours',type=float,default=24);parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-from-checkpoint',type=Path)
    parser.add_argument('--key-file',type=Path);parser.add_argument('--archive',type=Path)
    a=parser.parse_args(argv)
    if not 0<a.hours<=168:parser.error('--hours must be in (0,168]')
    if a.command=='restore':restore(a.key_file);return 0
    if a.command=='export':
        from wengua_course.runner import export_review
        path=a.archive or Path.home()/'Downloads'/('Wengua_V63_review_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
        print(export_review(a.output,path));return 0
    runtime.configure(a.app_root)
    if a.command=='check':print(d.dumps(runtime.verify_data()['summary']));return 0
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']=runtime.ALLOCATOR;os.environ['PYTORCH_ALLOC_CONF']=runtime.ALLOCATOR
    from .runner import run
    return run(a)
