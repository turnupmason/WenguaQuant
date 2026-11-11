"""Build, inspect, run or export the source-bound V64 continuation."""
from argparse import ArgumentParser
from pathlib import Path
import time,os
from wengua_course import data as d
from . import runtime


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('command',choices=['check','run','export'])
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v64/run1');p.add_argument('--app-root',type=Path)
    p.add_argument('--hours',type=float,default=24);p.add_argument('--resume',action='store_true');p.add_argument('--resume-from-checkpoint',type=Path);p.add_argument('--archive',type=Path)
    a=p.parse_args(argv)
    if not 0<a.hours<=168:p.error('--hours must be in (0,168]')
    if a.command=='export':
        from wengua_course.runner import export_review
        target=a.archive or Path.home()/'Downloads'/('Wengua_V64_review_'+time.strftime('%Y%m%d_%H%M%S')+'.zip');print(export_review(a.output,target));return 0
    runtime.configure(a.app_root);pack=runtime.verify_data()
    from .review import write_review
    write_review(pack,runtime.DATA);print(d.dumps(pack['summary']));print('课程审阅Excel与SQLite：'+str(runtime.DATA))
    if a.command=='check':return 0
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR)
    from .runner import run
    return run(a)
