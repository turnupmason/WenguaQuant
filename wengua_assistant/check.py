"""Preflight existing sources/model cache and only missing timezone data."""
import subprocess, sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def main():
    from wengua_course.runtime import configure
    configure()
    # Do not change Torch/Transformers/PEFT or download another model.
    missing = []
    try: ZoneInfo('Asia/Shanghai')
    except ZoneInfoNotFoundError: missing.append('tzdata==2025.2')
    if missing:
        print('安装缺少的时区资料（不改训练环境的模型库）：' + ', '.join(missing), flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', *missing], check=True)
    from .sources import checked_index, case_list
    checked_index()
    if len(case_list()) != 194: raise ValueError('两书案例登记数量不同，请保留原始资料并检查本地来源')
    from .inference import MODEL, REVISION
    from src.wengua.qwen_book_loop_v3 import official_model_cached
    if not official_model_cached({'model_id': MODEL, 'revision': REVISION}):
        raise FileNotFoundError('没有找到此前已下载的完整8B缓存。请使用之前训练的同一Windows账户和qwen_env。')
    from .prompt import cast
    packet = cast('2009-07-18T09:45', 'Asia/Shanghai')
    if packet['calendar']['four_pillars'] != {'year': '己丑', 'month': '辛未', 'day': '甲子', 'hour': '己巳'}:
        raise ValueError('既有排盘校验不同，停止而不修改原程序')
    late = cast('2000-09-21T23:45', 'Asia/Shanghai')
    if late['calendar']['four_pillars'] != {'year': '庚辰', 'month': '乙酉', 'day': '癸未', 'hour': '壬子'}:
        raise ValueError('原书23点换日校验不同')
    print('两书来源、194案目录、原有排盘及本地8B缓存已就绪。正在打开本地问答页面。', flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
