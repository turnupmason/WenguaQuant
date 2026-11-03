"""CLI: python -m wengua_quant --export <WenguaApp-V2-export> [--backend ...]."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dataset import load_v2_export
from .model import save_research_run


def main(argv=None):
    parser = argparse.ArgumentParser(description='WenguaQuant independent research pipeline')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--export', type=Path, help='Existing WenguaApp V2 export/data folder')
    source.add_argument('--app-root', type=Path, help='Local WenguaApp checkout with original source archive')
    parser.add_argument('--backend', choices=['logistic', 'catboost'], default='logistic')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='CPU')
    args = parser.parse_args(argv)
    if args.backend == 'logistic' and args.device == 'GPU':
        parser.error('Logistic uses CPU; GPU requires --backend catboost')
    if sys.version_info < (3, 11):
        parser.error('Python 3.11+ required')
    output = args.output or (Path('local/runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if output.exists():
        parser.error('Output already exists; never overwrite prior runs')
    if args.app_root:
        entry = args.app_root.resolve() / 'run_professional_training.py'
        if not entry.is_file():
            parser.error('WenguaApp V2 launcher not found in --app-root')
        export = output.resolve().parent / (output.name + '_source_export')
        if export.exists():
            parser.error('Export destination already exists')
        subprocess.run([sys.executable, str(entry), 'export', '--output', str(export)],
                       cwd=str(args.app_root.resolve()), check=True)
    else:
        export = args.export
    cohort = load_v2_export(export)
    report = save_research_run(cohort, output, backend=args.backend, device=args.device)
    print(json.dumps({'path': str(output), 'task': report['task'],
                      'events': report['event_families'], 'backend': report['backend'], 'device': report['device'],
                      'production_enabled': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
