"""Reuse exact completed parent answers only as evaluation, never training."""
from pathlib import Path
import json
from wengua_course import data as d
from . import runtime,evaluation,initialization


def seed(job,pack,*,testing=False,fixture_source=None):
    job=Path(job);folder=job/'generations';folder.mkdir(exist_ok=True);target=folder/'parent.jsonl';proof=folder/'parent.reuse.json'
    if testing:
        source=Path(fixture_source);raw=source.read_bytes()
    else:
        root=initialization.verify_source_run(runtime.ROOT/'local/course_v66/run1')
        old=d.read(root/'trials/ordinary_ce/job_contract.json')['config'];current=d.read(job/'job_contract.json')['config']
        if any(old[k]!=current[k] for k in ['model_id','revision','inference_max_context','eval_max_new_tokens','seed']):raise ValueError('Parent reuse generation budget or policy differs')
        source=root/'trials/ordinary_ce/generations/ordinary_ce.jsonl';raw=source.read_bytes()
        if d.digest(raw)!=d.read(runtime.ROOT/'data/reviews/v67_parent_sources.json')['files']['trials/ordinary_ce/generations/ordinary_ce.jsonl']:raise ValueError('Actual parent answers changed')
    rows=[json.loads(l) for l in raw.splitlines()];count=pack['summary']['retention_answers']
    if len(rows)!=count:raise ValueError('Parent answer count differs')
    evaluation.validate_cache(rows,pack,'ordinary_ce')
    value={'source_file':str(source),'source_sha256':d.digest(raw),'reused_rows':count,'old_answers_rewritten':False,'evaluation_only':True,'source_variant':'ordinary_ce'}
    if proof.exists() and d.read(proof)!=value:raise ValueError('Parent reuse source changed')
    if target.exists():
        if not target.read_bytes().startswith(raw):raise ValueError('Old parent bytes were changed')
    else:target.write_bytes(raw if raw.endswith(b'\n') else raw+b'\n')
    if not proof.exists():d.write(proof,value)
    return value
