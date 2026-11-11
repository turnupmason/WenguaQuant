"""Reuse the 74 actual parent answers without turning any answer into a teacher."""
from pathlib import Path
import json,time
from wengua_course import data as d
from research_audits.v63_checkpoint_probe import verify_cached
from . import runtime


def rows_from_sources():
    spec=d.read(runtime.ROOT/'data/reviews/v64_parent_answer_sources.json');rows=[];receipts=[]
    for label,info in spec.items():
        path=runtime.ROOT/info['path'];raw=path.read_bytes()
        if d.digest(raw)!=info['sha256']:raise ValueError('Earlier parent answers changed: '+str(path))
        records=list(map(json.loads,raw.splitlines()))
        if len(records)!=info['count']:raise ValueError('Earlier parent answer count differs')
        rows.extend(records);receipts.append({'path':info['path'],'sha256':info['sha256'],'answers':len(records)})
    if len(rows)!=74 or len({r['id'] for r in rows})!=74:raise ValueError('Exactly 74 known parent answers required')
    return rows,receipts


def seed(job,pack,config,source_sha256,*,testing=False,fixture_rows=None):
    job=Path(job);target=job/'generations/parent_review.jsonl';target.parent.mkdir(exist_ok=True)
    rows,receipts=(fixture_rows,[]) if testing else rows_from_sources()
    gold=runtime.review_rows(pack);by={r['id']:r for r in gold}
    if not verify_cached(rows,[by[r['id']] for r in rows]):raise ValueError('Known parent questions or gold changed')
    if not testing and source_sha256!=d.read(runtime.RELEASE)['reviewed_parent']['adapter_sha256']:
        raise ValueError('Parent answers belong to a different adapter')
    proof={'source_files':receipts,'reused_ids':[r['id'] for r in rows],'old_answers_rewritten':False,'weight_updates':0,
           'adapter_sha256':source_sha256}
    receipt=target.with_suffix('.reuse.json')
    if receipt.exists() and d.read(receipt)!=proof:raise ValueError('Parent answer reuse contract changed')
    runtime.resume_generation_file(target,gold)
    existing=[json.loads(s) for s in target.read_text(encoding='utf8').splitlines()] if target.exists() else []
    owner={r['id']:r for r in existing}
    for r in rows:
        if r['id'] in owner and owner[r['id']]!=r:raise ValueError('An actual reused answer was rewritten')
    if any(r['id'] not in owner for r in rows):
        if target.exists():target.rename(target.with_name(target.name+'.prefix-before-repair-'+str(time.time_ns())))
        reused={r['id'] for r in rows};existing=rows+[r for r in existing if r['id'] not in reused]
        target.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in existing),encoding='utf8')
    d.write(receipt,proof)
    # Establish the same whole-prompt/model receipt as the normal evaluator.
    # Its existing cache path can then safely continue from these old 74 rows.
    generation={'adapter_sha256':source_sha256,'variant':'inherited_adapter','model_id':config['model_id'],'revision':config['revision'],
        'prompt_sha256':d.digest(d.dumps([r['messages'] for r in gold])),'ids':[r['id'] for r in gold],
        'max_new_tokens':config['eval_max_new_tokens'],'eval_stage':'review'}
    path=target.with_suffix('.provenance.json')
    if path.exists() and d.read(path)!=generation:raise ValueError('Parent generation source/budget changed')
    d.write(path,generation);return proof
