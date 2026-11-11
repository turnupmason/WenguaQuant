"""Reuse exact existing answers; supply no answer text to the teacher builder."""
from pathlib import Path
import json,time
from wengua_course import data as d
from research_audits.v63_checkpoint_probe import verify_cached
from . import runtime


def source_rows(variant,root=None):
    spec=d.read(runtime.RELEASE)['baseline_sources'][variant];raw=((root or runtime.ROOT)/spec['relative_path']).read_bytes()
    if d.digest(raw)!=spec['sha256']:raise ValueError('Original baseline answers changed')
    rows=list(map(json.loads,raw.splitlines()))
    return rows,{'path':spec['relative_path'],'sha256':spec['sha256']}


def seed(job,pack,c,source_sha256,*,testing=False,fixture_rows=None):
    gold=runtime.review_rows(pack);by={r['id']:r for r in gold}
    if testing and isinstance(fixture_rows,dict):fixture_rows=fixture_rows.get(c['source_variant'],[])
    originals,source=(fixture_rows or [],None) if testing else source_rows(c['source_variant'])
    old={r['id']:r for r in originals};rows=[old[r['id']] for r in gold if r['id'] in old]
    if not testing and len(rows)!=(34 if c['source_variant']=='v63_trained' else 18):raise ValueError('Cached baseline subset differs')
    if not verify_cached(rows,[by[r['id']] for r in rows]):raise ValueError('Cached baseline prompt or teacher changed')
    if not testing and source_sha256!=d.read(runtime.RELEASE)['sources'][c['source_variant']]['sha256']:raise ValueError('Baseline model source differs')
    target=Path(job)/'generations/parent_review.jsonl';target.parent.mkdir(exist_ok=True)
    runtime.resume_generation_file(target,gold)
    existing=[json.loads(s) for s in target.read_text(encoding='utf8').splitlines()] if target.exists() else []
    owner={r['id']:r for r in existing}
    if any(r['id'] in owner and r!=owner[r['id']] for r in rows):raise ValueError('Original actual answer was rewritten')
    if any(r['id'] not in owner for r in rows):
        if target.exists():target.rename(target.with_name(target.name+'.prefix-'+str(time.time_ns())))
        ids={r['id'] for r in rows};existing=rows+[r for r in existing if r['id'] not in ids]
        target.write_bytes(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in existing).encode('utf8'))
    provenance={'adapter_sha256':source_sha256,'variant':'inherited_adapter','model_id':c['model_id'],'revision':c['revision'],
                'prompt_sha256':d.digest(d.dumps([r['messages'] for r in gold])),'ids':[r['id'] for r in gold],
                'max_new_tokens':c['eval_max_new_tokens'],'eval_stage':'review'}
    reuse={'source':source,'source_variant':c['source_variant'],'adapter_sha256':source_sha256,
           'reused_ids':[r['id'] for r in rows],'original_answers_rewritten':False,'weight_updates':0}
    for suffix,value in [('.provenance.json',provenance),('.reuse.json',reuse)]:
        path=target.with_suffix(suffix)
        if path.exists() and d.read(path)!=value:raise ValueError('Baseline cache provenance changed')
        d.write(path,value)
    return reuse
