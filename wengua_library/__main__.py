"""Restore and query a versioned book extension without changing active training."""
import argparse,json
from collections import Counter
from pathlib import Path
from . import intake
from wengua_corpus import book_rag as r
from wengua_corpus.__main__ import restore

RELEASE=r.ROOT/'data/corpus_updates/jinhan_batch03_20261109/release.json'

def load_release(path=RELEASE):
    release=r.read(path)
    if release['schema']!=intake.SCHEMA or release['automatic_training_inclusion'] or release['current_v60_training_changed']:
        raise ValueError('Invalid independent source-extension contract')
    for entry in release['entries']:
        r.private(r.ROOT/entry['destination'])
    r.private(r.ROOT/release['index'])
    return release

def libraries(release):
    found=[]
    for entry in release['entries']:
        path=r.private(r.ROOT/entry['destination']/entry['library'])
        if not path.is_file() or r.digest(path.read_bytes())!=entry['files'][entry['library']]:
            raise ValueError('Source library missing/changed; run restore: '+entry['id'])
        found.append(intake.load_library(path))
    return found

def ensure_index(release):
    libs=libraries(release);path=r.private(r.ROOT/release['index'])
    if not path.exists():r.build_index(libs,path)
    with r.Index(path) as index:
        actual=json.loads(index.db.execute("SELECT value FROM meta WHERE key='source_libraries'").fetchone()[0])
        if actual!=[r.digest(r.dumps(lib)) for lib in libs]:raise ValueError('Index belongs to another source release')
        return index.coverage

def query(index,question,*,study=False,section=None,book=None,limit=6):
    # Validate via the existing read-only retrieval API, including book and size.
    mode='study' if study else 'knowledge'
    result=index.search(question,mode=mode,book=book,limit=limit)
    if section:
        terms=set(r.terms(question));pool=[]
        for row in index.db.execute('SELECT payload FROM documents'):
            doc=json.loads(row[0])
            if doc.get('source_section')!=section or (not study and doc['kind']!='rule'):continue
            if doc.get('material_scope')=='structure':continue
            if book and any(index.chapters[p['chapter']]['book_id']!=book for p in doc['parts']):continue
            expanded=index.expand(doc)
            if all(intake.heading(line.strip()) for line in expanded['content'].splitlines() if line.strip()):continue
            # Audit questions must be able to retrieve a specific omission or
            # conflicting reading, rather than just a frequent chapter title.
            searchable=expanded['content']+' '+r.dumps(doc.get('source_uncertainties',[]))+' '+r.dumps(doc.get('remaining_parent_findings',[]))
            matched=terms&set(r.terms(searchable))
            if matched:pool.append((expanded,matched))
        frequency=Counter(t for _,matched in pool for t in matched)
        chosen=[((-sum(1/(1+frequency[t]) for t in matched),len(doc['content']),doc['id']),doc) for doc,matched in pool]
        result['documents']=[];seen=set()
        for _,doc in sorted(chosen,key=lambda x:x[0]):
            parts={r.dumps(p) for p in doc['parts']}
            if parts.issubset(seen):continue
            seen.update(parts);result['documents'].append(doc)
            if len(result['documents'])==limit:break
        result['retrieval_method']='explicit chapter scope + CPU Chinese-bigram overlap; literal evidence only'
        result['section']=section
    result.update(active_training_changed=False,automatic_training_inclusion=False,
                  source_truth_status='received_book_text_not_facsimile_or_predictive_verification')
    return result

def related_comparisons(release,docs,*,study=False):
    ids={x for d in docs for x in [d['id'],d.get('parent_material_id'),*d.get('source_unit_ids',[])] if x}
    groups=[]
    for entry in release['entries']:
        if 'reviewed_comparisons.json' not in entry['files']:continue
        path=r.private(r.ROOT/entry['destination']/'reviewed_comparisons.json')
        if r.digest(path.read_bytes())!=entry['files']['reviewed_comparisons.json']:raise ValueError('Comparison snapshot changed')
        groups.extend(g for g in r.read(path) if ids.intersection(g['new_unit_ids']))
    if not study:
        for group in groups:
            group['prior_evidence']=[p if p['kind']=='rule' else {k:v for k,v in p.items() if k!='text'}|{'requires_study_mode_for_literal_text':True} for p in group['prior_evidence']]
    return groups

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,default=RELEASE);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('restore');q.add_argument('--key-file',type=Path)
    s.add_parser('verify')
    q=s.add_parser('search');q.add_argument('--question',required=True);q.add_argument('--book');q.add_argument('--section');q.add_argument('--study',action='store_true');q.add_argument('--limit',type=int,default=6)
    a=p.parse_args(argv);release=load_release(a.release)
    if a.command=='restore':
        keytext=a.key_file.read_text().strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if len(keytext)!=64:raise ValueError('Expected existing 64-character delivery key')
        key=bytes.fromhex(keytext)
        for e in release['entries']:
            names=list(e['files']) if e['id']==release['new_entry_id'] else [e['library']]
            if e['id']!=release['new_entry_id'] and 'reviewed_comparisons.json' in e['files']:names.append('reviewed_comparisons.json')
            if all((r.ROOT/e['destination']/n).is_file() and r.digest((r.ROOT/e['destination']/n).read_bytes())==e['files'][n] for n in names):continue
            restore(e,key)
        print(json.dumps({'coverage':ensure_index(release),'training_started':False,'active_index_changed':False},ensure_ascii=False,indent=2))
    elif a.command=='verify':
        report=ensure_index(release)
        for e in release['entries']:
            if e['id']!=release['new_entry_id']:continue
            for name,expected in e['files'].items():
                path=r.private(r.ROOT/e['destination']/name)
                if r.digest(path.read_bytes())!=expected:raise ValueError('Extension member changed: '+name)
        print(json.dumps({'coverage':report,'active_training_changed':False,'current_training_eligible':False},ensure_ascii=False,indent=2))
    else:
        ensure_index(release)
        with r.Index(r.ROOT/release['index']) as index:result=query(index,a.question,study=a.study,section=a.section,book=a.book,limit=a.limit)
        result['source_comparisons']=related_comparisons(release,result['documents'],study=a.study)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
