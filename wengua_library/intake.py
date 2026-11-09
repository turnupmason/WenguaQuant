"""Literal-source intake; editorial findings never become observed outcomes."""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import gzip,json,re,sqlite3,unicodedata
from wengua_corpus import book_rag as r
from wengua_corpus.book_rag_sources import literal_selector

SCHEMA='wengua_source_extension_v1'
MAIN=('五假法','三诈法','九遁变化法','天德吉日','月德吉日','五符法','皇恩煞','月厌煞','洪范五行歌','奇门主客占验论','奇门占事')
SUB={x+'假' for x in '天地人神鬼'}|{x+'遁' for x in '天地人风云龙虎神鬼'}
DOMAINS={
 '占投军':['employment','military_history_source'],'占攻城':['military_history_source'],
 '占守城':['military_history_source'],'占盗贼':['military_history_source'],
 '占贼临境城可守否':['military_history_source'],'占胜败':['military_history_source'],
 '占远信':['missing_person','travel_abroad'],'占大计吉凶':['employment'],
 '占领批文迟速':['employment'],'占新任官员贤劣':['employment'],
 '占官事催提缓急':['litigation'],'占迁移吉凶':['geography_environment_historical_source'],
 '占雀噪':['miscellaneous_historical_source'],'占升迁':['employment'],
 '占乡会试':['education'],'占病何日愈':['health'],'占干谒':['employment'],
 '占捕捉':['criminal_case'],'占婚姻':['marriage'],
 '占走失':['lost_property','missing_person'],'占借贷':['business_profit'],
 '占武举':['education'],'占求财':['business_profit'],'占船主善恶':['travel_abroad'],
 '占在外人安否':['travel_abroad','missing_person'],'占访友':['travel_abroad','missing_person']}


def heading(line,profile=None):
    profile=profile or {}
    return (line in profile.get('main_headings',MAIN) or line in profile.get('subheadings',SUB) or line in profile.get('domains',DOMAINS) or line in profile.get('structural_headings',[]) or line in ['【白话译释】','【整理注】']
            or bool(re.fullmatch(r'【卷[一二三四五六七八九十]+】',line))
            or bool(profile.get('separator_volumes') and re.fullmatch(r'=+\s*(卷[一二三四五六七八九十]+末?|整理标记说明)\s*=+',line))
            or line.startswith('《金函奇门遁甲秘笈全书》') or line.startswith('范围：'))


def paragraphs(text,profile=None):
    """Separate headings even when a translation follows without a blank line."""
    lines=text.splitlines();out=[];start=None;profile=profile or {}
    breaks={int(n) for n in profile.get('heading_contexts',{})}|set(profile.get('break_before_lines',[]))
    for number,line in enumerate(lines+[''],1):
        is_head=heading(line.strip(),profile)
        if (not line.strip() or is_head or number in breaks) and start is not None:
            out.append((start,number-1));start=None
        if is_head:out.append((number,number))
        elif line.strip() and start is None:start=number
    return out


def load_library(path):return json.loads(gzip.decompress(Path(path).read_bytes()))


def source_document(chapter,id,title,ranges,*,kind='rule',domains=(),cues=(),phase=None,**extra):
    return dict(id=id,kind=kind,phase=phase or ('general_rule' if kind=='rule' else 'source_text_requires_collation'),
                title=title,domains=list(domains),cues=list(cues),case_ids=[],parent_source_ids=[],related_case_phases=[],
                parts=[literal_selector(chapter,*span) for span in ranges],
                topic='rule',review_status='reviewed_source_selector',**extra)


def text_units(raw,chapter,spec):
    lines=raw.splitlines();units=[];volume='文件说明';section='文件说明';mode='original';sub=''
    explicit=spec.get('unit_findings',{});profile=spec.get('source_profile',{});namespace=profile.get('namespace','jinhan.batch03')
    domains=profile.get('domains',DOMAINS);main=profile.get('main_headings',MAIN);subheads=profile.get('subheadings',SUB)
    study=profile.get('study_sections',['占病何日愈']);scopes=profile.get('unit_scopes',{})
    for i,(lo,hi) in enumerate(paragraphs(raw,profile),1):
        text='\n'.join(lines[lo-1:hi]);title=text.strip();is_head=lo==hi and heading(title,profile)
        separator=re.fullmatch(r'=+\s*(卷[一二三四五六七八九十]+末?|整理标记说明)\s*=+',title) if profile.get('separator_volumes') else None
        override=profile.get('heading_contexts',{}).get(str(lo))
        if override:
            section=override['section'];mode=override.get('source_mode',mode);sub=override.get('subsection','')
        elif separator:
            if separator[1]=='整理标记说明':section='整理注';mode='supplied_editorial_note'
            else:volume=separator[1];section='卷标题';mode='original';sub=''
        elif re.fullmatch(r'【卷[一二三四五六七八九十]+】',title):volume=title.strip('【】');section='卷标题';mode='original';sub=''
        elif title in main or title in domains:section=title;mode='original';sub=''
        elif title=='【白话译释】':mode='translation'
        elif title=='【整理注】':section='整理注';mode='supplied_editorial_note'
        elif title in subheads:sub=title
        supplied=re.findall(r'〔([^〕]+)〕',text)
        findings=deepcopy(explicit.get(str(lo),[]))
        for note in supplied:
            if any(x in note for x in ['存疑','待核','不一致','异文','出入','残','缺','推断','不明','句读','误','校正','修正']):
                findings.append({'kind':'supplied_editorial_note','detail':note,'verified_against_facsimile':False})
        scope='structure' if is_head else 'supplied_editorial_note' if mode=='supplied_editorial_note' or title.startswith('〔') else 'context_summary' if title.startswith('以上') else 'source_rule'
        scope=scopes.get(str(lo),scope)
        if title.startswith('例如谷雨'):scope='worked_example'
        blocking_findings=[f for f in findings if f.get('blocks_general_retrieval',True)]
        status='context_only' if scope!='source_rule' else 'quarantined_source_issue' if blocking_findings else 'source_grounded_candidate'
        if section in study:
            status='historical_source_study_only'
            findings.append({'kind':'source_scope','detail':profile.get('study_notice','古籍疾病断语仅作来源研究，不是临床结局标签；“八字内之日”所属八字需核对。')})
        u=dict(id=f"{namespace}.u{i:04d}",source_lines=[lo,hi],volume=volume,section=section,subsection=sub,
               source_mode=mode,scope=scope,title=' '.join(x for x in [volume,section,sub] if x),text=text,
               domains=domains.get(section,['formation'] if section in ['五假法','三诈法','九遁变化法'] else []),
               source_sha256=r.digest(text),findings=findings,course_state=status,case_ids=[],author_analysis_ids=[],
               current_training_eligible=False,rag_document_id=f"{namespace}.u{i:04d}")
        units.append(u)
    assigned=Counter(n for u in units for n in range(u['source_lines'][0],u['source_lines'][1]+1))
    if any(assigned[n]!=1 for n,line in enumerate(lines,1) if line.strip()):raise ValueError('Nonblank source not assigned exactly once')
    unknown=(set(explicit)|set(scopes)|set(profile.get('heading_contexts',{})))-{str(u['source_lines'][0]) for u in units}
    if unknown:raise ValueError('Finding does not start a material unit: '+','.join(sorted(unknown)))
    return units


def resolve_support(chapter,unit,fragments):
    """Every claimed premise/judgment is an exact, visible received-source span."""
    for value in fragments:
        if not isinstance(value,str) or not value or value not in unit['text']:raise ValueError('Structured source fragment not literal: '+repr(value))
    return literal_selector(chapter,*unit['source_lines'])


def compile_source(raw,book,spec):
    profile=spec.get('source_profile',{});namespace=profile.get('namespace','jinhan.batch03');domains=profile.get('domains',DOMAINS)
    if not re.fullmatch(r'jinhan\.batch[0-9]{2}',namespace):raise ValueError('Invalid source namespace')
    chapter=dict(book_id='jinhan',book_title=book['title'],chapter_id=profile.get('chapter_id','juan04_06_batch03'),
                 title=profile.get('title','五假三诈、卷五九遁与卷六占事·第三批节选'),method_id=book['method_id'],
                 source_revision='user_ocr_20261109_r1',original_text=raw,source_sha256=r.digest(raw),
                 source_form='received_edited_ocr_with_supplied_translation_and_notes',facsimile_verified=False,
                 source_file_line_numbers_not_printed_pages=True)
    units=text_units(raw,chapter,spec);byline={u['source_lines'][0]:u for u in units};byid={u['id']:u for u in units}
    first={}
    for u in units:
        if u['source_mode']=='original' and u['scope']=='source_rule':first.setdefault(u['section'],u)
    for u in units:
        context=first.get(u['section'])
        explicit_context=profile.get('section_context_lines',{}).get(u['section'])
        contexts=[byline[n] for n in explicit_context] if explicit_context is not None else ([context] if context else [])
        if any(c['section']!=u['section'] or c['scope']!='source_rule' for c in contexts):raise ValueError('Required context belongs to another topic or is not source text')
        u['required_context_ids']=([c['id'] for c in contexts if c['id']!=u['id']] if
            (u['section'] in domains or u['section']=='五符法') and u['scope']=='source_rule' else [])
        if u['required_context_ids'] and any(byid[k]['course_state']!='source_grounded_candidate' for k in u['required_context_ids']):
            u['course_state']='quarantined_context_dependency'
            u['findings'].append({'kind':'context_dependency','detail':'需同时阅读的本章原文存在待核条件。','source_unit_id':next(k for k in u['required_context_ids'] if byid[k]['course_state']!='source_grounded_candidate')})
    def support_ids(u):
        ids=[*u['required_context_ids'],u['id']]
        return sorted(ids,key=lambda k:byid[k]['source_lines']) if profile.get('source_ordered_contexts') else ids
    def support_parts(u):return [literal_selector(chapter,*byid[k]['source_lines']) for k in support_ids(u)]
    docs=[]
    for u in units:
        kind='rule' if u['course_state']=='source_grounded_candidate' else 'reference'
        docs.append(source_document(chapter,u['id'],u['title'],[byid[k]['source_lines'] for k in support_ids(u)],kind=kind,domains=u['domains'],
                    cues=[u['section'],u['subsection'],u['text'][:100]],source_mode=u['source_mode'],source_section=u['section'],
                    source_uncertainties=u['findings'],course_state=u['course_state'],
                    interpretation_limits=['同名符号按本书本专题取义；不自动替换WenguaApp排盘方法。'],
                    **({'material_scope':u['scope']} if profile.get('connected_evidence_required') else {})))
    for n,item in enumerate(spec.get('safe_excerpts',[]),1):
        u=byline[item['line']];quote=item['quote']
        if quote not in u['text'] or '〔' in quote:raise ValueError('Stable subpassage must be an exact unannotated source span')
        part=literal_selector(chapter,*u['source_lines'],quote=quote)
        doc=source_document(chapter,namespace+'.safe.'+str(n).zfill(3),item['title'],[u['source_lines']],
                  domains=u['domains'],cues=[item['title'],u['section']],source_section=u['section'],
                  reviewed_scope='literal_subpassage_not_whole_disputed_paragraph',parent_material_id=u['id'],
                  remaining_parent_findings=u['findings'])
        doc['parts']=[part]
        for extra in item.get('extra_quotes',[]):
            if extra not in u['text'] or '〔' in extra:raise ValueError('Required excerpt limit is not an exact unannotated source span')
            doc['parts'].append(literal_selector(chapter,*u['source_lines'],quote=extra))
        if item.get('extra_quotes'):doc['reviewed_scope']='literal_subpassages_with_required_limits'
        docs.append(doc)
    logic=[]
    for item in spec.get('logic_rules',[]):
        item=deepcopy(item);u=byline[item.pop('line')]
        snippets=item['literal_fragments'];part=resolve_support(chapter,u,snippets)
        item.update(source_unit_id=u['id'],source_parts=support_parts(u),source_section=u['section'],book_id='jinhan',
                    source_method_id=book['method_id'],current_training_eligible=False,
                    semantic_status='source_assertion_structured_not_expert_predictive_gold',observed_outcome=False)
        if profile.get('connected_evidence_required'):item['source_unit_ids']=support_ids(u)
        if not item.get('conditions') or 'conclusion' not in item:raise ValueError('Logic rule needs premises and scoped conclusion')
        logic.append(item)
    workflows=[]
    for section,topic_domains in domains.items():
        selected=[u for u in units if u['section']==section and u['scope']=='source_rule']
        if not selected:continue
        workflows.append(dict(id=namespace+'.workflow.'+str(len(workflows)+1).zfill(2),title=section,domains=topic_domains,
                              source_unit_ids=[u['id'] for u in selected],source_order_preserved=True,
                              logic_rule_ids=[x['id'] for x in logic if x['source_section']==section],
                              restrictions=[x for u in selected for x in u['findings']],
                              current_training_eligible=False,case_ids=[]))
    candidates=[]
    for item in spec.get('courses',[]):
        item=deepcopy(item);supports=[byline[n] for n in item.pop('source_lines')]
        allowed={'source_rule','supplied_editorial_note'} if item['format']=='source_audit_qa' else {'source_rule'}
        if any(u['scope'] not in allowed for u in supports):raise ValueError('Headings cannot be standalone teaching evidence')
        # Source membership is not semantic certification. Every explicit
        # target quotation is checked, and unresolved cases remain review-only.
        for fragment in item.get('literal_targets',[]):
            if not any(fragment in u['text'] for u in supports):raise ValueError('Course literal target is not in selected source')
        ids=list(dict.fromkeys(k for u in supports for k in [*u['required_context_ids'],u['id']]))
        if profile.get('source_ordered_contexts'):ids.sort(key=lambda k:byid[k]['source_lines'])
        item.update(source_unit_ids=ids,source_parts=[literal_selector(chapter,*byid[k]['source_lines']) for k in ids],
                    family_id=namespace+'.'+supports[0]['section'],
                    current_training_eligible=False,status='source_grounded_candidate_requires_semantic_and_style_review',
                    source_is_model_output=False,answer_origin='assistant_authored_source_grounded_draft_not_author_literal',
                    observed_outcome=False,case_ids=[])
        candidates.append(item)
    # No automatic teacher is made for structure, disputed text or an
    # application requiring an unstated month/chart/actor.
    covered={id for c in candidates for id in c['source_unit_ids']}
    for u in units:
        if u['id'] not in covered and u['course_state']=='source_grounded_candidate':
            candidates.append(dict(id=u['id']+'.reading',format='source_reading',source_unit_ids=[*u['required_context_ids'],u['id']],
                          source_parts=support_parts(u),family_id=namespace+'.'+u['section'],
                          question='《金函》'+u['section']+'本段原文怎样记载条件、用途与限制？',
                          answer=u['text'],current_training_eligible=False,
                          status='literal_source_reading_not_independent_application',source_is_model_output=False,answer_origin='literal_received_source',
                          observed_outcome=False,case_ids=[]))
    if profile.get('connected_evidence_required'):
        for item in candidates:
            item['source_family_ids']=sorted({namespace+'.'+byid[k]['section'] for k in item['source_unit_ids']})
            item['split_assigned']=False
            item['future_split_policy']='keep_connected_source_evidence_components_together; no new unseen-case claim'
    illustrations=[]
    for item in spec.get('illustrations',[]):
        item=deepcopy(item);u=byline[item.pop('line')];resolve_support(chapter,u,item['literal_fragments'])
        item.update(source_unit_id=u['id'],source_parts=[literal_selector(chapter,*u['source_lines'])],
                    example_type='worked_rule_illustration_not_observed_event',case_id=None,
                    complete_nine_palace_chart_given=False,observed_feedback_given=False)
        illustrations.append(item)
    for wf in workflows:
        selected=[byid[k] for k in wf['source_unit_ids']];clean=all(u['course_state']=='source_grounded_candidate' for u in selected)
        doc=source_document(chapter,wf['id'],wf['title']+'完整来源流程',[u['source_lines'] for u in selected],
                    kind='rule' if clean else 'reference',domains=wf['domains'],cues=[wf['title'],'取用神 条件 例外 判断顺序'],
                    source_unit_ids=wf['source_unit_ids'],source_section=wf['title'],source_uncertainties=wf['restrictions'],topic_scope='historical_source_workflow')
        docs.append(doc)
    library=dict(schema=r.SCHEMA,books={'jinhan':deepcopy(book)},chapters=[chapter],documents=docs,cases={})
    report=r.validate_library(library)
    return dict(schema=SCHEMA,library=library,units=units,logic_rules=logic,workflows=workflows,
                courses=candidates,illustrations=illustrations,gaps=deepcopy(spec.get('gaps',[])),
                source_structures=deepcopy(spec.get('source_structures',[])),report=report)


def exact_key(text):
    # Punctuation can reverse an OCR reading ("不，吉" versus "不吉"). Only
    # whitespace/width variants are eligible for automatic exact grouping.
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace())


def grams(text):
    key=''.join(c for c in exact_key(text) if not unicodedata.category(c).startswith('P'))
    return {key[i:i+3] for i in range(max(0,len(key)-2))}


def compare_sources(new,priors,max_near=3):
    old=[];exact=defaultdict(list);inverted=defaultdict(set)
    for lib in priors:
        chapters={r.chapter_key(c):c for c in lib['chapters']}
        for doc in lib['documents']:
            text=r.document_text(chapters,doc);g=grams(text);i=len(old)
            old.append(dict(id=doc['id'],book_ids=sorted({chapters[p['chapter']]['book_id'] for p in doc['parts']}),
                            kind=doc['kind'],text=text,parts=doc['parts'],grams=g))
            exact[exact_key(text)].append(i)
            if doc['kind']=='rule':
                for term in g:inverted[term].add(i)
    links=[]
    for u in new['units']:
        if u['scope']!='source_rule':continue
        key=exact_key(u['text'])
        if len(key)<12:continue
        for i in exact.get(key,[]):
            links.append(dict(new_unit_id=u['id'],prior_document_id=old[i]['id'],prior_book_ids=old[i]['book_ids'],
                              relation='exact_normalized_source_repetition',prior_document_kind=old[i]['kind'],
                              automatically_text_groupable=True,automatically_mergeable=False,
                              requires_condition_direction_and_scope_review=True,all_source_occurrences_retained=True))
        ng=grams(u['text']);counts=Counter(i for gram in ng for i in inverted.get(gram,()))
        candidates=[]
        for i,count in counts.most_common(40):
            if exact_key(old[i]['text'])==key:continue
            ratio=count/max(1,len(ng|old[i]['grams']))
            if ratio>=0.18 and count>=8:candidates.append((ratio,i))
        for ratio,i in sorted(candidates,reverse=True)[:max_near]:
            links.append(dict(new_unit_id=u['id'],prior_document_id=old[i]['id'],prior_book_ids=old[i]['book_ids'],
                              relation='near_source_candidate_not_semantic_equivalence',jaccard_3gram=round(ratio,6),
                              automatically_mergeable=False,requires_condition_direction_and_scope_review=True))
    # Same-upload original/translation correspondence is separate from a
    # duplicate verdict; a translation may omit a condition or reverse it.
    for u in new['units']:
        if u['source_mode']!='translation' or u['scope']!='source_rule':continue
        matches=[];g=grams(u['text'])
        for other in new['units']:
            if other['section']==u['section'] and other['source_mode']=='original' and other['scope']=='source_rule':
                og=grams(other['text']);over=len(g&og)/max(1,len(g|og))
                if over>=0.12:matches.append((over,other))
        for ratio,other in sorted(matches,key=lambda x:x[0],reverse=True)[:2]:
            links.append(dict(new_unit_id=u['id'],prior_document_id=other['id'],prior_book_ids=['jinhan'],
                              relation='same_upload_original_translation_alignment_candidate',jaccard_3gram=round(ratio,6),
                              automatically_mergeable=False,all_source_occurrences_retained=True))
    return old,links


def curated_comparisons(new,old,spec):
    result=[];byline={u['source_lines'][0]:u for u in new['units']}
    for finding in spec.get('comparisons',[]):
        finding=deepcopy(finding);needles=finding.pop('prior_all',[]);books=set(finding.pop('prior_books',[]))
        requested_ids=set(finding.pop('prior_document_ids',[]))
        allowed={'rule','reference'} if finding.pop('allow_reference_sources',False) else {'rule'}
        choices=[p for p in old if p['kind'] in allowed and (not requested_ids or p['id'] in requested_ids) and
                 (not books or books.intersection(p['book_ids'])) and all(x in p['text'] for x in needles)] if needles or requested_ids else []
        if requested_ids-{p['id'] for p in choices}:raise ValueError('Requested prior evidence missing or outside allowed scope')
        chosen=choices[:finding.pop('max_prior',4)]
        if needles and not chosen:raise ValueError('Curated comparison prior source missing: '+finding['id'])
        finding['new_unit_ids']=[byline[n]['id'] for n in finding.pop('new_lines')]
        finding['prior_evidence']=[{k:v for k,v in p.items() if k not in ['grams']} for p in chosen]
        finding['all_sources_retained']=True;finding['new_expert_gold_created']=False
        result.append(finding)
    return result
