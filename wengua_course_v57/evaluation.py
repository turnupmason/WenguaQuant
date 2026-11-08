"""Freeze small factual probes, full review and real RAG before local training."""
from copy import deepcopy
from wengua_course import data as d


def ordered(rows):return sorted(rows,key=lambda r:d.digest('v57-evaluation:'+r['id']))


def micro(records):
    rows=[r for r in records if r['split']=='validation'];selected=[]
    def take(skill,n,group=None,field='answer'):
        bucket=ordered([r for r in rows if r['skill']==skill]);picked=[]
        if group:
            for value in group:
                candidates=[r for r in bucket if r['checks'].get(field)==value]
                if candidates:picked.append(candidates[0])
        for r in bucket:
            if len(picked)>=n:break
            if r not in picked:picked.append(r)
        if len(picked)<n:raise ValueError('Insufficient evaluation coverage: '+skill)
        selected.extend(picked[:n])
    # Every elementary ordered pair, not only an easy majority relation.
    direction=ordered([r for r in rows if r['skill']=='direction']);pairs={}
    for r in direction:pairs.setdefault(tuple(r['checks']['elements']),r)
    if len(pairs)!=25:raise ValueError('All 25 elemental pairs required')
    selected.extend(pairs.values())
    take('given_direction',6,['生','受生','克','受克','比和']);take('chart_direction',6,['生','受生','克','受克','比和'])
    take('door_relation',5,['迫','制','和','义','比和'])
    take('property',5,['水','土','木','金','火'],field='element')
    people=[r for r in rows if r['provenance'].get('kind')=='explicit_person_condition'];people=ordered(people)
    chosen=[]
    for key in ['father_proxy','father_proxy_known','father_self','father_self_known','mother_proxy','mother_self','exam_self','exam_proxy']:
        chosen.append(next(r for r in people if r['provenance']['person_key']==key))
    selected.extend(chosen)
    typed=[r for r in rows if r['skill']=='typed_pair']
    for pair in ['戊戊','丁壬','庚戊']:
        selected.append(next(r for r in ordered(typed) if r.get('checks',{}).get('pair')==pair and r['checks']['kind']=='exact_name'))
    selected.append(next(r for r in ordered(typed) if r.get('checks',{}).get('kind')=='insufficient'))
    for element in ['水','火','木','金','土']:
        selected.append(next(r for r in ordered(rows) if r['provenance'].get('key')=='hetu_'+element and r.get('checks',{}).get('kind')=='number_pair'))
    if len(selected)!=64 or len({r['id'] for r in selected})!=64:raise ValueError('Micro coverage changed')
    return selected


def full(records,small,count=192):
    rows=ordered([r for r in records if r['split']=='validation']);picked={r['id']:r for r in small}
    for skill,n in [('positions',8),('condition',12),('typed_pair',8),('domain_rule',12),('workflow',12),('author_local',8)]:
        bucket=[r for r in rows if r['skill']==skill]
        if skill in ['domain_rule','workflow']:
            domains=['health','lost_property','employment','business_profit','marriage','education','pregnancy','travel_abroad','missing_person','litigation','criminal_case','weather']
            first=[next((r for r in bucket if r['domain']==x),None) for x in domains];bucket=[r for r in first if r]+[r for r in bucket if r not in first]
        for r in bucket[:n]:picked[r['id']]=r
    for domain in sorted({r['domain'] for r in rows if r['skill'] in ['case','author_reading']}):
        item=next(r for r in rows if r['domain']==domain and r['skill'] in ['case','author_reading']);picked[item['id']]=item
    for r in rows:
        if len(picked)>=count:break
        picked.setdefault(r['id'],r)
    if len(picked)>count:raise ValueError('Final domain coverage exceeds declared count')
    return list(picked.values())


def rag_contract(key):
    people={'exam_self':('日干',None,True),'exam_proxy':('时干',None,True),'health_daughter_father':('年干','乾宫',True),'health_father_self':('日干',None,False)}
    if key in people:
        role,helper,unknown=people[key]
        return {'kind':'person_contract','answer':role,'helper':helper,'unknown_year':unknown,'person_key':{'health_daughter_father':'father_proxy','health_father_self':'father_self'}.get(key,key)}
    if key=='hetu_金':return {'kind':'number_pair','answer':[4,9]}
    if key=='gui_yin_water':return {'kind':'literal_slots','required':[['阴'],['水'],['北']],'forbidden':['阳水','属金','属木','属火','属土','地网']}
    if key=='cross_domain_liuhe':return {'kind':'domain_map'}
    if key=='marriage_same_palace':return {'kind':'marriage_conditions'}
    required={'role_6':[['天芮']],'role_7':[['天心'],['乙奇']],'role_18':[['甲子戊','戊']],'role_20':[['六合']],'role_21':[['日干']],'role_22':[['时干']],'role_29':[['惊门']],'named_pair_戊戊':[['伏吟']],'named_pair_丁壬':[['奇仪相合']],'named_pair_庚戊':[['值符伏宫']]}
    return {'kind':'literal_slots','required':required[key]}


def legacy_rag(pack):
    result=[]
    for r in pack['legacy_actual_rag_rows']:
        item=deepcopy(r);key=r['id'].split(':')[-1];item.update(checks=rag_contract(key),skill='rag_identity' if key.startswith(('health_','exam_')) else 'rag_source_fact',domain='health' if key.startswith('health_') else 'education' if key.startswith('exam_') else 'foundation');item['provenance']={'required_citations':[]};result.append(item)
    return result


def pilot_training(records,count=1024):
    quotas={'positions':64,'direction':80,'given_direction':96,'chart_direction':96,'door_relation':64,'typed_pair':64,'property':48,'source_fact':48,'identity':120,'condition':120,'case':96,'workflow':48,'domain_rule':48,'author_reading':16,'author_local':16}
    chosen={}
    for skill,n in quotas.items():
        bucket=sorted((r for r in records if r['split']=='train' and r['skill']==skill),key=lambda r:(not r['provenance'].get('repair_priority',False),d.digest('v57-pilot:'+r['id'])))
        for r in bucket[:n]:chosen[r['id']]=r
    # A small category may contain fewer rows; fill unique mixed source lessons,
    # never manufacture duplicates to claim a larger knowledge base.
    for r in sorted((r for r in records if r['split']=='train'),key=lambda r:d.digest('v57-pilot-fill:'+r['id'])):
        if len(chosen)>=count:break
        chosen.setdefault(r['id'],r)
    if len(chosen)!=count:raise ValueError('Pilot unique core quota changed')
    return list(chosen.values())
