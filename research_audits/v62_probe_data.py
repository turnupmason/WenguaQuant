"""Input controls and exact trained teachers, evaluation-only; no model answers."""
from copy import deepcopy
from wengua_course import data as d
SCHEMA='wengua_v62_input_and_fit_probe_v1'

def build(pack):
    from research_audits.v61_probe_data import SYSTEM as neutral
    systems={'neutral':neutral,'course':pack['records'][0]['messages'][0]['content']}
    train=[r for r in pack['records'] if r['split']=='train'];records=[]
    def support(skill,term=None):
        return deepcopy(next(r for r in train if r['skill']==skill and (term is None or term in r['question']))['provenance']['support'])
    identity=support('identity');elements=support('direction');marriage=support('branch_conditions')
    cases=[
      ('self','这是按问事时刻排好的奇门取用练习。患者自己问健康，出生年未核实。起局年干乙，起局日干丁。先以谁代表患者，出生年未知怎么办？',
       {'实际主体':'患者本人','领域':'健康','起局年干':'乙','起局日干':'丁','出生年干':None},
       '本人取起局日干丁；出生年未知，只将出生年命留空，不改用起局年干乙。',identity),
      ('father','这是按问事时刻排好的奇门取用练习。女儿替父亲问病，起局年干乙、起局日干丁，父亲出生年干癸。乙、丁、癸分别怎么用？',
       {'实际主体':'女儿代问父亲','领域':'健康','起局年干':'乙','起局日干':'丁','父亲出生年干':'癸'},
       '代问父亲以起局年干乙为主，兼看乾宫；癸为父亲出生年命另看。不能以日干丁替代本题父亲的年干取用，也不把癸当日干。',identity),
      ('water_wood','只问直接五行关系：甲方属木，乙方属水。谁生谁？不要推事情吉凶。',
       {'甲方五行':'木','乙方五行':'水','判断范围':'仅直接关系，不断吉凶'},
       '水生木，所以乙方生甲方，甲方受生，不是木生水或水克木。',elements),
      ('marriage','婚事能否按本条可成分支判断：乙庚两宫相生，吉门吉格未提供，双方年命已提供。缺什么条件？能不能把未提供当作没有？',
       {'所问':'本条可成分支是否有足够条件','乙庚宫关系':'相生','吉门吉格':'未提供','双方年命':'已提供'},
       '还缺吉门吉格是否成立的资料。双方年命是辅助观察，不能替代该条件；未提供不等于不存在，也不能因此判婚事必败。',marriage)]
    def add(key,user,answer,system,category,provenance):
        records.append({'id':'v62_fit_probe:'+key,'question':user,'answer':answer,'messages':[{'role':'system','content':system},{'role':'user','content':user},{'role':'assistant','content':answer}],
         'format':'readonly_input_control','task':category,'group':key,'source_mode':'evaluation_only','source_ids':sorted({s['source_id'] for s in provenance}),'support':deepcopy(provenance),'not_training_or_rag_teacher':True})
    facts_text={
      'self':'患者本人问健康，出生年未核实，起局年干乙，起局日干丁。',
      'father':'女儿代问父亲健康，起局年干乙，起局日干丁，父亲出生年干癸。',
      'water_wood':'甲方属木，乙方属水。',
      'marriage':'乙庚两宫相生，吉门吉格未提供，双方年命已提供。'}
    questions={
      'self':'先以谁代表患者，出生年未知怎么办？',
      'father':'乙、丁、癸分别怎么用？',
      'water_wood':'谁生谁？不要推事情吉凶。',
      'marriage':'婚事能否按本条可成分支判断，还缺什么条件？能否把未提供当作没有？'}
    for name,plain,facts,answer,refs in cases:
        # Same evidence and facts; representation is the only changed user-input factor.
        frame='这是按问事时刻排好的奇门取用练习。' if name in ['self','father'] else '这是五行直接关系练习。' if name=='water_wood' else '这是奇门婚姻规则的条件练习。'
        plain=frame+'\n已核实条件：'+facts_text[name]+'\n所问：'+questions[name]
        structured=frame+'\n已核实条件：'+d.dumps(facts)+'\n所问：'+questions[name]
        for system_name,system in systems.items():
            for representation,user in [('plain',plain),('structured',structured)]:
                add(name+':'+system_name+':'+representation,user,answer,system,'input_2x2',refs)
            add(name+':'+system_name+':source',plain+'\n参考原文：\n'+'\n'.join(s['quote'] for s in refs),answer,system,'provided_source_control',refs)
    for skill in ['identity','direction','door_relation','branch_conditions','symbol_types','source_dialogue']:
        r=min((r for r in train if r['skill']==skill),key=lambda r:d.digest('v62-training-recall:'+r['id']))
        copied=deepcopy(r);copied.update(id='v62_exact_teacher:'+r['id'],task='exact_training_prompt_recall',source_mode='evaluation_only',split='diagnostic',not_training_or_rag_teacher=True)
        # Leave every message and the target byte-for-byte intact.
        if copied['messages']!=r['messages']:raise ValueError('Exact training prompt changed')
        records.append(copied)
    official_rag=deepcopy(pack['actual_rag_rows'])
    for r in records[:24]:
        for ref in r['support']:
            if ref['source_id'] not in pack['units'] or d.compact(ref['quote']) not in d.compact(pack['units'][ref['source_id']]['original_quote']):raise ValueError('Probe reference is not literal at source')
    if len(records)!=30 or len(official_rag)!=18:raise ValueError('Expected 30 input/recall controls and18 frozen RAG inputs')
    return {'schema':SCHEMA,'records':records,'official_rag_rows':official_rag,'ordinary_or_model_outputs_used_as_teachers':False,'inference_only':True,'weight_updates':0,'unseen_case_or_prediction_benchmark':False}
