"""Read-only, answer-bound review of the four-model diagnostic. Never a teacher."""
from pathlib import Path
from collections import Counter
import json
from wengua_course import data as d
from . import v61_probe as probe

ARCHIVE_SHA='1c3268d02f17ea6e8b1cbf8725bf2bb879a5c8dd179c99c11af3b5141aecbfba'
# Post-hoc limited reading judgments, not clinical/outcome labels or blind accuracy.
# Explicit index lists avoid transcription ambiguity in the display strings.
CORRECT={
 'official_base':[4,5,6,10,12,13,17,20,21,22,23],
 'retained_v58':[6,8,12,13,15,20,21,22,23],
 'v61_low31':[5,8,12,13,14,15,20,22,23],
 'v61_high31':[6,12,13,14,15,20,21,22,23],
}
PARTIAL={'official_base':[7,15],'retained_v58':[1,7,14],
         'v61_low31':[1,3,7,17,21],'v61_high31':[1,7,17]}
NOTES={
 'official_base':[
 '混入八字；把未知出生年当作不能确定本次日干/年干。',
 '先提起局年干，随后又因未知父亲出生年要求留空；两个年份混用。',
 '虚构日干癸、女儿出生乙，未按题面绑定。',
 '要求沿用前问用神，未切换本人日干。',
 '金生水方向正确；绕述其他五行可删。','同火无直接相克，主项正确。','纠正为土生金，方向正确。',
 '门克宫正确，缺所问名称门迫。','误称木克金、门迫。',
 '把盘中巽四的天英读成本位离九。','给定位置后水生木正确。',
 '开头不等同正确，后称天心属土、墓酉并混入紫微四化，生成到上限。',
 '借贷人克银行，对银行不利，正确。','给来源后方向和银行立场正确。',
 '把时克日卖得快答成卖慢，又推利润。','卖得快正确，仍越界推利润可能较高。',
 '混用六爻，三个六合角色均错。','有来源时媒人、证人证据、中介正确。',
 '八字夫妻宫等无关扩展，漏吉门吉格。','把媒人倾向当作本分支缺失条件，漏吉门吉格。',
 '自然改写合格。','两项已知事实正确。','算式和15正确。','仅保留第二步正确。'],
 'retained_v58':[
 '以年干代表本人，错误。','起局年干主项对；未清楚补兼乾宫和年命留空。',
 '把起局乙当年柱年命，未分开。','按出生年取命宫，漏本人日干。','答成水生金，方向反了。',
 '未直接回答同火无克，只要求顺序。','土生金正确。','方向正确，缺门迫。','宫克门、伤门受克正确。',
 '天英宫位读错，还说水生火。','比和错误。','重复不一定相同直至上限，无具体解释。',
 '银行不利正确。','银行不利正确。','卖得快及利润边界正确，但又称比和，与题面时克日矛盾。',
 '卖得快及利润边界正确。','错答天芮天心天辅。','复述其他用神，未回答六合三角色。',
 '未指出吉门吉格。','用双方年命替换本条吉门吉格。','自然改写合格。',
 '提取正确；额外提醒可删。','15正确。','只保留第二步正确。'],
 'v61_low31':[
 '先按出生年取年干，混淆主用神。','年干主项对，缺乾宫且年命补入措辞不清。',
 '把乙癸绑定至错误人物/年份。','只提年命癸，漏切换本人日干。','水生金错误。',
 '同火不取生克方向，主项正确。','既说土生金又说金生土，矛盾。','方向正确，漏门迫。',
 '宫克门、伤门受克，纠正主项正确。','天英宫位读错并扩展无关神煞。','答英宫生任宫，反向。',
 '虽说不同，却把乙奇也称为星，理由循环。','银行不利正确。','银行不利正确。',
 '卖得快且不能推出利润高，正确。','有来源同样正确。','三域角色错误。',
 '官司与交易角色对；婚姻未答六合媒人。','遗漏吉门吉格。','把年命当作本条所缺条件。',
 '自然改写合格。','事实提取正确，但又重复问题。','15正确。','只保留第二步正确。'],
 'v61_high31':[
 '混出生年、起局年干和代问对象。','年干主项对，缺乾宫且年命留空不清。',
 '补出题目没有的日干癸。','出生癸却又写年命庚，且漏本人日干。','水生金错误。',
 '将两个火说成前者克后者，错误。','明确否定金生土并纠正，主项正确。','方向正确，漏门迫。',
 '将门克宫、宫克门都称门迫，矛盾。','天英位置错误且方向错误。','木生水错误。',
 '把乙奇叫星，解释循环，没有分别判断的依据。','银行不利正确。','银行不利正确。',
 '卖得快且不能推出利润高，正确。','有来源同样正确。','错将天盘干用于所有六合角色。',
 '官司交易角色对，婚姻漏六合媒人。','遗漏吉门吉格。','年命替换本条吉门吉格。',
 '自然改写合格。','两项事实正确。','15正确。','只保留第二步正确。']}

def review(root,pack):
    root=Path(root);contract=d.read(root/'run_contract.json')
    if contract['code_sha256']!=probe.hashes():raise ValueError('Diagnostic code contract differs')
    if contract['software_fixture'] or contract['config']['eval_max_new_tokens']!=384 or contract['config']['inference_max_context']!=4096:
        raise ValueError('Not the actual uniform-budget diagnostic')
    from .v61_probe_data import build
    target=build(pack)
    if d.read(root/'prompts.json')!=target:raise ValueError('Diagnostic prompt pack differs')
    state=d.read(root/'run_status.json')
    if state['state']!='complete_review_required' or state['weight_updates']!=0:raise ValueError('Not the complete read-only probe')
    records=[];totals={}
    for label in CORRECT:
        rows=probe.read_rows(root/'answers'/(label+'.jsonl'))
        variant=label
        if not probe.verify_cached(rows,target['records'],variant):raise ValueError('Missing answer')
        by_id={r['id']:r for r in rows}
        rows=[by_id[r['id']] for r in target['records']]
        expected=probe.result_contract(contract['config'],contract['sources'][label],target['records'])
        if d.read(root/'answers'/(label+'.provenance.json'))!=expected:raise ValueError('Model/budget provenance differs')
        attention=d.read(root/'answers'/(label+'.attention.json'))
        if attention['software_fixture'] or not attention['cuda_calls'] or attention['math_fallback_allowed'] or attention['selected_backend']!='EFFICIENT_ATTENTION':raise ValueError('Actual CUDA attention route not proved')
        if len(rows)!=24 or len(NOTES[label])!=24:raise ValueError('Expected all24 read answers')
        for i,r in enumerate(rows):
            grade='C' if i in CORRECT[label] else 'P' if i in PARTIAL[label] else 'W'
            records.append({'model':label,'index':i,'id':r['id'],'answer_sha256':d.digest(r['answer']),
                'prompt_sha256':r['prompt_sha256'],'grade':grade,'bounded_reading_note':NOTES[label][i],
                'raw':r,'not_for_training_or_rag':True})
        totals[label]=dict(Counter(r['grade'] for r in records if r['model']==label))
    calls=list((root/'worker_calls').glob('*.json'))
    checked=0
    for request in calls:
        if any(t in request.stem for t in ['.result','.heartbeat','.allocator','.attention']):continue
        payload=d.read(request)
        if payload.get('schema')!=probe.SCHEMA or 'nonce' not in payload:continue
        probe.verify_reply(d.read(request.with_suffix('.result.json')),payload,request.read_bytes());checked+=1
    if checked!=4:raise ValueError('Four actual worker receipts required')
    return {'schema':'wengua_v61_dialogue_review_v1','input_archive_sha256':ARCHIVE_SHA,
        'records':records,'summary':{'answers':len(records),'actual_workers':checked,'bounded_manual_counts':totals,
        'weight_updates':0,'evaluation_only':True,'independent_case_accuracy':False,
        'old_weights_preserved':True,'base_is_professionally_qualified':False,'next_initialization':'official_fresh_lora'}}
