"""Read-only audit of the actual V65 paired/own-history/token probe.

Token fitting and limited answer observations are never case accuracy or
training teachers. GPU logits cannot be independently recomputed here.
"""
from pathlib import Path
import argparse,gzip,json,math,zipfile
from wengua_course import data as d
from wengua_course_v65 import runtime
from . import v65_learning_probe as probe,v65_learning_data as data,v65_token_diagnostics as token
from .v55_review import inspect_archive,read_rows
from .v64_review import verified_notes
from .v63_checkpoint_review import bound_note

UPLOAD_SHA='50f72a268430142dbfa8e39c8b0db24f8af51589ca4e4cfa933041aebd1b1cc4'
SOURCES={'v63_parent':'c12e455e0ba7ef3b923c48f23321f0da624b75de3f4eb88c0800aace24a2457d',
         'v65_epoch2':'d5fa4783d63f360303862757075458b2658aad46a310d058c14c2f891ccac063'}


def mean(values):return sum(values)/len(values) if values else None


def close(actual,expected):
    if expected is None:
        if actual is not None:raise ValueError('Expected absent empty-segment mean')
    elif type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8):
        raise ValueError('Token loss aggregate differs from full token records')


def verify_token_row(row,target,encoded):
    probe.validate_token_rows([row],[target]);records=row['tokens'];expected=encoded['full'][len(encoded['prefix']):]
    if (len(records)!=len(expected) or row['supervised_tokens']!=len(expected)
        or [r['target_id'] for r in records]!=expected
        or [r['in_decision_sentence'] for r in records]!=encoded['decision_mask']
        or [r['teacher_segment_index'] for r in records]!=encoded['token_segment_indices']
        or [r['character_offsets'] for r in records]!=[list(v) for v in encoded['target_offsets']]
        or row['professional_probability'] or not row['segment_boundary_tokens_assigned_by_start_offset']):
        raise ValueError('Whole source target tokens, offsets or masks changed')
    values=[]
    for record in records:
        value=record['nll'];predicted=record['gold_prefix_top1_id']
        if (type(value) not in (float,int) or not math.isfinite(value) or value<0 or type(predicted) is not int
            or predicted<0 or type(record['gold_prefix_top1_equals_target']) is not bool
            or record['gold_prefix_top1_equals_target']!=(predicted==record['target_id'])
            or (predicted!=record['target_id'] and value<math.log(2)-1e-5)):
            raise ValueError('Invalid token loss or top1 observation')
        values.append(value)
    decision=[v for v,m in zip(values,encoded['decision_mask'],strict=True) if m]
    rest=[v for v,m in zip(values,encoded['decision_mask'],strict=True) if not m]
    if row['decision_tokens']!=len(decision):raise ValueError('Decision token count changed')
    close(row['all_supervised_mean_nll'],mean(values));close(row['decision_sentence_mean_nll'],mean(decision));close(row['remaining_target_mean_nll'],mean(rest))
    if len(row['teacher_segments'])!=len(encoded['target_segments']):raise ValueError('Teacher segment count differs')
    for i,(actual,target_segment) in enumerate(zip(row['teacher_segments'],encoded['target_segments'],strict=True)):
        if any(actual[k]!=v for k,v in target_segment.items()):raise ValueError('Teacher segment text changed')
        values_i=[v for v,index in zip(values,encoded['token_segment_indices'],strict=True) if index==i]
        if actual['supervised_tokens']!=len(values_i):raise ValueError('Segment token count differs')
        close(actual['mean_nll'],mean(values_i))


def token_summary(rows,pack):
    by={r['id']:r for r in rows};out={}
    for style in ['original','paraphrase']:
        selected=[by[pair[style]] for pair in pack['pairs']]
        flat=[r for target in selected for r in target['tokens']]
        out[style]={'targets':len(selected),'supervised_tokens':len(flat),
            'gold_prefix_top1_errors':sum(not r['gold_prefix_top1_equals_target'] for r in flat),
            'targets_with_any_gold_prefix_top1_error':sum(any(not r['gold_prefix_top1_equals_target'] for r in t['tokens']) for t in selected),
            'token_weighted_mean_nll':mean([r['nll'] for r in flat]),
            'unweighted_target_mean_nll':mean([t['all_supervised_mean_nll'] for t in selected]),
            'decision_token_mean_nll':mean([r['nll'] for r in flat if r['in_decision_sentence']]),
            'conditional_on_gold_prefix_not_free_generation_accuracy':True}
    return out


def source_scope_inventory(course_pack,probe_pack):
    """Visible domain wording only; flags are not post-hoc target relabels."""
    import re
    missing=[]
    for row in course_pack['records']:
        if row['split']!='train' or row.get('skill')!='marriage':continue
        for index,message in enumerate(row['messages']):
            if message['role']!='assistant':continue
            visible='\n'.join(m['content'] for m in row['messages'][:index])
            if not re.search('婚|恋爱',visible):missing.append({'id':row['id'],'assistant_turn_index':index})
    probe_ids=[]
    for row in probe_pack['records']:
        c=row.get('provenance',{}).get('lesson_contract',{})
        if c.get('kind')=='marriage' and not any(re.search('婚|恋爱',m['content']) for m in row['messages'][:-1]):probe_ids.append(row['id'])
    return {'marriage_supervised_views_without_explicit_visible_domain':missing,
            'static_probe_marriage_questions_without_explicit_visible_domain':probe_ids,
            'flags_are_review_candidates_not_silent_target_relabels':True,'metadata_domain_is_not_model_visible_context':True}


def audit(archive,output,notes=None,tokenizer_root=None):
    archive=Path(archive)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact actual V65 learning ZIP required')
    runtime.configure();course_pack=runtime.verify_data();pack=data.build(course_pack);remarks=d.read(notes) if notes else {};details=[];workers=[];models={};tokens={};activations={}
    from transformers import AutoTokenizer
    config=d.read(runtime.CONFIG)
    tok=AutoTokenizer.from_pretrained(str(tokenizer_root) if tokenizer_root else config['model_id'],revision=None if tokenizer_root else config['revision'],local_files_only=True,trust_remote_code=False)
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        names=z.namelist();run=json.loads(z.read('run_contract.json'));state=json.loads(z.read('run_status.json'));prompts=json.loads(z.read('prompts.json'))
        if (len(names)!=26 or run['schema']!=probe.SCHEMA or run['code_sha256']!=probe.hashes()
            or prompts!=pack or run['prompts_sha256']!=d.digest(z.read('prompts.json'))
            or d.digest(d.dumps(pack))!=d.read(probe.PINS)['diagnostic_pack_sha256']
            or run['config']!=config or run['software_fixture'] or not run['inference_only'] or run['weight_updates']
            or run['professional_promotion'] or state['state']!='complete_review_required' or state['weight_updates']
            or state['professional_promotion'] or state['summary']!=pack['summary']):raise ValueError('Frozen actual source/data/probe contract differs')
        encoded={r['id']:token.encoded_target(tok,r,4096) for r in pack['records'] if r['teacher_forcing']}
        targets=[r for r in pack['records'] if r['teacher_forcing']]
        for label in probe.LABELS:
            source=run['sources'][label]
            if source['adapter_sha256']!=SOURCES[label] or source['files']['adapter_model.safetensors']!=SOURCES[label]:raise ValueError('Different saved source adapter')
            proof=json.loads(z.read(f'answers/{label}.provenance.json'))
            if proof!={'schema':probe.SCHEMA,'source':source,'config':config,'prompts_sha256':d.digest(d.dumps(pack)),'inference_only':True,'weight_updates':0}:
                raise ValueError('Per-model source or prompt contract changed')
            rows=read_rows(z,f'answers/{label}.jsonl');probe.validate_answers(rows,pack,label);models[label]=rows
            gold=data.sequence(pack,rows)
            if len(rows)!=81 or len(gold)!=81:raise ValueError('Missing actual own-history turns')
            if remarks:verified_notes(remarks,label,rows)
            for i,(row,g) in enumerate(zip(rows,gold,strict=True)):
                if any(row[k]!=g[k] for k in ['components','source_ids','task','group']):raise ValueError('Source component facts changed')
                details.append({'tag':label,'index':i,'row':row,'gold':g|{'skill':g['task']},'frozen_check':None,'annotation':bound_note(remarks,label,row)})
            scores=read_rows(z,f'answers/{label}.tokens.jsonl');tokens[label]=scores
            if len(scores)!=38:raise ValueError('Missing forced token targets')
            for row,target in zip(scores,targets,strict=True):
                verify_token_row(row,target,encoded[target['id']])
                if any(r['target_text']!=tok.decode([r['target_id']]) for r in row['tokens']):raise ValueError('Token text differs from pinned tokenizer')
            complete=json.loads(z.read(f'answers/{label}.complete.json'))
            if (complete['answers']!=81 or complete['teacher_forced_targets']!=38 or complete['weight_updates']
                or not complete['source_weights_unchanged'] or complete['answer_sha256']!=d.digest(z.read(f'answers/{label}.jsonl'))
                or complete['tokens_sha256']!=d.digest(z.read(f'answers/{label}.tokens.jsonl'))
                or {k:v for k,v in state['models'][label].items() if k not in ['state','cache_hit_without_model_load']}!=complete):
                raise ValueError('Complete answer/token bytes or completion receipt changed')
            activation=json.loads(z.read(f'answers/{label}.activation.json'));activations[label]=activation
            if (activation['software_fixture'] or activation['adapter_sha256']!=SOURCES[label]
                or activation['lora_B_matrices']!=252 or activation['nonzero_lora_B_matrices']!=252
                or activation['max_last_position_logit_delta']<=0 or not activation['parameters_frozen'] or activation['weight_updates']):
                raise ValueError('Actual nonzero frozen adapter effect was not verified')
            attention=json.loads(z.read(f'answers/{label}.attention.json'))
            if attention['software_fixture'] or not attention['cuda_calls'] or attention['math_fallback_allowed']:raise ValueError('Actual fused CUDA path missing')
        for name in (n for n in names if n.endswith('.result.json')):
            raw=z.read(name.replace('.result.json','.json'));request=json.loads(raw);reply=json.loads(z.read(name));result=probe.verify_reply(reply,request,raw)
            if (request['software_fixture'] or request['config']!=run['config'] or request['source']!=run['sources'][request['label']]
                or result!=state['models'][request['label']] or result['state']!='complete' or result['cache_hit_without_model_load']):
                raise ValueError('Actual new worker request or reply differs')
            workers.append({'label':request['label'],'pid':reply['pid'],'request_sha256':d.digest(raw),'weight_updates':0})
        if len(workers)!=2:raise ValueError('Exactly two actual model workers required')
    examples=[];after={r['id']:r for r in tokens['v65_epoch2']}
    free_after={r['id']:r for r in models['v65_epoch2']}
    for pair in pack['pairs']:
        row=after[pair['original']];bad=[r for r in row['tokens'] if not r['gold_prefix_top1_equals_target']]
        if bad:
            enc=encoded[pair['original']];first=bad[0];start=first['character_offsets'][0]-enc['answer_start_character']
            prefix=free_after[pair['original']]['expected'][:start];predicted=tok.decode([first['gold_prefix_top1_id']])
            examples.append({'pair_id':pair['id'],'supervised_tokens':row['supervised_tokens'],'mean_nll':row['all_supervised_mean_nll'],
                'first_missed_top1_matches_actual_free_prefix':free_after[pair['original']]['answer'].startswith(prefix+predicted),
                'missed_targets':[{'target':r['target_text'],'top1':tok.decode([r['gold_prefix_top1_id']]),'nll':r['nll']} for r in bad]})
    summary={'schema':'wengua_actual_v65_learning_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':26,'frozen_runtime_files':len(run['code_sha256']),
        'actual_workers':workers,'free_answers':162,'teacher_forced_targets':76,'weight_updates':0,'source_adapter_sha256':SOURCES,'activations':activations,
        'all_target_ids_offsets_segments_and_aggregates_verified':True,'actual_logits_independently_recomputed':False,
        'raw_source_weight_tensors_uploaded':False,'all_questions_known_development':True,'independent_case_accuracy_claimed':False,
        'whole_board_prediction_evaluated':False,'source_models_used_as_teachers':False,'original_test_used':False,'professional_promotion':False,
        'summary':pack['summary'],'notes_bound_to_answers':sum(bool(r['annotation']) for r in details),'token_diagnostics':{k:token_summary(v,pack) for k,v in tokens.items()},
        'source_scope_candidates':source_scope_inventory(course_pack,pack),
        'original_after_first_missed_token_examples':examples,
        'free_answer_shapes':{k:{'answers':len(v),'empty':sum(r['empty'] for r in v),'token_limit_hits':sum(r['reached_token_limit'] for r in v),
            'literal_original_matches':sum(r['answer']==r['expected'] for r in v if r['task']=='training_original')} for k,v in models.items()}}
    if remarks:
        summary['bounded_reading_observations']={}
        for label,rows in models.items():
            notes_=[remarks[label][r['id']] for r in rows]
            summary['bounded_reading_observations'][label]={'star_positions':sum(n.get('positions_supported',False) for n in notes_),
                'star_palace_directions':sum(n.get('directed_palace_relation_supported',False) for n in notes_),
                'star_position_and_direction_joint':sum(n.get('positions_supported',False) and n.get('directed_palace_relation_supported',False) for n in notes_),
                'dialogue_goal_and_current_state':sum(n.get('dialogue_goal_and_current_state_supported',False) for n in notes_),
                'bank_main_result':sum(n.get('bank_main_result_supported',False) for n in notes_),
                'known_development_manual_component_observation_only':True}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for r in details:f.write(d.dumps(r)+'\n')
    return summary,details


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--notes',type=Path);p.add_argument('--tokenizer-root',type=Path)
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/model_reviews/v65_learning');p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');a=p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected existing textbook key')
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261112_v65_learning/private_review_catalog.json')
        print(restore(entry,bytes.fromhex(value))/'review/Wengua_V65_学习定位逐答审阅.xlsx');return 0
    if a.archive is None:p.error('archive or --restore required')
    summary,details=audit(a.archive,a.output,a.notes,a.tokenizer_root)
    if a.workbook:
        from .v61_review import workbook
        workbook(a.output/'Wengua_V65_学习定位逐答审阅.xlsx',details)
    print(d.dumps({'free_answers':len(details),'forced_targets':76,'bound_notes':summary['notes_bound_to_answers']}));return 0


if __name__=='__main__':raise SystemExit(main())
