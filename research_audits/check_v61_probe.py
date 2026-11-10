"""Real tiny CPU four-variant inference, cache, export and native CMD fixture."""
from pathlib import Path
import tempfile,os,sys,subprocess
from wengua_course import data as d
from wengua_course_v61 import runtime
from . import v61_probe as probe

def main():
    runtime.configure()
    from wengua_course.fixtures import fixture_factory
    from wengua_course_v53 import runtime as old,initialization as oi
    tok,_,_=fixture_factory();ROOT=runtime.ROOT
    with tempfile.TemporaryDirectory(prefix='v61_probe_cpu_',dir=ROOT/'local') as tmp:
        root=Path(tmp);p=old.profile(testing=True);c=d.read(old.CONFIG)|{'initialization_only':True}
        job=p.engine.open_job(root/'seed/trials/init',c,resume=True);created=oi.create(p,job,c,factory=fixture_factory);source=Path(created['adapter']);sha=d.digest((source/'adapter_model.safetensors').read_bytes());original={p.name:d.digest(p.read_bytes()) for p in source.iterdir() if p.is_file()}
        rows=[]
        for i in range(3):
            rows.append({'id':'fixture:'+str(i),'question':'question one '+str(i),'answer':'answer two','messages':[{'role':'user','content':'question one '+str(i)},{'role':'assistant','content':'answer two'}],'source_ids':[],'format':'normal_dialogue_diagnostic','task':'fixture','group':'fixture','source_mode':'software_fixture_only'})
        pack={'records':rows,'software_fixture_only':True};sources={k:{'adapter':None if k=='official_base' else str(source),'adapter_sha256':None if k=='official_base' else sha,'variant':k,'software_fixture_only':True} for k in probe.LABELS}
        config=d.read(runtime.CONFIG)|{'max_seq_length':512,'inference_max_context':512}
        out=root/'probe'
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=config)==0
        state=d.read(out/'run_status.json');assert state['generated_answers']==12 and state['weight_updates']==0
        assert {p.name:d.digest(p.read_bytes()) for p in source.iterdir() if p.is_file()}==original
        first={p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}
        with (out/'answers/official_base.jsonl').open('ab') as f:f.write(b'{"interrupted":"\xff')
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=config)==0
        assert {p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}==first
        assert len(list((out/'answers').glob('*.partial-*')))==1
        assert all(m['cache_hit_without_model_load'] for m in d.read(out/'run_status.json')['models'].values())
        from wengua_course.runner import export_review
        assert export_review(out,root/'probe.zip').is_file()
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            r=subprocess.run(['cmd.exe','/d','/c',str(ROOT/'check_wengua_v61_dialogue.cmd'),'--help'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert r.returncode==0 and 'export' in r.stdout and 'run' in r.stdout
        result={'software_fixture_only':True,'four_variants_actual_tiny_inference':True,'all_source_adapter_bytes_unchanged':True,'cache_rerun_without_model_load':True,'export_complete':True,'weight_updates':0,'actual_8b_cuda_quality_claimed':False}
        d.write(ROOT/'local/v61_probe_cpu_verification.json',result);print(d.dumps(result))
if __name__=='__main__':main()
