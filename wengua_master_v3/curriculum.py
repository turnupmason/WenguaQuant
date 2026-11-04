"""Source-grounded Wengua Master V3 curriculum; separate from outcome research."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

PALACES={1:("坎","水"),2:("坤","土"),3:("震","木"),4:("巽","木"),6:("乾","金"),7:("兑","金"),8:("艮","土"),9:("离","火")}
GENERATES={"木":"火","火":"土","土":"金","金":"水","水":"木"}
CONTROLS={"木":"土","土":"水","水":"火","火":"金","金":"木"}
SYMBOL_PAIRS=(("天心星","天芮星"),("乙奇","天芮星"),("八神值符","天芮星"),("时干","天芮星"),("父亲的年干用神","天芮星"),("朋友的月干用神","天芮星"),("天心星","患者年干用神"),("乙奇","患者年干用神"))
REVERSE={"克":"被克","被克":"克","生":"被生","被生":"生","比和":"比和","同宫":"同宫"}
SYSTEM="你是问卦传统盘面事实练习助手。仅判给定宫位五行方向，不虚构完整九宫、出生年命、符号原始力量、医疗结论或应期。"
HISTORIC_SOURCE_SHA="21362e6f0266c03dcdc4a7ef35f45954ba274ae6458de0ba341132abca7d0405"
SPLIT_ORDER={"train":0,"validation":1,"test":2,"quarantine":3}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def dumps(o):return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def group_for_pair(a,b):
    x,y=sorted((a,b))
    return f"teaching-palace-pair:{x}:{y}"
def stable_split(group):
    v=int(sha(group.encode())[:8],16)%100
    return "test" if v>=90 else "validation" if v>=80 else "train"
def relation(a,b):
    if a not in PALACES or b not in PALACES:raise ValueError("Middle 5 cannot have an ordinary star/door/deity")
    if a==b:return "同宫"
    x,y=PALACES[a][1],PALACES[b][1]
    if x==y:return "比和"
    if GENERATES[x]==y:return "生"
    if GENERATES[y]==x:return "被生"
    if CONTROLS[x]==y:return "克"
    if CONTROLS[y]==x:return "被克"
    raise AssertionError("Missing elemental relation")
def _messages(question,answer,system=SYSTEM):
    return [{"role":"system","content":system},{"role":"user","content":question},{"role":"assistant","content":answer}]
def build_drills():
    """Hypothetical eight-palace fact drills, NOT real date-time casts or outcomes."""
    out=[]
    for a in PALACES:
        for b in PALACES:
            na,ea=PALACES[a];nb,eb=PALACES[b];r=relation(a,b);inv=REVERSE[r]
            for i,(left,right) in enumerate(SYMBOL_PAIRS):
                group=group_for_pair(a,b)
                context=f"教学假设：{left}在{na}{a}宫（{ea}），{right}在{nb}{b}宫（{eb}）。它不是一张真实排盘。"
                for mode,question,answer in (
                    ("direction",context+f"从{left}所在宫对{right}所在宫，五行关系是什么？",
                     f"从{na}{a}宫到{nb}{b}宫的物理宫位关系是「{r}」。这只确定五行方向，不能由此得出任何符号的源力量、最终吉凶或现实结果。"),
                    ("bidirectional",context+"分别从前者对后者、后者对前者说明关系；它们的符号力量能否直接互相复制？",
                     f"前者对后者为「{r}」，反向为「{inv}」。不同独立符号需另行读取各自源强。"+("同一物理宫的基础吉凶只计一次，不能按角色重复。" if a==b else "不能把宫位生克方向当作已算好的符号力量。"))
                ):
                    out.append({"id":f"v3.palace.{a}.{b}.roles.{i}.{mode}","task":"palace_"+mode,
                        "group":group,"split":stable_split(group),"curriculum_stage":"palace",
                        "source":"WenguaApp:data/knowledge/shenqi/shijia_symbols_ch04/symbol_index.json",
                        "evidence_kind":"synthetic_rule_exercise","real_chart_verified":False,
                        "fresh_prediction_gold":False,"medical_prediction_gold":False,
                        "messages":_messages(question,answer)})
    assert len(out)==1024
    return out
def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
def private_from_seed(seed,v2_dir=None):
    if seed.get("schema")!="wengua_master_v3_private_codex_source_review_v1":raise ValueError("Wrong seed schema")
    cases=seed["cases"];steps=seed["author_steps"];qas=seed["qa"]
    case={x["case_id"]:x for x in cases}
    if len(case)!=len(cases):raise ValueError("Duplicate case")
    splits={cid:x["source_split"] for cid,x in case.items()}
    out=[];seen=set();bycase=defaultdict(list)
    for s in steps:
        cid=s["case_id"];num=s["step_number"]
        if cid not in case:raise ValueError("Orphan author step")
        if (cid,num) in seen:raise ValueError("Duplicate author step")
        seen.add((cid,num));bycase[cid].append(s)
        answer=s.get("source_text","")
        if not answer:continue
        out.append({"id":f"v3.workbook.author.{cid}.{num}","task":"source_author_stage_study",
            "group":cid,"split":splits[cid],"citation":s.get("citation"),
            "source_review_status":s.get("workbook_review_status"),"user_authorized_use":True,
            "human_expert_review_done":False,"fresh_prediction_gold":False,
            "long_context_check_required":len(answer)>1700,
            "messages":_messages(f"复盘原书案例{cid}第{num}段（{s['topic']}），只作作者传统分析复盘。",answer,
              "保留原书来源与后验披露边界；不把作者事后反馈当事前事实。")})
    for cid,ss in bycase.items():
        joined="\n".join(f"第{x['step_number']}步（{x['topic']}）：{x['source_text']}" for x in sorted(ss,key=lambda s:s["step_number"]) if x.get("source_text"))
        if not joined:continue
        out.append({"id":f"v3.workbook.full.{cid}","task":"source_author_full_case_study",
            "group":cid,"split":splits[cid],"source_review_status":case[cid].get("workbook_review_status"),
            "user_authorized_use":True,"human_expert_review_done":False,"fresh_prediction_gold":False,
            "long_context_check_required":len(joined)>1700,
            "messages":_messages(f"复盘原书历史案例{cid}的全部原书作者步骤；不是对新时辰的预测。",joined,
             "历史作者分析研读；原有断语和后来反馈不得伪装成一次新的事前预测。")})
    for q in qas:
        sid=q["sample_id"]
        cid=sid.split(":step:",1)[0].removeprefix("author:") if sid.startswith("author:") else None
        split=splits[cid] if cid in case else q.get("source_split","quarantine")
        if split not in SPLIT_ORDER:split="quarantine"
        out.append({"id":"v3.workbook.qa."+sid,"task":q.get("task","qa"),
            "group":cid or ("qa-source:"+sid),"split":split,
            "source_review_status":q.get("workbook_review_status"),"user_authorized_use":True,
            "human_expert_review_done":False,"fresh_prediction_gold":False,
            "messages":_messages(q.get("question",""),q.get("answer",""),"仅依原书来源与具名规则回答，不编造条件。")})
    additional=0;reg_count=0
    if v2_dir:
        base=Path(v2_dir);registry=base/"case_registry.jsonl";units=base/"source_units.jsonl"
        if not registry.is_file() or not units.is_file():raise ValueError("Missing private V2 export")
        reg={x["case_id"]:x for x in load_jsonl(registry)};reg_count=len(reg)
        for u in load_jsonl(units):
            if u.get("kind")!="author_explicit_paragraph":continue
            cid=u.get("case_id");num=u.get("original_step_number")
            if cid not in reg:raise ValueError("Orphan original V2 author step")
            if (cid,num) in seen:continue
            text=u.get("text","")
            if not text:continue
            info=reg[cid]
            out.append({"id":f"v3.v2.author.{cid}.{num}","task":"source_author_stage_study",
                "group":cid,"split":info.get("split","quarantine"),
                "source_id":u.get("id"),"source_sha256":u.get("source_sha256"),
                "human_expert_review_done":False,"fresh_prediction_gold":False,
                "long_context_check_required":len(text)>1700,
                "messages":_messages(f"按原书复盘{cid}第{num}段作者分析。",text,
                  "只能复盘来源，不能将后来结果当事前预测。")})
            seen.add((cid,num));additional+=1
        # Consolidate overlapping source records and all linked event components.
        for row in out:
            cid=row["group"]
            if cid in reg:
                record=reg[cid]
                row["group"]="v2-component:"+str(record.get("evidence_component",record.get("event_family_id",cid)))
                row["split"]=max((row["split"],record.get("split","quarantine")),
                                 key=lambda x:SPLIT_ORDER[x])
        groups=defaultdict(list)
        for row in out:groups[row["group"]].append(row)
        for rows in groups.values():
            conservative=max((r["split"] for r in rows),key=lambda x:SPLIT_ORDER[x])
            for r in rows:r["split"]=conservative
    return out,{"workbook_cases":len(case),"workbook_author_steps":len(steps),
      "workbook_qa":len(qas),"v2_case_registry":reg_count,"v2_added_author_steps":additional}
def validate_release(rows):
    ids=set();groups=defaultdict(set)
    for r in rows:
        if r["id"] in ids:raise ValueError("Duplicate SFT id")
        ids.add(r["id"])
        if r.get("fresh_prediction_gold") or r.get("medical_prediction_gold"):raise ValueError("Unsupported predictive gold")
        if r.get("split") not in SPLIT_ORDER:raise ValueError("Invalid split")
        if not r.get("group"):raise ValueError("Missing source/event family")
        groups[r["group"]].add(r["split"])
        m=r.get("messages")
        if not m or m[-1].get("role")!="assistant" or not m[-1].get("content","").strip():raise ValueError("Empty assistant target")
    if any(len(s)!=1 for s in groups.values()):raise ValueError("Source family crossed splits")
    return {"rows":len(rows),"groups":len(groups),"splits":dict(Counter(r["split"] for r in rows)),
            "tasks":dict(Counter(r["task"] for r in rows)),"fresh_prediction_gold":0}
def prepare(output,private_seed=None,v2_dir=None):
    output=Path(output)
    if output.exists():raise FileExistsError("Refuse to overwrite V3 output")
    if v2_dir and not private_seed:raise ValueError("Require private reviewed source seed when linking V2")
    public=build_drills();pa=validate_release(public);private=[];stats={};seed=None
    if private_seed:
        seed=json.loads(Path(private_seed).read_text(encoding="utf-8"))
        private,stats=private_from_seed(seed,v2_dir);validate_release(private)
    output.mkdir(parents=True,exist_ok=False)
    (output/"public_rule_drills.jsonl").write_text("".join(dumps(x)+"\n" for x in public),encoding="utf-8")
    if private:
        for split in SPLIT_ORDER:
            (output/f"private_source_study_{split}.jsonl").write_text(
                "".join(dumps(x)+"\n" for x in private if x["split"]==split),encoding="utf-8")
    report={"schema":"wengua_master_v3_source_sft_candidate_v1",
       "generated_utc":datetime.now(timezone.utc).isoformat(),
       "public_rule_drills":pa,"private_study":validate_release(private) if private else None,
       "source_summary":stats,"workbook_sha256":seed.get("source_workbook_sha256") if seed else None,
       "private_seed_sha256":sha(Path(private_seed).read_bytes()) if private_seed else None,
       "historic_source_archive_sha256":HISTORIC_SOURCE_SHA,
       "expert_semantic_review_complete":False,"fresh_cast_full_reading_gold":0,
       "existing_qwen14b_v2_changed":False,"training_started":False,"production_enabled":False,
       "files":{p.name:sha(p.read_bytes()) for p in output.glob("*.jsonl")},
       "limits":["Palace drills are hypotheticals, not historical casts or outcomes.",
                 "Original book paragraphs remain in private local files.",
                 "Case source-study is NOT verified fresh-question prediction.",
                 "Full-board SFT awaits verified canonical cast facts and eligible expert answers."]}
    (output/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
