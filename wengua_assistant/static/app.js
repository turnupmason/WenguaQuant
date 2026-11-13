'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name=wengua-token]').content;
let session = null, busy = false;
const localZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
const civil = new Date(Date.now() - new Date().getTimezoneOffset()*60000).toISOString().slice(0,16);
$('time').value = civil;
async function api(path, payload) {
  const options = {headers:{'X-Wengua-Token':token}};
  if (payload !== undefined) { options.method='POST'; options.headers['Content-Type']='application/json'; options.body=JSON.stringify(payload); }
  const response = await fetch(path, options), value = await response.json();
  if (!response.ok) throw new Error(value.error || '请求失败');
  return value;
}
function text(parent, tag, value, cls) { const node=document.createElement(tag); node.textContent=value; if(cls)node.className=cls; parent.append(node); return node; }
function reset() {
  if(busy)return; session=null; $('messages').replaceChildren(); $('status').textContent='已新开对话。';
  $('chart').replaceChildren();$('sources').replaceChildren();$('positions').textContent='';$('budget').textContent='';
  $('clock').textContent='由原有排盘程序生成';$('relation').value='';$('background').value='';
  $('birthStatus').value='not_entered';$('birthStem').value='';
}
$('newChat').onclick=reset;
$('export').onclick=async()=>{
  if(!session||busy){$('status').textContent='请先完成一次问答，再下载本次对话。';return;}
  try{
    const response=await fetch('/api/export?session='+encodeURIComponent(session),{headers:{'X-Wengua-Token':token}});
    if(!response.ok)throw new Error((await response.json()).error);
    const url=URL.createObjectURL(await response.blob()),link=document.createElement('a');
    link.href=url;link.download='Wengua_问卦对话_'+new Date().toISOString().replace(/[:.]/g,'-')+'.zip';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
  }catch(error){$('status').textContent=error.message;}
};
$('mode').onchange=()=>{ const reading=$('mode').value==='reading'; $('timeLabel').hidden=!reading; $('zoneLabel').hidden=!reading; };
function preview(p) {
  if(p.chart) {
    const packet=p.chart, c=packet.chart, pillars=packet.calendar.four_pillars;
    $('clock').textContent=packet.calendar.reference_civil_local.slice(0,19).replace('T',' ')+' · '+['year','month','day','hour'].map(k=>pillars[k]).join(' ')+' · '+(c.dun==='yang'?'阳遁':'阴遁')+c.ju+'局';
    $('chart').replaceChildren();
    for(const n of [4,9,2,3,5,7,8,1,6]) {
      const cell=c.palaces[String(n)], node=document.createElement('div');node.className='palace';
      text(node,'strong',cell.gua+n+'宫');text(node,'div','地盘 '+cell.earth_stem);
      text(node,'div',cell.heaven_stars.map(s=>s.star+' '+s.carried_stem).join(' / ')||'中宫');
      text(node,'div',[cell.door,cell.deity].filter(Boolean).join(' · '));$('chart').append(node);
    }
    $('positions').textContent=JSON.stringify(p.pillar_positions,null,2);
  }
  $('sources').replaceChildren();
  for(const ref of p.references) {
    const detail=document.createElement('details');detail.className='source';
    text(detail,'summary','['+ref.citation+'] '+ref.title+' · '+ref.retrieval_stage);
    text(detail,'p',ref.sources.map(s=>'《'+s.book_title+'》'+s.chapter+' 第'+s.lines.join('—')+'行').join('；'));
    text(detail,'pre',ref.reading_content||ref.content);$('sources').append(detail);
  }
  $('budget').textContent=JSON.stringify({预算:p.budget,检索:p.retrieval_plan},null,2);
}
$('form').onsubmit=async event=>{
  event.preventDefault(); if(busy)return;
  const question=$('question').value.trim();if(!question)return;
  busy=true;$('send').disabled=true;
  for(const id of ['variant','mode','time','zone','newChat']) $(id).disabled=true;
  $('status').className='';
  const user=text($('messages'),'div','', 'message user');text(user,'b','你');text(user,'div',question);
  $('question').value='';$('status').textContent='准备本局与原书依据…';
  try {
    const start=await api('/api/ask',{session,question,variant:$('variant').value,mode:$('mode').value,domain:$('domain').value,
      time:$('mode').value==='reading'?$('time').value:null,zone:$('mode').value==='reading'?($('zone').value==='local'?localZone:$('zone').value):null,
      facts:{background:$('background').value,asker_relation:$('relation').value,birth_status:$('birthStatus').value,birth_stem:$('birthStem').value}});
    session=start.session;let shown=false;
    while(true) {
      const job=await api('/api/job?id='+encodeURIComponent(start.job));
      $('status').textContent=job.progress?.phase||job.phase||'等待本地模型…';
      if(job.preview&&!shown){preview(job.preview);shown=true;}
      if(job.state==='failed')throw new Error(job.error);
      if(job.state==='complete') {
        const answer=text($('messages'),'div','', 'message');text(answer,'b',$('variant').selectedOptions[0].textContent);
        text(answer,'div',job.result.raw_answer||'模型没有生成可显示的文字。');
        const check=job.result.checks;
        if(check.status!=='citations_checked_semantics_unverified')text(answer,'p','核对提示：'+check.status+'。原始回答已保留，请展开下面记录查看。','check');
        if(job.result.reached_token_limit)text(answer,'p','回答达到本轮长度上限，可以继续追问。','check');
        const detail=document.createElement('details');text(detail,'summary','原始答复核对与保存位置');text(detail,'pre',JSON.stringify({核对:check,保存:job.trace},null,2));answer.append(detail);
        $('status').textContent='回答完成。当前对话可以继续追问。';break;
      }
      await new Promise(resolve=>setTimeout(resolve,1500));
    }
  } catch(error){$('status').textContent=error.message;$('status').className='error';}
  finally{busy=false;$('send').disabled=false;for(const id of ['variant','mode','time','zone','newChat'])$(id).disabled=false;}
};
$('chatTab').onclick=()=>{$('chatPanel').hidden=false;$('casePanel').hidden=true;};
$('caseTab').onclick=()=>{$('chatPanel').hidden=true;$('casePanel').hidden=false;};
$('cases').onchange=async()=>{
  const parent=$('caseContent');parent.replaceChildren();if(!$('cases').value)return;
  try{const result=await api('/api/case?id='+encodeURIComponent($('cases').value));
    text(parent,'h2',result.case.case_id);text(parent,'p','原书转录；来源复盘，不是独立预测验收。');
    const chart=document.createElement('details');text(chart,'summary','原盘及原案登记');text(chart,'pre',JSON.stringify(result.case,null,2));parent.append(chart);
    for(const doc of result.documents){const detail=document.createElement('details');text(detail,'summary',doc.phase+' · '+doc.title);text(detail,'p',doc.sources.map(s=>'《'+s.book_title+'》'+s.chapter+' 第'+s.lines.join('—')+'行').join('；'));text(detail,'pre',doc.reading_content||doc.content);if(doc.analysis_annotations)text(detail,'pre',JSON.stringify(doc.analysis_annotations,null,2));parent.append(detail);}
  }catch(error){text(parent,'p',error.message,'error');}
};
api('/api/info').then(info=>{
  const permitted=new Set(['health','marriage','pregnancy','education','employment','business_profit','travel_abroad','missing_person','lost_property','criminal_case','litigation','sports_competition','weather','competition_evaluation','life_opportunity_book_study','military_history_source','geography_environment_historical_source','politics_elections','industrial_safety','world_affairs','miscellaneous_historical_source']);
  for(const [key,label] of Object.entries(info.domains)){if(!permitted.has(key))continue;const o=text($('domain'),'option',label);o.value=key;}
  text($('cases'),'option','选择原书案例（'+info.cases.length+'案）').value='';
  for(const c of info.cases){const o=text($('cases'),'option',(info.domains[c.domain]||c.domain)+' · '+c.id+' · '+c.author_steps+'段分析');o.value=c.id;}
  if(!info.candidate_present){$('variant').value='official';$('status').textContent='未找到V6.7第一轮目录，当前选择原模型；不会自动下载权重。';}
}).catch(error=>{$('status').textContent=error.message;});
