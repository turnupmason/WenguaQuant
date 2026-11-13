"""Select literal two-book evidence for actual questions, without SFT targets."""
from functools import lru_cache
import json, math, re
from wengua_course import data as d
from wengua_course_v54 import corpus
from wengua_course_v55.corpus import source_plan

# These are search queries, not newly authored professional rules.
STAGES = {
    'health': [('取用与人物关系', '疾病 患者自己求测 儿女辈求测长辈 年命'),
               ('疾病与治疗', '预测疾病 天芮 天心 乙奇 治疗 生克'),
               ('条件与应期', '疾病 应期 旺衰 空亡 入墓')],
    'lost_property': [('取用', '失物 日干 时干 失主'),
                      ('方位与环境', '失物 方位 内盘 外盘 物品'),
                      ('条件与应期', '失物 找回 空亡 应期')],
    'employment': [('取用', '工作 求职 日干 开门 杜门'),
                   ('条件', '求职 工作 生克 旺衰 空亡')],
    'marriage': [('取用', '婚姻 乙 庚 六合 日干 时干'),
                 ('条件', '婚姻 生克 空亡 合 冲')],
    'business_profit': [('取用', '求财 日干 戊 生门 本钱 利润'),
                        ('条件', '经营 求财 生克 旺衰 空亡')],
    'education': [('取用', '考试 日干 天辅 景门 丁奇'),
                  ('条件', '考试 录取 生克 旺衰 空亡')],
}


@lru_cache(maxsize=1)
def checked_index():
    # Existing read-only index validates chapter/selector hashes. No answers,
    # teacher summaries, old scoring tables or new books are added to it.
    library = corpus.source_library()
    corpus.r.validate_library(library)
    if set(library['books']) != {'shenqi', 'kaiwu'}:
        raise ValueError('只使用已经整理的两本书')
    return corpus.ensure_index()


def _bibliography(index, doc):
    for source in doc['sources']:
        key = source['chapter']
        source['chapter_id'] = key
        source['chapter'] = index.chapters[key]['title']
    return doc


@lru_cache(maxsize=1)
def literal_candidates():
    """A separate read-only ranker; do not patch the frozen old retriever."""
    with corpus.r.Index(checked_index()) as index:
        rows = []
        for raw, in index.db.execute('SELECT payload FROM documents'):
            doc = json.loads(raw)
            if doc['kind'] != 'rule' or doc.get('case_ids'): continue
            content = corpus.r.document_text(index.chapters, doc)
            rows.append((doc, content, set(corpus.r.terms(content))))
        return rows


def retrieve(question, domain=None, mode='knowledge'):
    if mode not in {'knowledge', 'reading'}:
        raise ValueError('实际问答只能检索通用规则；原案在独立页面查看')
    domains = corpus.r.domains_for(question, domain)
    plan = []
    with corpus.r.Index(checked_index()) as index:
        # The old timing ranker can put 九遁/五假 before a patient's actual
        # process. Rank all verified rule selectors here with explicit scope.
        # Ranking only selects original text; it does not invent an answer.
        def search(query, selected, anchors=()):
            query = corpus.r.normalize(query); terms = set(corpus.r.terms(query))
            wanted = [x for x in corpus.r.SYMBOLS if x in query]
            candidates = []
            for doc, content, words in literal_candidates():
                scoped = set(doc.get('domains', []))
                if selected and scoped and selected not in scoped: continue
                if selected not in {'criminal_case', 'litigation'} and any(x in content for x in ('刑侦', '案犯', '抢劫案')):
                    continue
                # Do not pass a dangling numbered dependency or a worked-case
                # preparation paragraph off as a self-contained general rule.
                if re.match(r'\s*[（(]?\d+[)）]?\s*如果没有第', content): continue
                if content.startswith(('第一项准备工作', '以上所述', '以上九宫所含')): continue
                if selected and not scoped:
                    active_words = corpus.r.DOMAINS.get(selected, ())
                    unrelated = [key for key, terms_ in corpus.r.DOMAINS.items()
                                 if key != selected and any(word in content for word in terms_)]
                    if unrelated and not any(word in content for word in active_words): continue
                if anchors and not any(x in content for x in anchors): continue
                overlap = len(terms & words)
                if not overlap: continue
                symbols = sum(x in content for x in wanted)
                # Penalize duplicated union bundles and unrelated casting;
                # whole original selectors are never cut into partial rules.
                score = (overlap + 4 * symbols + (5 if selected in scoped else 0)) / (1 + math.log2(1 + len(content)) / 8)
                score -= max(0, len(doc['parts']) - 1) * 2
                if doc.get('topic') == 'casting' and not any(x in query for x in ('排盘', '起局')): score -= 30
                candidates.append((-score, doc['id'], doc))
            docs = [index.expand(x[2]) for x in sorted(candidates)[:12]]
            return {'mode': mode, 'domains': domains, 'documents': docs,
                    'index_sha256': corpus.r.digest(index.path.read_bytes())}
        direct = search(question, domain)
        # Keep the actual question first. Add domain processes even when the
        # user did not literally ask for “流程”. No universal outcome rule.
        ranked = [(x, '当前问题') for x in direct['documents'][:1]]
        if len(domains) == 1:
            selected = domains[0]
            stages = STAGES.get(selected, [('领域取用', d.DOMAIN_ZH.get(selected, selected) + ' 用神 判断 生克 条件')])
            for label, query in stages:
                anchors = ('应期', '应验', '冲实', '填实', '出空') if '应期' in label else ()
                if selected == 'health' and label == '取用与人物关系': anchors = ('患者自己求测', '儿女辈', '长辈疾病')
                found = search(query, selected, anchors)
                plan.append({'stage': label, 'candidate_ids': [x['id'] for x in found['documents']]})
                ranked.extend((x, label) for x in found['documents'][:1])
            flows = [f for f in source_plan()['workflows'] if f['domain'] == selected]
            wanted = set(corpus.r.terms(question))
            flows.sort(key=lambda f: -len(wanted & set(corpus.r.terms(''.join(x['text'] for x in f['rules'])))))
            for flow in flows[:2]:
                if not wanted.intersection(corpus.r.terms(''.join(x['text'] for x in flow['rules']))): continue
                ranked.append((index.expand(flow['document']), '原书规则组'))
        ranked.extend((x, '当前问题补充') for x in direct['documents'][1:])
        kept = []; seen = set(); texts = []; duplicate_skips = []
        for doc, stage in ranked:
            text = d.compact(doc.get('reading_content', doc['content']))
            # A whole union containing only a repeated domain premise plus an
            # unrelated fragment is not useful extra evidence. Skip the whole
            # union rather than trimming a rule/exception out of its text.
            pieces = [d.compact(x) for x in doc['content'].split('\n\n') if d.compact(x)]
            overlap = sum(len(x) for x in pieces if any(x in old for old in texts))
            if doc['id'] in seen or any(text in old for old in texts) or overlap > len(text) * .6:
                duplicate_skips.append(doc['id'])
                continue
            if doc['kind'] != 'rule' or doc.get('case_ids'):
                raise ValueError('历史案例或反馈不能混入新问事')
            seen.add(doc['id']); texts.append(text)
            kept.append(_bibliography(index, doc) | {'retrieval_stage': stage})
        return direct | {'documents': kept, 'workflow_candidates': plan,
                         'domain_candidates': domains, 'workflow_is_source_selection': True,
                         'all_workflow_rules_loaded': False, 'duplicate_bundles_skipped': duplicate_skips}


def case_list():
    with corpus.r.Index(checked_index()) as index:
        return [{'id': k, 'domain': v['domain'],
                 'author_steps': len(v.get('author_step_source_records', []))}
                for k, v in sorted(index.cases.items())]


def case_source(case_id):
    with corpus.r.Index(checked_index()) as index:
        result = index.case(case_id)
        result['documents'] = [_bibliography(index, x) for x in result['documents']]
        # Explicit source viewing only: never pass this result to inference.
        return result | {'mode': 'historical_source_view', 'sent_to_model': False,
                         'original_chart_not_independently_verified': True}
