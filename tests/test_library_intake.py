"""Synthetic source-boundary regressions, independent of private book text."""
import json,tempfile,unittest
from pathlib import Path
from wengua_library import intake as i
from wengua_library.__main__ import query
from wengua_corpus import book_rag as r

BOOK={'title':'合成校对材料','method_id':'synthetic_not_runtime'}
SOURCE='【卷六】\n占升迁\n甲、乙、丙三项同时齐备。缺一不可。\n【白话译释】\n三项须同时有，缺项不可。\n'

class IntakeTests(unittest.TestCase):
    def setUp(self):(r.ROOT/'local').mkdir(exist_ok=True)
    def compile(self,source=SOURCE,**spec):return i.compile_source(source,BOOK,spec)
    def test_all_nonblank_lines_once_and_headings_never_teachers(self):
        pack=self.compile();assigned=[n for u in pack['units'] for n in range(u['source_lines'][0],u['source_lines'][1]+1)]
        self.assertEqual(assigned,list(range(1,6)))
        self.assertEqual(pack['report']['unaccounted_nonblank_lines'],0)
        byid={u['id']:u for u in pack['units']}
        for c in pack['courses']:
            self.assertFalse(c['current_training_eligible'])
            self.assertTrue(all(byid[k]['scope']=='source_rule' for k in c['source_unit_ids']))
    def test_conditional_negation_and_context_are_not_lost(self):
        pack=self.compile();translated=next(u for u in pack['units'] if u['source_mode']=='translation' and u['scope']=='source_rule')
        doc=next(d for d in pack['library']['documents'] if d['id']==translated['id'])
        chapters={r.chapter_key(c):c for c in pack['library']['chapters']}
        self.assertIn('缺一不可',r.document_text(chapters,doc))
        self.assertEqual(len(doc['parts']),2)
    def test_translation_cannot_bypass_disputed_original(self):
        p=self.compile(unit_findings={'3':[{'kind':'uncertain','detail':'原字待核'}]})
        translations=[u for u in p['units'] if u['source_mode']=='translation' and u['scope']=='source_rule']
        self.assertEqual(translations[0]['course_state'],'quarantined_context_dependency')
        self.assertFalse(p['courses'])
    def test_orphan_findings_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'does not start a material'):
            self.compile(unit_findings={'99':[{'detail':'wrong coordinate'}]})
    def test_punctuation_and_direction_never_exact_deduplicated(self):
        self.assertNotEqual(i.exact_key('不，吉'),i.exact_key('不吉'))
        self.assertNotEqual(i.exact_key('甲生乙'),i.exact_key('乙生甲'))
        self.assertNotEqual(i.exact_key('缺一不可'),i.exact_key('缺一可'))
        self.assertEqual(i.exact_key('Ａ 乙'),i.exact_key('A乙'))
    def test_invented_literal_target_or_premise_rejected(self):
        with self.assertRaisesRegex(ValueError,'literal target'):
            self.compile(courses=[{'id':'x','format':'conditional_qa','source_lines':[3],'literal_targets':['保证提升']}])
        with self.assertRaisesRegex(ValueError,'not literal'):
            self.compile(logic_rules=[{'id':'x','line':3,'literal_fragments':['丙生乙'],'conditions':['甲'],'conclusion':'吉'}])
    def test_source_audit_can_cite_missing_translation_note_without_sft(self):
        source='九遁变化法\n风遁\n甲乙是所给原文。\n【白话译释】\n〔所给译文缺失。〕\n'
        p=self.compile(source,courses=[{'id':'audit','format':'source_audit_qa','source_lines':[3,5],'question':'缺什么？','answer':'译文缺，原文在。'}])
        self.assertEqual(p['courses'][0]['source_unit_ids'],['jinhan.batch03.u0003','jinhan.batch03.u0005'])
        self.assertFalse(p['courses'][0]['current_training_eligible'])
        self.assertFalse(p['library']['cases'])
    def test_headings_cannot_masquerade_as_audit_evidence(self):
        with self.assertRaisesRegex(ValueError,'Headings'):
            self.compile(courses=[{'id':'bad','format':'source_audit_qa','source_lines':[2]}])
    def test_safe_literal_extract_does_not_clear_parent_issue(self):
        p=self.compile(unit_findings={'3':[{'kind':'pending','detail':'同段另句待核'}]},safe_excerpts=[{'line':3,'title':'确定子句','quote':'缺一不可。'}])
        safe=next(d for d in p['library']['documents'] if d['id'].startswith('jinhan.batch03.safe'))
        parent=next(d for d in p['library']['documents'] if d['id']==safe['parent_material_id'])
        self.assertEqual(parent['kind'],'reference');self.assertTrue(safe['remaining_parent_findings'])
        self.assertFalse(p['courses'])
    def test_worked_example_has_no_observed_case_or_outcome(self):
        p=self.compile('占胜败\n例如谷雨，甲乙同宫。\n',illustrations=[{'id':'example','line':2,'literal_fragments':['甲乙同宫']}])
        self.assertFalse(p['library']['cases'])
        self.assertFalse(p['illustrations'][0]['observed_feedback_given'])
        self.assertIsNone(p['illustrations'][0]['case_id'])
        self.assertFalse(p['courses'])
    def test_near_original_translation_alignment_is_not_merge_permission(self):
        p=self.compile('占升迁\n本篇甲乙丙丁戊己庚各条条件必须齐备。\n【白话译释】\n本篇甲乙丙丁戊己庚各条条件必须齐备，再看其他。\n')
        _,links=i.compare_sources(p,[])
        self.assertTrue(links)
        self.assertTrue(all(not x['automatically_mergeable'] for x in links))
    def test_index_modes_and_explicit_topic_do_not_leak_pending_text(self):
        p=self.compile(unit_findings={'3':[{'kind':'pending','detail':'待核'}]})
        with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as directory:
            path=Path(directory)/'sources.sqlite3';r.build_index([p['library']],path)
            with r.Index(path) as index:
                normal=query(index,'甲乙缺一',section='占升迁',book='jinhan')
                study=query(index,'甲乙缺一',section='占升迁',book='jinhan',study=True)
            self.assertFalse(normal['documents']);self.assertTrue(study['documents'])
            self.assertFalse(study['automatic_training_inclusion'])
    def test_scoped_audit_retrieval_prefers_specific_finding_over_header(self):
        source='三诈法\n甲乙条说明通行名目。\n\n乙丙条保留异文。\n'
        p=self.compile(source,unit_findings={'4':[{'kind':'source_conflict','detail':'名称用途有出入，待核。'}]})
        with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as directory:
            path=Path(directory)/'sources.sqlite3';r.build_index([p['library']],path)
            with r.Index(path) as index:result=query(index,'三诈名称用途',section='三诈法',book='jinhan',study=True,limit=1)
            self.assertEqual(result['documents'][0]['id'],'jinhan.batch03.u0003')

if __name__=='__main__':unittest.main()
