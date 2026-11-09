"""Synthetic tests for independent chapter profiles and complete source contexts."""
import tempfile,unittest,io
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path
from wengua_corpus import book_rag as r
from wengua_library import intake as i
from wengua_library.__main__ import related_comparisons,load_release,query,main
from wengua_library.build import bind_external_evidence

BOOK={'title':'合成材料','method_id':'synthetic_source_only'}
PROFILE={'namespace':'jinhan.batch04','chapter_id':'synthetic_scroll','title':'合成卷段','main_headings':[],
         'subheadings':[],'domains':{'占科举':['education']},'separator_volumes':True,
         'connected_evidence_required':True,'source_ordered_contexts':True,'study_sections':[]}
SOURCE='==================== 卷七 ====================\n占科举\n甲生乙，且得丙，方取。\n\n乙空亡则不取。\n【白话译释】\n正反条件同时看。\n'

class ProfileTests(unittest.TestCase):
    def setUp(self):(r.ROOT/'local').mkdir(exist_ok=True)
    def compile(self,**spec):
        profile=dict(PROFILE);profile.update(spec.pop('profile',{}))
        return i.compile_source(SOURCE,BOOK,{'source_profile':profile,**spec})
    def test_separator_volumes_and_namespace_do_not_collide_with_older_snapshot(self):
        p=self.compile()
        self.assertTrue(all(x['id'].startswith('jinhan.batch04.') for x in p['units']))
        self.assertTrue(all(x['volume']=='卷七' for x in p['units']))
        self.assertEqual(p['report']['unaccounted_nonblank_lines'],0)
        old=i.compile_source('【卷六】\n占升迁\n合成甲乙条件。\n',BOOK,{})['library']
        joined=r.merge_libraries([old,p['library']]);self.assertEqual(len(joined['chapters']),2)
    def test_later_exclusion_is_supplied_with_positive_source_in_original_order(self):
        p=self.compile(profile={'section_context_lines':{'占科举':[3,5]}},
            courses=[{'id':'question','format':'conditional_qa','source_lines':[3],'question':'是否取？','answer':'还看空亡限制。'}])
        lib=p['library'];cs={r.chapter_key(c):c for c in lib['chapters']}
        doc=next(d for d in lib['documents'] if d['id']=='jinhan.batch04.u0003')
        text=r.document_text(cs,doc)
        self.assertLess(text.index('甲生乙'),text.index('乙空亡'))
        course=p['courses'][0]
        self.assertEqual([x['lines'][0] for x in course['source_parts']],[3,5])
        self.assertFalse(course['split_assigned']);self.assertFalse(course['current_training_eligible'])
    def test_context_is_rejected_if_it_belongs_to_another_topic(self):
        with self.assertRaisesRegex(ValueError,'another topic'):
            self.compile(profile={'section_context_lines':{'占科举':[1]}})
    def test_scope_note_does_not_quarantine_clear_source_but_remains_visible(self):
        p=self.compile(unit_findings={'3':[{'detail':'按本篇解释，不全局合并。','blocks_general_retrieval':False}]})
        u=next(u for u in p['units'] if u['source_lines'][0]==3)
        self.assertEqual(u['course_state'],'source_grounded_candidate')
        self.assertTrue(u['findings'])
        pending=self.compile(unit_findings={'3':[{'detail':'原字不明。'}]})
        self.assertEqual(next(u for u in pending['units'] if u['source_lines'][0]==3)['course_state'],'quarantined_source_issue')
    def test_stable_excerpt_can_preserve_separate_negative_clause(self):
        source='占科举\n甲生乙则取；乙空亡不取。\n'
        p=i.compile_source(source,BOOK,{'source_profile':PROFILE,'safe_excerpts':[{'line':2,'title':'分句','quote':'甲生乙则取','extra_quotes':['乙空亡不取。']}]})
        doc=next(d for d in p['library']['documents'] if '.safe.' in d['id'])
        self.assertEqual(len(doc['parts']),2)
        cs={r.chapter_key(c):c for c in p['library']['chapters']}
        self.assertIn('乙空亡不取',r.document_text(cs,doc))
        with self.assertRaisesRegex(ValueError,'excerpt limit'):
            self.compile(safe_excerpts=[{'line':3,'title':'坏例','quote':'甲生乙','extra_quotes':['无条件录取']}])
    def test_undefined_namespace_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError,'namespace'):self.compile(profile={'namespace':'../../other'})
    def test_scoped_search_does_not_return_new_profile_heading_as_evidence(self):
        p=self.compile()
        with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as tmp:
            path=Path(tmp)/'index.sqlite3';r.build_index([p['library']],path)
            with r.Index(path) as index:result=query(index,'科举甲生乙',section='占科举',study=True)
            self.assertTrue(result['documents'])
            self.assertTrue(all(d.get('material_scope')!='structure' for d in result['documents']))
    def test_old_and_new_comparisons_accumulate_but_reference_text_requires_study(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as tmp:
            entry=[]
            for num in [3,4]:
                dest=Path(tmp)/str(num);dest.mkdir();path=dest/'reviewed_comparisons.json'
                groups=[{'id':str(num),'new_unit_ids':['source.'+str(num)],'prior_evidence':[{'id':'old','kind':'reference','text':'待核原文仅研究。'}]}]
                r.write(path,groups);entry.append({'id':str(num),'destination':dest.relative_to(r.ROOT).as_posix(),'files':{'reviewed_comparisons.json':r.digest(path.read_bytes())}})
            release={'entries':entry,'new_entry_id':'4'};docs=[{'id':'source.3'},{'id':'source.4'}]
            normal=related_comparisons(release,docs);study=related_comparisons(release,docs,study=True)
            self.assertEqual(len(normal),2);self.assertEqual(len(study),2)
            self.assertTrue(all('text' not in g['prior_evidence'][0] for g in normal))
            self.assertTrue(all(g['prior_evidence'][0]['text']=='待核原文仅研究。' for g in study))
    def test_versioned_release_never_activates_training_or_reuses_public_index_path(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as tmp:
            path=Path(tmp)/'release.json';base={'schema':i.SCHEMA,'automatic_training_inclusion':False,'current_v60_training_changed':False,'entries':[],'index':'local/rag/independent_test.sqlite3'}
            r.write(path,base);self.assertEqual(load_release(path)['index'],base['index'])
            base['automatic_training_inclusion']=True;r.write(path,base)
            with self.assertRaisesRegex(ValueError,'contract'):load_release(path)
    def test_cross_source_course_contains_literal_evidence_and_case_text_is_rejected(self):
        old=[{'id':'old.rule','kind':'rule','text':'原书另一条必要条件。','parts':[{'chapter':'old','lines':[2,2]}],'grams':set()},
             {'id':'old.case','kind':'case','text':'后来反馈。','parts':[],'grams':set()}]
        pack={'courses':[{'id':'c','prior_document_ids':['old.rule'],'source_parts':[],'source_family_ids':['new']}]}
        bind_external_evidence(pack,old);course=pack['courses'][0]
        self.assertEqual(course['external_source_documents'][0]['text'],'原书另一条必要条件。')
        self.assertNotIn('grams',course['external_source_documents'][0])
        self.assertIn('source_document:old.rule',course['source_family_ids'])
        bad={'courses':[{'id':'c','prior_document_ids':['old.case'],'source_parts':[]}]}
        with self.assertRaisesRegex(ValueError,'Historical cases'):bind_external_evidence(bad,old)
    def test_restore_does_not_skip_prior_comparisons_when_only_library_is_present(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as tmp:
            folder=Path(tmp);library=folder/'sources.json.gz';library.write_bytes(b'synthetic-library')
            key=folder/'key.txt';key.write_text('00'*32)
            entry={'id':'old','destination':folder.relative_to(r.ROOT).as_posix(),'library':library.name,
                   'files':{library.name:r.digest(library.read_bytes()),'reviewed_comparisons.json':'missing'}}
            release={'entries':[entry],'new_entry_id':'new'}
            with patch('wengua_library.__main__.load_release',return_value=release),patch('wengua_library.__main__.restore') as restore,patch('wengua_library.__main__.ensure_index',return_value={}),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['restore','--key-file',str(key)]),0)
            restore.assert_called_once()

if __name__=='__main__':unittest.main()
