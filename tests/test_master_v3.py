"""Wengua Master V3 contract tests; all synthesized case fixtures are fictional."""
import json,tempfile,unittest
from pathlib import Path
from wengua_master_v3.curriculum import PALACES,REVERSE,build_drills,relation,validate_release,private_from_seed,prepare

class V3Tests(unittest.TestCase):
    def test_direction(self):
        for a in PALACES:
            for b in PALACES:self.assertEqual(relation(b,a),REVERSE[relation(a,b)])
        self.assertEqual(relation(1,9),"克")
        self.assertEqual(relation(9,1),"被克")
        self.assertEqual(relation(6,1),"生")
        self.assertEqual(relation(3,4),"比和")
        self.assertEqual(relation(2,2),"同宫")
        with self.assertRaises(ValueError):relation(5,6)
    def test_public_not_observed_prediction(self):
        rows=build_drills();self.assertEqual(len(rows),1024)
        audit=validate_release(rows)
        self.assertEqual(audit["groups"],36)
        self.assertEqual(sum(audit["splits"].values()),1024)
        for r in rows:
            self.assertFalse(r["real_chart_verified"])
            self.assertFalse(r["fresh_prediction_gold"])
            self.assertIn("教学假设",r["messages"][1]["content"])
    def test_private_holdout_and_source_stage(self):
        seed={"schema":"wengua_master_v3_private_codex_source_review_v1",
          "cases":[{"case_id":"example-a","source_split":"test"},{"case_id":"example-b","source_split":"train"}],
          "author_steps":[{"case_id":"example-a","step_number":1,"source_text":"来源分析A","topic":"规则"},
                          {"case_id":"example-b","step_number":1,"source_text":"来源分析B","topic":"规则"}],
          "qa":[{"sample_id":"author:example-a:step:1","source_split":"train","question":"问","answer":"答","task":"author_stage"}]}
        rows,audit=private_from_seed(seed)
        self.assertEqual(audit["workbook_cases"],2)
        self.assertEqual(len(rows),5)
        self.assertEqual(sum(1 for r in rows if r["split"]=="test"),3)
        self.assertEqual(validate_release(rows)["fresh_prediction_gold"],0)
    def test_v2_family_lock_conservative(self):
        seed={"schema":"wengua_master_v3_private_codex_source_review_v1",
          "cases":[{"case_id":"old","source_split":"test"}],
          "author_steps":[{"case_id":"old","step_number":1,"source_text":"原作者阶段","topic":"复盘"}],
          "qa":[]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            registry=[{"case_id":"old","event_family_id":"fam","evidence_component":"connected","split":"train"},
                      {"case_id":"linked","event_family_id":"fam","evidence_component":"connected","split":"train"}]
            units=[{"kind":"author_explicit_paragraph","case_id":"linked","original_step_number":1,"text":"另一章的作者复盘","id":"original:1"}]
            (p/"case_registry.jsonl").write_text("".join(json.dumps(x)+"\n" for x in registry),encoding="utf-8")
            (p/"source_units.jsonl").write_text("".join(json.dumps(x)+"\n" for x in units),encoding="utf-8")
            rows,stats=private_from_seed(seed,p)
            self.assertEqual(stats["v2_case_registry"],2)
            self.assertEqual({r["split"] for r in rows},{"test"})
            self.assertEqual(validate_release(rows)["groups"],1)
    def test_leakage_and_rewrite_rejection(self):
        rows=build_drills()
        bad=dict(rows[0]);bad["id"]="other";bad["split"]="test" if bad["split"]!="test" else "train"
        with self.assertRaisesRegex(ValueError,"crossed splits"):validate_release(rows+[bad])
        bad=dict(rows[0]);bad["fresh_prediction_gold"]=True
        with self.assertRaisesRegex(ValueError,"predictive gold"):validate_release([bad])
    def test_build_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"new"
            x=prepare(path);self.assertFalse(x["training_started"])
            self.assertEqual(x["public_rule_drills"]["rows"],1024)
            with self.assertRaises(FileExistsError):prepare(path)

if __name__=="__main__":unittest.main()
