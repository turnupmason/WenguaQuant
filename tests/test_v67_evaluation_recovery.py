import ast,json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from wengua_course import data as d
from wengua_recovery import v67_evaluation as r


class V67EvaluationRecoveryBoundaries(unittest.TestCase):
    def test_async_accelerator_oom_is_recognized_but_unrelated_errors_are_not(self):
        self.assertTrue(r.is_cuda_oom(RuntimeError('CUDA error: out of memory')))
        self.assertTrue(r.is_cuda_oom({'type':'AcceleratorError','message':'CUDA error: out of memory'}))
        for text in ['CUDA error: illegal memory access','out of memory on CPU','unrelated failure']:
            self.assertFalse(r.is_cuda_oom(RuntimeError(text)))

    def test_existing_evaluation_receipt_and_source_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);adapter=job/'adapter';adapter.mkdir()
            for name in ['adapter_config.json','adapter_model.safetensors']:(adapter/name).write_bytes(b'software fixture')
            best={'tag':'epoch_01','adapter':str(adapter)};c={'fixed':True};pack={'fixed_questions':True}
            with patch.object(r.evaluation,'validate_cache',lambda *a:None):
                path,rows,contract=r.stage_contract(job,c,best,pack)
                self.assertEqual(rows,[]);self.assertEqual(contract['config'],c)
                path.write_bytes(b'');r.stage_contract(job,c,best,pack)
                (adapter/'adapter_model.safetensors').write_bytes(b'changed')
                with self.assertRaises(ValueError):r.stage_contract(job,c,best,pack)

    def test_frozen_v67_training_is_not_edited_to_adopt_recovery(self):
        manifest=r.frozen_check()
        self.assertEqual(r.runtime.hashes(),manifest['original_code_sha256'])
        self.assertFalse(manifest['training_code_modified']);self.assertEqual(manifest['training_updates'],0)
        self.assertEqual(r.POLICY['answers_per_worker'],24);self.assertEqual(r.POLICY['oom_attempts_per_position'],2)

    def test_recovery_module_has_no_training_call_or_truncation_switch(self):
        tree=ast.parse(Path(r.__file__).read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ['fit','train','backward','training_step']:self.assertNotIn(name,calls)
        self.assertNotIn('max_new_tokens',Path(r.__file__).read_text())
        self.assertIn('evaluation.one_answer(',Path(r.__file__).read_text())

    def test_recovery_receipts_preserve_existing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';r.freeze(p,{'saved':True});raw=p.read_bytes();r.freeze(p,{'saved':True})
            self.assertEqual(p.read_bytes(),raw)
            with self.assertRaises(ValueError):r.freeze(p,{'saved':False})

    def test_uploaded_prefix_survives_append_but_rewrite_or_deletion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'answers.jsonl';raw=b'{"answer":"one"}\r\n';proof={'rows':1,'sha256':d.digest(raw)}
            p.write_bytes(raw);r.verify_original_answer_prefix(p,proof)
            p.write_bytes(raw+b'{"answer":"next"}\n');r.verify_original_answer_prefix(p,proof)
            for bad in [b'',raw.replace(b'one',b'other')]:
                p.write_bytes(bad)
                with self.assertRaises(ValueError):r.verify_original_answer_prefix(p,proof)


if __name__=='__main__':unittest.main()
