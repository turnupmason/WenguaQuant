import json
import unittest
from research_audits.v59_review import verify_worker, prompt_sha, memory_summary
from wengua_course import data as d


class Review59Tests(unittest.TestCase):
    def pair(self):
        request = {'nonce':'synthetic','software_fixture':False}
        raw = json.dumps(request).encode()
        reply = {'request_sha256':d.digest(raw),'nonce':'synthetic','pid':3,'ok':True,'result':{'state':'complete'}}
        return request, raw, reply

    def test_success_and_interrupted_worker_remain_distinct(self):
        request, raw, reply = self.pair()
        verify_worker(reply, request, raw)
        failed = dict(reply,ok=False,error={'type':'KeyboardInterrupt','cuda_oom':False})
        verify_worker(failed, request, raw)
        self.assertFalse(failed['ok'])
        with self.assertRaises(ValueError):
            verify_worker(dict(failed,error={'type':'OutOfMemoryError','cuda_oom':True}),request,raw)

    def test_request_binding_is_not_replaced_by_launcher_pid(self):
        request, raw, reply = self.pair()
        for changed in [dict(reply,nonce='other'),dict(reply,request_sha256='0'*64),dict(reply,pid=0)]:
            with self.assertRaises(ValueError):verify_worker(changed,request,raw)

    def test_recorded_fixture_cannot_be_reported_as_real_run(self):
        request, raw, reply = self.pair()
        request['software_fixture'] = True
        with self.assertRaises(ValueError):verify_worker(reply,request,raw)

    def test_reused_prompt_dict_order_preserves_original_sha(self):
        a = [{'role':'user','content':'synthetic'}]
        b = [{'content':'synthetic','role':'user'}]
        self.assertEqual(prompt_sha(a),prompt_sha(b))
        self.assertNotEqual(prompt_sha(a),prompt_sha([{'role':'user','content':'changed'}]))

    def test_memory_comparison_uses_actual_training_samples(self):
        rows = [{'phase':'load','free_gib':0,'reserved_gib':16,'peak_allocated_gib':16},
                {'phase':'train','free_gib':4.8,'reserved_gib':9.8,'peak_allocated_gib':8.9,'allocator_ooms':0,'step':150},
                {'phase':'train','free_gib':5.0,'reserved_gib':9.0,'peak_allocated_gib':8.9,'allocator_ooms':0,'step':195}]
        result = memory_summary(rows)
        self.assertEqual(result['train_samples'],2); self.assertEqual(result['zero_free_samples'],0)
        self.assertEqual(result['max_sampled_reserved_gib'],9.8)
        self.assertEqual(result['first_step'],150); self.assertEqual(result['last_sampled_step'],195)
        self.assertEqual(memory_summary(rows[:1]),{'train_samples':0})


if __name__ == '__main__':unittest.main()
