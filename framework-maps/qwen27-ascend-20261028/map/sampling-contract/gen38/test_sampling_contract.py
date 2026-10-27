"""Meaningful saved-source sensitivity/rejection checks, not NPU tests."""
import copy, unittest
import audit_sampling_contract as a

class SamplingContract(unittest.TestCase):
    def setUp(self):
        self.body=a.payloads()[0][1]
        self.defaults={'temperature':1.,'top_k':20,'top_p':.95}
    def test_explicit_zero_overrides_model_temperature(self):
        p=a.convert_vllm(self.body,self.defaults);self.assertEqual(p['temperature'],0.)
    def test_vllm_inherits_then_clears_topk(self):
        p=a.convert_vllm(self.body,self.defaults);self.assertEqual(p['top_k'],20)
        n,t=a.normalize_vllm(p);self.assertEqual((n['top_k'],n['top_p'],t),(0,1.,'GREEDY'))
    def test_sg_defaults_override_model_topk(self):
        p=a.convert_sg(self.body,self.defaults);self.assertEqual(p['top_k'],-1)
        n,g=a.normalize_sg(p);self.assertEqual((n['top_k'],n['temperature'],g),(1,1.,True))
    def test_positive_temperature_changes_both_routes(self):
        b={**self.body,'temperature':1.}
        self.assertEqual(a.normalize_vllm(a.convert_vllm(b,self.defaults))[1],'RANDOM_SEED')
        self.assertFalse(a.normalize_sg(a.convert_sg(b,self.defaults))[1])
    def test_seed_change_cannot_select_random_at_zero(self):
        for seed in [None,1,20261638]:
            b={**self.body,'seed':seed}
            self.assertEqual(a.normalize_vllm(a.convert_vllm(b,self.defaults))[1],'GREEDY')
            self.assertTrue(a.normalize_sg(a.convert_sg(b,self.defaults))[1])
    def test_model_topk_changes_do_not_change_zero_route(self):
        for k in [1,20,100]:
            defaults={**self.defaults,'top_k':k}
            self.assertEqual(a.normalize_vllm(a.convert_vllm(self.body,defaults))[1],'GREEDY')
            self.assertTrue(a.normalize_sg(a.convert_sg(self.body,defaults))[1])
    def test_repetition_penalty_is_not_erased_by_greedy(self):
        b={**self.body,'repetition_penalty':1.1}
        self.assertEqual(a.normalize_vllm(a.convert_vllm(b,self.defaults))[0]['repetition_penalty'],1.1)
        self.assertEqual(a.normalize_sg(a.convert_sg(b,self.defaults))[0]['repetition_penalty'],1.1)
    def test_sg_empty_or_model_preference_same_conversion(self):
        self.assertEqual(a.convert_sg(self.body,{}),a.convert_sg(self.body,self.defaults))
    def test_missing_temperature_does_not_preserve_zero(self):
        b=copy.deepcopy(self.body);b.pop('temperature')
        self.assertEqual(a.normalize_vllm(a.convert_vllm(b,self.defaults))[1],'RANDOM_SEED')
        self.assertFalse(a.normalize_sg(a.convert_sg(b,self.defaults))[1])
    def test_source_schema_keeps_constraints_unset(self):
        v=a.convert_vllm(self.body,self.defaults);s=a.convert_sg(self.body,self.defaults)
        self.assertIsNone(v['structured_outputs']);self.assertIsNone(v['allowed_token_ids'])
        self.assertIsNone(s['regex']);self.assertIsNone(s['json_schema']);self.assertIsNone(s['logit_bias'])
    def test_original_sampler_control_flow_calls_argmax(self):
        self.assertEqual(a.replay_argmax_routing()['calls'],{'vllm_ascend_argmax':1,'sglang_argmax':1,'sglang_TP_sync_stub':1})

if __name__=='__main__':unittest.main(verbosity=2)
