# SPDX-License-Identifier: Apache-2.0
"""Device-free construction/preflight checks; no model or tensor allocation."""
from dataclasses import replace
import math
import sys
from types import SimpleNamespace
import unittest
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.glm5_next_plan import BackboneOptions,GlmBackbonePlan


def fixture():
    text={'model_type':'glm5_next_text','num_hidden_layers':4,'hidden_size':8,'vocab_size':64,'dtype':'bfloat16',
          'layer_types':['linear_attention']*3+['deepseek_sparse_attention'],'mlp_layer_types':['dense']*3+['sparse'],
          'linear_attn_config':{'kda_layers':[0,1,2],'full_attn_layers':[3],'num_heads':2,'head_dim':3,'short_conv_kernel_size':4},
          'num_attention_heads':2,'num_key_value_heads':2,'q_lora_rank':5,'kv_lora_rank':3,'qk_nope_head_dim':4,'qk_rope_head_dim':0,'v_head_dim':3,'mla_use_nope':True,
          'n_routed_experts':5,'n_shared_experts':1,'num_experts_per_tok':2,'moe_intermediate_size':6,'routed_scaling_factor':2.5,
          'scoring_func':'sigmoid','norm_topk_prob':True,'n_group':1,'topk_group':1,'topk_method':'noaux_tc','moe_router_dtype':'float32',
          'mhc':True,'hc_mult':2,'hc_eps':1e-6,'hc_sinkhorn_iters':5,'index_head_dim':4,'index_n_heads':2,'index_topk':8,
          'index_kpool':4,'index_kpool_compress':True,'index_kpool_always_select_tail':True,'num_nextn_predict_layers':1}
    config=Glm5NextConfig.from_hf_dict({'model_type':'glm5_next','architectures':['Glm5NextForConditionalGeneration'],'text_config':text,'vision_config':{'model_type':'fixture_vision'}})
    return replace(config,quantization_method='W8A8_DYNAMIC')


def factories(plan,events):
    def attention(spec):events.append(('attention',spec.layer_id));return object()
    def ffn(spec):events.append(('FFN',spec.layer_id));return object()
    def decoder(spec,a,f):
        events.append(('decoder',spec.layer_id));hc=lambda:SimpleNamespace(hidden_size=plan.hidden_size,streams=plan.streams)
        return SimpleNamespace(layer_id=spec.layer_id,attention_kind=spec.attention,mlp_kind=spec.mlp,attention=a,mlp=f,attn_hc=hc(),ffn_hc=hc())
    return {'attention_factories':{k:attention for k in ('linear_attention','deepseek_sparse_attention')},
            'ffn_factories':{k:ffn for k in ('dense','sparse')},'decoder_factory':decoder}


class PlanTests(unittest.TestCase):
    def test_topology_scope_and_no_torch_import(self):
        plan=GlmBackbonePlan.from_config(fixture());d=plan.to_dict()
        self.assertEqual(d['counts'],{'linear_attention':3,'deepseek_sparse_attention':1,'dense':3,'sparse':1})
        self.assertEqual(plan.layers[3].checkpoint_prefix,'model.language_model.layers.3')
        self.assertFalse(d['model_ready']);self.assertEqual(d['scope'],'text_backbone_components_only')
        self.assertTrue(plan.vision_config_present);self.assertEqual(plan.declared_mtp_layers,1)
        self.assertNotIn('torch',sys.modules)

    def test_unsupported_features_reject_before_any_factory(self):
        for kwargs in ({'logical_block_size':16},{'logical_block_size':192},{'tensor_parallel_size':2},{'expert_parallel_size':2},{'requests_per_step':2},{'mtp_tokens':1},{'vision':True},{'graphs':True},{'prefix_cache':True},{'requests_per_step':True}):
            with self.assertRaises(ValueError):GlmBackbonePlan.from_config(fixture(),BackboneOptions(**kwargs))

    def test_all_bindings_preflight_before_side_effects(self):
        config=fixture();plan=GlmBackbonePlan.from_config(config)
        for variant in ('missing','extra','noncallable','decoder'):
            events=[];bound=factories(plan,events)
            if variant=='missing':del bound['attention_factories']['deepseek_sparse_attention']
            if variant=='extra':bound['ffn_factories']['typo']=lambda s:None
            if variant=='noncallable':bound['ffn_factories']['sparse']=None
            if variant=='decoder':bound['decoder_factory']=None
            with self.assertRaises(ValueError):plan.assemble(config,**bound)
            self.assertEqual(events,[])

    def test_configuration_or_plan_mutation_prevents_factory_calls(self):
        config=fixture();plan=GlmBackbonePlan.from_config(config);events=[];bound=factories(plan,events)
        changed=replace(config,hidden_size=16)
        with self.assertRaises(ValueError):plan.assemble(changed,**bound)
        corrupt=replace(plan,layers=(replace(plan.layers[0],checkpoint_prefix='wrong'),)+plan.layers[1:])
        with self.assertRaises(ValueError):corrupt.assemble(config,**bound)
        self.assertEqual(events,[])

    def test_complete_assembly_order_and_immutable_result_map(self):
        config=fixture();plan=GlmBackbonePlan.from_config(config);events=[]
        result=plan.assemble(config,**factories(plan,events))
        self.assertEqual(list(result),[0,1,2,3]);self.assertEqual(events,[(kind,i) for i in range(4) for kind in ('attention','FFN','decoder')])
        with self.assertRaises(TypeError):result[0]=None

    def test_bad_decoder_kind_instance_or_hc_binding_rejects(self):
        config=fixture();plan=GlmBackbonePlan.from_config(config)
        for variant in ('kind','instance','geometry'):
            events=[];bound=factories(plan,events);original=bound['decoder_factory']
            def wrong(spec,a,f):
                result=original(spec,a,f)
                if variant=='kind':result.attention_kind='wrong'
                if variant=='instance':result.mlp=object()
                if variant=='geometry':result.ffn_hc.streams=4
                return result
            bound['decoder_factory']=wrong
            with self.assertRaises(ValueError):plan.assemble(config,**bound)
            self.assertEqual(len(events),3)

    def test_logical_byte_formula_independent_shape_products(self):
        plan=GlmBackbonePlan.from_config(fixture());actual=plan.logical_state_bytes(9,lookahead=3)
        shapes=[((3,3,2*3),2)]*3+[((2,3,3),4)]*3+[((9,3),2),((2,4),2),((2,7,4),4)]
        expected=sum(math.prod(shape)*size for shape,size in shapes)
        self.assertEqual(actual['logical_total'],expected);self.assertEqual(expected,834)
        with self.assertRaises(ValueError):plan.logical_state_bytes(-1)
        with self.assertRaises(ValueError):plan.logical_state_bytes(1,lookahead=True)

    def test_missing_identity_quant_or_layer_contracts_reject(self):
        config=fixture()
        for bad in (replace(config,quantization_method=None),replace(config,dtype='float16'),replace(config,architecture='Other'),replace(config,layers=config.layers[1:]),replace(config,layers=tuple(reversed(config.layers)))):
            with self.assertRaises(ValueError):GlmBackbonePlan.from_config(bad)

if __name__=='__main__':unittest.main(verbosity=2)
