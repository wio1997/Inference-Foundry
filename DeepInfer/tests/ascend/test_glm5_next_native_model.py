# SPDX-License-Identifier: Apache-2.0
"""Native factory call wiring with mocks only; never native/model acceptance."""
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import sys,unittest
from test_glm5_next_plan import fixture
from deepinfer.model_executor.models.glm5_next import native_model as m


class NativeModelTests(unittest.TestCase):
    def test_cpu_rejected_before_reader(self):
        reader=Mock()
        with self.assertRaisesRegex(ValueError,'authorized NPU'):
            m.load_native_decoder(reader,0,'cpu',max_expert_checkpoint_bytes=1,max_expert_resident_bytes=1)
        with self.assertRaisesRegex(ValueError,'authorized NPU'):
            m.load_native_text_runner(reader,'cpu',dsa_operations={},max_expert_checkpoint_bytes=1,max_expert_resident_bytes=1,vocab_start=0,vocab_stop=1)
        self.assertEqual(reader.mock_calls,[])

    def test_missing_DSA_rejects_before_expert_or_tensor_reads(self):
        c=fixture();reader=Mock(root='inert')
        with patch.object(m,'_native_device'),patch.object(m.Glm5NextConfig,'from_checkpoint',return_value=c),patch.object(m,'selected_expert_manifest') as manifest:
            with self.assertRaisesRegex(ValueError,'every configured DSA'):
                m.load_native_text_runner(reader,'npu',dsa_operations={},max_expert_checkpoint_bytes=1,max_expert_resident_bytes=1,vocab_start=0,vocab_stop=1)
            with self.assertRaisesRegex(ValueError,'native full-head'):
                m.load_native_decoder(reader,3,'npu',max_expert_checkpoint_bytes=1,max_expert_resident_bytes=1)
            manifest.assert_not_called();self.assertEqual(reader.mock_calls,[])

    def test_KDA_dense_and_DSA_complete_experts_use_real_factory_names(self):
        c=fixture();reader=Mock(root='inert');a=object();dense=object();router=object();experts=object();moe=object();operation=lambda *a:None
        built=[]
        def mock_decoder(*args,**kwargs):
            value=NS(attn_hc=NS(),ffn_hc=NS());built.append(value);return value
        with patch.object(m,'_native_device'),patch.object(m.Glm5NextKDA,'from_checkpoint',return_value=a) as kda,patch.object(m.GlmDenseMLP,'from_checkpoint',return_value=dense) as ffn,patch.object(m.GlmHybridDecoder,'from_checkpoint_components',side_effect=mock_decoder) as decoder:
            result=m.load_native_decoder(reader,0,'npu',config=c,max_expert_checkpoint_bytes=100,max_expert_resident_bytes=100);self.assertIs(result,built[-1]);self.assertTrue(result.native_tokenwise)
            kda.assert_called_once_with(reader,0,'npu',backend='ascendc');ffn.assert_called_once_with(reader,0,'npu');self.assertIs(decoder.call_args.kwargs['attention'],a);self.assertIs(decoder.call_args.kwargs['mlp'],dense)
        with patch.object(m,'_native_device'),patch.object(m,'selected_expert_manifest',return_value='manifest') as manifest,patch.object(m,'plan_expert_residency') as budget,patch.object(m.GlmMLAProjections,'from_checkpoint',return_value=a) as mla,patch.object(m.GlmIndexerProjections,'from_checkpoint',return_value='indexer'),patch.object(m,'PreparedDSAAttention',return_value='attention') as dsa,patch.object(m,'load_selected_expert_bank',return_value=experts) as bank,patch.object(m.GlmMoERouter,'from_checkpoint',return_value=router),patch.object(m,'GlmMoEMLP',return_value=moe) as adapter,patch.object(m.GlmHybridDecoder,'from_checkpoint_components',side_effect=mock_decoder) as decoder:
            m.load_native_decoder(reader,3,'npu',config=c,dsa_operation=operation,max_expert_checkpoint_bytes=100,max_expert_resident_bytes=90)
            manifest.assert_called_once_with(reader,3,(0,1,2,3,4),include_shared=True);budget.assert_called_once_with('manifest',max_checkpoint_bytes=100,max_resident_bytes=90)
            self.assertEqual(bank.call_args.args[2],(0,1,2,3,4));self.assertEqual(bank.call_args.kwargs['max_experts'],5);self.assertTrue(bank.call_args.kwargs['include_shared']);adapter.assert_called_once_with(router,experts,hidden_size=c.hidden_size,num_experts=5)
            self.assertIs(dsa.call_args.args[2],operation);self.assertIs(decoder.call_args.kwargs['mlp'],moe)

    def test_total_budget_rejects_before_decoder_allocations(self):
        c=fixture();reader=Mock(root='inert')
        with patch.object(m,'_native_device'),patch.object(m.Glm5NextConfig,'from_checkpoint',return_value=c),patch.object(m,'selected_expert_manifest'),patch.object(m,'plan_expert_residency',return_value=NS(checkpoint_payload_bytes=101,resident_logical_bytes=90)),patch.object(m,'load_native_decoder') as decoder:
            with self.assertRaisesRegex(ValueError,'total native expert'):
                m.load_native_text_runner(reader,'npu',dsa_operations={3:lambda *a:None},max_expert_checkpoint_bytes=100,max_expert_resident_bytes=100,vocab_start=0,vocab_stop=1)
            decoder.assert_not_called()

    def test_text_factory_reuses_topology_stack_boundary_head_runner(self):
        preexisting={n for n in ('torch_npu','fla_npu','_C_ascend') if n in sys.modules}
        c=fixture();reader=Mock(root='inert');events=[];hc=NS(hidden_size=c.hidden_size,streams=2)
        def decoder(reader,i,device,**kw):
            events.append((i,kw));spec=c.layers[i]
            return NS(layer_id=i,attention_kind=spec.attention,mlp_kind=spec.mlp,attention=object(),mlp=object(),attn_hc=hc,ffn_hc=hc)
        boundary=Mock();boundary.head_shard.return_value='head'
        with patch.object(m,'_native_device'),patch.object(m.Glm5NextConfig,'from_checkpoint',return_value=c),patch.object(m,'selected_expert_manifest'),patch.object(m,'plan_expert_residency',return_value=NS(checkpoint_payload_bytes=50,resident_logical_bytes=40)),patch.object(m,'load_native_decoder',side_effect=decoder),patch.object(m,'PreparedDecoderStack',return_value='stack') as stack,patch.object(m.GlmTextBoundary,'from_checkpoint',return_value=boundary),patch.object(m,'PreparedTextRunner') as runner:
            operation=lambda *a:None
            m.load_native_text_runner(reader,'npu',dsa_operations={3:operation},max_expert_checkpoint_bytes=100,max_expert_resident_bytes=100,vocab_start=2,vocab_stop=6,max_tokens=4)
            self.assertEqual([e[0] for e in events],[0,1,2,3]);self.assertIs(events[-1][1]['dsa_operation'],operation);self.assertEqual(len(stack.call_args.args[2]),4)
            boundary.head_shard.assert_called_once_with(2,6,max_rows=32);runner.assert_called_once_with('stack',boundary,'head',max_context_tokens=128,max_requests=1,native_tokenwise=True)
        self.assertEqual({n for n in ('torch_npu','fla_npu','_C_ascend') if n in sys.modules},preexisting)


if __name__=='__main__':unittest.main(verbosity=2)
