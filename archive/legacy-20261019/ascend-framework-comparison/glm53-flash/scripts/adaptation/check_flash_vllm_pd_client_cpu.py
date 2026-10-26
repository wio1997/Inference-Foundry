"""CPU protocol/error tests. No actual HTTP, model, SDK or NPU calls."""
import importlib.util,pathlib,json,unittest,io,urllib.error
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('flash_native_client',P/'scripts/adaptation/flash_vllm_native_pd_client.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Response(io.BytesIO):
 status=200
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
class Checks(unittest.TestCase):
 def client(self):return m.NativeVllmPDClient(prefill_url='http://172.16.12.14:19000/v1/completions',decode_url='http://172.16.12.15:19020/v1/completions',model='synthetic-CPU-fixture')
 def test_multi_token_delta_usage_only_final(self):
  e=m.StreamEvidence(100);e.event(json.dumps({'choices':[{'index':0,'text':'你好abc'}]}),200);e.event(json.dumps({'choices':[{'index':0,'text':'','finish_reason':'length'}]}),300);e.event(json.dumps({'choices':[],'usage':{'prompt_tokens':80000,'completion_tokens':600}}),350);e.event('[DONE]',400);r=e.report(500,80000,600);self.assertTrue(r['success']);self.assertEqual(r['text'],'你好abc');self.assertEqual(r['TTFT_s'],100/1e9);self.assertEqual(r['TPOT_s'],300/1e9/599)
 def test_midstream_EOF_not_success(self):
  e=m.StreamEvidence(100);e.event(json.dumps({'choices':[{'text':'partial'}]}),200);r=e.report(500,80000,600);self.assertFalse(r['success']);self.assertIsNone(r['usage']);self.assertEqual(r['TTFT_s'],100/1e9)
 def test_no_silent_local_recomputation(self):
  calls=[]
  def fake(req,**kw):calls.append(req);return Response(json.dumps({'usage':{'prompt_tokens':1},'choices':[]}).encode())
  with patch.object(m.urllib.request,'urlopen',fake):r=self.client().generate([42])
  self.assertEqual(len(calls),1);self.assertEqual(r['generation_POST_attempted'],1);self.assertIn('Missing actual native PD metadata',r['error']);self.assertFalse(r['success'])
 def test_decode_500_preserves_prefill_and_status(self):
  calls=[]
  def fake(req,**kw):
   calls.append(req)
   if len(calls)==1:return Response(json.dumps({'id':'CPU-p','usage':{'prompt_tokens':1,'completion_tokens':1},'kv_transfer_params':{'do_remote_prefill':True,'actual_fixture':'metadata'}}).encode())
   raise urllib.error.HTTPError(req.full_url,500,'nativeerror',{},io.BytesIO(b'actual CPU fixture error'))
  with patch.object(m.urllib.request,'urlopen',fake):r=self.client().generate([42])
  self.assertEqual(r['generation_POST_attempted'],2);self.assertEqual(r['HTTP_responses_received'],2);self.assertEqual(r['prefill_response_id'],'CPU-p');self.assertEqual(r['decode_HTTP_status'],500);self.assertEqual(r['native_HTTP_error_body'],'actual CPU fixture error');self.assertFalse(r['success'])
 def test_actual_tokenizer_bound_rejects_model_padding(self):
  with self.assertRaises(ValueError):self.client().generate([154856])
 def test_wrong_actual_output_usage_not_success(self):
  e=m.StreamEvidence(100);e.event(json.dumps({'choices':[{'text':'x','finish_reason':'length'}],'usage':{'prompt_tokens':80000,'completion_tokens':599}}),200);e.event('[DONE]',300);self.assertFalse(e.report(400,80000,600)['success'])
 def test_native_error_event_is_failure(self):
  with self.assertRaises(ValueError):m.StreamEvidence(0).event(json.dumps({'error':{'message':'nativefail'}}),1)
 def test_formal_payload_preserves_all_input_and_no_debug_echo(self):
  calls=[];ids=[154855,2,3]
  def fake(req,**kw):
   body=json.loads(req.data);calls.append(body)
   if len(calls)==1:return Response(json.dumps({'usage':{'prompt_tokens':3,'completion_tokens':1},'kv_transfer_params':{'do_remote_prefill':True,'native_key':[1,2]}}).encode())
   rows=[{'choices':[{'index':0,'text':'hello','finish_reason':'length'}]},{'choices':[],'usage':{'prompt_tokens':3,'completion_tokens':600}}];return Response((''.join('data: '+json.dumps(r)+'\n\n' for r in rows)+'data: [DONE]\n\n').encode())
  with patch.object(m.urllib.request,'urlopen',fake):r=self.client().generate(ids,retain_text=False)
  self.assertTrue(r['success']);self.assertNotIn('text',r);self.assertEqual([v['prompt'] for v in calls],[ids,ids]);self.assertEqual(calls[0]['max_tokens'],1);self.assertEqual(calls[1]['max_tokens'],600);self.assertTrue(calls[1]['stream']);self.assertTrue(calls[1]['ignore_eos']);self.assertFalse(any(k in calls[1] for k in ['return_token_ids','return_tokens_as_token_ids','logprobs','echo','cache_salt']));self.assertEqual(calls[1]['kv_transfer_params'],{'do_remote_prefill':True,'native_key':[1,2]})
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);res=unittest.TextTestRunner(verbosity=1).run(suite);out={'project_id':'glm53-flash','passed':res.wasSuccessful(),'tests':res.testsRun,'actual_HTTP_model_SDK_NPU_calls':0,'scope':'CPUprotocol fixture only; not runtimePD/performance acceptance'};p=P/'checks/flash-vllm-native-pd-client-cpu-20261018.json';p.write_text(json.dumps(out,indent=2)+'\n');raise SystemExit(0 if res.wasSuccessful() else 1)
