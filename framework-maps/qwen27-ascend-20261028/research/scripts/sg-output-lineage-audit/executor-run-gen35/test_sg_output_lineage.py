import unittest,copy,json
from audit_sg_output_lineage import commit_tokens,join_graph_tokens,request_rid
RID='full-request-ID'
def row(step,total,token,delta=1,mode='2'):
 return {'event':'scheduler_commit','_line':step,'pid':61,'rank_id':0,'forward_iter':step,'forward_mode':mode,'requests':[{'rid':RID}],'token_commit':[{'rid':RID,'committed_output_ids_total':total,'last_token_id':token,'model_sample_token_delta':delta}]}
def fixture():return [row(10,1,16,mode='1'),row(11,2,13),row(12,2,13,delta=0)]
def graph_fixture():
 rows=[];fences=[]
 for step in [11,12]:
  inv='exact-invocation-'+str(step);fences.append({'invocation':inv,'layer_rows':[{'layer_idx':i}for i in range(64)]})
  for side in ['start','end']:rows.append({'event':'sg_replay_outer_event','work_key':json.dumps({'step':step,'requests':[{'rid':RID}]}),'pid':61,'rank_id':0,'invocation':inv,'side':side,'_line':step})
 return rows,fences
class IdentityTests(unittest.TestCase):
 def test_normal_two_tokens_plus_tail(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture();out=join_graph_tokens(r,RID,c,z,f)
  self.assertEqual([x['last_token_id']for x in c],[16,13]);self.assertEqual(out[-1]['model_sample_token_delta'],0)
 def test_reject_multi_token_delta_cannot_use_last_only(self):
  r=fixture();r[1]['token_commit'][0]['model_sample_token_delta']=2
  with self.assertRaises(AssertionError):commit_tokens(r,RID,2)
 def test_reject_missing_or_gap_commit(self):
  r=fixture();r[1]['token_commit'][0]['committed_output_ids_total']=3
  with self.assertRaises(AssertionError):commit_tokens(r,RID,2)
 def test_reject_duplicate_commit_step(self):
  r=fixture();r[1]['forward_iter']=10
  with self.assertRaises(AssertionError):commit_tokens(r,RID,2)
 def test_reject_zero_delta_changed_token(self):
  r=fixture();r[-1]['token_commit'][0]['last_token_id']=999
  with self.assertRaises(AssertionError):commit_tokens(r,RID,2)
 def test_reject_mixed_request_batch(self):
  r=fixture();r[0]['requests'].append({'rid':'other'})
  with self.assertRaises(AssertionError):commit_tokens(r,RID,2)
 def test_reject_duplicate_RID_association(self):
  r={'event':'sg_request_id_link','external_request_id':'client','rid':RID}
  with self.assertRaises(AssertionError):request_rid([r,r],'client')
 def test_reject_prefix_ID_match(self):
  with self.assertRaises(AssertionError):request_rid([{'event':'sg_request_id_link','external_request_id':'client-long','rid':RID}],'client')
 def test_reject_graph_from_other_PID(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture()
  for x in r:x['pid']=999
  with self.assertRaises(AssertionError):join_graph_tokens(r,RID,c,z,f)
 def test_reject_missing_outer_endpoint(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture()
  with self.assertRaises(AssertionError):join_graph_tokens(r[1:],RID,c,z,f)
 def test_reject_mismatched_workstep_on_same_invocation(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture();r[1]['work_key']=json.dumps({'step':999,'requests':[{'rid':RID}]})
  with self.assertRaises(AssertionError):join_graph_tokens(r,RID,c,z,f)
 def test_reject_fence_without_exact_invocation(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture();f[0]['invocation']='similar-11'
  with self.assertRaises(AssertionError):join_graph_tokens(r,RID,c,z,f)
 def test_reject_63_native_layers(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture();f[0]['layer_rows'].pop()
  with self.assertRaises(AssertionError):join_graph_tokens(r,RID,c,z,f)
 def test_reject_mixed_work_request(self):
  c,z=commit_tokens(fixture(),RID,2);r,f=graph_fixture();r[0]['work_key']=json.dumps({'step':11,'requests':[{'rid':RID},{'rid':'other'}]})
  with self.assertRaises(AssertionError):join_graph_tokens(r,RID,c,z,f)
if __name__=='__main__':unittest.main()
