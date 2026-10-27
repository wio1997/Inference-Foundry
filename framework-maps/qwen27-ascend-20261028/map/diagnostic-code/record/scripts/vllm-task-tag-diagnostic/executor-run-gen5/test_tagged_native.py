"""Adversarial identity/coverage fixtures, including absent flow and duplicate keys."""
from pathlib import Path
import copy,gzip,json,tempfile,unittest
from audit_tagged_native import classify,load_indexes

def fixture():
 markers=[{'raw_task_key':[49,8,t],'marker_id':f'layer00:{side}','layer_idx':0,'marker_side':side}for t,side in [(0,'pre'),(3,'post')]]
 outer=[];events=[]
 def native(task,ts,model=49,stream=8,dur=1):
  return {'name':'event','ph':'X','pid':900,'tid':stream,'ts':ts,'dur':dur,'args':{'Task Type':'EVENT RECORD','Model Id':model,'Task Id':task,'Physic Stream Id':stream}}
 for side,task,ts,cts in [('start',10,100,1),('end',11,200,10)]:
  outer.append({'rank_id':0,'pid':335,'graph_python_id':123,'invocation':'explicit-replay-A','side':side,'event':'vllm_replay_outer_event','capture_status':0,'model_ri_nonnull':False,'stream_rc':0,'task_rc':0,'capture_rc':0,'callback_rc':0,'task_id':task,'acl_stream_id':37})
  events.extend([{'name':f'explicit-replay-A.{side}','ph':'X','pid':335,'tid':1,'ts':cts,'dur':3},
                 {'name':'record_event','ph':'X','pid':335,'tid':1,'ts':cts+1,'dur':1},
                 {'name':'torch_to_npu','cat':'async_npu','ph':'s','id':task,'pid':335,'tid':1,'ts':cts+1.5},
                 {'name':'torch_to_npu','cat':'async_npu','ph':'f','id':task,'pid':900,'tid':37,'ts':ts},
                 native(task,ts,2147483647,37)])
 events.extend([native(0,120),native(3,180),native(0,130,model=50),native(0,140,stream=9),native(33,195,dur=10)])
 return markers,outer,events

class IdentityTests(unittest.TestCase):
 def run_case(self,markers,outer,events):
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp)/'fixture.json.gz'
   with gzip.open(p,'wt')as f:json.dump(events,f)
   es,meta,scopes,cpu,flows,at,native,digest,size=load_indexes(p,temp)
   return classify(markers,outer,es,(meta,scopes,cpu,flows,at,native),'900','NPU 0')
 def test_exact_full_tuple_and_crossing_preserved(self):
  m,o,e=fixture();out,rest=self.run_case(m,o,e)
  self.assertEqual(out['counts']['resolved_endpoints'],2)
  self.assertEqual(out['counts']['exact_marker_occurrences'],2)
  self.assertEqual(out['layer_endpoint_intervals'][0]['endpoint_elapsed_us'],'59')
  self.assertTrue(any(x['classification']=='crossing'for x in rest))
  self.assertEqual(len(rest),5) # outer2 + wrong model + wrong stream + crossing
 def test_absent_flow_does_not_pair_by_time(self):
  m,o,e=fixture();e=[x for x in e if not(x.get('ph')=='f'and x.get('id')==11)]
  out,rest=self.run_case(m,o,e)
  self.assertEqual(out['counts']['explicit_outer_pairs'],0)
  self.assertEqual(out['counts']['layer_endpoint_intervals'],0)
  self.assertTrue(out['unresolved_outer_rows'])
 def test_duplicate_key_in_same_replay_stays_unknown(self):
  m,o,e=fixture();extra=copy.deepcopy(e[10]);extra['ts']=121;e.append(extra)
  out,rest=self.run_case(m,o,e)
  self.assertEqual(out['counts']['exact_marker_occurrences'],1)
  self.assertEqual(out['unknown_markers'][0]['candidate_count'],2)
  self.assertEqual(out['layer_endpoint_intervals'],[])
 def test_wrong_hardware_pid_cannot_match(self):
  m,o,e=fixture();e[10]['pid']=901
  out,rest=self.run_case(m,o,e)
  self.assertEqual(out['counts']['exact_marker_occurrences'],1)
  self.assertEqual(out['unknown_markers'][0]['candidate_count'],0)
 def test_missing_post_not_filled(self):
  m,o,e=fixture();del e[11]
  out,rest=self.run_case(m,o,e)
  self.assertEqual(out['layer_endpoint_intervals'],[])
  self.assertEqual(out['unknown_markers'][0]['candidate_count'],0)
 def test_reversed_layer_endpoints_rejected(self):
  m,o,e=fixture();e[10]['ts'],e[11]['ts']=180,120
  out,rest=self.run_case(m,o,e)
  self.assertEqual(out['layer_endpoint_intervals'],[])
  self.assertEqual(out['unknown_markers'][0]['reason'],'invalid_layer_endpoint_order')
 def test_duplicate_flow_finish_rejected(self):
  m,o,e=fixture();e.append(copy.deepcopy(e[3]))
  out,rest=self.run_case(m,o,e)
  self.assertEqual(out['counts']['explicit_outer_pairs'],0)
 def test_overlapping_explicit_replays_rejected(self):
  m,o,e=fixture();more=copy.deepcopy(o)
  for x in more:x['invocation']='explicit-replay-B'
  more_events=copy.deepcopy([x for i,x in enumerate(e[:10])if i not in(4,9)])
  for x in more_events:
   if x['name'].startswith('explicit-replay-A'):x['name']=x['name'].replace('A','B')
   if x.get('ph')in('s','f'):x['id']+=100
   if x['name']=='record_event'or x.get('ph')=='s'or x['name'].startswith('explicit-replay-B'):x['tid']=2
  out,rest=self.run_case(m,o+more,e+more_events)
  self.assertEqual(out['counts']['layer_endpoint_intervals'],0)
  self.assertTrue(out['outer_overlaps'])
 def test_reused_graph_keys_resolve_only_inside_explicit_replay(self):
  m,o,e=fixture();more=copy.deepcopy(o);extra=copy.deepcopy(e)
  for x in more:x['invocation']='explicit-replay-B'
  for x in extra:
   x['ts']+=300
   if x['name'].startswith('explicit-replay-A'):x['name']=x['name'].replace('A','B')
   if x.get('ph')in('s','f'):x['id']+=100
  out,rest=self.run_case(m,o+more,e+extra)
  self.assertEqual(out['counts']['exact_marker_occurrences'],4)
  self.assertEqual(out['counts']['layer_endpoint_intervals'],2)
  self.assertEqual({x['invocation']for x in out['layer_endpoint_intervals']},{'explicit-replay-A','explicit-replay-B'})

if __name__=='__main__':unittest.main(verbosity=2)
