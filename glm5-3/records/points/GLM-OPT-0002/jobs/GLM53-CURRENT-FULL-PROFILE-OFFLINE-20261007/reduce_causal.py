from pathlib import Path
import json,gzip,hashlib,collections
repo=Path('/Users/wio/work/Inference-Foundry-glm5-3');j=repo/'glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007'
def merge(iv):
 out=[]
 for s,e in sorted(iv):
  if e<=s:continue
  if out and s<=out[-1][1]:out[-1][1]=max(out[-1][1],e)
  else:out.append([s,e])
 return out
reports=[]
for rank in (0,13,15):
 geometry=json.loads((j/'rope_geometry'/('rank%d.json'%rank)).read_text())
 r=json.loads((j/'frontier'/('rank%d.json'%rank)).read_text())
 for ordinal in (2,3):
  p=j/'frontier'/('rank%d_period%d_events.json.gz'%(rank,ordinal));es=json.loads(gzip.decompress(p.read_bytes()));period=r['periods'][ordinal];lo,hi=period['lo_us'],period['hi_us'];links=[]
  for hw in [x for x in es if x['event']['name']=='aclnnIndex_SliceAiCore_Slice']:
   e=hw['event'];con=e['args']['connection_id'];node=next(x for x in es if x['event']['name']=='Node@launch' and x['event'].get('args',{}).get('connection_id')==con)
   n=node['event'];nt=float(n['ts']);deq=[x for x in es if x['event'].get('cat')=='dequeue' and x['event']['tid']==n['tid'] and float(x['event']['ts'])<=nt<float(x['event']['ts'])+float(x['event']['dur'])];assert len(deq)==1
   corr=deq[0]['event']['args']['correlation_id'];enq=next(x for x in es if x['event'].get('cat')=='enqueue' and x['event']['args']['correlation_id']==corr);en=enq['event'];et=float(en['ts']);cpu=[x for x in es if x['event']['name']=='aten::index' and x['event']['tid']==en['tid'] and float(x['event']['ts'])<=et<float(x['event']['ts'])+x['event']['dur']];assert len(cpu)==1
   ct=float(cpu[0]['event']['ts'])+cpu[0]['event']['dur'];nextcpu=sorted([x for x in es if x['event'].get('cat')=='cpu_op' and x['event']['tid']==en['tid'] and float(x['event']['ts'])>=ct],key=lambda x:float(x['event']['ts']));unsqueeze=[x for x in nextcpu if x['event']['name']=='aten::unsqueeze'][:2];assert len(unsqueeze)==2 and float(unsqueeze[-1]['event']['ts'])-ct<25
   geom=next(x for x in geometry['selected'] if x['original_event_index']==hw['index']);assert geom['kernel_row']['Output Shapes']=='"1048576,64"' and geom['kernel_row']['Input Shapes']=='"1048576,128;2;2"'
   links.append(dict(hardware=hw,node=node,dequeue=deq[0],enqueue=enq,index_scope=cpu[0],next_unsqueeze=unsqueeze,geometry=geom['kernel_row'],source_attribution='get_cos_and_sin_mla two cache[positions].unsqueeze(1).unsqueeze(2); unique source/call-pattern inference, no recorded Python line stack',device_offset_us=float(e['ts'])-lo))
  assert len(links)==4
  nonwait=[x for x in es if x['event'].get('args',{}).get('Task Type') in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','COMMUNICATION','SDMA_SQE','PCIE_DMA_SQE')]
  union=merge([(max(lo,float(x['event']['ts'])),min(hi,float(x['event']['ts'])+x['event']['dur'])) for x in nonwait]);used=sum(b-a for a,b in union)
  reports.append(dict(rank=rank,ordinal=ordinal,event_export_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),trace_sha256=r['sha256'],extent_us=hi-lo,nonwait_hardware_union_us=used,nonwait_complement_us=hi-lo-used,links=links,slice_inclusive_us=sum(x['hardware']['event']['dur'] for x in links),materialized_output_bytes_per_step=4*1048576*64*2,observed_saving_us=None))
allr=[json.loads((j/'frontier'/('rank%d.json'%rank)).read_text()) for rank in range(16)];skews=[]
assert all(len(r['graph_execute'])==12 for r in allr)
for ordinal in range(1,11):
 starts=[(float(r['graph_execute'][ordinal]['event']['ts']),r['rank']) for r in allr];skews.append(dict(ordinal=ordinal,skew_us=max(s for s,_ in starts)-min(s for s,_ in starts),latest_rank=max(starts)[1]))
result=dict(reducer='root independent connection -> dequeue -> enqueue -> ATen scope -> unique source pattern -> actual kernel geometry',all16_target_replay_count=12,steady_cross_rank_skew=skews,periods=reports,profile_off_on_calibration_valid=False,calibration_invalid_reason='Run276 strict post-request aggregate draft signature differs despite exact same natural23 output',performance_gain=None,Current=None)
p=j/'critical_path_reduced.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'periods':len(reports),'links':sum(len(z['links']) for z in reports),'steady_skews':skews,'slice_period3_us':{z['rank']:z['slice_inclusive_us'] for z in reports if z['ordinal']==3},'NPU_requests':0}))
