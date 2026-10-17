"""Same actual saved inputs: native convolution, bulk vs bounded native oracle.
Only24-token native serial oracle is used, for diagnosis, never performance.
65/2379 cases repeat actual post-conv vectors: labelled synthetic stress only.
"""
import pathlib,json,hashlib,importlib.util,traceback,time
R=pathlib.Path('/run-evidence');out={'project_id':'glm53-flash','passed':False,'completed':False,'generation_requests':0,'formal_requests':0,'scope':'ActualR049layer0/44T24same-input operatorreplay +labelledreal-derivedT65/2379stress; no model/PD/Graph/MTP/cache80K/performancequalification','cases':[]}
try:
 import torch,torch_npu
 torch.set_num_threads(2);torch.npu.set_device(0)
 for rel,sha in json.loads((R/'expected-native-sources.json').read_text()).items():assert hashlib.sha256((pathlib.Path('/vllm-workspace/vllm-ascend')/rel).read_bytes()).hexdigest()==sha,rel
 native=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/kda.py');assert hashlib.sha256(native.read_bytes()).hexdigest()=='d6027b2bbec786142184a85a0250bb5b9786ee2ec5d9e3aa9676a481cb4ff2ce'
 stable=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/fla_npu/ops/ascendc/_stable.py');assert hashlib.sha256(stable.read_bytes()).hexdigest()=='d6be8820bb54e0177cf92e396099011402f97066b0974a2f1a8a90187f08aae0'
 from vllm_ascend.ops.kda import run_recurrent_kda
 from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
 spec=importlib.util.spec_from_file_location('candidate_kda',R/'candidate_kda.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 private=R/'private-replay-inputs.pt';assert hashlib.sha256(private.read_bytes()).hexdigest()=='40447d146f74a022aabda64430d2e30ce0ad15c5e2b0d80e6d86494c297a7975';payload=torch.load(private,map_location='cpu',weights_only=True);assert len(payload)==2 and [p['layer'] for p in payload]==[0,44]
 def save():(R/'result.json').write_text(json.dumps(out,indent=2)+'\n')
 def compare(a,b,atol,rtol):
  a=a.detach().float().cpu();b=b.detach().float().cpu();d=(a-b).abs();return {'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'max_abs':float(d.max()),'RMS':float(d.square().mean().sqrt()),'within_frozen_gate':bool((d<=atol+rtol*b.abs()).all()),'atol':atol,'rtol':rtol,'bitwise_equal_diagnostic_only':bool(torch.equal(a,b)),'unequal_elements_diagnostic_only':int((a!=b).sum())}
 def native_cache_view():
  backing=torch.zeros(4,6,128,128,dtype=torch.float32,device='npu');view=backing.as_strided((4,4,128,128),(98304,16384,128,1));assert view.stride()==(98304,16384,128,1);return view
 def oracle(q,k,v,g,b,A,dt):
  q=q.double();k=k.double();v=v.double();q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt();k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt();state=torch.zeros(4,128,128,dtype=torch.float64);ys=[]
  for i in range(q.shape[1]):
   decay=(-5*torch.sigmoid(A.double().view(4,1).exp()*(g[0,i].double()+dt.double().view(4,128)))).exp();state=state*decay.unsqueeze(-2);dv=(v[0,i]-(state*k[0,i].unsqueeze(-2)).sum(-1))*b[0,i].double().sigmoid().unsqueeze(-1);state=state+dv.unsqueeze(-1)*k[0,i].unsqueeze(-2);ys.append((state*(q[0,i]/128**.5).unsqueeze(-2)).sum(-1))
  return torch.stack(ys).unsqueeze(0),state.unsqueeze(0)
 for p in payload:
  assert p['T']==24 and p['H']==4 and p['D']==128;layer=p['layer'];x=p['actual_qkv_before_conv'].to('npu');w=p['conv_weights'].squeeze(1).transpose(0,1).to(dtype=x.dtype,device='npu').contiguous();starts=torch.tensor([0,24],dtype=torch.int32,device='npu');slots=torch.tensor([3],dtype=torch.int32,device='npu');fresh=torch.tensor([False],dtype=torch.bool,device='npu');variants=[]
  for layout in ['compact','R049_strided_view']:
   backing=torch.zeros((4,3 if layout=='compact' else 128,1536),dtype=torch.bfloat16,device='npu');state=backing if layout=='compact' else backing.as_strided((4,3,1536),(196608,1536,1));z=causal_conv1d(x,w,state,starts,slots,run_mode=0,initial_state_mode=fresh);torch.npu.synchronize();assert torch.equal(state[3].cpu(),p['actual_qkv_before_conv'][-3:]) and bool((state[:3]==0).all());variants.append(z.cpu())
  assert x.dtype==w.dtype==torch.bfloat16 and tuple(w.shape)==(4,1536)
  conv_equal=bool(torch.equal(*variants));assert conv_equal,'Nativecompact/stridedconvoutputs differ';q0,k0,v0=[z.reshape(1,24,4,128).contiguous() for z in variants[1].split(512,dim=-1)];g0=p['actual_raw_gate'];b0=p['actual_raw_beta'];Ac=p['A_log'];dtc=p['dt_bias'];A=Ac.to('npu');dt=dtc.to('npu')
  for T in [24,65,2379]:
   idx=torch.arange(T)%24;qc,kc,vc,gc,bc=[z.index_select(1,idx).contiguous() for z in [q0,k0,v0,g0,b0]];expected_y,expected_s=oracle(qc,kc,vc,gc,bc,Ac,dtc);q,k,v,g,b=[z.to('npu') for z in [qc,kc,vc,gc,bc]];initial=torch.zeros(1,4,128,128,dtype=torch.float32,device='npu')
   y,s=m.run_chunk_kda(q,k,v,g,b.float().sigmoid(),initial.clone(),[0,T],None,A,dt,lower_bound=-5.)
   _,prefix_s=m.run_chunk_kda(q[:,:T-1],k[:,:T-1],v[:,:T-1],g[:,:T-1],b[:,:T-1].float().sigmoid(),initial.clone(),[0,T-1],None,A,dt,lower_bound=-5.)
   cache=native_cache_view();cache[3].copy_(prefix_s[0]);one=torch.tensor([0,1],dtype=torch.int32,device='npu');last=run_recurrent_kda(q[:,-1:],k[:,-1:],v[:,-1:],g[:,-1:],b[:,-1:],cache,one,slots,A,dt,lower_bound=-5.,beta_is_preprocessed=False);torch.npu.synchronize();assert bool((cache[:3]==0).all())
   comparisons={'bulk_state_vs_CPU64':compare(s,expected_s,1e-4,1e-4),'bulk_output_vs_CPU64':compare(y,expected_y,.02,.02),'bulk_vs_split_native_last_state':compare(s,cache[3:4],1e-4,1e-4),'bulk_vs_split_native_last_output':compare(y[:,-1:],last,.02,.02)}
   row={'layer':layer,'tokens':T,'source':'actualsaved24nativepostconv' if T==24 else 'syntheticrepeatofactual24postconv_notmodelhistory','native_conv_compact_strided_bitwise_equal':conv_equal,'NULL_slot_unchanged':True,'actual_slot':3,'native_conv_weight_shape':[4,1536],'native_conv_weight_dtype':'torch.bfloat16','native_conv_strided_cache_strides':[196608,1536,1],'native_recurrent_cache_strides':list(cache.stride()),'comparisons':comparisons}
   if T==24:
    native_cache=native_cache_view();parts=[]
    # Bounded native serial reference48calls TOTAL, onlydiagnosis/no timing.
    for token in range(24):parts.append(run_recurrent_kda(q[:,token:token+1],k[:,token:token+1],v[:,token:token+1],g[:,token:token+1],b[:,token:token+1],native_cache,one,slots,A,dt,lower_bound=-5.,beta_is_preprocessed=False))
    native_y=torch.cat(parts,dim=1);torch.npu.synchronize();assert bool((native_cache[:3]==0).all());comparisons.update(native_serial_state_vs_CPU64=compare(native_cache[3:4],expected_s,1e-4,1e-4),bulk_vs_native_serial_state=compare(s,native_cache[3:4],1e-4,1e-4),bulk_vs_native_serial_output=compare(y,native_y,.02,.02),native_replay_vs_saved_R049_state=compare(native_cache[3],p['saved_native_final_state'],1e-4,1e-4),native_replay_vs_saved_R049_last_output=compare(native_y[0,-1],p['saved_native_last_output'],.02,.02));row['native_serial_oracle_scope']='24tokensonly/diagnostic/notperformance';row['native_serial_calls']=24
   out['cases'].append(row);save()
 out['completed']=True;out['passed']=all(c['finite'] and c['within_frozen_gate'] for row in out['cases'] for c in row['comparisons'].values())
except BaseException as exc:out.update(error=repr(exc),traceback=traceback.format_exc())
(R/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
