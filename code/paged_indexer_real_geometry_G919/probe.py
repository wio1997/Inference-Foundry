"""Actual 910B FP4-domain equivalence and changed-input graph gate; no model."""
from pathlib import Path
import datetime,hashlib,json,os,traceback
import torch,torch_npu
import npu_paged_indexer as candidate
from sglang.kernels.ops.attention.dsv4 import fp4_indexer as native
from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4

rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{144+rank});torch.set_num_threads(1);torch.set_grad_enabled(False);torch.npu.set_device(rank)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'graphs':[],'score_checks':[],'scope':'NPUselectedFP4operanddomain component/changedinputgraph only; notmodelquality orE2Egain'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2)+'\n');q.replace(p)
def synchronize():torch.npu.synchronize()
def baseline(q,w,req,rt,lens,table,L,ratio):
 j=torch.arange(L,device=q.device);valid=j[None,:]<lens[:,None]
 slots=rt[req[:,None],(j*ratio)[None,:]].to(torch.int64)//ratio
 slots=slots.masked_fill(~valid,0)
 record['step']='stock_FP4_scores_pending';save();scores=native.fp4_index_logits_decode(q,w,slots,lens,table,table.shape[1]//68);synchronize();record['step']='stock_TopKV2';save()
 k=min(512,L);indices=scores.topk(k,dim=-1,sorted=False).indices.sort(dim=-1).values
 pages=torch.full((q.shape[0],512),-1,dtype=torch.int32,device=q.device)
 pages[:,:k]=torch.where(indices<lens[:,None],slots.gather(1,indices.clamp_max(L-1)),-1).to(torch.int32)
 synchronize();record['step']='stock_baseline_completed';save()
 return scores,pages
def proposed(q,w,req,rt,lens,table,L,ratio,pages=None):
 scores=candidate.score_paged(q,w,req,rt,lens,table,table.shape[1]//68,L,ratio)
 if pages is None:pages=torch.empty((q.shape[0],512),dtype=torch.int32,device=q.device)
 candidate.select_slots(scores,lens,req,rt,pages,ratio,512);return scores,pages
def compare(a,b):
 assert torch.equal(a[0].view(torch.int32),b[0].view(torch.int32)),'FP32scorebits differ; strictgatepreserved'
 assert torch.equal(a[1],b[1]),'SortedphysicalKVslots differ; strictgatepreserved'
try:
 save();pins=json.loads((ROOT/'paged_indexer_source_pins_g901.json').read_text());assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==pins['FP4_kernel']
 record.update(architecture=os.uname().machine,torch=torch.__version__,torch_npu=torch_npu.__version__,native_FP4_source_SHA256=pins['FP4_kernel'],candidate_source_SHA256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
 device=f'npu:{rank}';page_size=64;pool_rows=4096;max_requests=4
 cases=[] # Realgraphcapacitiesbelow, basedonactualD0TopKV2shapeevidenceG917
 for ratio in [1,2]:
  capacity=263435 if ratio==1 else 131717
  cases=[(1,1),(6,65),(16,513),(80,4097),(96,capacity),(1,capacity)]
  for number,(batch,L) in enumerate(cases):
   gen=torch.Generator().manual_seed(90200+100*rank+10*ratio+number)
   q=fake_quant_fp4((torch.randn(batch,32,128,generator=gen)*.125).bfloat16().to(device));weights=torch.rand(batch,32,generator=gen).bfloat16().to(device)
   req=(torch.arange(batch,device=device)%max_requests).to(torch.int64)
   rt=torch.randint(0,pool_rows*ratio,(max_requests,L*ratio),generator=gen,dtype=torch.int32).to(device)
   table=torch.randint(0,256,(pool_rows//page_size,page_size*68),generator=gen,dtype=torch.uint8).to(device);table[:,page_size*64:]=127
   lens=(torch.arange(batch,device=device)*37)%min(L+1,1025)
   if batch>1:lens[0]=0;lens[-1]=min(L,1024)
   elif L>4097:lens.fill_(L) # actualfull-visiblelongcontext, notonlycapacityallocation
   q0=q.clone();w0=weights.clone();rt0=rt.clone();t0=table.clone();l0=lens.clone()
   record.update(phase='case_running',current_case={'batch':batch,'capacity_L':L,'ratio':ratio},step='stock_baseline');save();synchronize();a=baseline(q,weights,req,rt,lens,table,L,ratio)
   record['step']='candidate_scores';save();score=candidate.score_paged(q,weights,req,rt,lens,table,page_size,L,ratio);synchronize();assert torch.equal(a[0].view(torch.int32),score.view(torch.int32)),'FP32scorebits differ'
   record['score_checks'].append({'rows':batch,'capacity_L':L,'ratio':ratio,'native_FP32_score_bits_equal':True});record['step']='candidate_selected_K_mapping';save()
   b=proposed(q,weights,req,rt,lens,table,L,ratio);synchronize();compare(a,b)
   assert torch.equal(q,q0) and torch.equal(weights,w0) and torch.equal(rt,rt0) and torch.equal(table,t0) and torch.equal(lens,l0)
   record['cases'].append({'rows':batch,'capacity_L':L,'ratio':ratio,'head32_dim128':True,'native_FP4_scores_bitwise_equal':True,'selected_physical_slots_equal':True,'inputs_unchanged':True,'domain':'Nativeper32UE8M0FP4query + actualpacked68BindexKpage layout; BF16headweights','largest_old_dense_slot_bytes':batch*L*8,'candidate_no_dense_slot_tensor':True});save()
   if batch==96 and L==capacity:
    # Capture only the candidate, compare every replay against fresh stock eager.
    # q, request-to-token contents, visible lengths and packed K contents change.
    out_pages=torch.empty((batch,512),dtype=torch.int32,device=device)
    for _ in range(2):c=proposed(q,weights,req,rt,lens,table,L,ratio,out_pages)
    synchronize();graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):captured=proposed(q,weights,req,rt,lens,table,L,ratio,out_pages)
    for replay,bound in enumerate([127,2048,63]):
     q.copy_(fake_quant_fp4((torch.randn(batch,32,128,generator=gen)*.125).bfloat16().to(device)))
     lens.copy_((torch.arange(batch,device=device)*53)%(bound+1));lens[0]=0;lens[-1]=bound
     rt.copy_(torch.randint(0,pool_rows*ratio,rt.shape,generator=gen,dtype=torch.int32).to(device))
     table[:,:page_size*64].bitwise_xor_(torch.tensor(17,dtype=torch.uint8,device=device))
     expected=baseline(q,weights,req,rt,lens,table,L,ratio);synchronize();graph.replay();synchronize();compare(expected,captured)
     record['graphs'].append({'ratio':ratio,'replay':replay,'capacity_L':L,'changed_live_bound':bound,'changed_q_req_table_packed_K':True,'native_eager_score_bits_and_physical_slots_equal':True});save()
    del graph,captured,c,out_pages,expected
   del a,b,score,q,weights,req,rt,table,lens,q0,w0,rt0,t0,l0;synchronize()
 record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except BaseException as e:
 record.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
