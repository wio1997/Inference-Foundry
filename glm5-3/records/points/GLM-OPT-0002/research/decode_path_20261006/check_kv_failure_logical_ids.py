"""CPU-only execution of actual transfer/error/aggregator/scheduler consumers."""
from pathlib import Path
import ast,copy,math,threading,queue,json,hashlib
from types import SimpleNamespace as NS
from dataclasses import dataclass,field
from collections import defaultdict,OrderedDict

ROOT=Path(__file__).resolve().parent
class Log:
    def __getattr__(self,k):return lambda *a,**kw:None
G={'threading':threading,'OrderedDict':OrderedDict,'logger':Log(),'copy':copy,'math':math,'dataclass':dataclass,'field':field,
   'RequestStatus':NS(WAITING_FOR_REMOTE_KVS='remote')}
def load_class(path,name,methods=None):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    if methods is not None:node.body=[n for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in methods];node.bases=[];node.decorator_list=[]
    code='from __future__ import annotations\n'+ast.unparse(node)
    exec(compile(code,str(path),'exec'),G);return G[name]
def load_func(path,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path),'exec'),G)

base=ROOT/'attention_sources/actual__distributed__kv_transfer__kv_p2p__mooncake_connector.py'
new=ROOT/'kv_failure_candidate/mooncake_connector.py'
load_func(base,'transfer_groups_need_independent_block_ids')
methods={'_expand_block_ids','_local_kernel_ids_for_shard','_get_kernel_block_scale','_get_group_kernel_params','_get_local_remote_cp_params','_get_kv_cache_group_id','_get_kv_split_metadata'}
Worker=load_class(base,'MooncakeConnectorWorker',methods)
w=Worker();w.tp_size=16;w.tp_rank=5;w.dcp_size=16;w.dcp_rank=5;w.pcp_size=1;w.pcp_rank=0;w.block_size=128;w.handshake_port=36205;w.side_channel_port=36200;w.use_mla=True;w.use_sparse=True;w._is_hma_required=False;w._prefill_tp_size=16
w.kv_group2layeridx={0:({'kv_cache_group_id':0,'kv_cache_spec_type':'AscendSFAIndexerCacheSpec'},[1]),1:({'kv_cache_group_id':0,'kv_cache_spec_type':'AscendMLAAttentionSpec'},[0])};w.block_size_scale=[[1,1],[16]];w.remote_port_send_num={};w.local_remote_block_port_mapping={};w.vllm_config=NS(kv_transfer_config=NS(kv_port=36200));w._get_tp_num_need_pulls=lambda n:1
meta=NS(remote_ptp_size=16,remote_pcp_size=1,remote_dcp_size=16,remote_port=36000,remote_block_size=128,remote_engine_id='P',remote_host='P',num_external_tokens=2334,num_computed_tokens=0,num_prompt_blocks=19,local_block_ids=([1,2],),remote_block_ids=([24,25],),remote_multi_nodes_meta_mapping={str(i):{'host':'P','handshake_port':36000+i} for i in range(16)})
w._get_remote_host_info_by_port=lambda *a:('P','P')
ports,local,remote=w._get_kv_split_metadata('remote',meta)
assert ports==[[36005]] and local==[(list(range(16,32)),[1])],(ports,local)
load_class(ROOT/'pd_graph_sources/vllm__vllm__v1__outputs.py','KVConnectorOutput')
Agg=load_class(ROOT/'pd_graph_sources/vllm__vllm__distributed__kv_transfer__kv_connector__utils.py','KVOutputAggregator',{'__init__','aggregate'})
Scheduler=load_class(ROOT/'pd_graph_sources/vllm__vllm__v1__core__sched__scheduler.py','Scheduler',{'_handle_invalid_blocks','_update_requests_with_invalid_blocks'})
Tracker=load_class(base,'KVCacheTaskTracker',{'__init__','add_req_to_process','update_done_task_count'})
recv_methods={'add_request','_handle_request','_is_failed_recv_request','_mark_failed_recv_request','_clear_failed_recv_request','get_and_clear_invalid_block_ids','_mark_request_task_submitted','_mark_request_task_done'}
OldRecv=load_class(base,'KVCacheRecvingThread',recv_methods);NewRecv=load_class(new,'KVCacheRecvingThread',recv_methods)

def transfer(cls,kind='transfer',ids=None,legacy=False):
    r=cls();r.failed_recv_requests=set();r.invalid_block_ids=set();r.failed_recv_requests_lock=threading.Lock();r.request_task_counts_lock=threading.Lock();r.request_task_counts=defaultdict(int);r.finished_request_markers=set();r.pending_reformat_lock=threading.Lock();r.pending_reformat={};r.proc_not_transfer_request_lock=threading.Lock();r.proc_not_transfer_request={};r.task_tracker=Tracker();r.task_tracker.add_req_to_process('r');r.request_queue=queue.Queue();cleanup=[]
    r._send_done_signal_to_free_remote_port=lambda *a:cleanup.append('free');r._send_done_recv_signal=lambda *a:cleanup.append('recv')
    def fail(*a):raise RuntimeError('controlled transfer/reformat failure')
    r._transfer_kv_cache_all_groups=fail if kind=='transfer' else lambda *a:None
    r._reformat_pending_kv_caches=fail if kind=='reformat' else lambda *a:None
    if kind=='skipped':r.failed_recv_requests.add('r')
    kwargs={} if cls==OldRecv or legacy else {'local_logical_block_ids':ids if ids is not None else meta.local_block_ids}
    r.add_request('r','remote',local[0],remote[0],[],'P','P',36005,all_task_done=True,**kwargs)
    m=r.request_queue.get();r._mark_request_task_submitted(m);r._handle_request(m)
    assert cleanup==['free','recv'] and r.task_tracker.finished_requests=={'r'} and r.request_queue.unfinished_tasks==0
    return r.get_and_clear_invalid_block_ids(),m

def consume(invalid,arrival='together',where='remote'):
    # Real all-worker aggregator; real scheduler invalid-block consumer.
    a=Agg(16);K=G['KVConnectorOutput']
    def output(ids=(),done=False):return NS(kv_connector_output=K(finished_recving={'r'} if done else None,invalid_block_ids=set(ids)))
    if arrival=='early':
        rows=[output() for _ in range(16)];rows[5]=output(invalid,True);z=a.aggregate(rows);assert z.kv_connector_output.invalid_block_ids==invalid and not z.kv_connector_output.finished_recving
        rows=[output(done=i!=5) for i in range(16)];last=a.aggregate(rows);assert last.kv_connector_output.finished_recving=={'r'}
    else:
        rows=[output(done=True) for _ in range(16)];rows[5]=output(invalid,True);z=a.aggregate(rows);assert z.kv_connector_output.finished_recving=={'r'}
    req=NS(request_id='r',status='remote',num_computed_tokens=2334)
    s=Scheduler();s.recompute_kv_load_failures=False;s.block_size=2048;s.skipped_waiting=[req] if where=='remote' else [];s.running=[req] if where=='running' else [];s.failed_recving_kv_req_ids=set();s.kv_cache_manager=NS(get_block_ids=lambda rid:([1,2],),evict_blocks=lambda ids:None)
    return s._handle_invalid_blocks(z.kv_connector_output.invalid_block_ids,{})

old,_=transfer(OldRecv);assert old==set(range(16,32)) and consume(old)==set()
rows=[]
for kind in ['transfer','reformat','skipped','success']:
 for arrival in ['early','together']:
  for where in ['remote','running']:
   invalid,_=transfer(NewRecv,kind);expected=set() if kind=='success' else {1,2};assert invalid==expected
   failed=consume(invalid,arrival,where);assert failed==(set() if kind=='success' else {'r'})
   rows.append(dict(kind=kind,arrival=arrival,request_state=where,invalid=sorted(invalid),fail_policy_matches=bool(failed)))
# Stable snapshot: asynchronous mutation cannot replace manager IDs.
r=NewRecv();r.request_queue=queue.Queue();ids=[[1,2]];r.add_request('r','remote',local[0],remote[0],[],'P','P',36005,local_logical_block_ids=ids);ids[0][:]=[99];assert r.request_queue.get()['local_logical_block_ids']==((1,2),)
# Explicit empty group stays empty; None/legacy maintains old ABI fallback only.
invalid,m=transfer(NewRecv,ids=[[]]);assert invalid==set() and m['local_logical_block_ids']==((),)
invalid,m=transfer(NewRecv,legacy=True);assert invalid==old
# Empty transfer plane still reports request logical identity for a failure in another plane.
saved=local[0];local[0]=([],saved[1]);invalid,_=transfer(NewRecv);assert invalid=={1,2};assert consume(invalid)=={'r'};local[0]=saved
out={'CPU_only':True,'NPU_initialized':False,'production_AST':True,'old_counterexample':{'ports':ports,'kernel_local':local,'scheduler_logical':meta.local_block_ids,'old_invalid':sorted(old),'old_fail_matches':False},'cases':rows,'immutable_snapshot':True,'explicit_empty_preserved':True,'legacy_fallback_only':True,'empty_transfer_group0':True,'correctness_candidate_only':True,'performance_claim':False,'limitations':['Reconstructed scale16/layout inputs match saved geometry; original Run263 req_meta was not recorded.','Scheduler invalid consumer and all-rank aggregator run as actual AST; API failure/worker lifecycle require one bounded device diagnostic.'],'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base,new,ROOT/'pd_graph_sources/vllm__vllm__v1__core__sched__scheduler.py',ROOT/'pd_graph_sources/vllm__vllm__distributed__kv_transfer__kv_connector__utils.py']}}
(ROOT/'kv_failure_CPU.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'cases':len(rows)+4,'old_invalid':sorted(old),'patched_invalid':[1,2],'CPU_only':True}))
