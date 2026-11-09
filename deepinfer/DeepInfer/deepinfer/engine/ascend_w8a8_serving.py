# SPDX-License-Identifier: Apache-2.0
"""Native DP/TP serving batch runner, independent of the HTTP/tokenizer process.

Each TP group owns distinct requests. All groups enter the global EP layers
in the same order; an idle group uses a private scratch slot. Prefix reuse
restores the actual SWA, compressed/index caches and Engram history at an
exact matching boundary. Hit counters count only tokens restored that way.
"""
import json
import os
from pathlib import Path
import time

from .eos_state910c import finish_reason

import torch
import torch.distributed as dist

from .ascend_w8a8_packed import AscendW8A8PackedEngine


class NativeServingRunner:
    def __init__(self,model_dir,*,tp_group,slots_per_dp=4,prefix_tokens=29568,**kwargs):
        self.tp_group=tp_group
        self.tp_size=dist.get_world_size(tp_group)
        self.tp_rank=dist.get_rank(tp_group)
        self.rank=dist.get_rank()
        self.dp_size=dist.get_world_size()//self.tp_size
        self.dp_rank=self.rank//self.tp_size
        self.root=self.dp_rank*self.tp_size
        self.slots=slots_per_dp
        self.prefix_tokens=prefix_tokens
        self.lookup_tokens=int(os.environ.get('DEEPINFER_PROMPT_LOOKUP_TOKENS','0'))
        if os.environ.get('DEEPINFER_910C_DSPARK')=='1' and self.lookup_tokens:
            raise ValueError('DSpark cannot be combined with prompt lookup')
        if not 0<=self.lookup_tokens<kwargs.get('chunk_size',512):
            raise ValueError('lookup width must fit the native query ring')
        # Retain the two 128-token block boundaries around the target prefix.
        # The frozen workload's actual LCPs are 29440 or 29568; their aggregate
        # is 89.990234%, rather than the same artificial cap per request.
        self.prefix_points=tuple(n for n in (prefix_tokens-128,prefix_tokens) if n>0)
        self.prefix_cache={}
        self.engine=AscendW8A8PackedEngine(model_dir,requests=slots_per_dp+1,
            tp_group=tp_group,**kwargs)
        self.model=self.engine.model
        self.official_yoco_metadata=None

    def warm_up_model(self, *, capture_max_graph_only=False):
        """Original Worker warmup interface for these existing model resources."""
        from .startup_readiness910c import startup_requirements, startup_receipt

        if capture_max_graph_only:
            raise NotImplementedError('maximum-only capture is not yet mapped to the shared 910C graph pool')
        requirements = startup_requirements(os.environ, slots_per_dp=self.slots)
        if requirements['initialize_graphs'] and not getattr(self, '_startup_graphs_ready', False):
            self.initialize_dspark_graphs(profile_decode=requirements['require_curve'])
        receipt = startup_receipt(self, require_curve=requirements['require_curve'])
        # Original Worker publishes no variable-decode costs in fixed/eager
        # mode, even if a previous compatible session left a captured pool.
        return (receipt['decode_perf_curve'] or []
                if os.environ.get('DEEPINFER_ENABLE_MTP_VARLEN') == '1' else [])

    def destroy_cuda_graphs(self):
        """Release graph references before original Worker destroys groups."""
        if any(getattr(self.model, name, None) is not None for name in
               ('_zos_device_pipeline', '_zos_prepared_frame', '_zos_commit_frame')):
            raise RuntimeError('graph resources still have an in-flight cache owner')
        torch.npu.synchronize()
        graphs = getattr(self.model, 'dspark_graphs', None)
        if graphs is not None:
            for frame in graphs.frames.values():
                frame.graph = None
            graphs.frames.clear()
        for name in ('_decode_graph', '_verify_graph'):
            frame = getattr(self.model, name, None)
            if frame is not None:
                frame.graph = None
            setattr(self.model, name, None)
        if self.model.dspark is not None:
            self.model.dspark.graphs.clear()
        self.model.dspark_graphs = None
        self._startup_graphs_ready = False

    @torch.inference_mode()
    def initialize_dspark_graphs(self, *, profile_decode=False):
        """Capture device graphs before declaring the resident engine ready.

        Initialization uses throwaway cache coordinates and mask tokens, not
        requests or saved predictions. Actual requests reset positions/history
        and write their own causal keys. Scores never include this startup.
        """
        self._startup_graphs_ready=False
        if self.model.dspark is None:return
        position=2*(min(32768,self.model.max_tokens-12)//2)
        if position<2*self.model.cfg['index_topk']:return
        noise=self.model.cfg['dspark_noise_token_id']
        self.model.mc2_global_bs=((self.slots*6+self.tp_size-1)//self.tp_size)*dist.get_world_size()
        self.model.next_positions[:4]=[position]*4
        variable=os.environ.get('DEEPINFER_ENABLE_MTP_VARLEN')=='1'
        if variable:
            from .dspark_graphs910c import DSparkTargetGraphs
            self.model.dspark_graphs=DSparkTargetGraphs(self.model)
        self.model.verify_chunks([[noise]*6 for _ in range(4)])
        self.model.dspark.propose([noise]*4,self.model.next_positions[:4],list(range(4)))
        if variable:
            self.model.dspark_graphs.capture_initial([noise]*4,[[noise]*5 for _ in range(4)])
            if profile_decode:
                curve=self.model.dspark_graphs.profile_startup()
                self.engine.progress({'stage':'startup_decode_profile_complete',
                    'curve':curve,'profile':self.model.dspark_graphs.startup_profile})
        torch.npu.synchronize()
        self.model.reset()
        self._startup_graphs_ready=True
        self.engine.progress({'stage':'dspark_graph_initialization_complete',
            'target_graph_captures':self.model._verify_graph.captures,
            'draft_graphs':len(self.model.dspark.graphs)})

    def _global_busy(self,value):
        tensor=torch.tensor([int(value)],dtype=torch.int32,device=self.model.device)
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def _global_zos_vote(self,busy,eligible,failed):
        # One collective replaces the existing _global_busy call in fast mode.
        # MAX implements any-busy, any-ineligible, and any-failed.
        if os.environ.get('DEEPINFER_910C_ZOS_VOTE_REUSE')=='1':
            buffers=getattr(self,'_zos_vote_reuse_buffers',None)
            if buffers is None:
                buffers=(torch.empty((3,),dtype=torch.int32,device='cpu',pin_memory=True),
                         torch.empty((3,),dtype=torch.int32,device=self.model.device))
                self._zos_vote_reuse_buffers=buffers
            host,tensor=buffers
            host[0]=int(busy);host[1]=int(not eligible);host[2]=int(failed)
            tensor.copy_(host,non_blocking=True)
            self.zos_vote_reuse_calls+=1
        else:
            tensor=torch.tensor([int(busy),int(not eligible),int(failed)],dtype=torch.int32,device=self.model.device)
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
        any_busy,any_ineligible,any_failed=tensor.cpu().tolist()
        return bool(any_busy),not bool(any_ineligible),bool(any_failed)

    @torch.inference_mode()
    def run(self,job,directory):
        if os.environ.get('DEEPINFER_910C_SCHEDULED_ROUTE') == '1':
            from .official_scheduled_serving910c import try_run
            if try_run(self, job, directory):
                return
        directory=Path(directory)
        self.zos_vote_reuse_calls=0
        requests=job['requests']
        owned=[(i,r) for i,r in enumerate(requests) if i%self.dp_size==self.dp_rank]
        # A single-request warmup also builds the other DP replica's cache.
        # Such internal work is excluded from API request/token/hit counters.
        internal=not owned
        if internal:
            owned=[(0,requests[0])]
        if len(owned)>self.slots:
            raise ValueError('batch exceeds native per-DP slot capacity')
        timing_on=(self.tp_rank==0 and (job.get('single_point_ablation') or {}).get('prefill_timing_split',False))
        cache_event_on=(self.tp_rank==0 and (job.get('single_point_ablation') or {}).get('cache_restore_event',False))
        cache_events=[] if cache_event_on else None
        reset_begin=time.monotonic() if timing_on else None
        self.model.reset()
        cache_timing={'reset_host_seconds':time.monotonic()-reset_begin,'slots':[],
            'first_token_host_seconds':None} if timing_on else None
        event_file=(directory/f'events-dp{self.dp_rank}.jsonl').open('a',buffering=1) if self.tp_rank==0 else None
        start=time.monotonic()
        official_requested=os.environ.get('DEEPINFER_910C_YOCO_OFFICIAL_PREFILL')=='1'
        def emit(event):
            if timing_on and cache_timing['first_token_host_seconds'] is None and event.get('type')=='token':
                cache_timing['first_token_host_seconds']=time.monotonic()-start
            if event_file is not None and not internal:
                event_file.write(json.dumps(event,separators=(',',':'))+'\n')
        try:
            if official_requested:
                from .official_yoco_metadata910c import OfficialYocoMetadata, configure_consumer
                configure_consumer(self.model)
                if self.official_yoco_metadata is None:
                    self.official_yoco_metadata=OfficialYocoMetadata(cp_size=self.tp_size)
                self.official_yoco_metadata.begin_job([r for _,r in owned])
            hits=[]
            for slot,(index,request) in enumerate(owned):
                ids=request['token_ids']
                if not ids or len(ids)+request['max_tokens']>self.model.max_tokens:
                    raise ValueError('request exceeds native context capacity')
                lookup_begin=time.monotonic() if timing_on else None
                snapshot=None
                for point in reversed(self.prefix_points):
                    if len(ids)>point:
                        snapshot=self.prefix_cache.get(tuple(ids[:point]))
                        if snapshot is not None:
                            break
                lookup_end=time.monotonic() if timing_on else None
                if snapshot is not None:
                    if cache_event_on:
                        event_begin=torch.npu.Event(enable_timing=True)
                        event_end=torch.npu.Event(enable_timing=True)
                        event_begin.record()
                    self.model.restore_prefix(snapshot,slot)
                    if cache_event_on:
                        event_end.record()
                        cache_events.append((slot,event_begin,event_end))
                if timing_on:
                    cache_timing['slots'].append({'slot':slot,'hit':snapshot is not None,
                        'lookup_host_seconds':lookup_end-lookup_begin,
                        'restore_host_seconds':time.monotonic()-lookup_end})
                hits.append(self.model.next_positions[slot])
                emit({'type':'start','index':index,'prompt_tokens':len(ids),'cached_tokens':hits[-1]})
            if timing_on: cache_timing['post_restore_host_seconds']=time.monotonic()-start
            last_logits={}
            prefill_slots=max(1,(len(requests)+self.dp_size-1)//self.dp_size)
            self.model.mc2_global_bs=((self.model.chunk_size*prefill_slots+self.tp_size-1)//self.tp_size)*dist.get_world_size()
            while self._global_busy(any(self.model.next_positions[s]<len(r['token_ids'])
                                         for s,(_,r) in enumerate(owned))):
                slots,chunks=[],[]
                for slot,(_,request) in enumerate(owned):
                    offset=self.model.next_positions[slot]
                    end=min(offset+self.model.chunk_size,len(request['token_ids']))
                    if not hits[slot]:
                        for point in self.prefix_points:
                            if offset<point<end:
                                end=point;break
                    if offset<end:
                        slots.append(slot);chunks.append(request['token_ids'][offset:end])
                if not slots:
                    self._reset_scratch()
                    slots,chunks=[self.slots],[[requests[0]['token_ids'][0]]]
                from .official_yoco_metadata910c import prepare_native_prefill
                bounded_keep,official_metadata=prepare_native_prefill(self,owned,slots,chunks,hits,internal=internal)
                logits=self.model.forward_tokens(chunks,slots,
                    sequence_parallel=self.model.sequence_parallel,bounded_keep=bounded_keep,official_yoco=official_metadata)
                for row,slot in enumerate(slots):
                    if slot==self.slots:
                        continue
                    ids=owned[slot][1]['token_ids']
                    # A restored parent prefix must not be displaced by
                    # request-specific children during C8 warmup. Preserve
                    # the cold seed anchors as the cache admission policy.
                    if not hits[slot] and self.model.next_positions[slot] in self.prefix_points:
                        key=tuple(ids[:self.model.next_positions[slot]])
                        if key not in self.prefix_cache:
                            # Bound cached prefix storage to two real anchors.
                            if len(self.prefix_cache)>=4:
                                del self.prefix_cache[next(iter(self.prefix_cache))]
                            self.prefix_cache[key]=self.model.capture_prefix(slot,key)
                    if self.model.next_positions[slot]==len(ids):
                        last_logits[slot]=logits[row:row+1]
                if self.tp_rank==0:
                    self.engine.progress({'stage':'serving_prefill','dp_rank':self.dp_rank,
                        'positions':self.model.next_positions[:len(owned)]})
            if timing_on: cache_timing['post_prefill_host_seconds']=time.monotonic()-start
            active=list(range(len(owned)))
            logits=torch.cat([last_logits[s] for s in active])
            self.model.mc2_global_bs=((self.slots+self.tp_size-1)//self.tp_size)*dist.get_world_size()
            uniform_graph=(len(requests)==self.slots*self.dp_size and
                len({r['max_tokens'] for r in requests})==1 and
                all(r.get('ignore_eos',False) and len(r['token_ids'])>=32768 for r in requests))
            self.model.use_decode_graph=uniform_graph
            generated=[[] for _ in owned]
            maximum=max(r['max_tokens'] for r in requests)
            # CANN graph verification needs at least index_topk complete C2
            # keys. This is a backend bound, independent of benchmark length.
            dspark_batch=(len(requests)==self.slots*self.dp_size and
                all(len(r['token_ids'])>=2*self.model.cfg['index_topk'] and
                    len(r['token_ids'])+r['max_tokens']+6<=self.model.max_tokens
                    for r in requests))
            if dspark_batch and self.model.dspark is not None:
                from .dspark910c import generate
                generated=generate(self,owned,logits,hits,emit,start)
                maximum=0
            if (uniform_graph and self.lookup_tokens and
                    all(len(r['token_ids'])+r['max_tokens']+self.lookup_tokens+1<=self.model.max_tokens for r in requests)):
                from .prompt_lookup910c import generate
                generated=generate(self,owned,logits,hits,emit,start,self.lookup_tokens)
                maximum=0
            for step in range(maximum):
                if active:
                    tokens=logits.argmax(-1).long() if self.tp_rank==0 else torch.empty(len(active),dtype=torch.int64,device=self.model.device)
                    dist.broadcast(tokens,self.root,group=self.tp_group)
                    values=tokens.cpu().tolist()
                else:
                    values=[]
                next_slots,next_chunks=[],[]
                for slot,value in zip(active,values):
                    index,request=owned[slot]
                    generated[slot].append(value)
                    finished=(len(generated[slot])>=request['max_tokens'] or
                        (not request.get('ignore_eos',False) and value==job['eos_token_id']))
                    emit({'type':'token','index':index,'token_id':value,'step':step,
                          'elapsed':time.monotonic()-start})
                    if finished:
                        emit({'type':'done','index':index,'prompt_tokens':len(request['token_ids']),
                              'cached_tokens':hits[slot],'output_tokens':len(generated[slot]),
                              'token_ids':generated[slot],
                              'finish_reason':finish_reason(value, len(generated[slot]), request['max_tokens'],
                                  ignore_eos=request.get('ignore_eos', False), eos_token_id=job['eos_token_id'])})
                    else:
                        next_slots.append(slot);next_chunks.append([value])
                if step+1==maximum:
                    break
                if not all(r.get('ignore_eos',False) for r in requests) and not self._global_busy(bool(next_slots)):
                    break
                # The number of global EP forwards is fixed for the submitted
                # batch even when individual requests reach EOS early.
                if not next_slots:
                    self._reset_scratch()
                    next_slots,next_chunks=[self.slots],[[requests[0]['token_ids'][0]]]
                    logits=self.model.forward_tokens(next_chunks,next_slots)
                    active=[]
                else:
                    logits=self.model.forward_tokens(next_chunks,next_slots)
                    active=next_slots
                if self.tp_rank==0 and (step+1)%32==0:
                    self.engine.progress({'stage':'serving_decode','dp_rank':self.dp_rank,'generated_steps':step+1})
            torch.npu.synchronize()
            if self.tp_rank==0:
                result={'state':'PASS','dp_rank':self.dp_rank,'internal_replica_work':internal,
                    'request_indices':[] if internal else [i for i,_ in owned],
                    'cached_tokens':[] if internal else hits,
                    'generated_token_ids':[] if internal else generated,
                    'seconds':time.monotonic()-start,'runtime_stats':dict(self.model.runtime_stats(),zos_official_scheduler_used=False),
                    'peak_npu_allocated_bytes':torch.npu.max_memory_allocated()}
                if timing_on: result['cache_restore_timing']=cache_timing
                if cache_event_on:
                    if not all(begin.query() and end.query() for _,begin,end in cache_events):
                        raise RuntimeError('cache timing event incomplete after existing device synchronize')
                    result['cache_restore_device_events']=[{'slot':slot,'current_stream_interval_ms':begin.elapsed_time(end)}
                        for slot,begin,end in cache_events]
                (directory/f'result-dp{self.dp_rank}.json').write_text(json.dumps(result,indent=2))
        finally:
            if official_requested and self.official_yoco_metadata is not None:
                self.official_yoco_metadata.end_job()
            if event_file is not None:
                event_file.close()
        dist.barrier()

    def _reset_scratch(self):
        slot=self.slots
        self.model.next_positions[slot]=0
        self.model.yoco_swa_valid_start[slot]=0
        for pending in self.model.pending.values():
            pending.pop(slot,None)
        if self.model.histories:
            self.model.histories[slot].reset()
