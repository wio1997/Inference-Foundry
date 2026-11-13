"""Bounded host metadata observation for the fixed cohort causal experiment.

No tensor reads, device events, synchronizations, requests, or model arithmetic.
Runtime imports are deferred so host ownership contracts can be checked alone.
"""
from collections import Counter
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from types import MethodType
import json
import hashlib
import threading
import time


def clock_identity():
    """Read host clock provenance once; never initialize any device runtime."""
    boot=Path('/proc/sys/kernel/random/boot_id').read_bytes()
    offsets=Path('/proc/self/timens_offsets').read_bytes()
    monotonic=time.get_clock_info('monotonic');perf=time.get_clock_info('perf_counter')
    return {'boot_id_sha256':hashlib.sha256(boot).hexdigest(),
            'timens_offsets_sha256':hashlib.sha256(offsets).hexdigest(),
            'monotonic_implementation':monotonic.implementation,
            'perf_counter_implementation':perf.implementation,
            'monotonic_resolution_seconds':monotonic.resolution,
            'recorded_monotonic_ns':time.monotonic_ns()}


def host_int(value):
    # Never int(Tensor), item(), or bool(Tensor): those could read the device.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError('expected a host integer')
    return int(value)


@lru_cache(maxsize=64)
def request_identity(rid):
    if not isinstance(rid,str) or not rid or len(rid)>256:
        raise TypeError('expected a bounded host request ID')
    return hashlib.sha256(rid.encode()).hexdigest()


def request_identities(runtime):
    ids=runtime.compute_worker.model_runner.input_batch.req_ids
    if not ids or len(ids)>8:
        raise ValueError('expected the fixed experiment batch of 1..8 requests')
    result=[request_identity(rid) for rid in ids]
    if len(set(result))!=len(result):raise ValueError('duplicate request identity')
    return result


def describe_batch(runtime, context):
    runner=runtime.compute_worker.model_runner
    descriptor=runtime.native_prefill_state.descriptor
    if descriptor is None:
        raise ValueError('target invocation has no owned schedule descriptor')
    spans=[];prefill=0;decode=0
    for rid in runner.input_batch.req_ids:
        state=runner.requests[rid]
        start=host_int(state.num_computed_tokens)
        q=host_int(descriptor.num_scheduled_tokens[rid])
        prompt=len(state.prompt_token_ids)
        if q<=0 or start<0:
            raise ValueError('invalid host schedule range')
        is_prefill=start<prompt
        prefill+=int(is_prefill);decode+=int(not is_prefill)
        spans.append([start,start+q,prompt])
    if sum(stop-start for start,stop,_ in spans)!=host_int(descriptor.total_num_scheduled_tokens):
        raise ValueError('request rows disagree with schedule total')
    mode=context.cudagraph_runtime_mode
    mode=mode.name if hasattr(mode,'name') else mode
    if not isinstance(mode,str):
        raise TypeError('graph mode must be a host enum/string')
    spec_reqs=len(descriptor.scheduled_spec_decode_tokens)
    kind=('mixed_prefill_decode' if prefill and (decode or spec_reqs)
          else 'pure_prefill' if prefill else 'decode')
    batch=getattr(context,'batch_descriptor',None)
    uniform=getattr(batch,'uniform',None)
    if uniform is not None and not isinstance(uniform,bool):
        raise TypeError('uniform must be a host boolean')
    return {'mode':mode,'kind':kind,'requests':len(spans),
            'scheduled_tokens':host_int(descriptor.total_num_scheduled_tokens),
            'prefill_requests':prefill,'decode_requests':decode,'spec_requests':spec_reqs,
            'uniform':uniform,'spans':spans,'request_id_sha256':request_identities(runtime)}


class ExecutionRouteObserver:
    def __init__(self,runtime,get_context,graph_class,*,cohort_enabled,event_limit=2048):
        self.runtime=runtime;self.runner=runtime.compute_worker.model_runner
        self.get_context=get_context;self.graph_class=graph_class
        self.clock=clock_identity()
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.cohort_enabled=bool(cohort_enabled)
        self.counts=Counter();self.histogram=Counter();self.events=[];self.graph_events=[];self.errors=[]
        if type(event_limit) is not int or not 1<=event_limit<=4096:
            raise ValueError('event limit must be a bounded positive host integer')
        self.event_limit=event_limit
        self.lock=threading.Lock();self.local=threading.local();self.closed=False
        self.previous_forward=self.runner._model_forward
        self.previous_graph=graph_class.__call__
        owner=self

        def forward(_runner,*args,**kwargs):
            sample=None
            try:sample=describe_batch(owner.runtime,owner.get_context())
            except (AttributeError,TypeError,ValueError,KeyError) as exc:
                with owner.lock:
                    owner.counts['metadata_errors']+=1
                    if len(owner.errors)<8:owner.errors.append(type(exc).__name__+': '+str(exc))
            before=time.monotonic_ns()
            depth=getattr(owner.local,'target_depth',0);owner.local.target_depth=depth+1
            try:
                result=owner.previous_forward(*args,**kwargs)
            except BaseException:
                with owner.lock:owner.counts['target_failures']+=1
                raise
            finally:
                owner.local.target_depth=depth
                returned=time.monotonic_ns();elapsed=returned-before
                with owner.lock:
                    owner.counts['target_invocations']+=1
                    owner.counts['target_host_elapsed_ns']+=elapsed
                    if sample is not None:
                        key=f"{sample['kind']}/{sample['mode']}/{sample['requests']}/{sample['scheduled_tokens']}"
                        owner.histogram[key]+=1
                        owner.counts['mode_'+sample['mode']]+=1
                        owner.counts['kind_'+sample['kind']]+=1
                        if len(owner.events)<owner.event_limit:
                            owner.events.append({'started_monotonic_ns':before,
                                'host_return_monotonic_ns':returned,**sample})
                        else:owner.counts['target_events_dropped']+=1
            return result

        def graph_call(wrapper,*args,**kwargs):
            draft=wrapper is getattr(getattr(owner.runner,'drafter',None),'_runnable',None)
            target=bool(getattr(owner.local,'target_depth',0))
            scope='draft' if draft else 'target' if target else None
            replay=False;request_ids=[]
            if scope:
                try:
                    request_ids=request_identities(owner.runtime)
                    context=owner.get_context()
                    entry=wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
                    replay=(context.cudagraph_runtime_mode==wrapper.runtime_mode
                            and entry is not None and entry.aclgraph is not None)
                except (AttributeError,TypeError,ValueError,KeyError) as exc:
                    with owner.lock:
                        owner.counts['graph_metadata_errors']+=1
                        if len(owner.errors)<8:owner.errors.append(type(exc).__name__+': '+str(exc))
            started=time.monotonic_ns() if scope else None
            result=owner.previous_graph(wrapper,*args,**kwargs)
            if scope:
                returned=time.monotonic_ns()
                with owner.lock:
                    owner.counts[scope+'_graph_wrapper_calls']+=1
                    owner.counts[scope+'_actual_replays' if replay else scope+'_non_replay_calls']+=1
                    if len(owner.graph_events)<owner.event_limit:
                        owner.graph_events.append({'scope':scope,'actual_replay':replay,
                            'request_id_sha256':request_ids,
                            'started_monotonic_ns':started,'host_return_monotonic_ns':returned})
                    else:owner.counts['graph_events_dropped']+=1
            return result

        self.installed_forward=MethodType(forward,self.runner)
        self.installed_graph=graph_call
        self.runner._model_forward=self.installed_forward
        self.graph_class.__call__=self.installed_graph

    def snapshot(self):
        with self.lock:
            return {'closed':self.closed,'cohort_enabled':self.cohort_enabled,
                    'clock':dict(self.clock),'observer_source_sha256':self.source_sha256,
                    'snapshot_monotonic_ns':time.monotonic_ns(),'counts':dict(self.counts),
                    'histogram':dict(self.histogram),'target_events':list(self.events),
                    'graph_events':list(self.graph_events),
                    'event_limit':self.event_limit,'errors':list(self.errors),
                    'scope':'host schedule/graph metadata;host elapsed includes waiting;not device kernel time'}

    def close(self):
        if self.closed:return
        if self.runner._model_forward is not self.installed_forward or self.graph_class.__call__ is not self.installed_graph:
            raise RuntimeError('execution observer ownership changed')
        self.runner._model_forward=self.previous_forward
        self.graph_class.__call__=self.previous_graph
        self.closed=True


def install_runtime_hooks(cohort_enabled):
    from deepinfer.model_executor.models.glm5_next.full_model import sdk_runtime,sdk_http
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    original_install=sdk_runtime.SDKComputeRuntime.install
    original_drain=sdk_runtime.SDKComputeRuntime.drain
    original_server=sdk_http.ThreadingHTTPServer
    holders=[]

    def install(runtime,worker):
        result=original_install(runtime,worker)
        if holders:raise RuntimeError('observer permits exactly one runtime per process')
        observer=ExecutionRouteObserver(runtime,get_forward_context,ACLGraphWrapper,
                                        cohort_enabled=cohort_enabled)
        runtime.execution_route_observer=observer;holders.append(observer)
        return result

    def drain(runtime):
        observer=getattr(runtime,'execution_route_observer',None)
        # Restore the shared MTP observer before its strict ownership close.
        if observer is not None:observer.close()
        try:return original_drain(runtime)
        finally:
            if observer is not None:
                path=Path(runtime.native_prefill_state_stats_path).with_name('execution-route-observer.json')
                temp=path.with_suffix('.tmp');temp.write_text(json.dumps(observer.snapshot())+'\n');temp.replace(path)

    def server(address,handler,*args,**kwargs):
        previous_reply=handler.reply
        def reply(instance,status,value):
            if instance.path=='/health' and status==200:
                if len(holders)!=1:raise RuntimeError('health has no unique execution observer')
                value={**value,'execution_route':holders[0].snapshot()}
            return previous_reply(instance,status,value)
        handler.reply=reply
        return original_server(address,handler,*args,**kwargs)

    sdk_runtime.SDKComputeRuntime.install=install
    sdk_runtime.SDKComputeRuntime.drain=drain
    sdk_http.ThreadingHTTPServer=server


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Record clock identity without importing an NPU runtime')
    parser.add_argument('--clock-output',type=Path,required=True)
    args=parser.parse_args()
    temp=args.clock_output.with_suffix('.tmp')
    temp.write_text(json.dumps(clock_identity())+'\n');temp.replace(args.clock_output)
