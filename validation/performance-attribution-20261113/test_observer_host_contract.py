"""Host metadata/delegation checks only; no model arithmetic or device code."""
from types import SimpleNamespace as NS
import unittest
from execution_route_observer import ExecutionRouteObserver,describe_batch,host_int


def fixture():
    context=NS(cudagraph_runtime_mode='FULL',batch_descriptor=type('Batch',(),{})())
    context.batch_descriptor.uniform=True
    marker=object();calls=[]
    class Graph:
        runtime_mode='FULL'
        def __init__(self):self.concrete_aclgraph_entries={context.batch_descriptor:NS(aclgraph=object())}
        def __call__(self,*args,**kwargs):calls.append((args,kwargs));return marker
    runner=NS(input_batch=NS(req_ids=['a']),requests={'a':NS(num_computed_tokens=1024,prompt_token_ids=[0]*1024)},
              model=Graph(),drafter=NS(_runnable=Graph()))
    runner._model_forward=runner.model
    descriptor=NS(num_scheduled_tokens={'a':6},total_num_scheduled_tokens=6,scheduled_spec_decode_tokens={'a':[1]*5})
    runtime=NS(compute_worker=NS(model_runner=runner),native_prefill_state=NS(descriptor=descriptor))
    return runtime,context,Graph,marker,calls


class Contracts(unittest.TestCase):
    def test_delegates_unchanged_and_restores_graph_owner_before_sdk_close(self):
        runtime,context,Graph,marker,calls=fixture();r=runtime.compute_worker.model_runner
        original_forward=r._model_forward;original_graph=Graph.__call__
        obs=ExecutionRouteObserver(runtime,lambda:context,Graph,cohort_enabled=True)
        arg=object();assert r._model_forward(arg,named=arg) is marker
        self.assertIs(calls[0][0][0],arg);self.assertIs(calls[0][1]['named'],arg)
        assert r.drafter._runnable(arg) is marker
        q=obs.snapshot()['counts'];self.assertEqual(q['target_actual_replays'],1);self.assertEqual(q['draft_actual_replays'],1)
        obs.close();self.assertIs(r._model_forward,original_forward);self.assertIs(Graph.__call__,original_graph)
        obs.close();self.assertTrue(obs.snapshot()['closed'])

    def test_rejects_non_host_scalar_without_conversion(self):
        class DeviceLike:
            def __int__(self):raise AssertionError('must not read device scalar')
        with self.assertRaises(TypeError):host_int(DeviceLike())
        runtime,context,Graph,marker,calls=fixture()
        runtime.compute_worker.model_runner.requests['a'].num_computed_tokens=DeviceLike()
        obs=ExecutionRouteObserver(runtime,lambda:context,Graph,cohort_enabled=False)
        try:
            self.assertIs(runtime.compute_worker.model_runner._model_forward(),marker)
            self.assertEqual(obs.snapshot()['counts']['metadata_errors'],1)
        finally:obs.close()

    def test_mixed_batch_records_host_spans(self):
        runtime,context,Graph,_,_=fixture();r=runtime.compute_worker.model_runner;d=runtime.native_prefill_state.descriptor
        r.input_batch.req_ids=['a','b'];r.requests['a'].num_computed_tokens=640
        r.requests['b']=NS(num_computed_tokens=1024,prompt_token_ids=[0]*1024)
        d.num_scheduled_tokens={'a':384,'b':6};d.total_num_scheduled_tokens=390;d.scheduled_spec_decode_tokens={'b':[1]*5}
        context.cudagraph_runtime_mode='NONE';context.batch_descriptor.uniform=False
        q=describe_batch(runtime,context)
        self.assertEqual((q['kind'],q['prefill_requests'],q['decode_requests']),('mixed_prefill_decode',1,1))
        self.assertEqual(q['spans'],[[640,1024,1024],[1024,1030,1024]])

    def test_failed_target_is_not_swallowed_and_depth_is_restored(self):
        runtime,context,Graph,_,_=fixture();r=runtime.compute_worker.model_runner
        def failed(*args,**kwargs):raise RuntimeError('original target failure')
        r._model_forward=failed
        obs=ExecutionRouteObserver(runtime,lambda:context,Graph,cohort_enabled=True)
        try:
            with self.assertRaisesRegex(RuntimeError,'original target failure'):r._model_forward()
            self.assertEqual(obs.local.target_depth,0)
            self.assertEqual(obs.snapshot()['counts']['target_failures'],1)
        finally:obs.close()

    def test_representative_events_bounded_counts_complete(self):
        runtime,context,Graph,_,_=fixture();r=runtime.compute_worker.model_runner;d=runtime.native_prefill_state.descriptor
        obs=ExecutionRouteObserver(runtime,lambda:context,Graph,cohort_enabled=True,event_limit=2)
        try:
            for q in (6,7,8,9,10):
                d.num_scheduled_tokens['a']=q;d.total_num_scheduled_tokens=q;r._model_forward()
            snap=obs.snapshot();self.assertEqual(snap['counts']['target_invocations'],5)
            self.assertEqual(len(snap['target_events']),2)
            self.assertEqual(snap['counts']['target_events_dropped'],3)
            self.assertEqual(sum(snap['histogram'].values()),5)
        finally:obs.close()


if __name__=='__main__':unittest.main()
