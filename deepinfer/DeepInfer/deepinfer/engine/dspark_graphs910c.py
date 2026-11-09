# SPDX-License-Identifier: Apache-2.0
"""Native DSpark target-graph pool and measured confidence-scheduler curve."""
import math
import torch
import torch.distributed as dist

from deepinfer.model_executor.models.w8a8_graph910c import NativeDecodeGraph
from deepinfer.model_executor.models.dspark_varlen910c import DSparkVarlenGraph


class DSparkTargetGraphs:
    def __init__(self,model):
        self.model=model
        # Keep sparse-MLA query spans positive in this pinned backend.
        # Reserve one real scratch query in padded graphs; curve keys are
        # supported REAL-query budget7, physical shape8.
        # The all-real six-per-request graph supports budget24 directly.
        # Reject the numerically inconsistent16-query graph (v13) and
        # dominated23-query graph (v11). Keep verified real budgets4/7/24.
        self.frames={4:NativeDecodeGraph(model,list(range(4))),
                     7:DSparkVarlenGraph(model,8)}
        self.full=NativeDecodeGraph(model,list(range(4)),query_tokens=6)
        model._decode_graph=self.frames[4]
        model._verify_graph=self.full
        self.curve=None
        self.startup_profile=None
        self.prefix_validation=None
        self.last_frame=None
        self.last_graph_size=None

    def reset_sequence(self):
        for frame in self.frames.values():frame.reset_sequence()
        self.full.reset_sequence()

    def select(self,lengths):
        total=sum(lengths)
        size=4 if total==4 else 7 if total<=7 else 24
        return size,self.full if size==24 else self.frames[size]

    def run(self,chunks,full_chunks):
        size,frame=self.select(list(map(len,chunks)))
        self.last_frame,self.last_graph_size=frame,size
        if frame is self.full:
            lengths=list(map(len,chunks))
            # Compute actual five-token proposals even outside the admitted
            # prefix, retaining exactly the full target's numerical shape.
            values=frame.step(full_chunks)
            if lengths==[6]*4:return values
            # Extra future positions are causal padding and never sampled.
            return torch.cat([values[slot*6:slot*6+length] for slot,length in enumerate(lengths)])
        return frame.step(chunks)

    def split_predictions(self,values,lengths):
        result=[];offset=0
        for length in lengths:
            result.append(values[offset:offset+length]);offset+=length
        return result

    def capture_initial(self,golden,proposals):
        """Capture every selectable EP graph in the same startup order.

        Request warmups then run after all graph-private memory is allocated.
        Curve timing and real-context prefix validation are separate gates.
        """
        bases=self.model.next_positions[:4]
        chunks=[[g,*p] for g,p in zip(golden,proposals)]
        cases=[(self.frames[4],[1]*4),(self.frames[7],[2,2,2,1]),
               (self.full,[6]*4)]
        for frame,lengths in cases:
            frame.step([row[:n] for row,n in zip(chunks,lengths)])
            frame.commit_prefixes(bases,[0]*4)
            dist.barrier()
        self.reset_sequence()

    def profile_startup(self):
        """Measure captured graphs before READY using the original timing procedure.

        Routing uses this backend's captured inputs. The original random-gate
        profiling input is not exposed here; record that remaining difference.
        Actual-context prefix validation still runs before the first decode.
        """
        if self.startup_profile is not None:
            return self.curve
        cases=[(24,self.full),(7,self.frames[7]),(4,self.frames[4])]
        if any(frame.graph is None for _,frame in cases):
            raise RuntimeError('startup profiling requires all captured target graphs')
        marker=torch.zeros(1,dtype=torch.float32,device=self.model.device)
        results=[]
        observations=[]
        for size,frame in cases:
            frame.graph.replay()
            dist.all_reduce(marker)
            events=[]
            for _ in range(5):
                begin=torch.npu.Event(enable_timing=True)
                end=torch.npu.Event(enable_timing=True)
                begin.record()
                frame.graph.replay()
                end.record()
                dist.all_reduce(marker)
                events.append((begin,end))
            times=[]
            for begin,end in events:
                begin.synchronize()
                end.synchronize()
                times.append(begin.elapsed_time(end))
            invalid=any(not math.isfinite(ms) or ms<=0 for ms in times)
            failure=torch.tensor([int(invalid)],dtype=torch.int32,device=self.model.device)
            dist.all_reduce(failure)
            if failure.item():
                raise RuntimeError('invalid startup graph event measurement')
            ordered=sorted(times)
            average=sum(ordered[1:-1])/3
            value=torch.full((1,),average,dtype=torch.float32,device=self.model.device)
            dist.all_reduce(value)
            results.append((size,value.item()/dist.get_world_size()))
            observations.append(dict(budget=size,physical_tokens=frame.tokens,local_event_ms=times))
        if any(not math.isfinite(ms) or ms<=0 for _,ms in results):
            raise RuntimeError('invalid measured startup decode curve')
        self.reset_sequence()
        self.curve=sorted(results)
        self.startup_profile=dict(source='original_profile_decode_perf_timing_procedure',
            warmups_per_graph=1,measurements_per_graph=5,trim_each_end=1,
            ep_size=dist.get_world_size(),input_origin='captured_native_startup_storage',
            gate_force_random_available=False,real_request_count=0,observations=observations)
        return self.curve

    def profile(self,golden,proposals):
        # A startup curve must never bypass the original real-context check.
        if self.prefix_validation is not None:return self.curve
        measure_curve=self.curve is None
        model=self.model
        bases=model.next_positions[:4]
        chunks=[[g,*p] for g,p in zip(golden,proposals)]
        # All ranks capture the same graph pool in the same order before
        # adaptive scheduling. Asymmetric lazy capture would break global EP
        # ordering when the two DP replicas choose different graph sizes.
        cases=[(4,self.frames[4],[1]*4),(7,self.frames[7],[2,2,2,1]),
               (24,self.full,[6]*4)]
        measured_curve={}
        prefix_predictions=[]
        for size,frame,lengths in cases:
            frame.forward_event_timing=False
            sample=[row[:n] for row,n in zip(chunks,lengths)]
            logits=frame.step(sample)
            from deepinfer.npu_ops.greedy910c import global_argmax
            predicted=global_argmax(logits[:sum(lengths)],model.tp_group,model.cfg['vocab_size']).cpu().tolist()
            offset=0;rows=[]
            for length in lengths:
                rows.append(predicted[offset:offset+length]);offset+=length
            prefix_predictions.append((size,lengths,rows))
            frame.commit_prefixes(bases,[0]*4)
            dist.barrier()
            if measure_curve:
                frame.graph.replay()
                dist.barrier()
                events=[]
                for _ in range(5):
                    start=torch.npu.Event(enable_timing=True)
                    end=torch.npu.Event(enable_timing=True)
                    start.record();frame.graph.replay();end.record()
                    events.append((start,end))
                    dist.barrier()
                torch.npu.synchronize()
                measured=sorted(start.elapsed_time(end) for start,end in events)
                average=sum(measured[1:-1])/3
                value=torch.tensor([average],dtype=torch.float32,device=model.device)
                dist.all_reduce(value)
                measured_curve[size]=max(measured_curve.get(size,0.),value.item()/dist.get_world_size())
            frame.forward_event_timing=True
        # These are the same real-context forwards already needed for the
        # startup cost curve. Check all shorter/nonuniform queries against
        # the full causal target; no extra API/model request is added.
        reference=prefix_predictions[-1][2]
        checked=0
        for size,lengths,rows in prefix_predictions[:-1]:
            for slot,(length,row) in enumerate(zip(lengths,rows)):
                if row!=reference[slot][:length]:
                    raise ValueError(f'variable DSpark target prefix mismatch: budget={size}, slot={slot}, actual={row}, full={reference[slot][:length]}')
                checked+=length
        validation={'state':'PASS','real_context_query_tokens_compared':checked,
            'budgets':[size for size,_,_ in prefix_predictions],
            'query_lengths':[lengths for _,lengths,_ in prefix_predictions],
            'reference':'same-context full causal target graph; no saved predictions',
            'additional_model_forwards':0}
        self.reset_sequence()
        if measure_curve:self.curve=sorted(measured_curve.items())
        self.prefix_validation=validation
        return self.curve
