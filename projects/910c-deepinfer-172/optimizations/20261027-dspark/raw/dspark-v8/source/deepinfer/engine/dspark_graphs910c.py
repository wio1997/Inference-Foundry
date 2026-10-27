# SPDX-License-Identifier: Apache-2.0
"""Native DSpark target-graph pool and measured confidence-scheduler curve."""
import torch
import torch.distributed as dist

from deepinfer.model_executor.models.w8a8_graph910c import NativeDecodeGraph
from deepinfer.model_executor.models.dspark_varlen910c import DSparkVarlenGraph


class DSparkTargetGraphs:
    def __init__(self,model):
        self.model=model
        self.frames={4:NativeDecodeGraph(model,list(range(4))),
                     8:DSparkVarlenGraph(model,8),16:DSparkVarlenGraph(model,16),
                     24:DSparkVarlenGraph(model,24)}
        self.full=NativeDecodeGraph(model,list(range(4)),query_tokens=6)
        model._decode_graph=self.frames[4]
        model._verify_graph=self.full
        self.curve=None
        self.last_frame=None
        self.last_graph_size=None

    def reset_sequence(self):
        for frame in self.frames.values():frame.reset_sequence()
        self.full.reset_sequence()

    def select(self,lengths):
        total=sum(lengths)
        size=4 if total==4 else 8 if total<=8 else 16 if total<=16 else 24
        return size,self.full if total==24 else self.frames[size]

    def run(self,chunks):
        size,frame=self.select(list(map(len,chunks)))
        self.last_frame,self.last_graph_size=frame,size
        return frame.step(chunks)

    def split_predictions(self,values,lengths):
        result=[];offset=0
        for length in lengths:
            result.append(values[offset:offset+length]);offset+=length
        return result

    def profile(self,golden,proposals):
        if self.curve is not None:return self.curve
        model=self.model
        bases=model.next_positions[:4]
        chunks=[[g,*p] for g,p in zip(golden,proposals)]
        # All ranks capture the same graph pool in the same order before
        # adaptive scheduling. Asymmetric lazy capture would break global EP
        # ordering when the two DP replicas choose different graph sizes.
        cases=[(4,self.frames[4],[1]*4),(8,self.frames[8],[2]*4),
               (16,self.frames[16],[4]*4),(24,self.frames[24],[5]*4),
               (24,self.full,[6]*4)]
        measured_curve={}
        for size,frame,lengths in cases:
            frame.forward_event_timing=False
            sample=[row[:n] for row,n in zip(chunks,lengths)]
            frame.step(sample)
            frame.commit_prefixes(bases,[0]*4)
            dist.barrier()
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
        self.curve=sorted(measured_curve.items())
        self.reset_sequence()
        return self.curve
