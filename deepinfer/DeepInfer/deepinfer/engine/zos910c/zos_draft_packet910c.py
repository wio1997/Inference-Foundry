"""Retained draft D2H packet for the native fixed24 local ZOS adapter.

Queue the producer's host copies before target submission, then wait only
that event after submission. This does not defer the previous target's CPU
commit and must not be described as complete upstream ZOS.
"""
import time
import torch


class DraftHostPacket:
    def __init__(self,device,proposal_dtype,confidence_dtype):
        if proposal_dtype not in (torch.int32,torch.int64) or confidence_dtype!=torch.float32:
            raise ValueError('unsupported native draft packet dtype')
        self.device=device
        self.proposal_dtype=proposal_dtype
        self.proposals=torch.empty((4,5),dtype=proposal_dtype,device='cpu',pin_memory=True)
        self.confidence=torch.empty((4,5),dtype=confidence_dtype,device='cpu',pin_memory=True)
        self.event=torch.npu.Event()
        self.pending=False
        self.retained=None
        self.stream=None

    def queue(self,proposals,confidence):
        if self.pending:raise RuntimeError('previous draft packet is still pending')
        for value,dtype in ((proposals,self.proposal_dtype),(confidence,torch.float32)):
            if (value.device!=self.device or value.dtype!=dtype or
                    tuple(value.shape)!=(4,5) or not value.is_contiguous()):
                raise ValueError('invalid fixed24 draft packet')
        self.stream=torch.npu.current_stream()
        self.retained=(proposals,confidence)
        self.proposals.copy_(proposals,non_blocking=True)
        self.confidence.copy_(confidence,non_blocking=True)
        self.event.record()
        self.pending=True
        return self

    def wait(self):
        if not self.pending:raise RuntimeError('draft packet was already consumed')
        if torch.npu.current_stream()!=self.stream:
            raise RuntimeError('draft packet changed its owner stream')
        self.event.synchronize()
        result=(self.proposals.tolist(),self.confidence.tolist())
        self.pending=False
        self.retained=None
        return result


def submit_before_draft_host(pool,state,packet,proposals,confidence,*,collectives_authorized=False):
    """Submit an agreed full target before parsing its held draft host data.

    Previous CPU positions/history/C2 must already be committed. The caller
    owns global eligibility, original sampling, policy lag and prefix commit.
    Native hash/UVA preparation contains TP collectives; no pre-vote call is
    permitted. A submit exception is fatal to the job, never a silent retry
    after cache writes.
    """
    if collectives_authorized is not True:
        raise RuntimeError('prepared target needs an agreed collective plan')
    if state.pending or state.window_pending:
        raise RuntimeError('previous device state or window is not settled')
    frame=pool.full
    if frame.graph is None or tuple(frame.slots)!=(0,1,2,3) or frame.query_tokens!=6:
        raise ValueError('prepared target requires the captured full graph')
    packet.queue(proposals,confidence)
    started=time.perf_counter()
    # prepare_engram_window fills state.input from the same held producer.
    window=state.prepare_engram_window(proposals)
    ticket=frame.prepare_target_inputs(state.input,window,
        held=(packet,proposals,confidence,state),collectives_authorized=True)
    pool.last_frame,pool.last_graph_size=frame,24
    logits=frame.replay_prepared_target(ticket)
    submitted_seconds=time.perf_counter()-started
    proposal_values,confidence_values=packet.wait()
    # CPU golden comes from the already committed host output, not a new D2H.
    # The caller publishes chunks built with that golden and this exact packet.
    return logits,ticket,proposal_values,confidence_values,submitted_seconds
