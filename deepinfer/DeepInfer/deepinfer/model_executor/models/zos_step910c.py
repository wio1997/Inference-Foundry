"""Local graph interface for held device preparation and host publication.

No model forward or cache write occurs during prepare. The graph replay is
the unchanged parent implementation. Host chunks must come from the same
retained draft packet that produced token_input; no placeholder history is
allowed. One caller NPU stream owns all operations and buffer reuse.
"""
from dataclasses import dataclass
import os
import torch
from .zos_device_step910c import guard_legacy


def _current_stream():
    return torch.npu.current_stream()


@dataclass(eq=False)
class PreparedTargetInputs:
    owner: object
    generation: int
    slots: tuple
    bases: tuple
    histories: tuple
    history_lengths: tuple
    tokens: object
    window: object
    held: tuple
    stream: object
    input_buffer: object
    row_buffers: tuple
    replay_before: int
    submitted: bool = False
    published: bool = False
    released: bool = False
    discarded: bool = False


def _ticket(frame,ticket):
    guard_legacy(frame,"old_ticket")
    if (not isinstance(ticket,PreparedTargetInputs) or ticket.owner is not frame or
            getattr(frame,'_zos_ticket',None) is not ticket or ticket.released or ticket.discarded or
            ticket.generation!=frame._zos_generation or
            getattr(frame.model,'_zos_prepared_frame',None) is not frame):
        raise RuntimeError('invalid, stale or already consumed target ticket')
    if _current_stream()!=ticket.stream:
        raise RuntimeError('target ticket changed its owner stream')
    if tuple(frame.slots)!=ticket.slots or frame.inputs is not ticket.input_buffer:
        raise RuntimeError('target graph input ownership changed')
    if tuple((k,id(v)) for k,v in sorted(frame.rows.items()))!=ticket.row_buffers:
        raise RuntimeError('target graph Engram input ownership changed')
    expected=tuple(p+(frame.query_tokens if ticket.submitted else 0) for p in ticket.bases)
    if tuple(frame.model.next_positions[s] for s in ticket.slots)!=expected:
        raise RuntimeError('host positions changed while target ticket was held')
    for slot,history,length in zip(ticket.slots,ticket.histories,ticket.history_lengths):
        if frame.model.histories[slot] is not history:
            raise RuntimeError('request history owner changed while ticket was held')
        want=length+(frame.query_tokens if ticket.published else 0)
        if len(history.history)!=want:
            raise RuntimeError('request history changed while ticket was held')


def _release(frame,ticket):
    if not (ticket.submitted and ticket.published):return
    ticket.released=True
    ticket.tokens=None;ticket.window=None;ticket.held=();ticket.histories=()
    frame._zos_ticket=None
    frame.model._zos_prepared_frame=None


def prepare_target_inputs(frame,token_input,window,*,held=(),collectives_authorized=False):
    """Prepare fixed24 device inputs without publishing CPU history.

    NativeUvaEngram.rows may issue TP all-gather. The caller must have reached
    a consistent eligibility/collective plan before invoking this full stage.
    A future pre-vote overlap must split off the purely local work first.
    Initial adapter still consumes the CPU-committed bases/C2 state. This is
    deliberately not a claim of complete official ZOS: a later adapter must
    accept the previous sampler's device positions/C2 to defer its host commit.
    """
    guard_legacy(frame,"old_prepare")
    if collectives_authorized is not True:
        raise RuntimeError('full target preparation requires an agreed collective plan')
    if getattr(frame,'_zos_ticket',None) is not None or getattr(frame,'_zos_commit_required',False):
        raise RuntimeError('previous target ticket or commit is outstanding')
    if (getattr(frame.model,'_zos_prepared_frame',None) is not None or
            getattr(frame.model,'_zos_commit_frame',None) is not None):
        raise RuntimeError('another graph publication/commit still owns the model')
    if (frame.count!=4 or frame.query_tokens!=6 or frame.tokens!=24 or
            tuple(frame.slots)!=(0,1,2,3) or frame.graph is None):
        raise ValueError('prepared target requires an already captured fixed4x6 graph')
    model=frame.model
    if (os.environ.get('DEEPINFER_910C_ENGRAM_DEVICE_HASH')!='1' or
            os.environ.get('DEEPINFER_910C_ENGRAM_UVA')!='1' or
            model.engram_device_hash is None or model.engram_uva_runtime is None or not frame.rows):
        raise ValueError('prepared target requires the existing device hash/UVA interface')
    for value,shape in [(token_input,(4,6)),(window,(24,4))]:
        if value.device!=model.device or value.dtype!=torch.int32 or tuple(value.shape)!=shape or not value.is_contiguous():
            raise ValueError('invalid prepared target input shape/dtype/device')
    protected=[frame.inputs,*frame.rows.values(),*[v for pair in frame.even.values() for v in pair]]
    if frame.engram_hash_output is not None:protected.append(frame.engram_hash_output)
    for producer in (token_input,window):
        a=producer.data_ptr();b=a+producer.numel()*producer.element_size()
        for target in protected:
            c=target.data_ptr();d=c+target.numel()*target.element_size()
            if a<d and c<b:
                raise ValueError('prepared producer overlaps graph/private output storage')
    if any(getattr(frame,k) is not None for k in ('device_input_override','device_engram_override','device_engram_fast_override')):
        raise RuntimeError('prepared target conflicts with an existing graph override')
    bases=tuple(model.next_positions[s] for s in frame.slots)
    if any(type(p) is not int or p<0 or p+6>model.max_tokens for p in bases):
        raise ValueError('prepared target outside native context capacity')
    histories=tuple(model.histories[s] for s in frame.slots)
    lengths=tuple(len(h.history) for h in histories)
    if lengths!=bases:
        raise ValueError('prepared target needs fully committed matching host histories')
    if model.engram_device_hash.width!=4:
        raise ValueError('prepared target requires the existing width4 Engram contract')
    # All validations above precede writes. These writes only affect graph
    # inputs/private C2 preparation, never the authoritative model cache.
    stream=_current_stream()
    try:
        if not frame.prepared:frame._prepare_sequence()
        frame.cpu_inputs[24:].copy_(torch.tensor([p+6 for p in bases],dtype=torch.int32))
        frame.inputs[24:].copy_(frame.cpu_inputs[24:],non_blocking=True)
        frame.inputs[:24].copy_(token_input.reshape(-1))
        device_hash=model.engram_device_hash
        if frame.engram_hash_output is None:
            frame.engram_hash_output=torch.empty((len(device_hash.layers),24,device_hash.heads),dtype=torch.int32,device=model.device)
        hashes=device_hash.compute(window,frame.engram_hash_output)
        for layer,buffer in frame.rows.items():model.engram_uva_runtime.rows(layer,hashes[layer],buffer)
        if _current_stream()!=stream:
            raise RuntimeError('target preparation changed its owner stream')
    except BaseException:
        frame.prepared=False
        raise
    frame.engram_hash_calls+=1;frame.engram_hash_tokens+=24
    frame.device_input_override_applied+=1;frame.device_engram_fast_calls+=1
    frame._zos_generation+=1
    ticket=PreparedTargetInputs(frame,frame._zos_generation,tuple(frame.slots),bases,histories,lengths,
        token_input,window,tuple(held),stream,frame.inputs,
        tuple((k,id(v)) for k,v in sorted(frame.rows.items())),frame.replays)
    frame._zos_ticket=ticket
    model._zos_prepared_frame=frame
    return ticket


def replay_prepared_target(frame,ticket):
    _ticket(frame,ticket)
    if ticket.submitted:raise RuntimeError('target ticket already submitted')
    if frame.replays!=ticket.replay_before:
        raise RuntimeError('target graph was replayed outside its ticket')
    result=frame._execute_graph()
    ticket.submitted=True
    frame._zos_commit_required=True
    frame.model._zos_commit_frame=frame
    _ticket(frame,ticket)
    _release(frame,ticket)
    return result


def publish_target_host(frame,ticket,chunks):
    """Publish actual host tokens exactly once, before or after graph submit.

    The owner draft packet must already have completed its original D2H event.
    This method performs no device readback and cannot validate arbitrary CPU
    tokens against a GPU producer; that association belongs to the packet
    factory and is independently checked in component/model correctness runs.
    """
    _ticket(frame,ticket)
    if ticket.published:raise RuntimeError('target host history already published')
    if len(chunks)!=4 or any(len(row)!=6 for row in chunks):
        raise ValueError('actual host chunks must match fixed4x6 target')
    for history,chunk in zip(ticket.histories,chunks):
        if any(type(x) is not int or not 0<=x<len(history.token_map) for x in chunk):
            raise ValueError('actual host chunks contain invalid vocabulary IDs')
    # Validate all requests before the first history mutation, matching the
    # parent raw_contexts all-or-nothing validation rule.
    for history,chunk in zip(ticket.histories,chunks):history.history.extend(chunk)
    ticket.published=True
    _ticket(frame,ticket)
    _release(frame,ticket)


def discard_target_inputs(frame,ticket):
    _ticket(frame,ticket)
    if ticket.submitted or ticket.published:
        raise RuntimeError('submitted/published target cannot be discarded')
    # A rejected global vote must not leave private C2 state marked prepared
    # across another graph's commit. The next original path reloads it.
    frame.prepared=False
    ticket.discarded=True;ticket.released=True
    ticket.tokens=None;ticket.window=None;ticket.held=();ticket.histories=()
    frame._zos_ticket=None
    frame.model._zos_prepared_frame=None
