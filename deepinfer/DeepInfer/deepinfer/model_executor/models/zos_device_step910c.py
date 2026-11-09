"""Fixed-slot graph interface: device input first, previous host commit later.

Uses existing window/hash/UVA/state/C2 implementations. No target model math.
The owning controller still supplies an agreed EP/TP plan and CPU acceptance.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import torch

def _stream():return torch.npu.current_stream()

def guard_legacy(frame,action):
    owner=getattr(frame.model,'_zos_device_pipeline',None)
    if owner is None:return
    if action=='commit' and owner._committing is not None and owner._committing.frame is frame:
        return
    raise RuntimeError('device target pipeline owns model; drain and close before legacy '+action)

def guard_execute(frame,advance_host_positions):
    owner=getattr(frame.model,'_zos_device_pipeline',None)
    if owner is None:
        if advance_host_positions is not True:raise RuntimeError('host advance may be deferred only by a device lease')
        return
    ticket=owner._submitting
    if (advance_host_positions is not False or ticket is None or ticket.frame is not frame or
            ticket is not owner.queue[-1]):
        raise RuntimeError('device graph submission lacks its owner lease')

@dataclass(eq=False)
class LocalInputs:
    owner:object
    generation:int
    stream:object
    previous:object
    state:object
    view:object
    proposals:object
    tokens:object
    positions:object
    window:object
    c2:object
    host:object
    event:object
    used:object
    status:object
    frame:object
    layout:object=None
    graph_inputs:object=None
    consumed:bool=False

@dataclass(eq=False)
class DeviceTargetLease:
    owner:object
    generation:int
    step:int
    local:object
    histories:tuple
    replay_before:int
    frame:object
    submitted:bool=False
    sealed:bool=False
    committed:bool=False
    discarded:bool=False
    state:object=None
    c2_packet:object=None
    state_event:object=None
    state_host:object=None
    state_retained:object=None
    target_packet:object=None
    before_snapshot:object=None
    rows_snapshot:object=None
    host_values:object=None
    target_values:object=None

class DeviceTargetPipeline:
    def __init__(self,frame,*,debug_c2=False,small_frame=None,compact_frame=None):
        model=frame.model
        if (getattr(model,'_zos_device_pipeline',None) is not None or
                getattr(model,'_zos_prepared_frame',None) is not None or
                getattr(model,'_zos_commit_frame',None) is not None or
                getattr(frame,'_zos_ticket',None) is not None or frame._zos_commit_required):
            raise RuntimeError('another target already owns model')
        if (frame.count!=4 or frame.query_tokens!=6 or frame.tokens!=24 or
                tuple(frame.slots)!=(0,1,2,3) or frame.graph is None):
            raise ValueError('device pipeline requires captured fixed4x6 graph')
        if set(frame.even)!={2,8,14}:raise ValueError('unexpected C2 sources')
        if (model.engram_device_hash is None or model.engram_uva_runtime is None or not frame.rows
                or model.engram_device_hash.width!=4):
            raise ValueError('existing device hash/UVA required')
        if any(len(model.histories[s].history)!=model.next_positions[s] for s in frame.slots):
            raise ValueError('seed requires fully committed host histories')
        if any(getattr(frame,k) is not None for k in ('device_input_override','device_engram_override','device_engram_fast_override')):
            raise RuntimeError('legacy input override still held')
        if small_frame is not None and (
                small_frame.model is not model or small_frame.count!=4 or small_frame.query_tokens!=1
                or small_frame.tokens!=4 or tuple(small_frame.slots)!=tuple(frame.slots)
                or small_frame.graph is None or set(small_frame.even)!=set(frame.even)
                or not small_frame.rows or getattr(small_frame,'_zos_ticket',None) is not None
                or small_frame._zos_commit_required
                or any(getattr(small_frame,k) is not None for k in
                       ('device_input_override','device_engram_override','device_engram_fast_override'))):
            raise ValueError('device pipeline requires captured fixed4x1 small graph')
        if compact_frame is not None and (
                compact_frame.model is not model or compact_frame.count!=5
                or compact_frame.tokens!=8 or compact_frame.capacity!=8
                or tuple(compact_frame.slots)!=(0,1,2,3,4)
                or compact_frame.graph is None or set(compact_frame.even)!=set(frame.even)
                or not compact_frame.rows or getattr(compact_frame,'_zos_ticket',None) is not None
                or compact_frame._zos_commit_required
                or any(getattr(compact_frame,k) is not None for k in
                       ('device_input_override','device_engram_override','device_engram_fast_override'))):
            raise ValueError('device pipeline requires captured 8-row graph with private scratch')
        # Reuse original private C2 seed while no pipeline owns the model.
        frame._prepare_sequence()
        self.initial_c2={s:tuple(t.clone() for t in pair) for s,pair in frame.even.items()}
        self.frame=frame;self.model=model;self.stream=_stream();self.debug_c2=debug_c2
        self.queue=[];self.local=None;self.generation=0;self.closed=False;self.poisoned=False
        self._submitting=None;self._committing=None;self._sampling=None
        self.histories=tuple(model.histories[s] for s in frame.slots)
        self.input_buffer=frame.inputs
        self.row_buffers=tuple((s,id(v)) for s,v in sorted(frame.rows.items()))
        self.small_frame=small_frame
        if small_frame is not None:
            self.small_input=small_frame.inputs
            self.small_rows=tuple((s,id(v)) for s,v in sorted(small_frame.rows.items()))
        self.compact_frame=compact_frame
        if compact_frame is not None:
            self.compact_input=compact_frame.inputs
            self.compact_rows=tuple((s,id(v)) for s,v in sorted(compact_frame.rows.items()))
        self.last_step=-1
        model._zos_device_pipeline=self
    def _live(self):
        if self.closed or getattr(self.model,'_zos_device_pipeline',None) is not self:
            raise RuntimeError('device pipeline is closed or replaced')
        if self.poisoned:raise RuntimeError('device pipeline failed; model must not resume')
        if _stream()!=self.stream:raise RuntimeError('device pipeline stream changed')
        if self.frame.inputs is not self.input_buffer or tuple((s,id(v)) for s,v in sorted(self.frame.rows.items()))!=self.row_buffers:
            raise RuntimeError('device graph input ownership changed')
        small=self.small_frame
        if small is not None and (small.inputs is not self.small_input
                or tuple(small.slots)!=(0,1,2,3)
                or tuple((s,id(v)) for s,v in sorted(small.rows.items()))!=self.small_rows):
            raise RuntimeError('small graph input ownership changed')
        compact=self.compact_frame
        if compact is not None and (compact.inputs is not self.compact_input
                or tuple(compact.slots)!=(0,1,2,3,4)
                or tuple((s,id(v)) for s,v in sorted(compact.rows.items()))!=self.compact_rows):
            raise RuntimeError('compact graph input ownership changed')
        if any(self.model.histories[s] is not h for s,h in zip(self.frame.slots,self.histories)):
            raise RuntimeError('request history identity changed')
    def _tensor(self,value,shape,dtype):
        if value.device!=self.model.device or tuple(value.shape)!=shape or value.dtype!=dtype or not value.is_contiguous():
            raise ValueError('invalid device input shape/dtype/device')
    def _lease(self,ticket):
        self._live()
        if (not isinstance(ticket,DeviceTargetLease) or ticket.owner is not self or
                ticket not in self.queue or ticket.committed or ticket.discarded
                or ticket.frame is not ticket.local.frame
                or ticket.frame not in (self.frame,self.small_frame,self.compact_frame)):
            raise RuntimeError('stale or foreign device lease')
    def _sample_status(self,ticket):
        self._lease(ticket)
        state=ticket.state
        if (not ticket.sealed or state is None or not state.pending or
                state.event is not ticket.state_event or state.host is not ticket.state_host or
                state.retained is not ticket.state_retained or state.c2_pending is not ticket.c2_packet):
            raise RuntimeError('pending sample ownership changed')
        state.event.synchronize()
        packed=state.host.tolist()
        expected_words=52 if getattr(state,'eos_enabled',False) else 20
        if len(packed)!=expected_words or any(packed[:8]):
            raise RuntimeError('device state reported an error before next target')
        out,host,event,_=ticket.c2_packet
        event.synchronize();values=host.tolist()
        if (len(values)!=24 or any(values[i*8+4:i*8+8]!=[0]*4 for i in range(3)) or
                any(values[i*8:i*8+4]!=values[:4] for i in range(3))):
            raise RuntimeError('device C2 reported an error before next target')
        if set(out)!={2,8,14}:raise RuntimeError('pending C2 source identity changed')
        tokens=state.target_tokens
        if any(a is not b for a,b in zip((tokens.event,tokens.host,tokens.device_result),ticket.target_packet)):
            raise RuntimeError('pending target token packet changed')
        return out
    def stage_local(self,state,proposals,*,previous=None,small4=False,layout=None):
        """Only local kernels/copies and explicit status waits. No TP exchange."""
        self._live()
        if self.local is not None or len(self.queue)>=2:raise RuntimeError('previous local/target step still held')
        if previous is None:
            if self.queue or state.pending:raise RuntimeError('seed state is not host paired')
            view=state.current;c2=self.initial_c2
        else:
            self._lease(previous)
            if self.queue!=[previous] or not previous.submitted or previous.state is not state:
                raise RuntimeError('next target must follow the one pending sample')
            c2={s:tuple(t for t in values[:2]) for s,values in self._sample_status(previous).items()}
            view=state.next
        self._tensor(proposals,(4,5),proposals.dtype)
        if proposals.dtype not in (torch.int32,torch.int64):raise ValueError('invalid draft dtype')
        self._tensor(view.pos,(4,),torch.int32);self._tensor(view.golden,(4,),torch.int32)
        self._tensor(view.tail,(4,3),torch.int32)
        graph_inputs=None
        if layout is not None and (small4 or self.compact_frame is None):
            raise ValueError('compact layout requires its registered small7 frame')
        if small4:
            frame=self.small_frame
            if frame is None:raise ValueError('small4 was not registered with this pipeline')
            from deepinfer.engine.zos_small_graph_layout910c import stage_small4_windows
            tokens=view.golden.reshape(4,1)
            context,used,status=stage_small4_windows(view.tail,view.golden,view.pos,
                vocab_size=state.vocab,max_pos=state.max_pos)
        else:
            frame=self.frame
            tokens=torch.empty((4,6),dtype=torch.int32,device=self.model.device)
            tokens[:,0].copy_(view.golden);tokens[:,1:].copy_(proposals)
            context=torch.empty((24,4),dtype=torch.int32,device=self.model.device)
            used=torch.empty((4,),dtype=torch.int32,device=self.model.device)
            status=torch.empty((4,),dtype=torch.int32,device=self.model.device)
            from deepinfer.npu_ops.engram_window910c import build_windows
            build_windows(view.tail,tokens,view.pos,context,used,status,
                          vocab_size=state.vocab,max_positions=state.max_pos)
            if layout is not None:
                from deepinfer.engine.zos_small_graph_layout910c import stage_small7_inputs
                frame=self.compact_frame
                graph_inputs,context=stage_small7_inputs(layout,tokens,context,view.pos,
                    scratch_base=frame.scratch_base,noise=self.model.cfg['dspark_noise_token_id'],
                    vocab_size=state.vocab,max_pos=state.max_pos)
                tokens=graph_inputs[:8]
        n=frame.tokens
        host=torch.empty((n+4,),dtype=torch.int32,device='cpu',pin_memory=True)
        host[:n].copy_(tokens.reshape(-1),non_blocking=True)
        host[n:].copy_(view.pos,non_blocking=True)
        metadata=torch.empty((8,),dtype=torch.int32,device='cpu',pin_memory=True)
        metadata[:4].copy_(used,non_blocking=True);metadata[4:].copy_(status,non_blocking=True)
        event=torch.npu.Event();event.record()
        event.synchronize();checked=metadata.tolist()
        if len(checked)!=8 or checked[4:]!=[0]*4:
            raise RuntimeError('device input/window rejected before global plan')
        if previous is None and host[n:].tolist()!=self.model.next_positions[:4]:
            raise RuntimeError('seed device positions do not match host')
        self.generation+=1
        local=LocalInputs(self,self.generation,self.stream,previous,state,view,proposals,tokens,view.pos,
                          context,c2,host,event,used,status,frame,layout,graph_inputs)
        self.local=local
        return local
    def _local(self,local):
        self._live()
        if (not isinstance(local,LocalInputs) or local.owner is not self or
                self.local is not local or local.consumed or local.stream!=self.stream
                or local.frame not in (self.frame,self.small_frame,self.compact_frame)
                or (local.layout is not None)!=(local.frame is self.compact_frame)):
            raise RuntimeError('stale or foreign local inputs')
        if local.previous is not None:
            self._sample_status(local.previous)
            if local.state.next is not local.view:raise RuntimeError('device state changed before prepare')
        elif local.state.pending or local.state.current is not local.view:
            raise RuntimeError('seed state changed before prepare')
    def refine_compact_inputs(self,local,lengths):
        """Consume original device Q correction without a host wait before forward."""
        self._local(local)
        if local.frame is not self.compact_frame:
            raise ValueError('device Q refinement requires the agreed eight-row graph')
        from deepinfer.engine.zos_small_graph_layout910c import stage_small7_device_inputs
        from deepinfer.npu_ops.engram_window910c import build_windows
        view=local.view;state=local.state
        tokens=torch.empty((4,6),dtype=torch.int32,device=self.model.device)
        tokens[:,0].copy_(view.golden);tokens[:,1:].copy_(local.proposals)
        windows=torch.empty((24,4),dtype=torch.int32,device=self.model.device)
        used=torch.empty((4,),dtype=torch.int32,device=self.model.device)
        status=torch.empty((4,),dtype=torch.int32,device=self.model.device)
        build_windows(view.tail,tokens,view.pos,windows,used,status,
                      vocab_size=state.vocab,max_positions=state.max_pos)
        inputs,compact,bad=stage_small7_device_inputs(lengths,tokens,windows,view.pos,
            scratch_base=local.frame.scratch_base,noise=self.model.cfg['dspark_noise_token_id'],
            vocab_size=state.vocab,max_pos=state.max_pos)
        local.graph_inputs=inputs;local.tokens=inputs[:8];local.window=compact
        local.host[:8].copy_(local.tokens,non_blocking=True)
        local.corrected_status=torch.empty((1,),dtype=torch.int32,device='cpu',pin_memory=True)
        local.corrected_status.copy_(bad.reshape(1),non_blocking=True)
        local.event=torch.npu.Event();local.event.record()
    def discard_local(self,local):
        self._local(local);local.consumed=True;self.local=None
    def prepare(self,local,step,*,collectives_authorized=False):
        self._local(local)
        if collectives_authorized is not True:raise RuntimeError('TP preparation requires an agreed global plan')
        if type(step) is not int or step<=self.last_step:raise ValueError('target step must advance')
        f=local.frame
        n=f.tokens
        compact=local.layout is not None
        if compact:self._tensor(local.graph_inputs,(35,),torch.int32)
        self._tensor(local.tokens,(8,) if compact else (4,f.query_tokens),torch.int32)
        self._tensor(local.positions,(4,),torch.int32)
        self._tensor(local.window,(n,4),torch.int32)
        protected=[f.inputs,*f.rows.values(),*[v for pair in f.even.values() for v in pair]]
        if f.engram_hash_output is not None:protected.append(f.engram_hash_output)
        input_data=local.graph_inputs if compact else local.tokens
        for value in (input_data,local.positions,local.window,*[t for pair in local.c2.values() for t in pair]):
            a=value.data_ptr();b=a+value.numel()*value.element_size()
            for target in protected:
                c=target.data_ptr();d=c+target.numel()*target.element_size()
                if a<d and c<b:raise ValueError('device producer overlaps private graph buffers')
        if set(local.c2)!=set(f.even):raise ValueError('C2 map differs from graph')
        for s,pair in local.c2.items():
            if len(pair)!=2:raise ValueError('C2 requires value and gate')
            for value,target in zip(pair,f.even[s]):self._tensor(value,(4,target.shape[1]),torch.float32)
        try:
            if compact:
                f.inputs.copy_(local.graph_inputs)
            else:
                f.inputs[:n].copy_(local.tokens.reshape(-1))
                f.inputs[n:].copy_(local.positions+f.query_tokens)
            for s,pair in local.c2.items():
                for value,target in zip(pair,f.even[s]):
                    target[:4].copy_(value)
                    if compact:target[4:].zero_()
            hashes=self.model.engram_device_hash
            if f.engram_hash_output is None:
                f.engram_hash_output=torch.empty((len(hashes.layers),n,hashes.heads),dtype=torch.int32,device=self.model.device)
            ids=hashes.compute(local.window,f.engram_hash_output)
            for layer,buffer in f.rows.items():self.model.engram_uva_runtime.rows(layer,ids[layer],buffer)
        except BaseException:
            self.poisoned=True
            raise
        f.prepared=True
        f.engram_hash_calls+=1;f.engram_hash_tokens+=n
        f.device_input_override_applied+=1;f.device_engram_fast_calls+=1
        ticket=DeviceTargetLease(self,local.generation,step,local,self.histories,f.replays,f)
        # The queue retains the old step until its host post. All GPU sources
        # needed by this step are held directly in local; no historical lease
        # chain is needed after the dependency check and input preparation.
        local.consumed=True;local.previous=None
        self.local=None;self.queue.append(ticket);self.last_step=step
        return ticket
    def submit(self,ticket):
        self._lease(ticket)
        if ticket is not self.queue[-1] or ticket.submitted:raise RuntimeError('target already submitted or out of order')
        if ticket.frame.replays!=ticket.replay_before:raise RuntimeError('target graph replayed outside lease')
        self._submitting=ticket
        try:output=ticket.frame._execute_graph(advance_host_positions=False)
        except BaseException:
            self.poisoned=True
            raise
        finally:self._submitting=None
        ticket.submitted=True
        return output
    def seal_sample(self,ticket,state):
        self._lease(ticket)
        if self._sampling is not ticket:raise RuntimeError('sample must enter through its guarded callback')
        if self.queue!=[ticket] or not ticket.submitted or ticket.sealed:
            raise RuntimeError('post-process previous host before current sample')
        if not state.pending or state.c2_pending is None:raise RuntimeError('sample/state/C2 not queued')
        if state.current.pos is not ticket.local.positions:raise RuntimeError('sample uses another device position state')
        ticket.state=state;ticket.c2_packet=state.c2_pending
        ticket.state_event=state.event;ticket.state_host=state.host;ticket.state_retained=state.retained
        tokens=state.target_tokens
        if tokens.device_result is None:raise RuntimeError('target token packet not queued')
        ticket.target_packet=(tokens.event,tokens.host,tokens.device_result)
        if self.debug_c2:
            ticket.before_snapshot={s:tuple(t.clone() for t in pair) for s,pair in ticket.frame.even_before.items()}
            ticket.rows_snapshot={s:tuple(t.clone() for t in pair) for s,pair in ticket.frame.even_rows.items()}
        ticket.sealed=True
    def queue_sample(self,ticket,state,callback):
        """Check old-host retirement before any current packet can be overwritten."""
        self._lease(ticket)
        if (self.queue!=[ticket] or not ticket.submitted or ticket.sealed or
                state is not ticket.local.state or state.pending or state.c2_pending is not None or
                not state.target_tokens.consumed):
            raise RuntimeError('previous host packet must be retired before current sampling')
        self._sampling=ticket
        try:
            result=callback()
            self.seal_sample(ticket,state)
            return result
        except BaseException:
            self.poisoned=True
            raise
        finally:self._sampling=None
    def _host_boundary(self,ticket,drain):
        self._lease(ticket)
        if ticket is not self.queue[0] or not ticket.sealed:raise RuntimeError('oldest sealed host step required')
        if drain:
            if len(self.queue)!=1 or self.local is not None:raise RuntimeError('discard next preparation before draining')
        elif len(self.queue)!=2 or not self.queue[1].submitted:
            raise RuntimeError('submit current target before previous host post-process')
    def host_inputs(self,ticket,*,drain=False):
        self._host_boundary(ticket,drain)
        if ticket.host_values is None:
            ticket.local.event.synchronize();values=ticket.local.host.tolist()
            n,q=ticket.frame.tokens,ticket.frame.query_tokens
            if len(values)!=n+4:raise RuntimeError('malformed held input packet')
            offsets=ticket.local.layout.offsets if ticket.local.layout is not None else tuple(i*q for i in range(5))
            ticket.host_values=(tuple(values[n:]),tuple(tuple(values[a:b]) for a,b in zip(offsets,offsets[1:])))
        return ticket.host_values
    def host_targets(self,ticket,*,drain=False):
        """Consume the original target packet once, at the old-host boundary."""
        self._host_boundary(ticket,drain);self._sample_status(ticket)
        if ticket.target_values is None:
            packet=ticket.state.target_tokens
            if not packet.consumed:
                values=packet.wait()
            else:
                # Existing controllers may already have waited on this same,
                # still-owned packet before computing their CPU acceptance.
                ticket.target_packet[0].synchronize()
                values=ticket.target_packet[1].tolist()
            if len(values)!=24:raise RuntimeError('malformed held target packet')
            ticket.target_values=tuple(values)
        return ticket.target_values
    def commit_oldest(self,ticket,accepted,commits,emitted,*,drain=False,remaining=None,finished=None):
        bases,chunks=self.host_inputs(ticket,drain=drain)
        self._sample_status(ticket)
        if (len(accepted)!=4 or len(commits)!=4 or len(emitted)!=4 or
                any(type(a) is not int or not 0<=a<=5 for a in accepted) or
                any(type(c) is not int or not 0<=c<=6 or c>a+1 or c!=len(row)
                    for a,c,row in zip(accepted,commits,emitted))):
            raise ValueError('invalid CPU acceptance/commit packet')
        frame=ticket.frame
        layout=ticket.local.layout
        eos=getattr(ticket.state,'eos_enabled',False)
        if eos and (layout is not None or frame.query_tokens!=6 or remaining is None or finished is None):
            raise ValueError('EOS target commit requires full graph and original stopping metadata')
        if layout is not None and any(a>=q or c>q for a,c,q in zip(accepted,commits,layout.lengths)):
            raise ValueError('compact commit exceeds the held real span')
        if layout is None and frame.query_tokens==1 and (any(accepted) or any(c>1 for c in commits)):
            raise ValueError('small4 may commit only its one real row per slot')
        if tuple(self.model.next_positions[:4])!=bases:
            raise RuntimeError('host position generation differs from oldest device input')
        for h,p,row in zip(self.histories,bases,chunks):
            if len(h.history)!=p or any(type(x) is not int or not 0<=x<len(h.token_map) for x in row):
                raise ValueError('old host history/actual chunks do not match')
        targets=self.host_targets(ticket,drain=drain)
        if len(targets)!=24 or any(tuple(row)!=tuple(targets[i*6:i*6+c]) for i,(row,c) in enumerate(zip(emitted,commits))):
            raise ValueError('emitted tokens differ from held target packet')
        check_frame=(SimpleNamespace(even_before=ticket.before_snapshot,even_rows=ticket.rows_snapshot)
                     if self.debug_c2 else frame)
        try:
            check_layout={'layout':layout} if layout is not None else {}
            out,c2_ok,_=ticket.state.settle_c2(bases,commits,check_frame,check=self.debug_c2,**check_layout)
            stopping={'remaining':remaining,'finished':finished} if eos else {}
            state_ok=ticket.state.settle(accepted,commits,emitted,**stopping)
            if not c2_ok or not state_ok:raise RuntimeError('CPU acceptance differs from pending device state')
            for h,row in zip(self.histories,chunks):h.history.extend(row)
            self._committing=ticket
            try:frame.commit_prefixes(bases,commits,device_pending=out)
            finally:self._committing=None
        except BaseException:
            self.poisoned=True
            raise
        # A paired (in-place) caller can prepare again with previous=None.
        # Seed it from this validated result, rather than construction-time
        # C2. Pending callers still consume their previous ticket's packet.
        self.initial_c2={s:tuple(values[:2]) for s,values in out.items()}
        self.queue.pop(0);ticket.committed=True
        ticket.state=None;ticket.c2_packet=None;ticket.state_retained=None;ticket.target_packet=None
        ticket.before_snapshot=None;ticket.rows_snapshot=None
        return bases,chunks
    def discard_prepared(self,ticket):
        self._lease(ticket)
        if ticket.submitted or ticket.sealed or ticket is not self.queue[-1]:
            raise RuntimeError('only latest unsubmitted target may be discarded')
        self.queue.pop();ticket.discarded=True;ticket.frame.prepared=False
    def close(self):
        self._live()
        if self.queue or self.local is not None:raise RuntimeError('drain all target steps before close')
        self.frame.prepared=False;self.initial_c2=None
        if self.small_frame is not None:self.small_frame.prepared=False
        if self.compact_frame is not None:self.compact_frame.prepared=False
        self.model._zos_device_pipeline=None;self.closed=True
