# SPDX-License-Identifier: Apache-2.0
"""Fixed-profile device DSpark state and event-ordered host packet.

Only the state used by the next draft is kept on device. CPU retains model,
Engram history, C2, output and scheduling authority. Callers must verify the
packet against those CPU decisions and participate in the next global vote
before consuming a prelaunched draft.
"""
import torch
from deepinfer.npu_ops.greedy_prefix910c import GreedyPrefixBuffers, verify_greedy_prefix
from deepinfer.npu_ops.state910c import StateBuffers, commit_state

class PendingTokens:
    def __init__(self, device):
        self.device=device
        self.host=torch.empty((24,),dtype=torch.int64,device='cpu',pin_memory=True)
        self.event=torch.npu.Event();self.consumed=True;self.device_result=None
    def queue(self,device_result):
        if not self.consumed:raise RuntimeError('target packet still in flight')
        if (device_result.device!=self.device or device_result.dtype!=torch.int64 or
                tuple(device_result.shape)!=(24,) or not device_result.is_contiguous()):
            raise ValueError('invalid fixed24 target packet')
        self.device_result=device_result
        self.host.copy_(device_result,non_blocking=True)
        self.event.record();self.consumed=False
        return self
    def wait(self):
        if self.consumed:raise RuntimeError('target packet already consumed')
        self.event.synchronize();self.consumed=True
        return self.host.tolist()

class AsyncDraftState:
    def __init__(self,model,golden,remaining,finished,*,eos_token_id=None,ignore_eos=None):
        self.device=model.device;self.vocab=int(model.cfg['vocab_size']);self.max_pos=int(model.max_tokens)
        width=int(model.cfg['engram_max_ngram_size'])-1
        if width!=3 or len(golden)!=4 or len(remaining)!=4 or len(finished)!=4:
            raise ValueError('unsupported fixed-profile state')
        positions=list(model.next_positions[:4]);tails=[]
        for i,pos in enumerate(positions):
            history=model.histories[i].history
            if len(history)!=pos:raise ValueError('history and position differ')
            tail=list(history[-width:]);tails.append([-1]*(width-len(tail))+tail)
        if any(type(x) is not int or not 0<=x<self.vocab for x in golden):raise ValueError('invalid golden')
        if any(type(x) is not int or x<0 for x in remaining):raise ValueError('invalid remaining')
        self.eos_enabled=False
        if eos_token_id is None:
            if ignore_eos is not None:raise ValueError('EOS mask needs an EOS token')
        else:
            if (type(eos_token_id) is not int or not 0<=eos_token_id<self.vocab or
                    not isinstance(ignore_eos,(tuple,list)) or len(ignore_eos)!=4 or
                    any(type(value) is not bool for value in ignore_eos)):
                raise ValueError('invalid EOS configuration')
            self.eos_enabled=not all(ignore_eos)
            self.eos_token_id=eos_token_id
            self.eos_ignore_host=list(ignore_eos)
        if self.eos_enabled:
            for done,left,token,ignore in zip(finished,remaining,golden,ignore_eos):
                if (type(done) is not bool or (left==0 and not done) or
                        (done and left>0 and (ignore or token!=eos_token_id))):
                    raise ValueError('invalid EOS finished state')
        else:
            if any(type(x) is not bool or x!=(left==0) for x,left in zip(finished,remaining)):raise ValueError('invalid finished state')
        if any(type(x) is not int or x<0 or x+6>self.max_pos for x in positions):raise ValueError('invalid position')
        if any(any(type(x) is not int or x < -1 or x>=self.vocab for x in row) for row in tails):raise ValueError('invalid history tail')
        self.current=StateBuffers.allocate(self.device,width)
        self.next=StateBuffers.allocate(self.device,width)
        for dest,values,dtype in ((self.current.pos,positions,torch.int32),
             (self.current.remaining,remaining,torch.int32),(self.current.golden,golden,torch.int32),
             (self.current.finished,finished,torch.bool),(self.current.tail,tails,torch.int32)):
            dest.copy_(torch.tensor(values,dtype=dtype,device=self.device))
        self.current.status.zero_()
        self.greedy=GreedyPrefixBuffers.allocate(self.device)
        self.lengths=torch.full((4,),6,dtype=torch.int32,device=self.device)
        self.input=torch.empty((4,6),dtype=torch.int32,device=self.device)
        self.packet_words=52 if self.eos_enabled else 20
        self.packed=torch.empty((self.packet_words,),dtype=torch.int32,device=self.device)
        self.host=torch.empty((self.packet_words,),dtype=torch.int32,device='cpu',pin_memory=True)
        self.eos_commit=None;self.eos_decision=None;self.c2_retained=None
        if self.eos_enabled:
            self.eos_ignore=torch.tensor(ignore_eos,dtype=torch.bool,device=self.device)
            self.eos_host={'pos':list(positions),'remaining':list(remaining),
                          'finished':list(finished),'golden':list(golden),
                          'tail':[list(row) for row in tails]}
        self.event=None;self.pending=False;self.retained=None
        self.target_tokens=PendingTokens(self.device)
        self.c2_pending=None
        self.window_context=torch.empty((24,4),dtype=torch.int32,device=self.device)
        self.window_used=torch.empty((4,),dtype=torch.int32,device=self.device)
        self.window_status=torch.empty((4,),dtype=torch.int32,device=self.device)
        self.window_host=torch.empty((8,),dtype=torch.int32,device='cpu',pin_memory=True)
        self.window_event=torch.npu.Event()
        self.window_pending=False

    def prepare_engram_window(self,proposals):
        """Produce the next full-graph windows from published device state."""
        if self.window_pending:raise RuntimeError('previous window is not settled')
        from deepinfer.npu_ops.engram_window910c import build_windows
        tokens=self.target_graph_input(proposals)
        build_windows(self.current.tail,tokens,self.current.pos,self.window_context,
                      self.window_used,self.window_status,vocab_size=self.vocab,
                      max_positions=self.max_pos)
        self.window_host[:4].copy_(self.window_used,non_blocking=True)
        self.window_host[4:].copy_(self.window_status,non_blocking=True)
        self.window_event.record();self.window_pending=True
        return self.window_context

    def verify_engram_window(self,bases):
        if not self.window_pending:raise RuntimeError('no window is pending')
        self.window_event.synchronize();values=self.window_host.tolist()
        self.window_pending=False
        return values[:4]==[x+6 for x in bases] and values[4:]==[0]*4

    def queue_c2(self,frame,compact_cu=None):
        """Compute C2 pending values on device for the current full graph.

        Fresh payloads are retained by model.pending until a later commit or
        snapshot. This deliberately avoids recycling storage across graph
        changes, request termination, and prefix-cache snapshot creation.
        """
        if compact_cu is not None or frame.query_tokens==1:
            return self._queue_small_c2(frame,compact_cu)
        if not self.pending or self.c2_pending is not None:
            raise RuntimeError('C2 transition requires one pending greedy result')
        if frame.count!=4 or frame.query_tokens!=6 or tuple(frame.slots)!=(0,1,2,3):
            raise ValueError('C2 device path requires the four-slot full graph')
        if set(frame.even_rows)!={2,8,14} or set(frame.even_before)!={2,8,14}:
            raise ValueError('unexpected C2 graph sources')
        from deepinfer.npu_ops.c2_pending910c import allocate,launch
        before_valid=(self.current.pos.remainder(2)!=0).contiguous()
        out={}; host=torch.empty((24,),dtype=torch.int32,device='cpu',pin_memory=True)
        for index,source in enumerate((2,8,14)):
            old_v,old_g=frame.even_before[source]
            rows_v,rows_g=frame.even_rows[source]
            outputs=allocate(old_v,old_g)
            launch(old_v,old_g,before_valid,rows_v,rows_g,self.current.pos,
                   self.eos_commit if self.eos_enabled else self.greedy.commit,outputs,max_pos=self.max_pos)
            host[index*8:index*8+4].copy_(outputs[2],non_blocking=True)
            host[index*8+4:index*8+8].copy_(outputs[3],non_blocking=True)
            out[source]=outputs
        event=torch.npu.Event();event.record()
        self.c2_pending=(out,host,event,before_valid)
        if self.eos_enabled:self.c2_retained=(frame,frame.even_before,frame.even_rows,self.eos_commit)
        return out

    def _queue_small_c2(self,frame,compact_cu):
        if self.eos_enabled:
            raise ValueError('ordinary EOS retains the full target graph')
        if not self.pending or self.c2_pending is not None:
            raise RuntimeError('small C2 transition requires one pending sample')
        if compact_cu is not None:
            if frame.count!=5 or frame.tokens!=8 or tuple(frame.slots)!=(0,1,2,3,4):
                raise ValueError('invalid registered compact graph')
            from .zos_small_graph_layout910c import select_small7_c2
            out=select_small7_c2(frame.even_before,frame.even_rows,self.current.pos,self.greedy.commit,
                                 compact_cu,max_pos=self.max_pos)
        else:
            if frame.count!=4 or frame.query_tokens!=1 or tuple(frame.slots)!=(0,1,2,3):
                raise ValueError('invalid registered four-row graph')
            from .zos_small_graph_layout910c import select_small4_c2
            out=select_small4_c2(frame.even_before,frame.even_rows,self.current.pos,self.greedy.commit,
                                 max_pos=self.max_pos)
        host=torch.empty((24,),dtype=torch.int32,device='cpu',pin_memory=True)
        for index,source in enumerate((2,8,14)):
            host[index*8:index*8+4].copy_(out[source][2],non_blocking=True)
            host[index*8+4:index*8+8].copy_(out[source][3],non_blocking=True)
        event=torch.npu.Event();event.record()
        self.c2_pending=(out,host,event,None)
        return out

    def settle_c2(self,bases,commits,frame,*,check=False,layout=None):
        if self.c2_pending is None:raise RuntimeError('no pending C2 transition')
        out,host,event,_=self.c2_pending
        event.synchronize();observed=host.tolist()
        expected=[int((base+count)%2!=0) for base,count in zip(bases,commits)]
        ok=all(observed[i*8:i*8+4]==expected and
               observed[i*8+4:i*8+8]==[0]*4 for i in range(3))
        checked_rows=0
        if ok and check:
            # Diagnostic only: compare actual device C2 payload to the old
            # Python selection, bit-for-bit, before it can be published.
            for source in (2,8,14):
                for slot,(base,count) in enumerate(zip(bases,commits)):
                    if (base+count)%2:
                        for k in range(2):
                            if not count:
                                expected_value=frame.even_before[source][k][slot:slot+1]
                            elif layout is not None:
                                row=layout.c2_row(slot,count)
                                expected_value=frame.even_rows[source][k][row:row+1]
                            else:
                                expected_value=frame.even_rows[source][k][slot:slot+1,count-1]
                            actual=out[source][k][slot:slot+1]
                            if not torch.equal(actual.contiguous().cpu().view(torch.int32),
                                               expected_value.contiguous().cpu().view(torch.int32)):
                                ok=False
                        checked_rows+=1
        self.c2_pending=None
        if self.eos_enabled:self.c2_retained=None
        return out,ok,checked_rows

    def target_graph_input(self, proposals):
        """Build the full target input from the published device state.

        The caller keeps proposals alive and orders all uses on the same NPU
        stream. CPU state is still authoritative for scheduling and fallback.
        """
        if self.pending:raise RuntimeError('device state transition is pending')
        if (proposals.device!=self.device or tuple(proposals.shape)!=(4,5)
                or proposals.dtype not in (torch.int32,torch.int64)):
            raise ValueError('expected device draft tokens [4,5]')
        self.input[:,0].copy_(self.current.golden)
        self.input[:,1:].copy_(proposals)
        return self.input
    def queue(self,proposals,target):
        if self.pending:raise RuntimeError('previous async state not settled')
        if self.eos_enabled and self.c2_pending is not None:raise RuntimeError('previous EOS C2 still pending')
        draft=proposals.to(dtype=torch.int32).contiguous()
        target32=target.reshape(4,6).to(dtype=torch.int32).contiguous()
        self.input[:,0].copy_(self.current.golden);self.input[:,1:].copy_(draft)
        g=verify_greedy_prefix(draft,target32,self.lengths,self.current.remaining,self.current.finished,self.greedy)
        if self.eos_enabled:
            from .eos_state910c import clip_output
            self.eos_decision=clip_output(target32,g.commit,self.current.finished,self.eos_ignore,
                                          eos_token_id=self.eos_token_id)
            self.eos_commit=self.eos_decision['commit']
            commit,last=self.eos_commit,self.eos_decision['last']
        else:
            commit,last=g.commit,g.last
        commit_state(self.current.pos,self.current.remaining,self.current.golden,self.current.finished,
            self.current.tail,self.input,commit,last,self.next,vocab_size=self.vocab,max_positions=self.max_pos)
        if self.eos_enabled:self.next.finished.logical_or_(self.eos_decision['eos_hit'])
        for i,value in enumerate((g.status,self.next.status,g.accepted,commit,last)):
            self.packed[4*i:4*i+4].copy_(value)
        if self.eos_enabled:
            for first,value in ((20,self.next.remaining),(24,self.next.finished),
                                (28,self.eos_decision['eos_hit']),(32,self.next.pos),
                                (36,self.next.tail.reshape(-1)),(48,self.next.golden)):
                self.packed[first:first+value.numel()].copy_(value)
        self.host.copy_(self.packed,non_blocking=True)
        self.event=torch.npu.Event();self.event.record()
        self.retained=(draft,target32,target,proposals);self.pending=True
        if self.eos_enabled:self.retained+= (self.eos_decision,)
        return self.next.golden,self.next.pos
    def settle(self,accepted,commits,emitted,*,remaining=None,finished=None):
        if self.eos_enabled:return self._settle_eos(accepted,commits,emitted,remaining,finished)
        if not self.pending:raise RuntimeError('no pending async state')
        self.event.synchronize();v=self.host.tolist()
        if len(v)!=20:raise RuntimeError('malformed async state packet')
        statuses=v[:8];actual_accepted=v[8:12];actual_commits=v[12:16];actual_last=v[16:20]
        expected_last=[row[-1] if row else -1 for row in emitted]
        ok=(not any(statuses) and actual_accepted==list(accepted) and
            actual_commits==list(commits) and actual_last==expected_last)
        self.pending=False;self.retained=None
        if ok:self.current,self.next=self.next,self.current
        return ok

    def _settle_eos(self,accepted,commits,emitted,remaining,finished):
        if not self.pending:raise RuntimeError('no pending EOS state')
        if self.c2_pending is not None:raise RuntimeError('settle EOS C2 before state publication')
        self.event.synchronize();v=self.host.tolist()
        old=self.eos_host;expected=None
        metadata_ok=(all(isinstance(x,(list,tuple)) and len(x)==4 for x in
                         (accepted,commits,emitted,remaining,finished)) and
                     all(type(x) is int and 0<=x<=5 for x in accepted) and
                     all(type(x) is int and 0<=x<=6 for x in commits) and
                     all(type(x) is int and x>=0 for x in remaining) and
                     all(type(x) is bool for x in finished))
        if metadata_ok:
            expected={'pos':[],'remaining':[],'finished':[],'golden':[],'tail':[]}
            eos_hits=[];last=[]
            for slot,(a,c,row) in enumerate(zip(accepted,commits,emitted)):
                if (not isinstance(row,(tuple,list)) or len(row)!=c or
                        any(type(t) is not int or not 0<=t<self.vocab for t in row) or
                        c>old['remaining'][slot] or c>a+1 or
                        (old['finished'][slot] and (a!=0 or c!=0)) or
                        (not self.eos_ignore_host[slot] and self.eos_token_id in row[:-1])):
                    metadata_ok=False;break
                hit=bool(row) and not self.eos_ignore_host[slot] and row[-1]==self.eos_token_id
                left=old['remaining'][slot]-c
                done=old['finished'][slot] or hit or (c>0 and left==0)
                expected['pos'].append(old['pos'][slot]+c)
                expected['remaining'].append(left);expected['finished'].append(done)
                expected['golden'].append(row[-1] if row else old['golden'][slot])
                inputs=[old['golden'][slot],*row[:-1]] if c else []
                expected['tail'].append((old['tail'][slot]+inputs)[-3:])
                eos_hits.append(int(hit));last.append(row[-1] if row else -1)
        ok=(metadata_ok and len(v)==52 and not any(v[:8]) and
            v[8:12]==list(accepted) and v[12:16]==list(commits) and v[16:20]==last and
            v[20:24]==expected['remaining']==list(remaining) and
            v[24:28]==list(map(int,expected['finished']))==list(map(int,finished)) and
            v[28:32]==eos_hits and v[32:36]==expected['pos'] and
            v[36:48]==[t for row in expected['tail'] for t in row] and
            v[48:52]==expected['golden'])
        self.pending=False;self.retained=None;self.eos_commit=None;self.eos_decision=None
        if ok:
            self.current,self.next=self.next,self.current
            self.eos_host=expected
        return bool(ok)
