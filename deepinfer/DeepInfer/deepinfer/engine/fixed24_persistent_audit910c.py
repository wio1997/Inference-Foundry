# SPDX-License-Identifier: Apache-2.0
"""Diagnostic consecutive device state against the existing CPU model state.

Opt-in only, for four full six-query rows and ignore-EOS. This adapter never
installs device state into the model, affects sampling, or adds EP collectives.
Every iteration synchronizes and compares; no performance claim is possible.
"""
import traceback

from .fixed24_initial_snapshot910c import inspect_model, stage_snapshot
from .fixed24_state_candidate910c import Fixed24StateCandidate


class PersistentStateAudit:
    def __init__(self, limit=8):
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError('audit limit must be 1..16')
        self.limit = limit
        self.candidate = None
        self.staged_draft = None
        self.window_context = None
        self.window_used = None
        self.window_status = None
        self.disabled = False
        self.stats = {'enabled':True, 'limit':limit, 'calls':0, 'passes':0,
                      'failures':0, 'consecutive':0, 'details':[]}

    def _fail(self, stage):
        self.disabled = True
        self.stats['failures'] += 1
        self.stats['details'].append({'stage':stage, 'error':traceback.format_exc()})

    def draft_inputs(self, golden, positions, *, allow_after_limit=False):
        """Return consecutive device state for the draft graph, with CPU oracle.

        This is a correctness gate: the CPU remains authoritative and readback
        is intentional. A mismatch aborts the diagnostic instead of silently
        advancing from stale or divergent device state.
        """
        if self.disabled or self.candidate is None or (self.stats['calls'] >= self.limit and not allow_after_limit):
            return None
        state = self.candidate.current.state
        if state.golden.cpu().tolist() != list(golden) or state.pos.cpu().tolist() != list(positions):
            raise RuntimeError('consecutive draft device state differs from CPU')
        self.stats['draft_device_inputs_checked'] = self.stats.get('draft_device_inputs_checked', 0) + 1
        return state.golden, state.pos

    def interrupt(self):
        """Do not silently carry fixed-shape state across an unobserved step."""
        if self.candidate is not None and self.stats['calls'] < self.limit:
            self.disabled = True
            self.stats['details'].append({'stage':'shape_gap', 'state':'STOPPED'})

    def before(self, model, frame, proposals, golden, remaining, finished, chunks):
        if self.disabled or self.stats['calls'] >= self.limit:
            return False
        try:
            import torch
            host = inspect_model(model, frame, golden, remaining, finished)
            if self.candidate is None:
                snap = stage_snapshot(host, model, frame)
                self.candidate = Fixed24StateCandidate(snap, ignore_eos=True,
                    fixed_query_tokens=6, sources=(2,8,14))
            draft = proposals.to(dtype=torch.int32).contiguous()
            staged=self.candidate.next_graph_input(draft)
            if staged.cpu().tolist() != [list(row) for row in chunks]:
                raise ValueError('device-constructed input differs from CPU target input')
            if frame.device_input_override is not None:
                raise ValueError('unconsumed target-input override')
            from deepinfer.npu_ops.engram_window910c import build_windows
            if self.window_context is None:
                self.window_context=torch.empty((24,4),dtype=torch.int32,device=model.device)
                self.window_used=torch.empty((4,),dtype=torch.int32,device=model.device)
                self.window_status=torch.empty((4,),dtype=torch.int32,device=model.device)
            state=self.candidate.current.state
            build_windows(state.tail,staged,state.pos,self.window_context,self.window_used,
                          self.window_status,vocab_size=int(model.cfg['vocab_size']),
                          max_positions=int(model.max_tokens))
            if self.window_status.cpu().tolist()!=[0]*4 or self.window_used.cpu().tolist()!=[p+6 for p in host.positions]:
                raise ValueError('device window status or used position mismatch')
            if frame.device_engram_override is not None:
                raise ValueError('unconsumed Engram-window override')
            self.override_count_before=frame.device_input_override_applied
            self.engram_count_before=frame.device_engram_override_applied
            frame.device_input_override=staged
            frame.device_engram_override=self.window_context
            self.staged_draft = draft
            return True
        except Exception:
            frame.device_input_override=None
            frame.device_engram_override=None
            self._fail('before')
            return False

    def prelaunch_state(self, model, frame, target_device):
        """Queue the device transition for a same-stream draft prelaunch.

        Caller is responsible for a collective eligibility vote, and a second
        collective success vote before the speculative result is consumed.
        CPU remains authoritative; this method alone does not publish state.
        """
        if self.disabled or self.staged_draft is None:
            return None
        try:
            import torch
            if frame is not model.dspark_graphs.full:
                raise ValueError('prelaunch requires the full fixed24 frame')
            target=target_device.reshape(4,6).to(dtype=torch.int32).contiguous()
            rows={s:frame.even_rows[s] for s in (2,8,14)}
            self.candidate.queue_transition(self.staged_draft,target,rows,
                vocab_size=int(model.cfg['vocab_size']),max_positions=int(model.max_tokens))
            self.stats['prelaunch_transitions']=self.stats.get('prelaunch_transitions',0)+1
            return self.candidate.pending_state_for_prelaunch()
        except Exception:
            self._fail('prelaunch')
            return None

    def after(self, model, frame, target_device, commits, accepted, emitted,
              generated, maximum, finished):
        if self.disabled or self.staged_draft is None:
            return
        self.stats['calls'] += 1
        call = self.stats['calls']
        try:
            import torch
            if frame is not model.dspark_graphs.full:
                raise ValueError('target frame changed during diagnostic')
            if frame.device_input_override is not None or frame.device_input_override_applied != self.override_count_before+1:
                raise ValueError('target graph did not consume exactly one device input')
            if frame.device_engram_override is not None or frame.device_engram_override_applied != self.engram_count_before+1:
                raise ValueError('Engram hash did not consume the verified device windows')
            staged = self.candidate._input.cpu().tolist()
            actual_graph = frame.inputs[:24].reshape(4,6).cpu().tolist()
            if staged != actual_graph:
                raise ValueError('staged device input differs from actual target graph input')
            if not self.candidate._pending:
                target = target_device.reshape(4,6).to(dtype=torch.int32).contiguous()
                rows = {s: frame.even_rows[s] for s in (2,8,14)}
                self.candidate.queue_transition(self.staged_draft, target, rows,
                    vocab_size=int(model.cfg['vocab_size']), max_positions=int(model.max_tokens))
            # This diagnostic never publishes state to the service. It uses a
            # local vote and adds no collective to the live EP ordering.
            packet = self.candidate.settle(lambda local_ok: local_ok)
            self.staged_draft = None
            if (list(packet.committed) != list(commits) or
                    list(packet.accepted) != list(accepted) or
                    [list(x) for x in packet.emitted] != [list(x) for x in emitted]):
                raise ValueError('device packet differs from actual CPU sampling')
            current = self.candidate.current
            actual = {key:getattr(current.state,key).cpu().tolist() for key in
                      ('pos','remaining','golden','finished','tail','status')}
            h = int(model.cfg['engram_max_ngram_size']) - 1
            tails=[]
            for slot in range(4):
                history = model.histories[slot].history
                if len(history) != model.next_positions[slot]:
                    raise ValueError('post-commit CPU history length mismatch')
                tail=list(history[-h:]); tails.append([-1]*(h-len(tail))+tail)
            expected={'pos':list(model.next_positions[:4]),
                'remaining':[maximum[i]-len(generated[i]) for i in range(4)],
                'golden':[x[-1] for x in generated], 'finished':list(finished),
                'tail':tails, 'status':[0]*4}
            if actual != expected:
                raise ValueError('device state differs from CPU state: '+repr({
                    k:{'actual':actual[k], 'expected':expected[k]} for k in actual
                    if actual[k] != expected[k]}))
            valid_slots={}
            for source in (2,8,14):
                item=current.pending[source]
                valid=item.valid.cpu().tolist()
                valid_slots[str(source)]=sum(valid)
                source_pending=model.pending.get(source,{})
                if valid != [i in source_pending for i in range(4)]:
                    raise ValueError('C2 valid flags differ at source '+str(source))
                if item.status.cpu().tolist() != [0]*4:
                    raise ValueError('C2 status nonzero at source '+str(source))
                for slot in range(4):
                    if valid[slot]:
                        for device_rows, host_value in zip((item.value,item.gate),source_pending[slot]):
                            got=device_rows[slot].contiguous().cpu().view(torch.int32).tolist()
                            want=host_value.contiguous().cpu().view(torch.int32).reshape(-1).tolist()
                            if got != want:
                                raise ValueError('C2 FP32 bits differ at source '+str(source)+
                                                 ' slot '+str(slot))
            self.stats['passes'] += 1
            self.stats['consecutive'] += 1
            self.stats['details'].append({'call':call, 'state':'PASS',
                                          'c2_sources':[2,8,14], 'c2_valid_slots':valid_slots, 'device_input_consumed':True, 'device_engram_window_consumed':True})
        except Exception:
            self._fail('after')
