# SPDX-License-Identifier: Apache-2.0
"""Unconnected fixed-24 device-state candidate. Not a serving or ZOS entrypoint.

All work is submitted to the caller's current NPU stream. The caller must make
producer/consumer streams wait, retain all input tensors until settle returns,
and invoke the *same* consensus operation on every EP rank. The default service
never imports this module. The current implementation deliberately reads back a
small packet and status per iteration: correctness gate, not a performance path.
"""
from dataclasses import dataclass
import os


@dataclass
class PendingC2:
    value: object
    gate: object
    valid: object
    status: object


@dataclass
class Snapshot:
    state: object
    pending: dict


@dataclass(frozen=True)
class OutputPacket:
    accepted: tuple
    committed: tuple
    emitted: tuple
    last: tuple


class Fixed24StateCandidate:
    """A single uncommitted transition with fail-closed device-buffer swapping.

    Initial state and C2 pending buffers must be initialized by the caller after
    prefill, be resident on one NPU, and remain dedicated to this instance. The
    caller is responsible for preserving full CPU histories for fallback and
    establishing coherent model/graph state before and after any invocation.
    This class does not implement EOS, dynamic graph selection, draft execution,
    Engram lookup, distributed termination, or a server.
    """
    SOURCES=(2,8,14)

    def __init__(self, initial, *, ignore_eos, fixed_query_tokens, sources):
        if os.environ.get('DEEPINFER_910C_DEVICE_STATE_FIXED24')!='1':
            raise RuntimeError('fixed24 device state requires explicit opt-in')
        if ignore_eos is not True or fixed_query_tokens != 6 or tuple(sorted(sources)) != self.SOURCES:
            raise ValueError('only four slots, 6 query rows, C2 sources 2/8/14 and ignore-eos are supported')
        if not isinstance(initial,Snapshot) or set(initial.pending)!=set(self.SOURCES):
            raise ValueError('complete initial state and C2 snapshots required')
        import torch
        from deepinfer.npu_ops.greedy_prefix910c import GreedyPrefixBuffers
        from deepinfer.npu_ops.state910c import StateBuffers
        from deepinfer.npu_ops.c2_pending910c import allocate
        state=initial.state
        if state.pos.device.type!='npu':raise ValueError('device state must be on NPU')
        device=state.pos.device
        self._current=initial
        c2={}
        for source in self.SOURCES:
            c=initial.pending[source]
            value,gate,valid,status=allocate(c.value,c.gate)
            c2[source]=PendingC2(value,gate,valid,status)
        self._candidate=Snapshot(StateBuffers.allocate(device,state.tail.shape[1]),c2)
        self._greedy=GreedyPrefixBuffers.allocate(device)
        self._lengths=torch.full((4,),6,dtype=torch.int32,device=device)
        self._input=torch.empty((4,6),dtype=torch.int32,device=device)
        self._pending=False
        self._staged=False
        self._staged_draft=None
        self._poisoned=False
        self._device=device

    def _check_ready(self):
        if self._poisoned:raise RuntimeError('candidate is poisoned; fail this request')
        if self._pending:raise RuntimeError('previous transition has not settled')

    def next_graph_input(self, draft):
        """Enqueue 4x6 [current golden, five drafts]; return persistent input.

        A caller may read this output only on the same stream or after an
        explicit event dependency. Draft and output must not be modified until
        the corresponding graph and queue_transition have completed.
        """
        self._check_ready()
        import torch
        if self._staged:raise RuntimeError('staged input has not been consumed')
        if draft.device!=self._device or draft.dtype!=torch.int32 or tuple(draft.shape)!=(4,5) or not draft.is_contiguous():
            raise ValueError('draft must be contiguous int32 [4,5] on the same NPU')
        self._input[:,0].copy_(self._current.state.golden)
        self._input[:,1:].copy_(draft)
        self._staged=True
        self._staged_draft=draft
        return self._input

    @staticmethod
    def _assert_disjoint(tensors):
        spans=[]
        for value in tensors:
            if not value.is_contiguous():raise ValueError('candidate tensors must be contiguous')
            start=value.data_ptr();end=start+value.numel()*value.element_size()
            spans.append((start,end))
        spans.sort()
        if any(end>next_start for (_,end),(next_start,_) in zip(spans,spans[1:])):
            raise ValueError('cross-source or state/output storage alias')

    def queue_transition(self, draft, targets, rows, *, vocab_size, max_positions):
        """Enqueue one target sampling/state transition; do not use candidate yet.

        ``rows`` contains all three sources, each (value, gate) from this exact
        24-row graph replay. No rank may proceed to a next graph before settle.
        """
        self._check_ready()
        if not self._staged:raise RuntimeError('build input from current state first')
        if draft is not self._staged_draft:raise ValueError('draft must be the same tensor staged into the graph input')
        if set(rows)!=set(self.SOURCES):raise ValueError('missing or extra C2 source')
        tensors=[draft,targets,self._input,self._lengths]
        tensors.extend(vars(self._greedy).values())
        for snap in (self._current,self._candidate):
            tensors.extend(vars(snap.state).values())
            for source in self.SOURCES:tensors.extend(vars(snap.pending[source]).values())
        for source in self.SOURCES:tensors.extend(rows[source])
        self._assert_disjoint(tensors)
        from deepinfer.npu_ops.greedy_prefix910c import verify_greedy_prefix
        from deepinfer.npu_ops.state910c import commit_state
        from deepinfer.npu_ops.c2_pending910c import launch
        # Once enqueuing starts, any failure makes the candidate unusable: GPU
        # commands might already be in flight and cannot be rolled back here.
        self._pending=True
        try:
            current=self._current;dest=self._candidate
            g=verify_greedy_prefix(draft,targets,self._lengths,current.state.remaining,current.state.finished,self._greedy)
            commit_state(current.state.pos,current.state.remaining,current.state.golden,current.state.finished,
                current.state.tail,self._input,g.commit,g.last,dest.state,vocab_size=vocab_size,max_positions=max_positions)
            for source in self.SOURCES:
                old=current.pending[source];new=dest.pending[source];v,gates=rows[source]
                launch(old.value,old.gate,old.valid,v,gates,current.state.pos,self._greedy.commit,
                       (new.value,new.gate,new.valid,new.status),max_pos=max_positions)
        except BaseException:
            self._poisoned=True
            raise

    def pending_state_for_prelaunch(self):
        """Expose device state only for an opt-in speculative draft diagnostic.

        The queued transition and its consumer must use the same stream. The
        caller must establish all-rank eligibility, keep these buffers alive,
        and reject the whole job if any rank fails validation before the
        prelaunched output is used. This does not publish the snapshot.
        """
        if self._poisoned or not self._pending:
            raise RuntimeError('no viable pending transition for draft prelaunch')
        return self._candidate.state

    def settle(self, consensus):
        """Synchronize status/output and publish only after a global success vote.

        Every participating rank must call ``consensus(local_ok)`` in identical
        collective order; it must return the same bool on every rank. Failures in
        enqueue or collective execution require externally terminating the job;
        this class cannot recover an asymmetric EP failure. This method returns a
        copied immutable CPU packet, never a view of reusable device storage.
        """
        if self._poisoned or not self._pending:raise RuntimeError('no viable pending transition')
        if not callable(consensus):raise TypeError('global consensus callback required')
        packet=None;error=None
        try:
            statuses=[self._greedy.status.cpu().tolist(),self._candidate.state.status.cpu().tolist()]
            statuses.extend(self._candidate.pending[s].status.cpu().tolist() for s in self.SOURCES)
            if any(len(values)!=4 or any(type(x) is not int or x!=0 for x in values) for values in statuses):
                raise RuntimeError('device status is nonzero or malformed')
            accepted=self._greedy.accepted.cpu().tolist();commits=self._greedy.commit.cpu().tolist()
            emitted=self._greedy.emitted.cpu().tolist();last=self._greedy.last.cpu().tolist()
            if not all(len(x)==4 for x in (accepted,commits,emitted,last)):
                raise RuntimeError('invalid output dimensions')
            for slot in range(4):
                c=commits[slot];a=accepted[slot];row=emitted[slot]
                if (type(c) is not int or not 0<=c<=6 or type(a) is not int or not 0<=a<=5 or len(row)!=6
                    or any(type(x) is not int or x<0 for x in row[:c])
                    or any(x!=-1 for x in row[c:]) or last[slot]!=(row[c-1] if c else -1)):
                    raise RuntimeError('invalid clipped output packet')
            packet=OutputPacket(tuple(accepted),tuple(commits),tuple(tuple(row[:c]) for row,c in zip(emitted,commits)),tuple(last))
        except BaseException as exc:
            error=exc
        try:
            global_ok=consensus(error is None)
            if type(global_ok) is not bool:raise TypeError('consensus must return a bool')
        except BaseException:
            self._poisoned=True
            raise
        if not global_ok or error is not None:
            self._poisoned=True
            raise RuntimeError('fixed24 transition rejected by local status or EP consensus') from error
        self._current,self._candidate=self._candidate,self._current
        self._pending=False
        self._staged=False
        self._staged_draft=None
        return packet

    @property
    def current(self):
        """Published snapshot; unavailable during pending or poisoned states."""
        self._check_ready()
        return self._current
