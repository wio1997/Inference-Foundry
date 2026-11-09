"""Drive existing 910C target/DFlash/C2 kernels from original Scheduler rows."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from deepinfer.engine.dspark910c import target_tokens
from deepinfer.engine.async_draft_state910c import AsyncDraftState
from deepinfer.model_executor.models.zos_device_step910c import DeviceTargetPipeline
from .official_qk_reference910c import (
    compute_seqlens_k_zos_ref,
    compute_seqlens_q_by_confidence_rates_ref,
)
from .scheduled_target910c import pack_decode_sample, physical_commits
from .zos_draft_packet910c import DraftHostPacket


@dataclass(frozen=True)
class TargetSample:
    step: object
    ticket: object
    logical_q: tuple[int, ...]
    logical_confidence: torch.Tensor
    logical_accepted: torch.Tensor


class SampleInputStaging:
    """Own integer inputs until the original Worker has posted their sample.

    The Worker synchronizes sample_end_evt before post_process/commit. That
    existing completion point permits the next sample to rewrite pinned input
    storage; this adapter adds no event or device synchronization.
    """

    def __init__(self, device, *, slots=4, pin_memory=True):
        self.slots = slots
        self.host_positions = torch.empty(slots, dtype=torch.int64, pin_memory=pin_memory)
        self.host_lengths = torch.empty(slots, dtype=torch.int32, pin_memory=pin_memory)
        self.device_positions = torch.empty(slots, dtype=torch.int64, device=device)
        self.pending = None
        self.submissions = 0

    def stage(self, rows, logical_q, ticket):
        if self.pending is not None:
            raise RuntimeError("sample input staging requires the previous sample's commit")
        lengths = [1] * self.slots
        positions = []
        for row, q in zip(rows, logical_q, strict=True):
            if not 0 <= row.slot < self.slots or row.slot in positions or not 1 <= q <= 6:
                raise ValueError("invalid physical sample input row")
            positions.append(row.slot)
            lengths[row.slot] = q
        count = len(positions)
        self.host_positions.numpy()[:count] = positions
        self.host_lengths.numpy()[:] = lengths
        view = self.device_positions[:count]
        view.copy_(self.host_positions[:count], non_blocking=True)
        self.pending = ticket
        self.submissions += 1
        return view

    def queue_lengths(self, destination):
        if self.pending is None:
            raise RuntimeError("sample inputs were not staged")
        destination.copy_(self.host_lengths, non_blocking=True)

    def release(self, ticket):
        if self.pending is not ticket:
            raise RuntimeError("sample input commit does not own the pending storage")
        self.pending = None


class ScheduledNativeTarget:
    """One fixed-four-slot graph owner on a 910C Worker stream."""

    def __init__(self, runner, engine, *, debug_c2=False, async_sample_inputs=False, context_view=None, multigraph=False):
        self.runner = runner
        self.model = runner.model
        self.engine = engine
        self.debug_c2 = debug_c2
        self.context_view = context_view
        self.state = None
        self.pipeline = None
        self.previous = None
        self.draft_packet = None
        self.async_sample_inputs = async_sample_inputs
        self.sample_inputs = None
        self.multigraph = multigraph
        self.graph_records = []

    def seed(self, golden, remaining, finished, *, eos_token_id, ignore_eos, proposals):
        if self.pipeline is not None:
            raise RuntimeError("close the old target generation before seeding")
        model = self.model
        if self.context_view is not None:
            self.context_view.prepare_decode(finished)
        positions = model.next_positions[:4]
        # The CANN fixed graph expects a real C2 span even for padding rows.
        minimum = 2 * model.cfg["index_topk"]
        if len(positions) != 4 or any(not minimum <= p <= model.max_tokens - 11 for p in positions):
            raise ValueError("fixed target requires four initialized long-context slots")
        if len(proposals) != 4 or any(len(row) != model.dspark.nextn for row in proposals):
            raise ValueError("seed requires the fixed physical draft layout")
        # Startup timing has no real causal context. Preserve the repaired
        # short/full graph check on the first initialized request context.
        model.dspark_graphs.profile(golden,proposals)
        proposals = torch.tensor(proposals, dtype=torch.int64, device=model.device)
        frame = model._verify_graph
        self.state = AsyncDraftState(model, golden, remaining, finished,
                                     eos_token_id=eos_token_id, ignore_eos=ignore_eos)
        graphs = model.dspark_graphs
        self.pipeline = DeviceTargetPipeline(frame, debug_c2=self.debug_c2,
            small_frame=graphs.frames[4] if self.multigraph else None,
            compact_frame=graphs.frames[7] if self.multigraph else None)
        self.proposals = proposals
        self.draft_packet = DraftHostPacket(model.device, proposals.dtype, torch.float32)
        if self.async_sample_inputs:
            self.sample_inputs = SampleInputStaging(model.device)

    def requested_graph(self, step):
        if self.state is None:
            raise RuntimeError('graph selection requires the active original slot generation')
        if self.state.eos_enabled:
            return 24
        lengths = step.metadata.decode_query_lengths
        total = sum(lengths)
        return 4 if total == 4 else 8 if total <= 7 else 24

    def prepare_decode(self, step, capacity=24):
        if step.metadata.is_prompt or self.pipeline is None:
            raise RuntimeError("fixed target requires an authorized decode and initialized graph")
        if capacity not in (4,8,24) or (self.state.eos_enabled and capacity!=24):
            raise ValueError('incompatible graph capacity or ordinary EOS generation')
        layout = None
        if capacity==8:
            from deepinfer.engine.zos_small_graph_layout910c import SmallTargetLayout
            layout=SmallTargetLayout(step.metadata.decode_query_lengths,capacity=8)
        local=self.pipeline.stage_local(self.state,self.proposals,previous=self.previous,
            small4=capacity==4,layout=layout)
        if self.multigraph:
            local.graph_record=dict(step=step.number,logical_budget=sum(r.q for r in step.metadata.rows),
                initial_lengths=step.metadata.decode_query_lengths,physical_rows=capacity,
                curve_key=7 if capacity==8 else capacity,ordinary_eos=self.state.eos_enabled)
        return local

    def refine_decode(self, local, lengths):
        self.pipeline.refine_compact_inputs(local,lengths)

    def forward_decode(self, step, local):
        ticket = self.pipeline.prepare(local, step.number, collectives_authorized=True)
        logits=self.pipeline.submit(ticket)
        if self.multigraph:self.graph_records.append(local.graph_record)
        return ticket, logits

    def sample_decode(self, step, ticket, logits, logical_q):
        rows = step.metadata.rows
        lengths = [1] * 4
        for row, q in zip(rows, logical_q, strict=True):
            if not 1 <= q <= 6:
                raise ValueError("original Q exceeds fixed verification width")
            lengths[row.slot] = q
        state, model = self.state, self.model
        positions = (self.sample_inputs.stage(rows, logical_q, ticket)
                     if self.sample_inputs is not None else
                     torch.tensor([r.slot for r in rows], dtype=torch.int64, device=model.device))

        def queue():
            capacity=ticket.frame.tokens
            pred=target_tokens(self.runner,logits,defer_host=True).reshape(capacity).contiguous()
            if capacity!=24:
                from deepinfer.engine.zos_small_graph_layout910c import SmallTargetLayout
                layout=SmallTargetLayout(tuple(lengths),capacity=capacity)
                if capacity==8:
                    ticket.local.event.synchronize()
                    if ticket.local.corrected_status.tolist()!=[0]:
                        raise RuntimeError('original device Q does not fit the agreed compact graph')
                    ticket.local.layout=layout
                pred=layout.pack_predictions(pred)
                self.graph_records[-1]['sample_lengths']=tuple(lengths)
            state.target_tokens.queue(pred)
            if self.sample_inputs is not None:
                self.sample_inputs.queue_lengths(state.lengths)
            else:
                state.lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device=model.device))
            golden, starts = state.queue(ticket.local.proposals, pred)
            self.proposals, confidence = model.dspark.propose(golden, starts, list(range(4)))
            self.draft_packet.queue(self.proposals, confidence)
            state.queue_c2(ticket.frame,compact_cu=ticket.frame.cu_query if capacity==8 else None)
            return TargetSample(step, ticket, tuple(logical_q), confidence.index_select(0, positions),
                                state.greedy.accepted.index_select(0, positions))

        sample = self.pipeline.queue_sample(ticket, state, queue)
        self.previous = ticket
        return sample

    def read_sampler(self, sample):
        """Consume only the oldest packet, after the current forward if present."""
        pipe, ticket = self.pipeline, sample.ticket
        drain = len(pipe.queue) == 1
        _, held_chunks = pipe.host_inputs(ticket, drain=drain)
        held_target = pipe.host_targets(ticket, drain=drain)
        accepted = self.state.host[8:12].clone()
        drafts, confidence = self.draft_packet.wait()
        packed = pack_decode_sample(sample.step, sample.logical_q,
            proposals=torch.tensor([list(row[1:])+[0]*(6-len(row)) for row in held_chunks], dtype=torch.int32),
            targets=torch.tensor(held_target, dtype=torch.int32).reshape(4, 6),
            accepted_counts=accepted, next_drafts=torch.tensor(drafts, dtype=torch.int32),
            confidence=torch.tensor(confidence, dtype=torch.float32))
        return packed, tuple(accepted.tolist())

    def commit(self, sample, physical_accepted, rows, *, remaining, finished):
        counts, emitted = physical_commits(sample.step, rows)
        pipe = self.pipeline
        pipe.commit_oldest(sample.ticket, physical_accepted, counts, emitted,
                           drain=len(pipe.queue) == 1, remaining=remaining, finished=finished)
        if self.sample_inputs is not None:
            self.sample_inputs.release(sample.ticket)
        if self.previous is sample.ticket:
            self.previous = None

    def close(self):
        if self.pipeline is not None:
            if self.sample_inputs is not None and self.sample_inputs.pending is not None:
                raise RuntimeError("close requires committed sample input storage")
            if self.draft_packet.pending:
                raise RuntimeError("close requires settled draft readback")
            self.pipeline.close()
            self.pipeline = self.state = self.previous = self.draft_packet = None
            self.sample_inputs = None
        if self.context_view is not None:
            self.context_view.restore()

    def sample_input_stats(self):
        staging = self.sample_inputs
        return dict(enabled=self.async_sample_inputs,
                    submissions=0 if staging is None else staging.submissions,
                    pinned=False if staging is None else
                    staging.host_positions.is_pinned() and staging.host_lengths.is_pinned())

    def correct_qk(self, metadata, previous_metadata, previous_sample, total_q):
        """Official C++ seq_id remap and pinned CPU buffers; NPU Torch Q/K math."""
        remap_cpu, host_q, host_k, overlap = self.engine.prepare_zos_inputs(
            metadata, previous_metadata, len(metadata), 5)
        if not overlap:
            return None
        device = self.model.device
        remap = remap_cpu.to(device, non_blocking=True)
        q = host_q.to(device, non_blocking=True)
        k = host_k.to(device, non_blocking=True)
        compute_seqlens_q_by_confidence_rates_ref(total_q, remap, previous_sample.logical_confidence, q)
        compute_seqlens_k_zos_ref(k, q, remap, previous_sample.logical_accepted, nextn=5)
        host_q.copy_(q, non_blocking=True)
        event = torch.npu.Event()
        event.record()
        return SimpleNamespace(metadata_zos=SimpleNamespace(remap_cpu=remap_cpu, remap=remap,
            seqlens_q=q, seqlens_k=k), seqlens_q_cpu=host_q, d2h_evt=event)
