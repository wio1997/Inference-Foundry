"""Rank-local NativeServingRunner integration for the isolated Worker entry."""

from dataclasses import replace
from types import SimpleNamespace as NS
import time

import numpy as np
import torch
import torch.distributed as dist

from deepinfer.engine.dspark910c import target_tokens

from .scheduled_worker910c import check_native_prompt, forward_native_prompt
from deepinfer.worker.execution_resources910c import ExistingW8A8Resources


class ScheduledNativeModelRunner(ExistingW8A8Resources):
    def __init__(self, runner, target, controller, *, driver, groups, registry, scratch_token):
        super().__init__(runner, registry)
        self.target = target
        self.controller = controller
        self.driver = driver
        self.tp_group, self.control_group = groups
        self.registry = registry
        self.scratch_token = scratch_token
        self.step = None
        self.local = None
        self.driver_steps = {}
        self.admission = None
        self.trace = None
        self.phase_marks=[]
        self.device_timings=[]





    def destroy_cuda_graphs(self):
        self.target.close()
        super().destroy_cuda_graphs()
        self.local = self.step = None
        self.driver_steps.clear()
        self.worker = None

    def mark(self, number, phase):
        self.phase_marks.append((number,phase,time.monotonic_ns()))
        if self.trace is not None:
            self.trace.write(f"{time.monotonic_ns()} {number} {phase}\n")

    def project_step(self, step, metadata):
        if step.metadata.is_prompt or self.target.pipeline is not None:
            return step
        if self.worker.prev_step_state is not None:
            raise RuntimeError("cannot seed a target from unsettled Scheduler drafts")
        width = self.model.dspark.nextn
        vocab = self.model.cfg["vocab_size"]
        pad = (vocab - 1,) * width
        drafts = [pad] * step.metadata.num_slots
        for row, meta in zip(step.metadata.rows, metadata, strict=True):
            if row.mtp_read_slot:
                # With the local Worker drained, the host row holds the
                # committed proposals for the original MTP pool read slot.
                tokens = tuple(meta.last_next_n_token)
                if len(tokens) != width or any(not 0 <= t < vocab for t in tokens):
                    raise ValueError("invalid committed Scheduler drafts")
                drafts[row.slot] = tokens
        return replace(step, seed_drafts=tuple(drafts))

    @torch.inference_mode()
    def prepare_step(self, step):
        if step.metadata.is_prompt:
            if self.admission is not None:
                return self.admission.prepare_step(step)
            check_native_prompt(self.runner, step)
            return False
        if self.admission is not None:
            self.admission.prepare_step(step)
        if self.target.pipeline is None:
            return True
        self.local = self.target.prepare_decode(step,
            self.target.requested_graph(step) if self.target.multigraph else 24)
        return False

    @torch.inference_mode()
    def activate_step(self, step):
        if step.metadata.is_prompt:
            self.admission.activate_step(step)
            return
        if self.worker.prev_step_state is not None or step.seed_drafts is None:
            raise RuntimeError("decode activation requires committed Scheduler drafts")
        self.target.seed(**self.controller.decode_seed(step), proposals=step.seed_drafts)
        if not self.target.multigraph:
            self.local = self.target.prepare_decode(step)

    @property
    def multigraph(self):
        return self.target.multigraph

    def requested_graph(self, step):
        return self.target.requested_graph(step)

    @property
    def prepared_graph_size(self):
        return 0 if self.local is None else self.local.frame.tokens

    def prepare_graph(self, step, capacity):
        if self.local is not None and self.local.frame.tokens!=capacity:
            self.target.pipeline.discard_local(self.local)
            self.local=None
        if self.local is None:
            self.local=self.target.prepare_decode(step,capacity)

    def bind_step(self, step, metadata, outputs):
        self.controller.bind_step(step)
        if self.driver is not None:
            self.driver_steps[step.number] = (metadata, outputs)

    def correct_seqlens_qk(self, metadata, is_prompt, total_q):
        prev = self.worker.prev_step_state
        if is_prompt or prev is None or prev.is_placeholder:
            return None
        self.mark(self.step.number, "correct-qk-enter")
        result = self.target.correct_qk(metadata, prev.seq_metadata_list, prev.sampler_output.target, total_q)
        self.mark(self.step.number, "correct-qk-exit")
        return result

    def _placeholder_seq_metadata(self, is_prompt):
        return NS(seq_id=-1, seqlen_q=1, is_placeholder=True)

    def execute_main_model(self, metadata, gpu_cache, *, metadata_zos=None, total_q=None, scheduler_outputs=None):
        step = self.step
        self.mark(step.number, f"forward-enter prompt={step.metadata.is_prompt} rows={len(step.metadata.rows)}")
        if step.metadata.is_prompt:
            logits = forward_native_prompt(self.runner, step, scratch_token=self.scratch_token,
                                           bounded_keep=0 if getattr(self.runner,'_scheduled_official_yoco',False) else self.controller.bounded_keep(step))
            ticket = None
        else:
            if getattr(self.model,"eplb_runtime",None) is not None:
                from deepinfer.engine.eplb_model910c import bind_decode
                bind_decode(self.model,step,metadata,metadata_zos,driver=self.driver is not None,
                            root=self.runner.root,tp_group=self.runner.tp_group)
            self.model.mc2_global_bs = 3 * dist.get_world_size()
            if self.multigraph and self.local.frame.tokens==8:
                # The original Worker supplies corrected Q on its TP driver.
                # Broadcast only four integer lengths, without a CPU readback.
                lengths=torch.ones(4,dtype=torch.int32,device=self.model.device)
                if self.driver is not None and not step.metadata.is_placeholder:
                    q=(metadata_zos.seqlens_q[:len(step.metadata.rows)] if metadata_zos is not None else
                       torch.tensor([m.seqlen_q for m in metadata],dtype=torch.int32,device=self.model.device))
                    slots=torch.tensor([r.slot for r in step.metadata.rows],dtype=torch.int64,device=self.model.device)
                    lengths.index_copy_(0,slots,q.to(torch.int32))
                dist.broadcast(lengths,src=self.runner.root,group=self.runner.tp_group)
                error=None
                try:
                    self.target.refine_decode(self.local,lengths)
                except Exception as exc:
                    error=str(exc)
                failure=torch.tensor([int(error is not None)],dtype=torch.int32)
                dist.all_reduce(failure,group=self.control_group)
                if failure.item():
                    raise RuntimeError(f'incompatible corrected graph input: {error}')
            self.local.frame.forward_event_timing = True
            ticket, logits = self.target.forward_decode(step, self.local)
            self.registry.set("device_forward", ticket.frame.last_forward_events)
            self.local = None
        self.mark(step.number, "forward-exit")
        return NS(graph_size=0 if ticket is None else 7 if ticket.frame.tokens==8 else ticket.frame.tokens,
                  input_metadata=(step, ticket), model_output=logits)

    def _broadcast_tp(self, value):
        box = [value]
        dist.broadcast_object_list(box, src=self.runner.root, group=self.tp_group, device=torch.device("cpu"))
        return box[0]

    def _verify_tp(self, error, phase):
        flag = torch.tensor([int(error is not None)], dtype=torch.int32)
        dist.all_reduce(flag, group=self.tp_group)
        if flag.item():
            raise RuntimeError(f"native scheduled {phase} failed: {error}")

    def execute_sample(self, metadata, input_metadata, movable, gpu_cache):
        step, ticket = input_metadata
        logits = movable.move()
        self.mark(step.number, "sample-enter")
        if ticket is not None:
            q = self._broadcast_tp(tuple(m.seqlen_q for m in metadata)
                                   if self.driver is not None and not step.metadata.is_placeholder else ())
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            value = self.target.sample_decode(step, ticket, logits, q)
            end.record()
            self.registry.set("device_sample", (begin, end))
            self.mark(step.number, "sample-decode-exit")
            result=NS(step=step.number, target=value, physical_slots=(0, 1, 2, 3))
            if getattr(self.model,"eplb_runtime",None) is not None:
                self.model.eplb_runtime.step(False)
            return result
        outputs = target_tokens(self.runner, logits) if step.chunks else ()
        n = len(step.chunks)
        if len(outputs) != n:
            raise RuntimeError("native prompt logits differ from original row count")
        pad = self.model.cfg["vocab_size"] - 1
        self.mark(step.number, "sample-prompt-exit")
        result=NS(step=step.number, target=None, physical_slots=tuple(c.slot for c in step.chunks),
                  tokens=torch.tensor([t if c.is_last else -1 for c, t in zip(step.chunks, outputs)],
                                      dtype=torch.int32),
                  accepted=torch.empty(0, dtype=torch.int32),
                  drafts=torch.full((n, 5), pad, dtype=torch.int32), confidence=None)
        if getattr(self.model,"eplb_runtime",None) is not None:
            self.model.eplb_runtime.step(True)
        return result

    def post_process(self, metadata, sample, is_prompt):
        runtime=getattr(self.model,"eplb_runtime",None)
        if runtime is not None:
            sample.moe_load_balance_ratio=runtime.manager.take_moe_load_balance_ratio()
        error = None
        rows = ()
        stop = None
        step = self.controller.step_for(sample.step)
        self.mark(step.number, "post-enter")
        try:
            packed, accepted = self.target.read_sampler(sample.target) if sample.target is not None else (sample, ())
            if self.driver is not None:
                cpp_meta, outputs = self.driver_steps.pop(sample.step)
                rows = self.driver.commit_sample(NS(scheduler_outputs=outputs, sampler_output=packed,
                    seq_metadata_list=cpp_meta, is_placeholder=step.metadata.is_placeholder))
                stop = self.controller.original_stopping(step, outputs.scheduled_seqs, rows)
        except Exception as exc:
            error = str(exc)
        self._verify_tp(error, "original Scheduler commit")
        rows, stop = self._broadcast_tp((rows, stop))
        self.mark(step.number, "original-post-exit")

        try:
            if sample.target is not None:
                self.target.commit(sample.target, accepted, rows, remaining=stop.remaining, finished=stop.finished)
        except Exception as exc:
            error = str(exc)
        self._verify_tp(error, "native target/C2 settle")
        self.controller.publish_committed(step, rows, stop)
        self.mark(step.number, "post-exit")

    def post_already_committed(self, state):
        # Rank-local model post issued the C++ commit once before applying its
        # authoritative physical envelope. The Worker still owns step order.
        if state.forward_events is not None and state.sample_events is not None:
            forward_ms=state.forward_events[0].elapsed_time(state.forward_events[1])
            sample_ms=state.sample_events[0].elapsed_time(state.sample_events[1])
            self.driver.scheduler.update_mtp_runtime(
                len(state.seq_metadata_list), state.graph_size, forward_ms, sample_ms)
            self.device_timings.append((state.sampler_output.step,state.graph_size,forward_ms,sample_ms))
        return np.empty(0, dtype=np.uint64)
