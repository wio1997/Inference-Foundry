"""Bridge an original Scheduler and Worker to the 910C rank-local model runner."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from .native_step_metadata910c import NativeStepMetadata, capture_step
from .scheduled_prefill910c import PromptChunk, prompt_chunks


@dataclass(frozen=True)
class ScheduledStep:
    number: int
    metadata: NativeStepMetadata
    chunks: tuple[PromptChunk, ...]
    retired_ids: tuple[int, ...] = ()
    seed_drafts: tuple[tuple[int, ...], ...] | None = None
    admissions: tuple[tuple[int, int, int, int], ...] = ()


def create_scheduler_groups(tp_size):
    """Every rank calls this after the resident default group is initialized."""
    world = dist.get_world_size()
    if tp_size < 1 or world % tp_size:
        raise ValueError("TP groups must divide the EP world")
    # The original ZOS engine also uses a CPU DP group for its host decisions.
    roots = list(range(0, world, tp_size))
    control_group = dist.new_group(list(range(world)), backend="gloo")
    driver_group = dist.new_group(roots, backend="gloo")
    rank = dist.get_rank()
    for root in roots:
        group = dist.new_group(list(range(root, root + tp_size)), backend="gloo")
        if root <= rank < root + tp_size:
            local = group
    return (driver_group if rank in roots else None), local, control_group


class ScheduledDriver:
    """Keep pybind metadata on its owner until the original Worker posts it."""

    def __init__(self, engine, scheduler, pool, *, nextn, on_commit, on_finished, on_schedule=None):
        self.engine = engine
        self.scheduler = scheduler
        self.pool = pool
        self.nextn = nextn
        self.on_commit = on_commit
        self.on_finished = on_finished
        self.on_schedule = on_schedule
        self._release = []

    def schedule(self, number, bindings, sequences, positions, *, slots, chunk_size):
        meta, outputs = self.scheduler.schedule(self.pool)
        admissions = ()
        if self.on_schedule is not None:
            positions, admissions = self.on_schedule(meta, outputs, bindings, positions)
        step = capture_step(meta, outputs, bindings, num_slots=slots)
        if step.clone_blocks or step.clone_states or step.zero_states or outputs.ignored_seqs:
            raise ValueError("unsupported Scheduler cache action or ignored request")
        chunks = prompt_chunks(step, sequences, positions, chunk_size=chunk_size) if step.is_prompt else ()
        retired = tuple(seq.seq_id for seq in outputs.finished_seqs)
        return ScheduledStep(number, step, chunks, retired, admissions=admissions), meta, outputs

    def commit_sample(self, state):
        outputs = state.scheduler_outputs
        if state.is_placeholder:
            rows = []
        else:
            sample = state.sampler_output
            seqs = outputs.scheduled_seqs
            before = [seq.get_len() for seq in seqs]
            self.engine.process_MTPInfo(outputs, self.nextn > 0, sample.drafts, sample.confidence)
            _, counts = self.scheduler.process_outputs(outputs, sample.accepted, sample.tokens, None)
            rows = []
            for seq, start, count in zip(seqs, before, counts.tolist(), strict=True):
                tokens = ()
                if count:
                    drafts = seq.last_accepted_token
                    tokens = tuple(drafts[:count])
                    if len(tokens) < count:
                        tokens += (seq.data.get_last_token_id(),)
                if len(tokens) != count or seq.get_len() - start != count:
                    raise RuntimeError("original Scheduler commit disagrees with token data")
                rows.append((seq.seq_id, tokens, count, seq.num_cached_tokens))
        if not state.is_placeholder:
            self._release.extend(state.seq_metadata_list)
        return rows

    def post(self, state):
        rows = self.commit_sample(state)
        self.on_commit(state.sampler_output.step, rows)
        return np.empty(0, dtype=np.uint64)

    def release_metadata(self):
        if self._release:
            self.pool.release(self._release)
            self._release.clear()

    def pre_sampler_sync(self, outputs):
        self.release_metadata()
        for seq in outputs.finished_seqs:
            self.scheduler.free_seq(seq)
            self.on_finished(seq)


def run_scheduled_step(worker, model_runner, number, *, driver, tp_group, control_group, root, schedule):
    """Only the driver has live C++ references. Followers consume frozen rows."""
    if worker.is_driver_worker != (driver is not None):
        raise ValueError("Scheduler may run on the TP driver only")
    mark = getattr(model_runner, "mark", lambda *_: None)
    mark(number, "schedule-enter")
    metadata, outputs, step = None, None, None
    error = None
    if driver is not None:
        try:
            step, metadata, outputs = driver.schedule(number, **schedule)
            if hasattr(model_runner, "project_step"):
                step = model_runner.project_step(step, metadata)
        except Exception as exc:
            error = str(exc)
    message = [step, error]
    dist.broadcast_object_list(message, src=root, group=tp_group, device=torch.device("cpu"))
    step, error = message
    mark(number, f"project-exit error={error}")

    # Reach the same stop decision before any rank enters the model's EP.
    activation = False
    if error is None:
        try:
            if step is None or step.number != number:
                raise ValueError("stale TP Scheduler projection")
            if hasattr(model_runner, "receive_step"):
                model_runner.receive_step(step)
            mark(number, "prepare-enter")
            activation = bool(model_runner.prepare_step(step))
            mark(number, f"prepare-exit activation={activation}")
        except Exception as exc:
            error = str(exc)
    multigraph=getattr(model_runner,"multigraph",False)
    values=[int(error is not None),0 if step is None else int(step.metadata.is_prompt),int(activation)]
    if multigraph:
        prepared=model_runner.prepared_graph_size
        values.extend((int(prepared==8),int(prepared==24)))
    status = torch.tensor(values,dtype=torch.int32)
    dist.all_reduce(status, group=control_group)
    if status[0].item() or status[1].item() not in (0, dist.get_world_size()):
        raise RuntimeError(f"incompatible distributed Scheduler step {number}: {error}")

    if status[2].item():
        if activation:
            try:
                mark(number, "activate-enter")
                model_runner.activate_step(step)
                mark(number, "activate-exit")
            except Exception as exc:
                error = str(exc)
        failure = torch.tensor([int(error is not None)], dtype=torch.int32)
        dist.all_reduce(failure, group=control_group)
        if failure.item():
            raise RuntimeError(f"incompatible physical slot activation {number}: {error}")

    if multigraph and not step.metadata.is_prompt:
        # Fused EP graphs use one physical shape. Retain original Q/budgets;
        # only local padding and the corresponding backend graph are agreed.
        if status[2].item():
            # New slot generations need their real EOS state after activation.
            requested=24
            try:
                requested=model_runner.requested_graph(step)
                if requested not in (4,8,24):raise ValueError('unknown physical graph request')
            except Exception as exc:
                error=str(exc)
            vote=torch.tensor([int(error is not None),int(requested==8),int(requested==24)],dtype=torch.int32)
            dist.all_reduce(vote,group=control_group)
            if vote[0].item():raise RuntimeError(f'incompatible graph plan {number}: {error}')
            capacity=24 if vote[2].item() else 8 if vote[1].item() else 4
            recheck=True
        else:
            # Uniform steady-state steps reuse the existing readiness vote;
            # they add no control collective before the original Worker.
            count8,count24=int(status[3].item()),int(status[4].item())
            capacity=24 if count24 else 8 if count8 else 4
            count=count24 if capacity==24 else count8 if capacity==8 else dist.get_world_size()
            recheck=count!=dist.get_world_size()
        if recheck:
            try:
                model_runner.prepare_graph(step,capacity)
            except Exception as exc:
                error=str(exc)
            failure=torch.tensor([int(error is not None)],dtype=torch.int32)
            dist.all_reduce(failure,group=control_group)
            if failure.item():raise RuntimeError(f'incompatible graph inputs {number}: {error}')

    mark(number, "pre-ep-ready")
    model_runner.step = step
    if hasattr(model_runner, "bind_step"):
        model_runner.bind_step(step, metadata, outputs)
    follower_metadata = step.metadata.rows if driver is None else metadata
    try:
        result = worker.execute_model_step(step.metadata.is_prompt, step.metadata.is_placeholder,
                                           follower_metadata, outputs)
    except Exception as exc:
        error = str(exc)
    mark(number, f"worker-exit error={error}")
    # Old-state post and explicit slot drains are TP-local. Agree at the
    # outer step boundary, where all EP ranks arrive exactly once.
    failure = torch.tensor([int(error is not None)], dtype=torch.int32)
    dist.all_reduce(failure, group=control_group)
    if failure.item():
        raise RuntimeError(f"native Worker step {number} failed: {error}")
    return result


def check_native_prompt(runner, step):
    """Check local physical capacity before any rank enters EP."""
    if not step.metadata.is_prompt:
        raise ValueError("scheduled prefill requires the prompt phase")
    model = runner.model
    chunks = step.chunks
    if any(not 0 <= c.slot < runner.slots or model.next_positions[c.slot] != c.start
           or not 0 < len(c.tokens) <= model.chunk_size for c in chunks):
        raise ValueError("native prefill cursor differs from the scheduled slice")


def forward_native_prompt(runner, step, *, scratch_token, bounded_keep):
    """Reuse NativeServingRunner's packed prefill forward for one agreed phase."""
    model = runner.model
    chunks = step.chunks
    # MC2 capacity must be identical across EP, including the placeholder DP.
    model.mc2_global_bs = ((model.chunk_size * runner.slots + runner.tp_size - 1)
                           // runner.tp_size) * runner.dp_size * runner.tp_size
    if not chunks:
        runner._reset_scratch()
    actual_chunks=[c.tokens for c in chunks] or [[scratch_token]]
    actual_slots=[c.slot for c in chunks] or [runner.slots]
    official_metadata=None
    if getattr(runner,'_scheduled_official_yoco',False):
        from deepinfer.engine.official_yoco_metadata910c import prepare_native_prefill
        bounded_keep,official_metadata=prepare_native_prefill(runner,runner._scheduled_owned,
            actual_slots,actual_chunks,runner._scheduled_hits,internal=not chunks)
    logits = model.forward_tokens(actual_chunks,actual_slots,
                                  sequence_parallel=model.sequence_parallel,
                                  bounded_keep=bounded_keep,official_yoco=official_metadata)
    if getattr(runner, '_scheduled_cold_enabled', False):
        from .cold_prefill910c import capture_native_anchors
        capture_native_anchors(runner, chunks)
    if any(model.next_positions[c.slot] != c.end for c in chunks):
        raise RuntimeError("native prefill did not consume the scheduled slice")
    return logits
