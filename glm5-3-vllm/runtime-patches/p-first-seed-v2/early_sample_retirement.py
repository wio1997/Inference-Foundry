"""Keep the native KV/CoW fence tied to all-worker GPU retirement."""
import functools
import json
import os
from pathlib import Path
import re


def maybe_abort_test(self, output):
    root = getattr(self, '_glm53_retirement_root', None)
    if root is None or (root / 'disabled').exists() or not output.glm53_gpu_early_sample:
        return
    command = root / 'abort-test-once.json'
    if not command.exists():
        return
    value = json.loads(command.read_text())
    assert set(value) == {'client_request_id'}
    needle = value['client_request_id']
    assert isinstance(needle, str) and re.fullmatch(r'earlypubabort-[0-9a-f]{32}', needle)
    matched = [rid for rid in output.req_ids if needle in rid and rid in self.requests]
    if not matched:
        return
    assert len(matched) == 1
    request = self.requests[matched[0]]
    assert request.last_sched_seq > self.processed_step_seq
    before = len(self.deferred_frees)
    result = self.finish_requests(matched, self._glm53_abort_status)
    assert len(result) == 1 and result[0] is request and request.status == self._glm53_abort_status
    assert len(self.deferred_frees) > before, 'Native abort did not retain in-flight KV blocks'
    receipt = dict(client_request_id=needle, native_request_id=matched[0],
                   native_finish_requests_called=True, status='FINISHED_ABORTED',
                   issued_through=self.sched_step_seq, processed_before=self.processed_step_seq,
                   last_request_step=request.last_sched_seq, output_step=output.glm53_gpu_step_seq,
                   retired_in_reply=output.glm53_gpu_retired_through,
                   deferred_before=before, deferred_after=len(self.deferred_frees),
                   scope='One explicit native abort of the nonce-matched task test request before applying its early-output retirement ACK. This does not time the exact physical GPU conflict window.')
    dest = root / ('abort-test-receipt-' + needle + '.json')
    assert not dest.exists()
    dest.write_text(json.dumps(receipt, sort_keys=True) + '\n')
    command.unlink()


def aggregate_metadata(outputs):
    assert len(outputs) == 16 and all(o is not None for o in outputs)
    steps = [o.glm53_gpu_step_seq for o in outputs]
    retired = [o.glm53_gpu_retired_through for o in outputs]
    assert len(set(steps)) == 1 and type(steps[0]) is int and steps[0] >= 0
    assert all(type(v) is int and 0 <= v <= steps[0] for v in retired)
    assert all(o.glm53_gpu_retire_rank_count == 1 for o in outputs)
    return steps[0], min(retired), any(o.glm53_gpu_early_sample for o in outputs)


def apply_ack(self, scheduled, output):
    step, retired = output.glm53_gpu_step_seq, output.glm53_gpu_retired_through
    assert type(step) is int and step == scheduled.glm53_gpu_step_seq
    assert type(retired) is int and self.processed_step_seq <= retired <= step <= self.sched_step_seq
    assert output.glm53_gpu_retire_rank_count == 16
    previous = self.processed_step_seq
    maybe_abort_test(self, output)
    self.processed_step_seq = retired
    self._drain_deferred_frees()
    root = getattr(self, '_glm53_retirement_root', None)
    if root is not None and not (root / 'disabled').exists() and self._glm53_retirement_records < 6000:
        self._glm53_retirement_records += 1
        row = dict(kind='retirement_ack', step=step, retired=retired, previous=previous,
                   scheduled_through=self.sched_step_seq, deferred_entries=len(self.deferred_frees),
                   early_sample=bool(output.glm53_gpu_early_sample), rank_count=16,
                   nonempty=scheduled.total_num_scheduled_tokens > 0,
                   gpu_step_work=scheduled.glm53_gpu_step_has_work)
        with (root / ('scheduler-%d.jsonl' % os.getpid())).open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')


def install_scheduler(ns, root=None):
    Scheduler = ns['Scheduler']
    root = Path('/out/early-sample-publication-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    init_native, schedule_native, has_native = Scheduler.__init__, Scheduler.schedule, Scheduler.has_requests

    @functools.wraps(init_native)
    def init(self, *args, **kwargs):
        init_native(self, *args, **kwargs)
        cfg = self.vllm_config
        kv = cfg.kv_transfer_config
        assert kv is not None and kv.is_kv_consumer and not kv.is_kv_producer
        assert cfg.max_concurrent_batches == 2 and self.defer_block_free
        assert cfg.parallel_config.tensor_parallel_size == 16 and cfg.parallel_config.pipeline_parallel_size == 1
        self._glm53_retirement_split = True
        self._glm53_retirement_root = root
        self._glm53_retirement_records = 0
        self._glm53_abort_status = ns['RequestStatus'].FINISHED_ABORTED
    Scheduler.__init__ = init

    @functools.wraps(schedule_native)
    def schedule(self, *args, **kwargs):
        before = self.sched_step_seq
        output = schedule_native(self, *args, **kwargs)
        nonempty = output.total_num_scheduled_tokens > 0
        writes = bool(output.kv_cache_block_copies or output.new_block_ids_to_zero)
        assert self.sched_step_seq == before + int(nonempty)
        if not nonempty and writes:
            # CoW fences assigned by the original scheduler target this next
            # sequence. A zero-token write must also have a real completion.
            self.sched_step_seq += 1
        output.glm53_gpu_step_seq = self.sched_step_seq
        output.glm53_gpu_step_has_work = nonempty or writes
        return output
    Scheduler.schedule = schedule

    @functools.wraps(has_native)
    def has_requests(self):
        return bool(has_native(self) or self.sched_step_seq > self.processed_step_seq or self.deferred_frees)
    Scheduler.has_requests = has_requests
    Scheduler._glm53_apply_retirement_ack = apply_ack


def install_aggregator(ns):
    Aggregator = ns['KVOutputAggregator']
    native = Aggregator.aggregate

    @functools.wraps(native)
    def aggregate(self, outputs, output_rank=0):
        assert len(outputs) == 16
        if all(output is None for output in outputs):
            # Generative execute_model defers its result to sample_tokens.
            # This is not a data-ready or retirement acknowledgement.
            output = native(self, outputs, output_rank)
            assert output is None
            return output
        step, retired, early = aggregate_metadata(outputs)
        output = native(self, outputs, output_rank)
        assert output is not None
        output.glm53_gpu_step_seq = step
        output.glm53_gpu_retired_through = retired
        output.glm53_gpu_early_sample = early
        output.glm53_gpu_retire_rank_count = 16
        return output
    Aggregator.aggregate = aggregate
