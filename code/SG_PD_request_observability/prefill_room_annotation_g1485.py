"""Opt-in rank-zero Prefill annotations for a separate P/D diagnostic capture."""

from functools import wraps
from hashlib import sha256
from pathlib import Path


PREFILL_SHA256 = 'b185dcc192b765a19ac2048e204ac24cb49c09116bab7773a37f2a4925aa5bb1'
STEP_SPAN_SHA256 = 'aef800c2f362094103ff57f04231267c27d2f3d6ae6be00af348d3a1d6651340'
SEND_PREFIX = 'ds41.profile.prefill_last_kv_send '
POLL_PREFIX = 'ds41.profile.prefill_transfer_observed '


def marker(prefix, req, fake_host, abort_type):
    room = req.bootstrap_room
    if not room or req.bootstrap_host == fake_host or isinstance(req.finished_reason, abort_type):
        return None
    # Identical room serialization to the existing rank-zero Decode marker.
    fingerprint = sha256(str(room).encode()).hexdigest()[:16]
    return f'{prefix}room={fingerprint} prompt={len(req.origin_input_ids)}'


def annotate_send(original, active, record, fake_host, abort_type):
    @wraps(original)
    def scoped(self, req, last_chunk=False, end_idx=None):
        name = marker(SEND_PREFIX, req, fake_host, abort_type) if last_chunk and active() else None
        if name is None:
            return original(self, req, last_chunk=last_chunk, end_idx=end_idx)
        with record(name):
            return original(self, req, last_chunk=last_chunk, end_idx=end_idx)

    scoped._ds41_prefill_room_annotation = True
    return scoped


def annotate_poll(original, active, record, fake_host, abort_type):
    @wraps(original)
    def scoped(self, rids_to_check=None):
        result = original(self, rids_to_check=rids_to_check)
        if active():
            for req in result:
                name = marker(POLL_PREFIX, req, fake_host, abort_type)
                if name is not None:
                    with record(name):
                        pass
        return result

    scoped._ds41_prefill_room_annotation = True
    return scoped


def install(*, role):
    """Install only in the chosen Prefill diagnostic process at startup."""
    if role != 'prefill':
        raise ValueError('Prefill role required')

    import torch
    from sglang.srt.disaggregation import prefill
    from sglang.srt.model_executor import step_span_utils

    for module, pinned in ((prefill, PREFILL_SHA256), (step_span_utils, STEP_SPAN_SHA256)):
        if sha256(Path(module.__file__).read_bytes()).hexdigest() != pinned:
            raise ValueError(f'Unexpected source for {module.__name__}')

    def rank_zero_active():
        if not step_span_utils.detailed_annotations_enabled():
            return False
        ranks = prefill.get_parallel()
        return ranks.tp_rank == 0 and ranks.pp_rank == 0

    scheduler = prefill.SchedulerDisaggregationPrefillMixin
    send = scheduler.send_kv_chunk
    poll = scheduler.process_disagg_prefill_transfer_queue
    installed = [getattr(fn, '_ds41_prefill_room_annotation', False) for fn in (send, poll)]
    if all(installed):
        return
    if any(installed):
        raise ValueError('Partial Prefill room annotations already installed')
    options = (rank_zero_active, torch.profiler.record_function, prefill.FAKE_BOOTSTRAP_HOST, prefill.FINISH_ABORT)
    scheduler.send_kv_chunk = annotate_send(send, *options)
    scheduler.process_disagg_prefill_transfer_queue = annotate_poll(poll, *options)
