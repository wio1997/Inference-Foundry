"""Copy official Scheduler outputs before the next schedule reuses storage."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StepRow:
    seq_id: int
    slot: int
    epoch: int
    q: int
    scheduled_k: int
    block_table: tuple[int, ...]
    state_id: int
    mtp_read_slot: int
    mtp_write_slot: int


@dataclass(frozen=True)
class NativeStepMetadata:
    is_prompt: bool
    rows: tuple[StepRow, ...]
    clone_blocks: tuple[tuple[int, int], ...]
    clone_states: tuple[tuple[int, int], ...]
    zero_states: tuple[int, ...]
    num_slots: int

    @property
    def is_placeholder(self):
        return not self.rows

    @property
    def decode_query_lengths(self):
        if self.is_prompt:
            raise ValueError("prompt rows do not use the fixed target graph")
        lengths = [1] * self.num_slots
        for row in self.rows:
            lengths[row.slot] = row.q
        return tuple(lengths)


def capture_step(metadata, outputs, slot_bindings, *, num_slots):
    """Project original logical rows through a driver-owned physical slot map."""
    if type(num_slots) is not int or num_slots < 1:
        raise ValueError("invalid native slot count")
    by_id = {}
    for slot, binding in slot_bindings.items():
        if type(slot) is not int or not 0 <= slot < num_slots:
            raise ValueError("binding cannot refer to the private scratch slot")
        if binding.seq_id in by_id or binding.seq_id < 0 or binding.epoch < 0:
            raise ValueError("invalid or duplicate native binding")
        by_id[binding.seq_id] = slot, binding.epoch

    ids = tuple(m.seq_id for m in metadata)
    if ids != tuple(seq.seq_id for seq in outputs.scheduled_seqs) or len(set(ids)) != len(ids):
        raise ValueError("metadata and official scheduled order disagree")
    rows = []
    for m in metadata:
        if m.is_placeholder or m.is_prompt != outputs.prompt_run or not 1 <= m.seqlen_q <= m.seqlen_k:
            raise ValueError("invalid official logical row")
        if m.seq_id not in by_id:
            raise ValueError("official request has no reserved native slot")
        slot, epoch = by_id[m.seq_id]
        rows.append(StepRow(m.seq_id, slot, epoch, m.seqlen_q, m.seqlen_k, tuple(m.block_table),
                            m.state_id, m.mtp_info_read_slot, m.mtp_info_write_slot))
    return NativeStepMetadata(outputs.prompt_run, tuple(rows),
                              tuple(tuple(pair) for pair in outputs.clone_blocks.tolist()),
                              tuple(tuple(pair) for pair in outputs.clone_states.tolist()),
                              tuple(outputs.zero_states.tolist()), num_slots)
