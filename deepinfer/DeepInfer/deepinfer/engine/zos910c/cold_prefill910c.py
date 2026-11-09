"""Local cold-prefix and physical-padding interfaces; no scheduler policy."""


def can_prefill(request, snapshot, *, cold, minimum):
    if snapshot is not None and len(snapshot["ids"]) >= minimum:
        return True
    return cold and len(request["token_ids"]) >= minimum


def select_admission_dp(candidates, hint):
    """Keep the existing hot-prefix preference; zero hit can still be eligible."""
    if not 0 <= hint < len(candidates):
        raise ValueError("invalid admission DP hint")
    if any(len(row) != 2 or row[0] not in (0, 1) or row[1] < 0
           or (not row[0] and row[1]) for row in candidates):
        raise ValueError("availability and cached length differ")
    hits = [row[1] for row in candidates]
    if any(hits):
        return hint if hits[hint] else max(range(len(hits)), key=hits.__getitem__)
    if candidates[hint][0]:
        return hint
    return next((dp for dp, row in enumerate(candidates) if row[0]), None)


def bootstrap_padding(runner, prompt, *, capture, context_tokens=None):
    """Initialize real C2 addresses in the existing scratch slot on every EP rank.

    This is fixed-graph storage setup, never a Sequence or a cache hit. The
    caller agrees the need globally before entering this fixed chunk loop.
    """
    model = runner.model
    minimum = 2 * model.cfg["index_topk"]
    if len(prompt) < minimum or model.chunk_size < 1:
        raise ValueError("padding setup needs a real long-context prefix")
    length = minimum if context_tokens is None else context_tokens
    if not minimum <= length <= model.max_tokens - 11:
        raise ValueError("physical padding context exceeds backend capacity")
    prefix = tuple((list(prompt) * ((length + len(prompt) - 1) // len(prompt)))[:length])
    runner._reset_scratch()
    model.mc2_global_bs = ((model.chunk_size * runner.slots + runner.tp_size - 1)
                           // runner.tp_size) * runner.dp_size * runner.tp_size
    forwards = 0
    for start in range(0, length, model.chunk_size):
        chunk = prefix[start:start + model.chunk_size]
        model.forward_tokens([chunk], [runner.slots],
                             sequence_parallel=model.sequence_parallel)
        forwards += 1
    if model.next_positions[runner.slots] != length:
        raise RuntimeError("physical padding prefix cursor differs")
    snapshot = model.capture_prefix(runner.slots, prefix) if capture else None
    return snapshot, length, forwards


def capture_native_anchors(runner, chunks):
    """Use the already validated native cold-prefix admission policy."""
    model = runner.model
    for chunk in chunks:
        if runner._scheduled_hits[chunk.slot] or model.next_positions[chunk.slot] not in runner.prefix_points:
            continue
        owned = runner._scheduled_owned[chunk.slot]
        if owned is None or model.next_positions[chunk.slot] != chunk.end:
            raise RuntimeError("cold prefix capture has no real prompt owner")
        key = tuple(owned[1]["token_ids"][:chunk.end])
        if key not in runner.prefix_cache:
            if len(runner.prefix_cache) >= 4:
                del runner.prefix_cache[next(iter(runner.prefix_cache))]
            runner.prefix_cache[key] = model.capture_prefix(chunk.slot, key)
            runner._scheduled_cold_prefix_captures += 1


def reset_cold_context(model, slot):
    """The existing native zero-prefix reset, without changing request identity."""
    model.next_positions[slot] = 0
    model.yoco_swa_valid_start[slot] = 0
    model.restored_prefix_slots.discard(slot)
    for pending in model.pending.values():
        pending.pop(slot, None)
    if model.histories:
        model.histories[slot].reset()


class ColdContextView:
    """Project waiting cold contexts onto private valid physical decode storage.

    Real request positions remain in the Scheduler/ledger. Only an empty-output
    cold slot can borrow padding; the original Worker must drain before restore.
    """

    def __init__(self, runner, ledger):
        self.runner, self.model, self.ledger = runner, runner.model, ledger
        self.minimum = 2 * self.model.cfg["index_topk"]
        self.padding = None
        self.saved = {}
        self.bootstrap_tokens = self.bootstrap_forwards = 0
        self.swaps = []
        self.restores = 0

    def initialize(self, prompt):
        padding = getattr(self.runner, "_cold_context_padding", None)
        if padding is None:
            padding, self.bootstrap_tokens, self.bootstrap_forwards = bootstrap_padding(
                self.runner, prompt, capture=True, context_tokens=4096)
            self.runner._cold_context_padding = padding
        if not self.minimum <= len(padding["ids"]) <= self.model.max_tokens - 11:
            raise RuntimeError("private padding storage is incompatible")
        self.padding = padding

    def logical_positions(self):
        positions = list(self.model.next_positions)
        for slot, row in self.saved.items():
            positions[slot] = row["position"]
        return positions

    def prepare_decode(self, finished):
        if self.padding is None or self.saved:
            raise RuntimeError("cold context view requires initialized, restored storage")
        if getattr(self.model, "_zos_device_pipeline", None) is not None:
            raise RuntimeError("release old target before changing physical context")
        for slot, (binding, request, hit) in self.ledger.requests.items():
            if hit >= self.minimum or self.ledger.results[binding.seq_id]:
                continue
            if not finished[slot]:
                # Do not replace a live customer's context to satisfy a graph.
                continue
            position = self.model.next_positions[slot]
            if not 0 <= position <= len(request["token_ids"]):
                raise RuntimeError("waiting cold context exceeds its real prompt")
            snapshot = (self.model.capture_prefix(slot, tuple(request["token_ids"][:position]))
                        if position else None)
            self.saved[slot] = dict(binding=binding, position=position, snapshot=snapshot,
                                    restored=slot in self.model.restored_prefix_slots)
            self.model.restore_prefix(self.padding, slot)
            self.swaps.append(dict(slot=slot, seq_id=binding.seq_id, epoch=binding.epoch,
                                   logical_position=position, physical_position=len(self.padding["ids"])))

    def restore(self):
        if getattr(self.model, "_zos_device_pipeline", None) is not None:
            raise RuntimeError("cold context restore requires the target pipeline closed")
        for slot, row in self.saved.items():
            current = self.ledger.requests.get(slot)
            if current is None or current[0] != row["binding"]:
                raise RuntimeError("cold context restore has a stale physical generation")
            if row["snapshot"] is None:
                reset_cold_context(self.model, slot)
            else:
                self.model.restore_prefix(row["snapshot"], slot)
            if row["restored"]:
                self.model.restored_prefix_slots.add(slot)
            else:
                self.model.restored_prefix_slots.discard(slot)
            self.restores += 1
        self.saved.clear()

    def stats(self):
        return dict(private_padding_tokens=0 if self.padding is None else len(self.padding["ids"]),
                    swaps=list(self.swaps), restores=self.restores, pending_slots=sorted(self.saved),
                    bootstrap_tokens=self.bootstrap_tokens, bootstrap_forwards=self.bootstrap_forwards)
