#!/usr/bin/env python3
"""No-NPU contract gate for ghost publication before a live continuation."""
import json
import importlib.util
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
MODULE = ROOT / 'runtime/ghost_publication.py'
spec = importlib.util.spec_from_file_location('ghost_publication', MODULE)
assert spec is not None and spec.loader is not None
ghost_publication = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = ghost_publication
spec.loader.exec_module(ghost_publication)
GhostPublicationLedger = ghost_publication.GhostPublicationLedger

OUT = ROOT / 'evidence/20260926_loop074_refill/run344/protocol_gate.json'
names = [f'cohortA-{i}' for i in range(12)]
checks = []

def check(name, func):
    func()
    checks.append(name)

def expect_raises(exc, func):
    try:
        func()
    except exc:
        return
    raise AssertionError(f'expected {exc.__name__}')

def normal():
    ledger = GhostPublicationLedger(generation=7, request_ids=names, max_tokens=1024)
    prior = {names[0]: (), names[1]: tuple(range(17)), names[2]: (51,)}
    partial = {}
    for i in range(3):
        rid = names[i]
        seg = ledger.publish(request_id=rid, client_index=i,
            prior_token_ids=prior[rid], runtime_token_ids=tuple(range(1024)))
        assert len(seg.prior_token_ids) + len(seg.new_token_ids) == 1024
        partial[rid] = seg.new_token_ids
    assert [len(partial[names[i]]) for i in range(3)] == [1024, 1007, 1023]
    assert len(ledger.early_request_ids) == 3
    # The original twelve-request bulk settlement occurs exactly once. The
    # external output layer suppresses its three already-published rows.
    final = {rid: partial.get(rid, tuple(range(1024))) for rid in names}
    suppress = ledger.settle_once(final)
    assert suppress == frozenset(names[:3])
    expect_raises(RuntimeError, lambda: ledger.settle_once(final))
    expect_raises(RuntimeError, lambda: ledger.publish(request_id=names[3],
        client_index=3, prior_token_ids=(), runtime_token_ids=tuple(range(1024))))

def duplicate_and_foreign():
    ledger = GhostPublicationLedger(generation=8, request_ids=names, max_tokens=1024)
    data = dict(request_id=names[0], client_index=0, prior_token_ids=(),
                runtime_token_ids=tuple(range(1024)))
    ledger.publish(**data)
    expect_raises(ValueError, lambda: ledger.publish(**data))
    expect_raises(ValueError, lambda: ledger.publish(**{**data, 'request_id':'other'}))
    expect_raises(ValueError, lambda: ledger.settle_once({names[0]:()}))

def divergence():
    ledger = GhostPublicationLedger(generation=9, request_ids=names, max_tokens=1024)
    ledger.publish(request_id=names[0], client_index=0,
        prior_token_ids=(99,), runtime_token_ids=tuple(range(1024)))
    final = {rid: tuple(range(1024)) for rid in names}
    expect_raises(ValueError, lambda: ledger.settle_once(final))
    expect_raises(RuntimeError, lambda: ledger.settle_once(final))

def failed_continuation():
    ledger = GhostPublicationLedger(generation=10, request_ids=names, max_tokens=1024)
    ledger.publish(request_id=names[0], client_index=0,
        prior_token_ids=(), runtime_token_ids=tuple(range(1024)))
    ledger.fail_continuation()
    expect_raises(RuntimeError, lambda: ledger.publish(request_id=names[1],
        client_index=1, prior_token_ids=(), runtime_token_ids=tuple(range(1024))))
    expect_raises(RuntimeError, lambda: ledger.settle_once(
        {rid:tuple(range(1024)) for rid in names}))

def insufficient_or_invalid():
    ledger = GhostPublicationLedger(generation=11, request_ids=names, max_tokens=1024)
    expect_raises(ValueError, lambda: ledger.publish(request_id=names[0],
        client_index=0, prior_token_ids=tuple(range(18)),
        runtime_token_ids=tuple(range(1000))))
    expect_raises(ValueError, lambda: ledger.publish(request_id=names[0],
        client_index=0, prior_token_ids=tuple(range(1024)),
        runtime_token_ids=tuple(range(1024))))
    expect_raises(ValueError, lambda: ledger.publish(request_id=names[0],
        client_index=0, prior_token_ids=(), runtime_token_ids=(-1,)*1024))

for name, fn in [
    ('p0_p17_p1_partial3_final12_once', normal),
    ('duplicate_foreign_missing_final_rejected', duplicate_and_foreign),
    ('final_divergence_invalidates', divergence),
    ('continuation_failure_invalidates', failed_continuation),
    ('short_complete_negative_rejected', insufficient_or_invalid),
]:
    check(name, fn)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'status':'pass','checks':checks,
    'scope':'pure protocol ledger only; no worker continuation, Scheduler transaction, KV retention, HTTP arrival or Product E2E'}, indent=2)+'\n')
print(json.dumps({'status':'pass','checks':checks}))
