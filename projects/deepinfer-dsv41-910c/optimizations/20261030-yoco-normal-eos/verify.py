"""Check the candidate's eligibility boundary without an NPU."""
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import argparse
import hashlib


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
args = parser.parse_args()
assert hashlib.sha256(args.base.read_bytes()).hexdigest() == '24dd794197b78427fe0a730e353a8e79200c6dc79189af1be64d6597751496cd'
assert hashlib.sha256(args.candidate.read_bytes()).hexdigest() == 'f406ff043d9b34a6a9a3a7a6f43656d80a0c588b3b226d7dd3ab56c54595f508'
base = load('baseline_bounded', args.base)
candidate = load('candidate_bounded', args.candidate)
model = SimpleNamespace(tail_ffn=True, sequence_parallel=True,
    layers=[SimpleNamespace(context_parallel=True)], sp=SimpleNamespace(world=8),
    dspark=SimpleNamespace(nextn=5), cfg={'sliding_window': 128, 'index_topk': 512},
    restored_prefix_slots={0, 1, 2, 3}, next_positions=[29440, 29568, 29440, 29568])
slots = list(range(4))
chunks = [[0] * (32768 - position) for position in model.next_positions]
hits = model.next_positions.copy()


@contextmanager
def enabled(value):
    key = 'DEEPINFER_910C_YOCO_NORMAL_EOS'
    previous = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


tested = 0
for keep, flags in product((140, 256), product((False, True), repeat=4)):
    owned = [(i, {'token_ids': [0] * 32768, 'ignore_eos': flag}) for i, flag in enumerate(flags)]
    args = (model, owned, slots, chunks, hits)
    reference = base.eligible(*args, internal=False, keep=keep)
    for value in (None, '0', 'other'):
        with enabled(value):
            assert candidate.eligible(*args, internal=False, keep=keep) == reference
            tested += 1
    with enabled('1'):
        actual = candidate.eligible(*args, internal=False, keep=keep)
        assert actual == (all(flags) or not any(flags)), (keep, flags, actual)
        assert not candidate.eligible(*args, internal=True, keep=keep)
        tested += 2

owned = [(i, {'token_ids': [0] * 32768, 'ignore_eos': False}) for i in range(4)]
with enabled('1'):
    assert not candidate.eligible(model, owned, slots, chunks, [0, *hits[1:]], internal=False)
    assert not candidate.eligible(model, owned, [0, 1, 2, 4], chunks, hits, internal=False)
    shortened = [x[:] for x in chunks]
    shortened[0].pop()
    assert not candidate.eligible(model, owned, slots, shortened, hits, internal=False)
    assert not candidate.eligible(model, owned, slots, chunks, hits, internal=False, keep=139)
print('ELIGIBILITY_CONTRACT_PASS', tested, 'mixed EOS and all original geometry gates retained')
