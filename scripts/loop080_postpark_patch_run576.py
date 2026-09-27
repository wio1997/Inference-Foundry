#!/usr/bin/env python3
"""SHA-pinned reversible preview for one post-park Target/Draft witness.

Check is read only. Install/restore require a stopped service and an offline
controller. Never interpret diagnostic throughput as formal E2E.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASCEND = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = {
    'runner': ASCEND/'worker/model_runner_v1.py',
    'graph': ASCEND/'compilation/acl_graph.py',
    'w4a8': ASCEND/'quantization/methods/w4a8.py',
    'comm': ASCEND/'ops/fused_moe/moe_comm_method.py',
    'draft': ASCEND/'spec_decode/dspark_proposer.py',
    'dsa': ASCEND/'attention/context_parallel/dsa_cp.py',
    'runtime': ROOT/'runtime/extreme_decode.py',
    'serving': ROOT/'runtime/fixed_serving.py',
}
ORIGINAL = {
    'runner': '004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba',
    'graph': '6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6',
    'w4a8': '1cda64ddb13a88f2fe86ccdb59b6555864148e3994022631915720114d7fb2e5',
    'comm': '043f3c28518254347589f7af03793810262d0b29a9a82ca17dfd5fe5f9559f2e',
    'draft': 'e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f',
    'dsa': '27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e',
    'runtime': 'eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499',
    'serving': '137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a',
}
MARK = '# EXTREME_LOOP080_POSTPARK_RUN576'
HELPER = ROOT/'scripts/loop080_postpark_capture_run576.py'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(src: str, anchor: str, block: str, *, before=False) -> str:
    if src.count(anchor) != 1:
        raise ValueError(f'anchor count {src.count(anchor)} != 1: {anchor!r}')
    return src.replace(anchor, block + anchor if before else anchor + block)


def patch(key: str, src: str) -> str:
    if MARK in src:
        raise ValueError('source already patched')
    hook = f'{MARK}\n'
    imp = 'from scripts import loop080_postpark_capture_run576 as _postpark576\n'
    if key == 'runner':
        return once(src,
                    '                _cohort_output = FixedCohortServing(\n'
                    '                    _extreme_runtime,\n'
                    '                    initial_output_counts=[0] * _cfg.batch_size,\n'
                    '                    max_output_tokens=1024,\n'
                    '                ).run()\n',
                    f'                {hook}                {imp}'
                    '                _postpark576.bind_cohort(\n'
                    '                    len(self._extreme_served_cohorts), _extreme_req_ids,\n'
                    '                    os.getenv("RUN_TS"))\n', before=True)
    if key == 'graph':
        src = once(src, '                        output = self.runnable(*args, **kwargs)\n',
                   f'                        {hook}'
                   '                        '+imp+
                   '                        _postpark576.graph_begin(entry)\n'
                   '                        try:\n'
                   '                            output = self.runnable(*args, **kwargs)\n'
                   '                        finally:\n'
                   '                            _postpark576.graph_end()\n', before=True)
        # Remove original call, now wrapped above.
        src = once(src, '                        output = self.runnable(*args, **kwargs)\n', '', before=False) if False else src
        # The original call is the second occurrence after insertion.
        call = '                        output = self.runnable(*args, **kwargs)\n'
        if src.count(call) != 2:
            raise ValueError('Graph runnable wrapper count')
        i = src.rfind(call)
        src = src[:i] + src[i+len(call):]
        return once(src, '        entry.aclgraph.replay()\n',
                    f'        {hook}        {imp}'
                    '        _postpark576.graph_replay(entry)\n', before=True)
    if key == 'w4a8':
        return once(src, '        moe_comm_method = _EXTRA_CTX.moe_comm_method\n',
                    f'        {hook}        {imp}'
                    "        _postpark576.route(layer, topk_ids, log2phy, enable_force_load_balance,\n"
                    "                           self.dynamic_eplb, x, router_logits,\n"
                    "                           dict(w1=w1, w2=w2, w1_scale=w1_scale,\n"
                    "                                w2_scale=w2_scale, w1_scale_bias=w1_scale_bias,\n"
                    "                                w2_scale_bias=w2_scale_bias),\n"
                    "                           self.quant_type, _EXTRA_CTX.use_mega_moe)\n", before=True)
    if key == 'comm':
        return once(src,
                    '        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)\n',
                    f'        {hook}        {imp}'
                    '        _postpark576.dispatched(fused_experts_input.topk_ids, token_dispatch_output,\n'
                    '                                self.token_dispatcher, fused_experts_input.routing.expert_map)\n')
    if key == 'draft':
        return once(src, '        return num_query_total, token_indices_to_sample, cad, None\n',
                    f'        {hook}        {imp}'
                    '        _postpark576.draft_inputs(self, num_query_total, token_indices_to_sample, cad)\n',
                    before=True)
    if key == 'dsa':
        return once(src, '        if self.compress_ratio <= 1:\n            attn_output = attn_op(\n',
                    f'        {hook}        {imp}'
                    '        _postpark576.attention(layer_name, q, local_seq_lengths_query,\n'
                    '                               local_seq_lengths_key, self.compress_ratio,\n'
                    '                               full_gather_wo_a_enabled)\n', before=True)
    if key == 'runtime':
        src = once(src, '                    target_output = self.target.execute(self.state)\n',
                   f'                    {hook}                    {imp}'
                   '                    _postpark576.before_target(self)\n', before=True)
        src = once(src, '                    target_output = self.target.execute(self.state)\n',
                   f'                    {hook}                    {imp}'
                   '                    _postpark576.after_target(self)\n')
        src = once(src, '            mark("acceptance")\n',
                   f'            {hook}            {imp}'
                   '            _postpark576.after_acceptance(acceptance_output)\n')
        src = once(src, '            with self._scope("extreme::proposer"):\n',
                   f'                {hook}                {imp}'
                   '                _postpark576.draft_begin()\n')
        return once(src, '            mark("proposer")\n',
                    f'            {hook}            _postpark576.draft_end(next_draft)\n', before=True)
    src = once(src, '        cycles = 0\n',
               f'        {hook}        {imp}'
               '        _postpark576.cohort_start(self.runtime)\n', before=True)
    src = once(src, '        for slot in slots:\n            self._parked[slot] = True\n',
               f'        {hook}        {imp}'
               '        _postpark576.after_park(self, slots)\n')
    return once(src, '        counts_cpu = count_history[:cycles].cpu()\n',
                f'        {hook}        _postpark576.export(self, counts_cpu, cycles)\n')


def prepared():
    helper = HELPER.read_bytes()
    compile(helper, str(HELPER), 'exec')
    rows, originals, patched = {}, {}, {}
    for key, path in SOURCES.items():
        old = path.read_bytes()
        if sha(old) != ORIGINAL[key]:
            raise ValueError(f'{key} source SHA drift: {sha(old)}')
        new = patch(key, old.decode()).encode()
        compile(new, str(path), 'exec')
        originals[key], patched[key] = old, new
        rows[key] = dict(path=str(path), original=sha(old), patched=sha(new))
    return dict(helper_sha=sha(helper), files=rows), originals, patched


def atomic_write(path: Path, data: bytes):
    tmp = path.with_name(path.name + '.run576.tmp')
    if tmp.exists():
        raise ValueError(f'stale temporary file: {tmp}')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('check', 'install', 'restore'))
    ap.add_argument('--state-dir', type=Path)
    ap.add_argument('--record', type=Path, required=True)
    ap.add_argument('--offline-confirmed', action='store_true')
    a = ap.parse_args()
    if a.action != 'check' and not a.offline_confirmed:
        raise ValueError('offline confirmation required')
    if a.action in ('check', 'install'):
        manifest, originals, patched = prepared()
        if a.action == 'install':
            if a.state_dir is None or a.state_dir.exists():
                raise ValueError('fresh state dir required')
            a.state_dir.mkdir(parents=True)
            for key, data in originals.items():
                atomic_write(a.state_dir/f'{key}.orig', data)
            atomic_write(a.state_dir/'manifest.json',
                         (json.dumps(manifest, indent=2)+'\n').encode())
            try:
                for key, path in SOURCES.items():
                    if path.read_bytes() != originals[key]:
                        raise ValueError(f'{key} changed during install')
                    atomic_write(path, patched[key])
            except BaseException:
                for key, path in SOURCES.items():
                    if sha(path.read_bytes()) == manifest['files'][key]['patched']:
                        atomic_write(path, originals[key])
                raise
    else:
        if a.state_dir is None:
            raise ValueError('state dir required')
        manifest = json.loads((a.state_dir/'manifest.json').read_text())
        if set(manifest['files']) != set(SOURCES):
            raise ValueError('restore manifest source set drift')
        if manifest['helper_sha'] != sha(HELPER.read_bytes()):
            raise ValueError('helper changed since install')
        for key, row in manifest['files'].items():
            if str(SOURCES[key]) != row['path'] or row['original'] != ORIGINAL[key]:
                raise ValueError('source identity mismatch')
            if sha(SOURCES[key].read_bytes()) not in (row['patched'],row['original']):
                raise ValueError(f'{key} installed source drift')
            if sha((a.state_dir/f'{key}.orig').read_bytes()) != row['original']:
                raise ValueError(f'{key} backup drift')
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) != ORIGINAL[key]:
                atomic_write(path, (a.state_dir/f'{key}.orig').read_bytes())
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(dict(action=a.action, **manifest), indent=2)+'\n')
    print(json.dumps(dict(action=a.action, files=len(manifest['files']))))


if __name__ == '__main__':
    main()
