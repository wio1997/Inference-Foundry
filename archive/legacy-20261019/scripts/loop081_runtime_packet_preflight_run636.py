#!/usr/bin/env python3
"""Run636: fail-closed source gate for the next fixed-work Runtime Bound packet.

This is source/measurement design only; it does not install instrumentation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
FILES = {
    'cycle': (ROOT / 'runtime/extreme_decode.py', 'eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499'),
    'target': (ROOT / 'runtime/target_adapter.py', 'c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5'),
    'serving': (ROOT / 'runtime/fixed_serving.py', '137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a'),
    'draft': (ROOT / 'bootstrap/vllm_dspark_handoff.py', 'fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e'),
    'target_handoff': (ROOT / 'bootstrap/vllm_target_handoff.py', '2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5'),
    'runner': (ASC / 'worker/model_runner_v1.py', '004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'),
    'graph': (ASC / 'compilation/acl_graph.py', '6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6'),
}
ANCHORS = {
    'cycle': ('def step(self)', 'self.target.execute(self.state)', 'self.acceptance.execute(',
              'self._state_machine.advance_state(acceptance_output)', 'self.proposer.execute(',
              'self._commit_next_target_metadata()', 'self._launch_next_target_metadata()'),
    'target': ('model_output = self.binding.forward(', 'logits = self.binding.compute_logits(sample_hidden)',
               'aux_hidden_states=tuple(aux_hidden_states)'),
    'serving': ('result = self.runtime.step()', 'token_history[index].copy_',
                'self.runtime.invalidate_scheduled_metadata()', 'tokens_cpu = token_history[:cycles].cpu()'),
    'draft': ('self._host_copy_stream.wait_stream(current)', 'self._host_copy_event.record()',
              'self._host_copy_event.synchronize()', 'target.aux_hidden_states',
              'self.proposer._propose('),
    'target_handoff': ('output = self.model(', 'output = _gather_output(output)'),
    'runner': ('_extreme_wall_start = time.perf_counter()', '_cohort_output = FixedCohortServing(',
               'torch.npu.synchronize()', '"wall_seconds": _extreme_wall_seconds'),
    'graph': ('torch.npu.current_stream().synchronize()', 'entry.aclgraph.replay()'),
}
OUT = ROOT / 'evidence/20260928_loop081_bound/run636/runtime_packet_preflight.json'


def main():
    source = {}
    for key, (path, expected) in FILES.items():
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise AssertionError(f'installed source drift: {key}')
        body = payload.decode()
        for anchor in ANCHORS[key]:
            if anchor not in body:
                raise AssertionError(f'missing {key} anchor {anchor}')
        source[key] = {'path': str(path), 'sha256': digest,
                       'validated_anchors': list(ANCHORS[key])}
    packet = {
        'status': 'source_preflight_only_NOT_LIVE_READY',
        'contract': 'fixed DSpark7 acceptance/cycles/output/model work; 8x910B3 DP1TP8 warm48+measured48 32K->1024 c12',
        'source': source,
        'primary_branch': 'Runtime adjacent-cycle actual readiness and first-consumer chain, conditional on low-overhead complete admission',
        'priority_evidence': {
            'run99_formal_rank0_Runtime_arithmetic_s': '69.040-69.457 per formal repeat; not a disjoint E2E phase or removable time',
            'run606_same_W0_current': 'built→allrank existing-sync wide Host envelopes sum about69.28s over four cohorts; preparation own Target NONE Host sum8.334-8.984s/rank. Neither is a causal saving budget.',
            'fallback': 'Measure Target NONE preparation if Runtime boundary cannot be certified/kept low overhead or a same-W0 sensitivity test ranks it higher.',
        },
        'minimum_Runtime_edges': [
            'selected cycle Target input/metadata Host submission and stream order; separately actual producer-ready -> first native Target read',
            'Target main hidden producer/gather completion -> logits producer completion -> acceptance first logits read',
            'Target aux producer/gather completion -> first DSpark aux pack read; no invented aux-to-logits prerequisite',
            'acceptance output -> state advance -> actual Draft query-input producers and first consumers',
            'Draft model/proposer result producer -> draft-commit copy completion -> next Target first real read',
            'acceptance counts producer -> existing side-stream Host count copy event -> next-cycle existing synchronize and mirror consumer',
            'when configured: private next-target metadata producer on schedule stream -> existing wait/commit and next target first use; preserve Host park/metadata invalidation',
            'last-cycle staged output producer -> existing CPU copy/final sync -> ModelRunner bulk output -> Scheduler/API/client publication',
        ],
        'measurement_rules': [
            'Pin actual all8 rank/cohort/request IDs, selected ordinal/cycle, acceptance/output ledger, tensor identity, stream ID and exact source mode in one W0.',
            'Record current-stream and relevant side-stream producer and existing wait/consumer events without adding any new synchronize on the measured path; preserve existing waits.',
            'Retain predecessor and successor cycles, Graph replay and HCCL task ownership; event on one stream is not a certificate of all dependent streams.',
            'Sample bounded strata after actual W0 admission; do not impose a prior Run606 acceptance trajectory as a gate for a new W0.',
            'Keep OFF/ON/OFF timing and output/cycle/acceptance comparison; only exact matched W0 can support direct cost transfer, otherwise compare distributions with declared observer uncertainty.',
            'No numerical Resource/Scheduling/Product bound or gap from source anchor presence, Host scope widths, event spacing or profiled task sums.',
        ],
        'unmet_live_gates': [
            'reversible patch plus restore hashes and all8 idle check',
            'bounded event storage and no new measured-path synchronize',
            'preflight implementation plan for HCCL/Graph/native and side-stream owner binding; post-run actual completion ownership remains evidence to acquire',
            'independent Astra High preflight, same-W0 identity/correctness admission and OFF/ON/OFF effect threshold',
        ],
        'strict_resource_floor_s': None, 'strict_scheduling_floor_s': None,
        'product_e2e_ceiling_tps': None, 'numeric_current_to_credible_limit_gap': None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(packet, indent=2) + '\n')
    print(json.dumps({'status': packet['status'], 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
