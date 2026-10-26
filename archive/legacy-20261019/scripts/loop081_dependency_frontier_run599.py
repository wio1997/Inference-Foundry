#!/usr/bin/env python3
"""Pin source and evidence for one Target-tail / Draft-context DAG split.

Read-only structural audit. No readiness timing, overlap or TPS conclusion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
HIST = Path('/data/wio/performance_history/dsv4f-w4a8-ascend-logs')
OUT = ROOT / 'evidence/20260928_loop081_bound/run599'
SOURCES = {
    'target_adapter': ROOT/'runtime/target_adapter.py',
    'target_handoff': ROOT/'bootstrap/vllm_target_handoff.py',
    'decode': ROOT/'runtime/extreme_decode.py',
    'state': ROOT/'runtime/fixed_decode.py',
    'acceptance': ROOT/'runtime/fixed_acceptance.py',
    'greedy': ROOT/'runtime/greedy_accept.py',
    'dspark_handoff': ROOT/'bootstrap/vllm_dspark_handoff.py',
    'base_proposer': ASC/'spec_decode/llm_base_proposer.py',
    'dspark_proposer': ASC/'spec_decode/dspark_proposer.py',
    'dflash_proposer': ASC/'spec_decode/dflash_proposer.py',
    'dspark_model': ASC/'models/deepseek_v4_dspark.py',
    'dspark_inputs': ASC/'ops/triton/spec_decode/utils.py',
}
EVIDENCE = {
    'basis_admission': ROOT/'evidence/20260928_loop081_bound/run597/live/b/basis_admission.json',
    'work_ledger': ROOT/'evidence/20260928_loop081_bound/run598/summary.json',
    'historical_R09': HIST/'rounds/R09_draft_forward_decomposition/REPORT.md',
    'historical_R21': HIST/'rounds/R21_dsa_cp_compressor_wkv_overlap/REPORT.md',
    'current_Run579': ROOT/'evidence/20260928_loop080_bound/run579/findings.md',
    'current_Run580': ROOT/'evidence/20260928_loop080_bound/run580/findings.md',
}
PREDICATES = {
    'target_adapter': ['model_output = self.binding.forward(',
                       'logits = self.binding.compute_logits(sample_hidden)'],
    'target_handoff': ['output = self.model(',
                       'output = _gather_output(output)',
                       'return output'],
    'decode': ['target_output = self.target.execute(self.state)',
               'acceptance_output = self.acceptance.execute(',
               'self._state_machine.advance_state(acceptance_output)',
               'next_draft = self.proposer.execute('],
    'state': ['def advance_state(self, acceptance: AcceptanceOutput)',
              'state.last_sampled_tokens.copy_(',
              'state.num_computed_tokens.add_(state.num_sampled)'],
    'acceptance': ['predicted = target.logits.argmax(dim=-1)',
                   'accepted, counts = greedy_accept('],
    'greedy': ['leading_match = torch.cumprod(',
               'recovery = torch.gather(target_argmax',
               'final_token = torch.where(all_accepted, bonus_tokens, recovery)'],
    'dspark_handoff': ['def _refresh_draft_context_slots(',
                      'positions = state.target_positions.view(',
                      'target_hidden_states = hidden[token_indices]',
                      'next_token_ids=state.last_sampled_tokens',
                      'acceptance.num_sampled,'],
    'base_proposer': ['target_hidden_states = self.model.combine_hidden_states(',
                      'self.set_inputs_first_pass(',
                      'self.build_model_inputs_first_pass(num_input_tokens, self._context_slot_mapping_buffers)'],
    'dspark_proposer': ['self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states',
                       'num_query_total = batch_size * self.num_query_per_req'],
    'dflash_proposer': ['self.model.precompute_and_store_context_kv('],
    'dspark_model': ['def combine_hidden_states(',
                     'def precompute_and_store_context_kv(',
                     'for layer_idx, layer in enumerate(self.layers.values()):'],
    'dspark_inputs': ['tl.store(out_context_positions_ptr + ctx_pos_idx, pos)',
                      'tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)',
                      'num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)',
                      'bonus_token = tl.load(next_token_ids_ptr + req_idx)'],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if (OUT/'source_gate.json').exists():
        raise ValueError('run599 output exists')
    assert (HIST/'.git').exists()
    source_hashes = {}
    for key, path in SOURCES.items():
        text = path.read_text()
        for fragment in PREDICATES[key]:
            assert fragment in text, (key, fragment)
        source_hashes[key] = dict(path=str(path), sha256=digest(path),
                                  predicates=PREDICATES[key])
    evidence_hashes = {key:dict(path=str(path),sha256=digest(path))
                       for key,path in EVIDENCE.items()}
    admission = json.loads(EVIDENCE['basis_admission'].read_text())
    ledger = json.loads(EVIDENCE['work_ledger'].read_text())
    assert admission['status']=='diagnostic_compact_basis_all8_admitted'
    assert ledger['status']=='diagnostic_W0_current_execution_class_ledger'
    assert admission['measured_rank0_totals']['cycles']==ledger['totals']['cycles']==1200
    assert ledger['totals']['accepted_staged_tokens']==49523
    assert ledger['totals']['target8_active_class_rows']==99976
    result = dict(status='structural_dependency_split_source_gate',
        source_sha256=source_hashes,evidence_sha256=evidence_hashes,
        target_forward_cut='after binding.forward including conditional FlashComm gather; Python return is not device completion',
        context_branch='Target aux + old target positions + context slot mapping -> combine -> three-layer context KV store',
        query_branch='Target logits -> greedy acceptance; raw acceptance.num_sampled -> rejected-count/query geometry, while masked-or-preserved state.last_sampled_tokens -> query seed; old Target positions/seq_len also feed query geometry',
        join='context store and query preparation must both precede Draft query consumer',
        active_prefix_decision_witness_incidents=49523,
        active_prefix_scope='not minimal logits or fresh Target model work; parked raw counts and DSpark context are outside',
        inferred_legal_overlap=False,measured_node_ready=False,
        measured_mixed_capacity=False,finite_scheduling_bound_s=None,
        formal_current_tps=571.681)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Run599 source gate PASS',len(source_hashes),'sources',len(evidence_hashes),'evidence')


if __name__=='__main__':main()
