#!/usr/bin/env python3
"""Reproduce the bounded F01 archive review. Reads files; never accesses an NPU/server."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'records/history/continuation'


def main():
    sources = []

    def read(relative, compressed=False):
        path = HISTORY / relative
        raw = path.read_bytes()
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(gzip.decompress(raw) if compressed else raw)

    source = read('profile-first-20261031/HC-attention-fork-source-contract-v93.json')
    contract = read('profile-first-20261031/HC-operator-numerical-and-dependency-contract-v59.json')
    variants = read('profile-first-20261031/HC-attention-fork-variants-v93.json')
    profile = read('profile-first-20261031/HC-operator-profile-and-edges-v58.json.gz', compressed=True)
    bridge = read('shared-profile-analysis-v166/graph-replay-containment-focus.json')
    header_path = HISTORY / 'HC-operator-source-v58/vllm-workspace/vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h'
    header = header_path.read_text()
    sources.append({'path': str(header_path.relative_to(ROOT)), 'sha256': hashlib.sha256(header_path.read_bytes()).hexdigest()})
    model = '\n'.join(source['source'])
    checks = {
        'source_selects_external_premix_for_Y': 'tilingData->hasPreMix != 0 ? preMixLocal : mixes01ReduceLocal' in header,
        'source_Y_before_Post_and_comb_per_processed_block': header.index('ProcessPre(') < header.index('ProcessY(') < header.index('ProcessPost(') < header.index('SoftmaxFP32Perf('),
        'source_global_sync_after_first_stage': 'SyncAll<false>();' in header,
        'model_norm_after_hc_pre': model.index('self.hc_pre(') < model.index('self.input_layernorm(x)'),
        'model_current_coeffs_live_at_post_and_next_ffn_pre': 'self.hc_post(x, residual, attn_post, attn_comb)' in model and '            attn_pre,' in model,
        'archive_new_variant_not_implemented': variants['implementation'] is False and variants['native_test'] is False,
    }
    edge = profile['edge_summary']['ATTENTION_PRE_TO_NORM']
    result = {
        'candidate': 'F01', 'variant': 'HC93-EARLY-Y', 'review_date': '2026-11-03',
        'kind': 'ARCHIVED_SOURCE_AND_PROFILE_REVIEW_NOT_NATIVE_TEST',
        'status': 'PASS' if all(checks.values()) else 'REVIEW_FAILURE',
        'checks': checks, 'evidence': sources,
        'source_claim': '有外部pre_mix时，源码中Y由X和该pre_mix生成，不依赖本次新系数；当前每块Y先于post/comb写出，但下游按整体HcPre完成执行。',
        'profile_166_20261030': {
            'population': 'root-bound target graph attention HC->norm;16logicaldevices/2windows;not171',
            'pairs': edge['root_bound'], 'same_stream_all_registered_edges': edge['same_stream'] == edge['all_edges'],
            'HcPre_median_us': profile['root_bound_stage_ns']['ATTENTION_HcPre']['median'] / 1000,
            'gap_median_us': edge['gap_ns']['median'] / 1000,
            'RmsNorm_median_us': profile['root_bound_stage_ns']['ATTENTION_Norm']['median'] / 1000,
            'does_not_prove': ['Y内部就绪时刻', '全部HcPre时间可省', 'CPU不是全局瓶颈', '端到端关键路径收益'],
        },
        'profile_171_20261102': {
            'profile': bridge['profile'],
            'recorded_runtime_modes': sorted({e['identity']['runtime_mode'] for e in bridge['scope_records'] if 'runtime_mode' in e.get('identity', {})}),
            'claim': '记录有FULL图回放；不借此迁移166耗时或证明每个native HcPre的creator绑定。',
        },
        'not_established': ['171加载二进制与归档底层源码逐指令一致', '多队列任务可同时驻留执行', '新接口和事件交接可实现', '改动后数值正确性', '有效资源重叠与同条件性能收益', '独立代理复核'],
        'no_new_profile_no_server_action_no_kernel_implementation': True,
        'source_contract_notes': contract['facts'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(not all(checks.values()))


if __name__ == '__main__':
    main()
