#!/usr/bin/env python3
"""Supersede generic-pool interpretation with actual frozen dispatch facts."""
from hashlib import sha256
import ast
import itertools
import json
from pathlib import Path

OUT=Path('audit/execution-search')


def main():
    ident=json.loads((OUT/'actual-tokenizer-execution-identity.json').read_text())
    by={r['runtime_path']:r for r in ident['sources']}
    for r in ident['sources']:assert sha256(Path(r['local_path']).read_bytes()).hexdigest()==r['runtime_sha256']
    tr=by['/vllm-workspace/vllm/vllm/tokenizers/deepseek_v41.py'];m=ast.parse(Path(tr['local_path']).read_text())
    factory=next(n for n in m.body if isinstance(n,ast.FunctionDef) and n.name=='get_deepseek_v41_tokenizer')
    inner=next(n for n in factory.body if isinstance(n,ast.ClassDef));override=[n.name for n in inner.body if isinstance(n,ast.FunctionDef)]
    assert set(override)=={'apply_chat_template','num_special_tokens_to_add','get_added_vocab','__reduce__'}
    rr=by['/vllm-workspace/vllm/vllm/renderers/deepseek_v4.py'];m=ast.parse(Path(rr['local_path']).read_text());renderer=next(n for n in m.body if isinstance(n,ast.ClassDef))
    assert ast.unparse(renderer.bases[0])=='BaseRenderer[DeepseekV4Tokenizer]'
    methods=[n.name for n in renderer.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    assert 'maybe_make_thread_pool' not in Path(rr['local_path']).read_text()
    scalar=json.loads((OUT/'actual-tokenizer-unpooled-replay.json').read_text());assert scalar['native_deep_copies_in_pool']==0 and scalar['same_generic_replay_ID_hashes']
    flags=json.loads((OUT/'actual-tokenizer-runtime-flags.json').read_text());assert all(r['VLLM_USE_FASTOKENS'] is None for r in flags['process_flags'])
    native=json.loads((OUT/'shared-prefix-tokenizer-point.json').read_text());handoff=json.loads((OUT/'prefix-encoding-handoff-point.json').read_text())
    fifo=json.loads((OUT/'input-renderer-frontier.json').read_text())['FIFO']['jobs'];ttfts=json.loads((OUT/'frontend-output-frontier.json').read_text())['clock']['TTFTs_ms']
    scenarios=[];perms=list(itertools.permutations(range(8)))
    for warm in [False,True]:
        for rho,h in [(.5,.5),(.9,.5)]:
            ends=[]
            for i,j in enumerate(fifo):
                gap=j['start_ms']-(fifo[i-1]['end_ms'] if i else 0)
                start=max(j['submit_ms'],(ends[-1] if i else 0)+gap)
                fraction=rho if warm or i>0 else 0
                insertion=0 if warm or i>0 else handoff['median_insertion_ms']
                ends.append(start+j['parent_ms']-fraction*j['native_ms']+h+insertion)
            deltas=[j['end_ms']-e for j,e in zip(fifo,ends)]
            credits=[max(ttfts)-max(t-deltas[k] for t,k in zip(ttfts,p)) for p in perms]
            scenarios.append(dict(cache_warm=warm,native_fraction_removed_on_hits=rho,new_ms_per_job=h,
              first_miss_retains_native=not warm,first_miss_insertion_ms=0 if warm else handoff['median_insertion_ms'],
              FIFO_completion_acceleration_ms=deltas,last_input_acceleration_ms=deltas[-1],
              fixed_downstream_lag_P90_best_pairing_ms=max(credits),fixed_downstream_lag_P90_worst_pairing_ms=min(credits)))
    out=dict(status='ACTUAL_DSV41_COMPLETION_DISPATCH_CORRECTION_AND_LIMITED_POINTS',whole_chain_complete=False,
      explicit_mode='deepseek_v41',actual_tokenizer='TokenizersBackend.from_pretrained -> get_deepseek_v41_tokenizer -> get_cached_tokenizer',
      actual_renderer='RendererRegistry deepseek_v41 -> DeepseekV4Renderer -> BaseRenderer, NOT HfRenderer',
      tokenizer_overrides=override,renderer_methods=methods,actual_renderer_auto_copy_pool=False,
      offset_contract='BaseRenderer._can_produce_offsets=False inherited; current request offsetsFalse, generic HfRenderer offset capability does not apply.',
      correction=dict(withdrawn='Frozen actual renderer already has HfRenderer independent tokenizer pool; thread-count flag alone is safe parallelism',
        retained='FIFO timestamps/native duration, all raw ID points and guarded numeric-prefix proof remain valid in declared primitive domains',
        previous_generic_pool_and_actual_wrapper_pool_replays='Counterfactual pool/factory capability only, superseded by actual unpooled replay; no service changes were made'),
      scalar_replay=dict(path='actual-tokenizer-unpooled-replay.json',type=scalar['pooled_type'],pool_copies=0,all8_ID_hashes_equal=True,
        scope='Actual V41 factory+cached backend+Base scalar methods over direct configured generic backend. Full from_pretrained bootstrap, loaded serving object/wire not asserted.'),
      fastokens=dict(source_default=False,process_initial_exec_flags=[r['VLLM_USE_FASTOKENS'] for r in flags['process_flags']],
        runtime_scope=flags['scope'],profile='Native tokenizers.Tokenizer.encode_batch and DecodeStream steps observed; config/source support native branch, not in-memory backend attestation.'),
      source_pins=ident['sources'],
      decisions=[dict(state='已确认可实现',scope='Guarded known stable-prefix native IDs and first Encoding handoff primitives only',
          proof='shared-prefix-tokenizer-proof.md',native_point='shared-prefix-tokenizer-point.json',handoff_point='prefix-encoding-handoff-point.json',
          native_cases=native['counts']['cases'],ID_elements=native['counts']['ID_elements'],mismatches=native['counts']['mismatches'],
          independent_output_lists=True,source_cold_miss='Retain original full Encoding/IDs; extract prefix tuple by character token_to_chars binary search, no extra prefix encoding',
          local_diagnostic=handoff['local_CPU'],median_insertion_ms=handoff['median_insertion_ms']),
        dict(state='待确认',scope='Online prefix cache integration, worthy of development',
          gaps=['General online prefix discovery/insertion, byte-LRU/epoch/actual tokenizer fingerprint and invalidation',
            'Original from_pretrained resolved assets and serving-object/wire equality before enabling cache',
            'Real exposure, CPU affinity/contention, cold misses and six ordinary workload eligibility; no request speedup admitted']),
        dict(state='已停止',scope='Only increasing renderer-num-workers on actual mode and relying on nonexistent auto-pool',
          reason='Selected DeepseekV4Renderer inherits BaseRenderer directly; HF encoding mutates backend config. Explicit copy-pool/scheduling rewrite remains a separate pending option.'),
        dict(state='已停止',scope='Arbitrary prefix/suffix concat or UTF8-byte offset used as Encoding character cut',
          reason='Actual naive-cut counterexamples; char117067 != UTF8byte117187, wrong-cut control rejected.')],
      conditional_FIFO_scenarios=scenarios,
      price_contract='Existing profiled queue/jobs/native durations retained. Cold first miss retains ALL original native work, pays insertion, only7 later hits credited. Warm credits8. rho/h are scenarios; measured insertion comes from separate isolated CPU cohort and is illustrative, not a matching service cost. Pairing range assumes downstream lag fixed and is NOT a causal request bound/expected speedup. All resource/scheduling interactions with parallelism and old seed budget remain non-additive.',
      next='Startup kernel_warmup/workspace helper bodies and inherited actual MM budget; selected text scalar source dispatch reviewed in bounded scope, bootstrap/cache integration pending.',
      no_new_serving_or_model_or_NPU_admission=True,service_changes=[],new_profile_collection=0,
      import_scope='Earlier rejected AutoTokenizer attempts imported model/NPU packages; no explicit model/device computation requested, import-time device initialization not observed. Successful generic/actual scalar replays use configured backend and assert torch_npu absent; native points assert Torch absent.',
      cache_inventory=dict(path='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/cpu-input-renderer-cache',observed_exists=False,files=[]))
    (OUT/'actual-tokenizer-dispatch-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    report=OUT/'输入renderer与公共前缀交付判断.md';s=report.read_text();banner='> **2026-11-10 更正：实际mode为deepseek_v41，renderer是DeepseekV4Renderer→BaseRenderer，不经过HfRenderer，自动副本池不存在。撤回本文初稿“固定服务已有独立池、调线程数即安全并行”的解释。通用pool源码/点证只属反事实能力；时间/ID原件保留。以[实际分派更正](实际tokenizer分派与prefix原语判断.md)为准。**\n\n'
    if banner not in s:
        h,t=s.split('\n',1);report.write_text(h+'\n\n'+banner+t.lstrip('\n'))
    print(json.dumps(dict(actual_renderer_pool=False,scalar_IDs='PASS unpooled',new_primitive_scope='Limited known-prefix/producer only',cold_half_h05=scenarios[0],whole_chain_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
