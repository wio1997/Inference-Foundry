#!/usr/bin/env python3
"""Source/archived-evidence qualification and same-C8 whole delivery pricing."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zlib

sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from price_vocab_selection_fusion import records
from price_attention_send_epilogue import ENDPOINTS
from analyze_final_producer_entry import latest_dp_change

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):return dict(n=len(v),min=min(v),median=statistics.median(v),max=max(v))

def main():
    identity=read(OUT/'embedding-context-execution-identity.json')
    assert len(identity['sources'])==15
    for r in identity['sources']:assert sha(r['local_path'])==r['runtime_sha256']
    receipt=read(OUT/'context-parent-receipt.json')
    assert sha(receipt['raw_path'])==receipt['raw_sha256'] and not receipt['unqualified']
    raw=json.loads(zlib.decompress(Path(receipt['raw_path']).read_bytes()))
    waves,hist,schedule,_,_=records();offsets=hist['clock_contract']['offset_upper_ns']
    by={(w['wave'],p['rank']):w for p in raw for w in p['windows'] if w['active']}
    assert len(by)==3624 and len(raw)==16 and receipt['windows']==3696
    costs={};local=defaultdict(list);groups=[]
    for wave in waves:
        dp=[]
        for d in [0,1]:
            ix=list(range(d*8,d*8+8))
            if not all((wave,i) in by for i in ix):continue
            rows=[by[wave,i] for i in ix];o=[[offsets[i][j] for j in ix] for i in ix]
            n,={w['selected_rows'] for w in rows};assert n==sum(waves[wave][ix[0]]['scheduled'].values())
            def bound(start,end):return latest_dp_change([end(w) for w in rows],[start(w) for w in rows],o)
            bounds=dict(
                whole_parent=bound(lambda w:w['parent_start_ns'],lambda w:w['parent_end_ns']),
                main_GEMM_gather_RMS=bound(lambda w:w['main_task'][1],lambda w:w['main_norm_task'][2]),
                aux_selection_through_pre_context=bound(lambda w:w['parent_start_ns'],lambda w:w['first_context_ns']))
            group=dict(wave=wave,DP=d,rows=n,kind=rows[0]['kind'],bounds_ms=bounds,
                main_tasks=[w['main_task'][0] for w in rows],source_route=rows[0]['route'])
            groups.append(group);dp.append(bounds)
            for k in bounds:local[n,k].append(bounds[k][1])
        if dp:costs[wave]={k:max(v[k][1] for v in dp) for k in dp[0]}
    assert len(groups)==453 and len(costs)==227
    endpoints=read(ENDPOINTS)['endpoint_ms'];client=schedule['client']
    def price(case,rho=1.,added=0.):
        result={}
        for metric,base in endpoints.items():
            gains=[];counts=[]
            for c in client:
                deadline=c['end_ns'] if metric=='makespan' else c['nonempty_chunk_ns'][0]
                selected=[w for w in costs if min(r['entry'] for r in waves[w].values() if 'entry' in r)<=deadline]
                counts.append(len(selected));gains.append(sum(rho*costs[w][case]-added for w in selected))
            saved=max(base)-max(a-b for a,b in zip(base,gains))
            result[metric]=dict(signed_saved_ms=saved,signed_percent=100*saved/max(base),
                signed_per_request_ms=gains,eligible_waves_per_request=counts)
        return result
    gross={case:price(case) for case in next(iter(costs.values()))}
    paid=[dict(removable_exposed_fraction=rho,added_ms_per_wave=h,result=price('whole_parent',rho,h))
          for rho in [0,.25,.5,.75,1] for h in [.05,.1,.25,.5]]
    thresholds=[]
    for h in [.05,.1,.25,.5]:
        for metric in endpoints:
            lo=0.;hi=1.
            if price('whole_parent',hi,h)[metric]['signed_percent']<1:
                threshold=None
            else:
                for _ in range(45):
                    mid=(lo+hi)/2
                    if price('whole_parent',mid,h)[metric]['signed_percent']>=1:hi=mid
                    else:lo=mid
                threshold=hi
            thresholds.append(dict(added_ms_per_wave=h,metric=metric,
                required_removable_exposed_fraction_for_historical_1pct_priority=threshold))
    # Qualify exact old artifacts before reusing their narrower screens or
    # primitive counterexamples. Current-script mismatch remains explicit.
    old_names=['EMBEDDING-ENGRAM-FUSION-ESTIMATES.json','ORDINARY-EMBEDDING-FIXED-ROUTE.json',
               'DSPARK-CONTEXT-FUSION-ESTIMATES.json','DSPARK-AUX-DELIVERY.json']
    qualified=[]
    for name in old_names:
        p=BASE/'full-review'/name;d=read(p);pins=d.get('sources',d.get('input_sha256',{}));checks=[]
        for source,digest in pins.items():
            q=Path(source)
            if not q.is_absolute():
                # This older reader's read(name) resolves from full-review;
                # newer sources manifests resolve from the analysis root.
                q=(BASE/'full-review'/q) if 'input_sha256' in d else BASE/q
            if q.is_file():checks.append(dict(path=str(q),sha256=sha(q),expected=digest,match=sha(q)==digest))
            else:checks.append(dict(path=str(q),expected=digest,missing=True,match=False))
        qualified.append(dict(path=str(p),sha256=sha(p),pins=checks,
            input_pin_match=all(r['match'] for r in checks),pin_count=len(checks)))
    embed=read(BASE/'full-review'/old_names[0]);ordinary=read(BASE/'full-review'/old_names[1]);ctx=read(BASE/'full-review'/old_names[2]);aux=read(BASE/'full-review'/old_names[3])
    embedding=embed['C8_saved_ms']['clock_bounds_tuned']['embedding_named_free']
    native_receipt=BASE/'full-review'/aux['native_receipt']
    assert sha(native_receipt)==aux['input_sha256'][aux['native_receipt']]
    nr=read(native_receipt)
    assert nr['image_id']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
    current={r['runtime_path']:r['runtime_sha256'] for r in identity['sources']}
    staging=read(OUT/'staging-execution-identity.json')
    current.update({r['runtime_path']:r['runtime_sha256'] for r in staging['sources']})
    old_sources=nr['result']['source_sha256']
    assert all(current.get(k)==v for k,v in old_sources.items())
    old_native=[dict(name=r['name'],same_M=r['same_M_main_projection'],early_full_M=r['early_full_M_main_projection']) for r in nr['result']['checks']]
    assert any(r['early_full_M']['mismatches']>0 for r in old_native)
    # Ordinary entries below are existing conditional models, NOT matching new
    # host-prefix timing; the new whole-parent profile does not close that gap.
    ordinary_summaries=[]
    for mode,case in ordinary['cases'].items():
        if mode not in ['keep_external','skip_external','external_only','shared_local']:
            continue # Engram gate joint cases are not embedding-only budgets.
        for r in case.get('ordinary_requests',[]):
            estimates=r['estimates'];pct=max(max(x['response_pct']) for x in estimates.values())
            ordinary_summaries.append(dict(case=mode,request=r['request'],upper_conditional_response_percent=pct))
    result=dict(status='PARTIAL_INPUT_EMBEDDING_AND_CONTEXT_PARENT_SOURCE_PRICE_REVIEW',whole_chain_complete=False,
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',source_identity='embedding-context-execution-identity.json',
        current_binding=[
            dict(entry='gpu_model_runner.py:3502-3655;model_runner_v1.py:4192-4206;model.py:1519-1552',
                finding='Text MM preprocessing computes embedding into persistent bank and passes raw IDs. FULL raw-ID capture can still contain its own lookup; eager supplied-embed path does not. Raw ID clamp/history/Engram observers survive any bank bypass.'),
            dict(entry='llm_base_proposer.py:381-390,486-560,1192-1203,2679-2697;dspark.py:249,404-413,461-503',
                finding='Load disables proposer MM embedding for this text-only draft API. DSpark directly embeds IDs once; wrapper ignoring inputs_embeds is not a duplicate active lookup. Own embedding flags/weights retain target and draft as distinct values.'),
            dict(entry='model_runner_v1.py:2063-2106;llm_base_proposer.py:1053-1084,2388-2444',
                finding='Padded context selector is unchanged INT32 arange/QSL extent, while accepted count separately produces rejection and sample-row selectors. No-spec prefix and nonpadded late selector are separate; K0 returns before main.'),
            dict(entry='dspark_proposer.py:229-338;dflash_proposer.py:301-320;dspark.py:281-311;dsa_v1.py:428-525',
                finding='Three distinct context Wkv/norm/RoPE stores consume main output. Corrected prefix P=seq_len-rejected, Q5 queries see [max(0,P-128),P+5). All query stores precede own attention. Physical per-group slots, future prefix/CoW, queues and c=0 still gate row retirement.'),
            dict(entry='models/common/ops/sequence_parallel.py:25-99;model.py:1541-1558;dspark.py:249-274',
                finding='Target shards before HC4 repeat; draft repeats then shards/pads. No-pad sp_shard clones, padded branch returns a view. Virtual broadcast cannot reassociate the original HC/GEMM reductions or change residual ownership.')],
        current_C8=dict(receipt='context-parent-receipt.json',rank_windows=3696,active_rank_windows=3624,
            active_TP8_groups=453,active_waves=227,groups=groups,
            scope='Start at original aux IndexCheck triplet or Cat through main GEMM/feature AG/RMS, input/DP metadata, three context panels, draft token embedding/pad/premix to first draft HcPre. Earlier target aux mean/SP gathers and three draft bodies/scores are outside.',
            qualification='Bounded immutable original C8 SQL with exact saved first-context/first-HcPre IDs and timestamps, same stream/model, unique main/Cat/RMS and exact row count. No tensor-address/ownership proof or new profiling. Active/idle paths explicitly separated.',
            gross_favorable_endpoint_sensitivity=gross,paid_whole_parent_scenarios=paid,
            thresholds=thresholds,shape_group_upper_ms={f'{n}:{k}':stats(v) for (n,k),v in local.items()},
            pricing_contract='Each wave chooses independently most favorable active DP eight-rank frontier under saved clocks; propagate all waves whose earliest host entry precedes each client deadline. Necessary work, idle-peer limits, exposure and replacement all relaxed in gross. Signed rho/h scenarios are sensitivities, not speedup or hardware bounds. Do not add this parent to metadata-host/early-output/terminal budgets.'),
        archived_qualification=qualified,
        archived_narrow_embedding=dict(reference_ms=embed['C8_reference_ms'],gross_ms=embedding,
            qualification='Old named-only component model. Check exact input_pin_match before reuse; excludes broader host publisher and large HC arithmetic.',
            ordinary_conditional_models=ordinary_summaries),
        archived_main_geometry=dict(cases=old_native,native_receipt=dict(path=str(native_receipt),sha256=sha(native_receipt),
            original_scope=nr['scope'],current_source_hashes_match=len(old_sources),
            qualification='Native receipt independently hash-verified and all7 implementation source pins match current identity. Old derived aux-report has one stale flow-index pin; no old broad source-coverage claim reused.'),
            gate='Same-M samples only. Full-M early projection followed by different-M late selection has explicit 4/2-bit mismatch examples; generic recipe stopped. This is an old primitive counterexample, not a new native test or all frozen shape validation.'),
        candidates=[
            dict(name='Draft MM duplicate-embedding deletion',state='已停止',reason='Load-time API gate disables this upstream path; no active duplicate to delete'),
            dict(name='Target/draft same-ID embedding value sharing',state='已停止',reason='Distinct loaded embeddings; common IDs do not authorize common values'),
            dict(name='Standalone named embedding/SP delivery including guarded external-lookup deletion',state='已停止',reason='Qualified old original-schedule whole named embedding free C8 .399% TTFT/.501% batch, six ordinary embedding-only conditional variants below .620%. This does not close broader host publisher/HC arithmetic or admit an alias-unsafe patch'),
            dict(name='Standalone main GEMM/feature gather/RMS erasure or early-main-only route',state='已停止',reason='Even whole original stage free is narrow low priority; it does not prune the wider host/context parent. No new per-shape numerical probe for this budget'),
            dict(name='Generic early full-M projection then changed-M selector',state='已停止',reason='Archived primitive bit counterexamples; same-M case must preserve original geometry'),
            dict(name='Aux/main plus metadata/context/input combined delivery',state='待确认',reason='Whole-parent favorable1.55% TTFT/1.46% batch is material enough to retain, but required computation/communication/guards and net cost unproven; overlaps existing metadata-host candidate, not independent added gain'),
            dict(name='Demand/physical-slot-aware context main/cache producer',state='待确认',reason='Requires all three actual cache readers, prefix/CoW lifetime and original-M/tile primitive bits; row count alone is not execution time')],
        next_action='Continue actual three-layer DSpark decoder/TP-EP schedule and large submission/megakernel parent; retain combined host/context net-cost and physical-cache/numeric gaps. Do not re-probe named embedding, small cache writers or failed generic changed-M main.',
        inputs=[dict(path=str(p),sha256=sha(p)) for p in [OUT/'embedding-context-execution-identity.json',OUT/'context-parent-receipt.json',Path(ENDPOINTS),Path(__file__)]],
        new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    (OUT/'embedding-context-frontier.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(gross=gross,half_rho_h01=next(r for r in paid if r['removable_exposed_fraction']==.5 and r['added_ms_per_wave']==.1),
        old_qualification=[dict(path=r['path'],count=r['pin_count'],match=r['input_pin_match'],mismatches=sum(not p['match'] for p in r['pins'])) for r in qualified],
        new_NPU_work=0)))

if __name__=='__main__':main()
