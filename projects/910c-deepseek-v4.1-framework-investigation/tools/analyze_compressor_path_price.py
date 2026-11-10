#!/usr/bin/env python3
"""Match complete C1/C2 source publication before pricing its children."""
from pathlib import Path
from hashlib import sha256
from statistics import mean, median
import json, zlib
from analyze_o_projection_parent_price import start, dur, union
R=Path('audit/execution-search')
B=Path('/Users/wio/work/branch-fusion-audit-20261104/full-review')
def main():
    files=[R/'compressor-path-parent-primary.json',R/'compressor-path-dependencies.json',R/'swa-publication-parent-calls.json.zlib',R/'qkv-joint-delivery-price.json',R/'index-key-publication-profile.json',R/'index-key-decode-publication-profile.json',B/'C2-POOL-RING-OWNER.json',B/'CLEAN-C2-PUBLICATION-VALUE.json',B/'C2-RETENTION-HORIZON.json']
    raw=json.loads(zlib.decompress(files[2].read_bytes())); windows=json.loads(files[3].read_text()); profiles=[]
    suffix=['RmsNorm','MatMulV2','RmsNorm','InplacePartialRotaryMul','DynamicQuant','ScatterNdUpdateSk','Cast','ScatterNdUpdateSk','InplacePartialRotaryMul','ScatterNdUpdateSk']
    for src in raw:
        by={r['csv_row']:r['data'] for r in src['selected']}; heads=[i for i,r in by.items() if r['Type']=='MatMulV2' and r['Input Shapes'].strip('"').split(';')==['24,512','128,512']]
        assert len(heads)==48; chains=[]
        for j,h in enumerate(heads):
            assert [by[h+k]['Type'] for k in range(-1,9)]==suffix
            c2=by[h-3]['Type']=='_pool_kernel'; layer=[2,8,14,20][j%4]; assert c2==(layer!=20)
            if c2:
                offsets=[-8,-7,-5,-4,-3,-2]+list(range(-1,9)); root=h-8
                assert [by[h+k]['Type'] for k in [-8,-7,-5,-4,-3,-2]]==['Cast','MatMulV3','MatMulV3','ZerosLike','_pool_kernel','_cache_update_kernel']
                gemm=[h-7,h-5,h]; pool_ring=list(range(h-4,h-1))
            else:
                offsets=[-3]+list(range(-1,9));root=h-3;gemm=[h-3,h];pool_ring=[]
                assert by[root]['Type']=='MatMulV2' and by[root]['Input Shapes'].strip('"').split(';')==['24,5120','512,5120']
            ids=[h+k for k in offsets]; assert len({by[i]['Stream ID'] for i in ids})==1
            assert [by[h+k]['Type'] for k in range(9,13)]==['Cast','VllmAscendQuantLightningIndexerV2','_prepare_indexer_indices_kernel','SparseFlashMla']
            total=float(union([by[i] for i in ids])); mandatory_gemm=sum(float(dur(by[i])) for i in gemm); nongemm=total-mandatory_gemm
            end=float(start(by[h+8])+dur(by[h+8])); begin=float(start(by[root]));reader=float(start(by[h+12])+dur(by[h+12]))
            chains.append(dict(layer=layer,ratio=2 if c2 else 1,root_csv_row=root,Wk_csv_row=h,cache_done_csv_row=h+8,source_native_us=total,source_span_us=end-begin,retained_GEMM_us=mandatory_gemm,non_GEMM_native_us=nongemm,suffix10_native_us=float(union([by[i] for i in range(h-1,h+9)])),pool_ring3_native_us=float(union([by[i] for i in pool_ring])) if pool_ring else 0.,source_to_selected_SMLA_end_span_us=reader-begin,source_plus_selected_QLI_SMLA_native_us=float(union([by[i] for i in ids+list(range(h+9,h+13))])),paid_scenarios=[dict(removable_non_GEMM_fraction=rho,added_us_per_source=added,net_us=rho*nongemm-added) for rho,added in [(0.25,5.),(0.5,10.),(0.5,20.),(0.75,20.)]]))
        w=next(x['window_us'] for x in windows['profiles'] if x['ep']==src['ep']); keys=[k for k in chains[0] if k.endswith('_us')]
        sums={k:sum(c[k] for c in chains) for k in keys}
        profiles.append(dict(ep=src['ep'],window_us=w,complete_source_chains=len(chains),C2=36,C1=12,sum_us=sums,window_pct={k:v/w*100 for k,v in sums.items()},median_us={k:median(c[k] for c in chains) for k in keys},paid_scenarios=[dict(**{k:v for k,v in chains[0]['paid_scenarios'][n].items() if k!='net_us'},net_us=sum(c['paid_scenarios'][n]['net_us'] for c in chains),net_window_pct=sum(c['paid_scenarios'][n]['net_us'] for c in chains)/w*100) for n in range(4)],chains=chains))
    first=json.loads(files[4].read_text());decode=json.loads(files[5].read_text());owner=json.loads(files[6].read_text());old=json.loads(files[7].read_text());horizon=json.loads(files[8].read_text())
    current=json.loads(files[1].read_text());pins={s['runtime_path']:s['runtime_sha256'] for s in current['sources']}; oldpins=owner['current171_source_hashes'];assert all(pins[k]==v for k,v in oldpins.items())
    values=[{k:v for k,v in x.items() if k!='requests'} for x in first['conditional_values']]
    out=dict(status='COMPLETE_SOURCE_RING_AND_DUAL_CACHE_CHAIN_PRICED_WITH_READER_ENDPOINTS',profiles=profiles,mean_window_pct={k:mean(p['window_pct'][k] for p in profiles) for k in profiles[0]['window_pct']},original_C8_first_token_fixed_arrival_history=values,original_C8_decode_history=decode['summary'],old_C2_stage_bound=dict(scope=old['scope'],summary=old['summary'],decision=old['decision'],interpretation=old['interpretation'],retained_open=old['retained_open']),exact_SHA_native_ring_history=dict(source_pins=oldpins,full_native_comparisons=owner['full_native_comparisons'],saved_native_work=owner['saved_native_work'],contracts=owner['contracts'],hazard_examples=owner['hazard_examples'],limits=owner['limits'],qualification='Reuse bounded historical numerical/owner evidence. It does not admit arbitrary all-input full dual-sink fusion or replace current graph/alias/lifetime acceptance.'),compact_history=dict(retention=horizon['retention'],projection_and_publication=horizon['projection_and_publication'],qualification='Only predecessor-retention representation under explicit retreat/owner/reader guards. Full ring bytes remain observable until all consumers are closed; SWA can still determine allocated page size.'),decisions=['Same standalone postprojection/pool-ring/compact-publication children retain original C8 screening stops. Do not repeat old native fixtures for unchanged variants.','The larger input/query/SWA/QLI/SMLA placement parent is distinct, not bounded by standalone native cost; actual resource waits and mandatory original arithmetic stay priced.','Owner retains original full-M computation and adds communication. Eliminating duplicate FLOPs alone is not8x source latency gain. No request credit without exposed critical-path and paid packet cost.'],request_credit=0,ordinary_price='UNBOUND: no matched sixordinary complete-delivery measurements in this source pass; no C8 extrapolation',limits=['Mirrored price: old C8 EP0/8 M24/global andM3/local,96 complete source deliveries,72C2/24C1. No full16rank/current request gain measured.','Native union uses exact source task list, excludes aux query quantization already on another stream. Root-to-reader span includes mandatory QLI/SMLA. Neither whole span nor all nonGEMM work can disappear.','Paid fractions are sensitivity parameters, not demonstrated reductions or numerical acceptance. RMS/Rotary/quant/cache math remain necessary.','Original C8 first-token fixed-arrival and112decode-period histories are separate loads/denominators; not additive to mirrored windows or an all-wave/client prediction.','Existing C1 slot writer pair is metadata, not the C1 BF16 compressor GEMM and dual-sink source here. Do not transfer its budget by name.'],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in files],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,every_replacement_bitwise=True,new_NPU_work=0,new_profile=0,remote_writes=[],service_requests=0,new_primitive_admissions=0)
    (R/'compressor-path-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(mean_window_pct=out['mean_window_pct'],paid=[p['paid_scenarios'] for p in profiles],request_credit=0)))
if __name__=='__main__':main()
