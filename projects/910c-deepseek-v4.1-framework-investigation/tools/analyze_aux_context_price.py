#!/usr/bin/env python3
"""Price aux-to-three-cache branch, then paid copy/RoPE sharing children."""
from pathlib import Path
from hashlib import sha256
from statistics import mean,median
import json,zlib
from analyze_o_projection_parent_price import start,dur,union
R=Path('audit/execution-search');B=Path('/Users/wio/work/branch-fusion-audit-20261104/full-review')
def p90(xs):
    a=sorted(xs);p=.9*(len(a)-1);i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def main():
    paths=[R/'aux-context-parent-primary.json',R/'aux-context-dependencies.json',R/'aux-context-registry-dependencies.json',R/'swa-publication-parent-calls.json.zlib',R/'qkv-joint-delivery-price.json',B/'ORDERED-DSPARK-VALUE.json',B/'CURRENT-HELD-STAGE-VALUE.json',B/'DSPARK-CURRENT-OUTPUT-PUBLICATION.json',R/'embedding-context-frontier.json']
    raw=json.loads(zlib.decompress(paths[3].read_bytes()));windows=json.loads(paths[4].read_text());profiles=[]
    panel_signature=['MatMulV2','RmsNorm']+['IndexCheck','Index']*4+['Cast','Cast','Slice','RotaryPositionEmbedding','ViewCopy','Slice','Slice','ConcatD','Cast','GreaterEqual','ClipByValueV2','Cast','FloorDiv','Cast','BroadcastTo','FloorMod','Pack','LogicalNot','MaskedFill','ScatterNdUpdateSk']
    assert len(panel_signature)==30
    for src in raw:
        by={r['csv_row']:r['data'] for r in src['selected']};ids=sorted(by);heads=[i for i in ids if by[i]['Type']=='MatMulV2' and by[i]['Input Shapes'].strip('"').split(';')==['24,15360','640,15360']];assert len(heads)==12
        chains=[]
        for h in heads:
            means=[i for i in ids if h-600<i<h and by[i]['Type']=='ReduceMean' and by[i]['Input Shapes'].strip('"')=='3,4,5120;1'];assert len(means)==3
            aux_gathers=[]
            for i in means:
                group=[j for j in ids if i<j<=i+2];assert len(group)==2 and all(by[j]['Type']=='hcom_allGather_' for j in group)
                assert len({(start(by[j]),dur(by[j])) for j in group})==1;aux_gathers.append(group[0])
            selection=list(range(h-7,h));assert [by[i]['Type'] for i in selection]==['IndexCheck','Index']*3+['ConcatD']
            assert by[h+1]['Type']==by[h+2]['Type']=='hcom_allGather_' and start(by[h+1])==start(by[h+2]) and dur(by[h+1])==dur(by[h+2]);assert by[h+3]['Type']=='Transpose' and by[h+4]['Type']=='RmsNorm'
            wkv=[i for i in ids if h<i<h+180 and by[i]['Type']=='MatMulV2' and by[i]['Input Shapes'].strip('"').split(';')==['24,5120','512,5120']];assert len(wkv)==3 and wkv==[wkv[0]+30*j for j in range(3)]
            for i in wkv:assert [by[i+j]['Type'] for j in range(30)]==panel_signature
            panel_tasks=[by[i+j] for i in wkv for j in range(30)];rope_families=[[by[i+j] for j in range(2,10)] for i in wkv];rope=[float(union(x)) for x in rope_families]
            aux_cost=float(union([by[i] for i in aux_gathers+selection]));mean_cost=float(union([by[i] for i in means]));full_context=float(union(panel_tasks))
            branch=[by[i] for i in means+aux_gathers+selection+[h,h+1,h+3,h+4]]+panel_tasks
            share_saved=rope[1]+rope[2];one_config_keep=max(float(union(rope_families[0][:4])),float(union(rope_families[0][4:])))
            active_saved=sum(rope)-one_config_keep
            scenarios=[]
            for pack,gather,select,extra in [(10,30,25,5),(20,50,40,10),(50,100,60,20)]:
                packed=aux_cost-pack-gather-select;shared=share_saved-extra
                scenarios.append(dict(new_pack_us=pack,new_packed_AG_us=gather,new_selected_delivery_us=select,new_RoPE_sharing_control_us=extra,aux_pack_net_us=packed,RoPE_same_full_proxy_net_us=shared,joint_disjoint_net_us=packed+shared))
            chains.append(dict(main_proj=h,aux_mean_rows=means,aux_AG_rows=aux_gathers,context_Wkv_rows=wkv,all3_cache_write_done=wkv[-1]+29,whole_branch_native_union_us=float(union(branch)),aux_mean_retained_us=mean_cost,aux_AG_selection_concat_us=aux_cost,main_GEMM_retained_us=float(dur(by[h])),main_feature_AG_layout_norm_retained_us=float(union([by[h+1],by[h+3],by[h+4]])),context_all90_native_us=full_context,context_span_us=float(start(by[wkv[-1]+29])+dur(by[wkv[-1]+29])-start(by[wkv[0]])),context_Wkv3_retained_us=sum(float(dur(by[i])) for i in wkv),RoPE_full_getter3_us=sum(rope),RoPE_keep_first_full_getter_remove_later2_us=share_saved,RoPE_one_active_config_optimistic_removed_us=active_saved,source_order_earliest_aux_to_cache_end_span_us=float(start(by[wkv[-1]+29])+dur(by[wkv[-1]+29])-start(by[means[0]])),scenarios=scenarios))
        window=next(p['window_us'] for p in windows['profiles'] if p['ep']==src['ep']);sums={k:sum(c[k] for c in chains) for k in chains[0] if k.endswith('_us')}
        profiles.append(dict(ep=src['ep'],window_us=window,calls=len(chains),sum_us=sums,window_pct={k:v/window*100 for k,v in sums.items()},median_us={k:median(c[k] for c in chains) for k in sums},scenarios=[dict(**{k:v for k,v in chains[0]['scenarios'][j].items() if not k.endswith('_net_us')},**{k:sum(c['scenarios'][j][k] for c in chains) for k in ['aux_pack_net_us','RoPE_same_full_proxy_net_us','joint_disjoint_net_us']},joint_net_window_pct=sum(c['scenarios'][j]['joint_disjoint_net_us'] for c in chains)/window*100) for j in range(3)],chains=chains))
    old=json.loads(paths[5].read_text());held=json.loads(paths[6].read_text());request=json.loads(paths[7].read_text());carrier={}
    for name in ['context_KV_panels','context_start_to_bookkeeping','target_last_post_end_to_draft_fill']:
        if name not in request['current_carrier_sensitivity']:continue
        carrier[name]={}
        for kind,part in request['current_carrier_sensitivity'][name].items():
            reqs=part['requests'];before=p90([x['TTFT_ms'] for x in reqs]);after=p90([x['TTFT_ms']-x['max_current_credit_ms'] for x in reqs]);carrier[name][kind]=dict(whole_parent_free_upper_ms=part['upper_ms'],max_request_pct=max(x['max_current_credit_ms']/x['TTFT_ms']*100 for x in reqs),before_P90_TTFT_ms=before,fixed_carrier_P90_credit_ms=before-after,fixed_carrier_P90_credit_pct=(before-after)/before*100)
    out=dict(status='COMPLETE_AUX_BRANCH_AND_CONTEXT_CACHE_PARENT_MATCHED_BEFORE_NESTED_PRUNING',profiles=profiles,mean_window_pct={k:mean(p['window_pct'][k] for p in profiles) for k in profiles[0]['window_pct']},original_C8_all16_decode_history=dict(coverage=old['coverage'],context_KV=old['summary']['context_KV'],aux_selection_projection_copy=old['summary']['aux_selection_projection_copy'],old_exact_stage_decisions=old['decisions'],held_context_summary={k:v for k,v in held['summary'].items() if k.startswith('draft_context')},qualification='Separate112 full16rank local periods. All90context tasks free remain at most0.866501%of those periods; preserve exact old stopped stage-confined row/RoPE/slot recipes. Not a whole-request or larger host/context upper bound.'),fixed_current_carrier_request_history=carrier,fixed_carrier_scope=request['scope'],whole_wave_propagated_host_context_history=json.loads(paths[8].read_text()),traffic=dict(TP=8,HC=4,H=5120,aux_layers=3,full_rows=24,SP_local_rows=3,aux_mean_output_bytes_per_rank_per_layer=3*5120*2,all3_global_aux_bytes=3*24*5120*2,original_concat_bytes=24*15360*2,qualification='Packing changes collective launches/layout, not3aux logical payload volume. Original means/GEMM/RMS/Rotary and cache bytes stay live; no raw byte count is a timing gain.'),decisions=['Preserve previous stopped independent context row/RoPE/slot and changed-M projection variants under their original budget; no repeated native fixture for these children.','New aux packing plus same-proxy RoPE reuse is a copy/shared-preparation child; charge new pack, equal-volume AG, selected delivery and control/lifetime overhead explicitly. Conditions are not attained speedups.','Retain larger aux/host/context common producer, original-tile row ownership or megakernel as distinct parent with native original-rounding/capture/cache/readiness/paid-value gaps; do not transfer small child ceilings to it.'],request_credit=0,limits=['Local price only old mirrored C8 EP0/8 M24/localM3,24 complete aux/main/3cache chains72panels. This is not all16rank or6ordinary timing.','Earliest aux-to-cache span includes intervening mandatory target layers, target head/rejection and metadata; never credit that span as removable aux cost. Branch-native union retains only explicitly matched native tasks, not all CPU/DMA/host latency.','Repeated hcom rows with matching start/duration are one event, not two charges. New packed allgather keeps equal total logical payload.','RoPE shared-once scenario preserves original first full helper family and all3 original numerical consumers, casts/copies/stores; active-config scenario uses a conservative retained pair estimate, not observed config-label mapping or a native implementation.','Fixed-current-carrier histories freeze every earlier iteration/request start. Their P90/max request credits cannot be used as all-wave/full-request or client completion forecasts. Old wave-propagated larger host/context scenario has a separate denominator and mandatory work.','No new native bits, workload population or request gain measured. Sixordinary and unbound exposure default0.'],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),every_replacement_bitwise=True,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],whole_chain_complete=False,new_primitive_admissions=0)
    (R/'aux-context-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(mean_window_pct=out['mean_window_pct'],paid_joint=[p['scenarios'] for p in profiles],fixed_carrier=carrier['context_KV_panels'],request_credit=0)))
if __name__=='__main__':main()
