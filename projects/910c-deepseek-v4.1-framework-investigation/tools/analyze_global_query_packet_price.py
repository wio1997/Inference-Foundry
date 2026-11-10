#!/usr/bin/env python3
"""Reprice missing head8 query packets using saved native metadata and explicit overlap."""
from pathlib import Path
from hashlib import sha256
from decimal import Decimal as D
import json,zlib,itertools,statistics

ROOT=Path('audit/execution-search')
SMLA='aclnnSparseFlashMlaMetadata_SparseFlashMlaMetadata_SparseFlashMlaMetadata'
QLI='aclnnVllmAscendQuantLightningIndexerV2Metadata_QuantLightningIndexerV2Metadata_VllmAscendQuantLightningIndexerV2Metadata'
TARGET_SHAPES={'"5;;;;4;;;;"','"5;;;;4;4;4;;"','"5;;;;4;4;;;"'}
def main():
    rawpath=ROOT/'swa-publication-parent-calls.json.zlib';raw=json.loads(zlib.decompress(rawpath.read_bytes()))
    pricepath=ROOT/'qkv-joint-delivery-price.json';prices=json.loads(pricepath.read_text());out=[]
    for p in prices['profiles']:
        original=next(r for r in raw if r['ep']==p['ep']);calls=original['selected'];byid={r['csv_row']:r['data'] for r in calls}
        metas=[r for r in calls if r['data']['Name']==SMLA and r['data']['Input Shapes'] in TARGET_SHAPES]
        assert len(metas)==36
        durations=sum(D(r['data']['Duration(us)']) for r in metas)
        heads=sorted([r for r in p['deliveries'] if r['rows']==24],key=lambda r:D(r['root_start']));assert len(heads)==480
        firsts=heads[::40];assert len(firsts)==12
        c0=sorted([r for r in metas if r['data']['Input Shapes']=='"5;;;;4;;;;"'],key=lambda r:D(r['data']['Start Time(us)'].strip()))
        frames=[];matched=set()
        for i,first in enumerate(firsts):
            lo=D(c0[i]['data']['Start Time(us)'].strip());hi=D(first['root_start'])
            group=[r for r in calls if r['data']['Name'] in [SMLA,QLI] and lo<=D(r['data']['Start Time(us)'].strip())<hi]
            assert len(group)==5 and sum(r['data']['Name']==SMLA for r in group)==3
            assert {r['data']['Input Shapes'] for r in group if r['data']['Name']==SMLA}==TARGET_SHAPES
            assert not (matched&{r['csv_row'] for r in group});matched.update(r['csv_row'] for r in group)
            last=max(D(r['data']['Start Time(us)'].strip())+D(r['data']['Duration(us)']) for r in group)
            ref=sum(D(r['data']['Duration(us)']) for r in group if r['data']['Name']==SMLA)
            firstsmla=D(byid[first['smla']]['Start Time(us)'].strip())
            frames.append(dict(index=i,target_and_QLI_metadata_csv_rows=[r['csv_row'] for r in group],first_QA_csv_row=first['qa'],first_SMLA_csv_row=first['smla'],last_metadata_end_us=str(last),first_QA_us=str(hi),last_metadata_to_first_QA_us=float(hi-last),last_metadata_to_first_SMLA_us=float(firstsmla-last),three_original_target_packet_duration_reference_us=float(ref)))
        scenarios=[]
        for swa,mode,ratio,exposure,pack in itertools.product([0,.5],['replace_original_three','coexist_with_original_three'],[.5,1,2],[0,1],[0,5,20]):
            old=next(s for s in p['head_partition_through_attention']['scenarios'] if s['new_qb_duration_ratio']==.25 and s['new_qr_and_own_index_ag_cost_vs_larger_input_ag']==1 and s['SWA_native_fraction_saved']==swa and s['new_SMLA_duration_ratio']==1 and s['added_pack_metadata_sync_us_per_delivery']==5)
            meta_delta=exposure*float(durations)*(ratio-(1 if mode=='replace_original_three' else 0))
            net=old['net_us']+5*480-pack*480-meta_delta
            scenarios.append(dict(SWA_native_fraction_saved=swa,query_packet_mode=mode,new_three_packet_cost_vs_original_reference=ratio,metadata_delta_exposure=exposure,nonmetadata_pack_sync_us_per_delivery=pack,charged_metadata_delta_us=meta_delta,conditional_net_us=net,conditional_window_pct=net/p['window_us']*100))
        out.append(dict(ep=p['ep'],window_us=p['window_us'],original_target_packet_calls=len(metas),draft_packet_excluded=12,original_target_packet_duration_sum_us=float(durations),original_reference_per_iteration_us=float(durations)/12,original_reference_per_target_delivery_us=float(durations)/480,matched_frames=frames,slack_summary=dict(min_to_first_QA_us=min(r['last_metadata_to_first_QA_us'] for r in frames),median_to_first_QA_us=statistics.median(r['last_metadata_to_first_QA_us'] for r in frames),min_to_first_SMLA_us=min(r['last_metadata_to_first_SMLA_us'] for r in frames)),scenarios=scenarios))
    result=dict(status='SOURCE_CONSISTENT_ORIGINAL_PACKET_REFERENCE_AND_PAID_HEAD_SENSITIVITY',profiles=out,
      input_availability='Existing cp_global batch already retains original globalQSL and device-authoritative global lengths/positions/C2lengths. It omits querypacket generation. A new guarded head8 query packet can use those inputs without recomputing CP suffix; still validate complete original/compressed causal mask and nativebit.',
      pricing='Original loop038 h5 scenario is first restored to h0 by adding480*5, then explicit nonmetadata pack/sync and metadata delta charged. Replace-mode delta assumes original three packets truly replaced with identical exposure; coexist mode retains old and adds new. All assumptions are sensitivities, not measured newpacket cost or critical timing.',
      limits=['Native CSV rows match source-consistent12x(3targetSMLA+2QLI) frames before first targetQA; this is not pointer/event binding.','Durations of original head64/localquery packets are references only; head8/globalquery planner and kernel may differ.','Ready-to-first-reader slack is original schedule only, not guaranteed overlap for new work or a removable/casual bound. Other producer tasks, input-ready, stream launch and allrank changes still matter.','Replacing packets requires queryroute-specific graph metadata buffers/attrs and coherent fallback/frontier sets. Do not keep old consumers while overwriting head64 packets withhead8.','No request extrapolation or repeated perlayer packet charge; packets are once per source-consistent iteration.'],
      whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [rawpath,pricepath]],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'global-query-packet-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(ep=r['ep'],sum_us=r['original_target_packet_duration_sum_us'],reference_per_iteration=r['original_reference_per_iteration_us'],slack=r['slack_summary'],paid_examples=[s for s in r['scenarios'] if s['new_three_packet_cost_vs_original_reference']==1 and s['metadata_delta_exposure']==1 and s['nonmetadata_pack_sync_us_per_delivery']==5]) for r in out],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
