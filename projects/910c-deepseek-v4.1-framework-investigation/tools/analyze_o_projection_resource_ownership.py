#!/usr/bin/env python3
"""Screen full rounded OA panel ownership and distinct N-owner recomputation, no NPU."""
from pathlib import Path
from hashlib import sha256
import math,json,itertools

ROOT=Path('audit/execution-search');OLD=Path('/Users/wio/work/branch-fusion-audit-20261104/full-review/OA-OB-SHARED-CUBE-SCREEN.json')
def main():
    old=json.loads(OLD.read_text());hw=old['hardware'];clocks_path=ROOT/'o-projection-saved-clocks.json';clocks=json.loads(clocks_path.read_text());freq=[]
    for row in clocks['saved_profile_clocks']:
        assert row['csv_exists']
        freq += [r[2] for db in row['clock_records'] for tab in db['clock_tables'] if tab['table']=='AICORE_FREQ' for r in tab['rows']]
    assert len(freq)==6 and set(freq)=={1800};GHz=1.8;bandwidth=hw['L1_to_L0B_B_per_AIC_cycle']*GHz*1000
    assert hw['AIC']==24 and hw['L1_usable_bytes']==524032 and bandwidth==230400
    pfile=ROOT/'o-projection-parent-price.json';parents=json.loads(pfile.read_text());proposals=[]
    for owners in [1,2,4,8,16,24]:
        ntiles=40;max_tiles=math.ceil(ntiles/owners);maxN=128*max_tiles
        per_owner_weight=4096*1024*2+1024*maxN*2
        ingress_floor=(per_owner_weight-65536)/bandwidth
        # MAC reference only; currently dominated by right-operand ingress.
        mac_floor=(32*4096*1024+32*1024*maxN)/(4096*GHz*1000)
        aggregate_weight=owners*4096*1024*2+1024*5120*2
        proposals.append(dict(N_owners=owners,proposed_M_panel=32,max_OB_N_per_owner=maxN,OA_computation_replication=owners,per_owner_weight_operand_bytes=per_owner_weight,allow_resident_L0B_bytes=65536,L1_to_L0B_reference_floor_us=ingress_floor,MAC_reference_floor_us=mac_floor,ideal_reference_floor_us=max(ingress_floor,mac_floor),aggregate_logical_weight_operand_bytes=aggregate_weight,aggregate_service_TB_s_at_reference_floor=aggregate_weight/max(ingress_floor,mac_floor)/1e6))
    outcomes=[]
    for p in parents['profiles']:
        vals=[r for r in p['deliveries'] if r['rows']==24];assert len(vals)==480
        observed=sum(r['OA_OB_span_us'] for r in vals);single=proposals[0]['ideal_reference_floor_us'];scenarios=[]
        for proposal,eta,h in itertools.product(proposals,[1,.9,.8],[0,5,20]):
            new=proposal['ideal_reference_floor_us']/eta
            net=observed-len(vals)*(new+h)
            scenarios.append(dict(N_owners=proposal['N_owners'],uniform_reference_capacity_fraction=eta,additional_nonoverlap_us=h,conditional_new_pair_reference_us=new,conditional_net_us=net,conditional_window_pct=net/p['window_us']*100))
        oracle=sum(max(0,r['OA_OB_span_us']-single) for r in vals)
        outcomes.append(dict(ep=p['ep'],window_us=p['window_us'],target_pairs=480,original_pair_span_sum_us=observed,original_pair_mean_us=observed/480,original_pair_minmax_us=[min(r['OA_OB_span_us'] for r in vals),max(r['OA_OB_span_us'] for r in vals)],single_owner_ideal_floor_us=single,single_owner_signed_net_if_all_sites_replaced_us=observed-480*single,single_owner_per_site_free_fallback_oracle_us=oracle,single_owner_oracle_window_pct=oracle/p['window_us']*100,keep_both_original_native_kernels_only_handoff_gap_free_us=sum(r['OA_OB_handoff_gap_us'] for r in vals),scenarios=scenarios))
    result=dict(status='REFERENCE_RESOURCE_SCREEN_NOT_NATIVE_PERFORMANCE_OR_BIT_PROOF',
      hardware_reference=dict(clock_source=str(clocks_path),clock_MHz_values=freq,clock_scope='Six recorded AICORE_FREQ rows cover the two saved representative windows; not continuous measurement of allranks/live device.',L1_to_L0B_B_per_AIC_cycle=128,L1_to_L0A_B_per_AIC_cycle=256,L1_usable_bytes=524032,L0B_bytes=65536,L0C_bytes=131072,reference_MAC_per_AIC_cycle=4096,MAC_constant_scope='Historical reference model assumption; primary linked page verifies storage/path bandwidth, not this MAC-throughput number.',primary_url=hw['primary_document'],primary_scope='Current2201 guide identifies A2/A3, per-core L1/L0 sizes and 128B/cycle right operand path. InstalledCANN9.2 SDK separately verifies legal FixpipeL0C->L1 BF16 NZ route.'),
      source_mechanism='Preserve complete1024-wide rounded BF16 OA panel within each owner using F322BF16 L0C->L1 NZ, then feed OB leftoperand. Local L1 cannot be read by another AIC; splitting OA-R ownership and summing new OB partials changes FP32 association unless original schedule exactly reproduced.',
      proposals=proposals,profiles=outcomes,
      static_buffer_illustration=dict(scope='Illustrative32-row ownership, not native tiling or full allocation/flags proof.',OA_BF16_complete_panel_L1_bytes=32*1024*2,OA_A_pingpong_L1_bytes=2*32*64*2,OA_weight_pingpong_L1_bytes=2*64*256*2,OB_weight_pingpong_L1_bytes=2*256*128*2,other_L1_placeholder_bytes=16384,L1_sum_bytes=65536+8192+65536+131072+16384,L0B_OA_pingpong_bytes=2*64*256*2,L0C_OA_pingpong_bytes=2*32*256*4,phase_reuse='Finish all OA columns into roundedpanel before OB; phase-local L0C can be reused, no unpriced retained OB FP32 accumulator across OA phases.'),
      decisions=[dict(scope='One complete rounded OA panel per AIC, allOBcolumns on thatowner, currentM24',status='已停止同一局部schedule',result='Reference optimistic ingress floor81.636us excludes many real costs. Fresh recorded clocks match1.8GHz; exact saved currentpair oracle below disclosed threshold, no need repeatoldsingle-owner NPU recipe.'),dict(scope='N-sharded owners each recompute fullOA, local roundedpanel then originalK-ordered OB Ntiles',status='待确认',result='Different geometry from stopped singleowner. Can lower perowner weight ingress, but duplicates OA and aggregate weight operand demand; ideal local screen not enough without L2/fanout service, originalMMA bit and paid cost.'),dict(scope='Native replacement primitive',status='已确认可实现',result='本轮0；SDK route/finite scalar/resource model is not nativebit admission.')],
      limits=['No use of peak bandwidth as achieved bandwidth. L1-to-L0B floor is only a necessary reference constraint; additional core/L2/MTE2/flags/launch/padding costs omitted.','Replicated weight bytes are logical operand traffic, not HBM reads or new persistent weight copies; source weights remain same rank shards.','Singleowner stops only the explicit currentM24/recordedclock ownership family. Largeprefill and multiowner geometry keep separate gates and originalM/weight requirements.','Newkernel must retain active original OA and OB K accumulation and BF16 conversion; available source/compiledkey list does not identify active runtimekey.','No fullrequest or sixordinary extrapolation; reference1%window quantities are sensitivities, not user acceptance criteria.'],
      whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [OLD,clocks_path,pfile]],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'o-projection-resource-ownership.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(proposals=proposals,profiles=[{k:v for k,v in r.items() if k!='scenarios'} for r in outcomes],N24_ideal_h5=[s for r in outcomes for s in r['scenarios'] if s['N_owners']==24 and s['uniform_reference_capacity_fraction']==1 and s['additional_nonoverlap_us']==5]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
