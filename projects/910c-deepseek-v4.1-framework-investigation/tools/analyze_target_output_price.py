#!/usr/bin/env python3
"""Incremental paid summary price with unchanged projection and downstream work."""
from pathlib import Path
from hashlib import sha256
from statistics import mean, median
import json,zlib
from analyze_o_projection_parent_price import start,dur,union

R=Path('audit/execution-search');B=Path('/Users/wio/work/branch-fusion-audit-20261104/full-review')
def main():
    paths=[R/'target-output-parent-primary.json',R/'target-output-dependencies.json',R/'target-output-aux-dependencies.json',R/'swa-publication-parent-calls.json.zlib',R/'qkv-joint-delivery-price.json',B/'ORDERED-TERMINAL-HEAD-VALUE.json',B/'ORDERED-SAMPLING-VALUE.json']
    raw=json.loads(zlib.decompress(paths[3].read_bytes()));windows=json.loads(paths[4].read_text());profiles=[]
    signature=['MatMulV2','hcom_allGather_','Transpose','IndexCheck','Index','Cast','Cast','ArgMaxV2','Cast','Cast','IndexCheck','Index','Cast','TensorMove','Fill','ArgMaxV2','Cast','rejection_greedy_sample_triton']
    seed=['NotEqual','Cast','Less','LogicalAnd','Cast','ReduceSum','Cast','Sub','ClipByValueV2','Cast','GatherElementsV2','NotEqual','SelectV2','prepare_inputs_padded_kernel']
    for src in raw:
        by={r['csv_row']:r['data'] for r in src['selected']};ids=sorted(by)
        heads=[i for i in ids if by[i]['Type']=='MatMulV2' and by[i]['Input Shapes'].strip('"').split(';')==['24,5120','16160,5120']];assert len(heads)==12
        chains=[]
        for h in heads:
            assert [by[h+j]['Type'] for j in range(18)]==signature
            assert [by[h+j]['Type'] for j in range(18,32)]==seed
            assert by[h+4]['Input Shapes'].strip('"')=='24,129280;1;2;4'
            assert by[h+11]['Input Shapes'].strip('"')=='24,129280;1;2;20'
            assert 'DT_BF16' in by[h]['Output Data Types']
            before=max(i for i in ids if h-70<i<h and by[i]['Type']=='HcPost')
            ready=start(by[before])+dur(by[before]);rejectend=start(by[h+17])+dur(by[h+17]);seedend=start(by[h+31])+dur(by[h+31])
            removable=[by[h+j] for j in range(1,17) if j!=14]
            cost=float(union(removable));scenarios=[]
            for local,comm,merge in [(20,10,10),(50,30,20),(100,100,40),(100,200,50)]:
                scenarios.append(dict(new_local_reduce_us=local,new_collective_us=comm,new_merge_packet_us=merge,new_total_us=local+comm+merge,net_us=cost-local-comm-merge))
            chains.append(dict(head_csv_row=h,last_target_Post=before,final_reject=h+17,final_seed_padded=h+31,original_projection_us=float(dur(by[h])),removable_score_delivery_native_us=cost,head_gather_us=float(dur(by[h+1])),target_Post_ready_to_reject_done_us=float(rejectend-ready),target_Post_ready_to_seed_padded_done_us=float(seedend-ready),score_delivery_span_us=float(rejectend-start(by[h+1])),unchanged_reject_us=float(dur(by[h+17])),unchanged_seed_native_us=float(union([by[h+j] for j in range(18,32)])),scenarios=scenarios))
        window=next(x['window_us'] for x in windows['profiles'] if x['ep']==src['ep'])
        sums={k:sum(c[k] for c in chains) for k in chains[0] if k.endswith('_us')}
        profiles.append(dict(ep=src['ep'],calls=len(chains),window_us=window,sum_us=sums,window_pct={k:v/window*100 for k,v in sums.items()},median_us={k:median(c[k] for c in chains) for k in sums},scenario_totals=[dict(**{k:v for k,v in chains[0]['scenarios'][j].items() if k!='net_us'},net_us=sum(c['scenarios'][j]['net_us'] for c in chains),net_window_pct=sum(c['scenarios'][j]['net_us'] for c in chains)/window*100) for j in range(4)],chains=chains))
    oldhead=json.loads(paths[5].read_text());oldseed=json.loads(paths[6].read_text())
    out=dict(status='MATCHED_TARGET_FULL_PARENT_AND_INCREMENTAL_SCORE_SUMMARY_PAID_PRICE',profiles=profiles,mean_window_pct={k:mean(p['window_pct'][k] for p in profiles) for k in profiles[0]['window_pct']},
      traffic=dict(M=24,bonus_rows=4,verification_rows=20,V=129280,local_V=16160,TP=8,original_local_BF16_score_bytes=24*16160*2,original_full_BF16_score_bytes=24*129280*2,summary_each_rank_logical_bytes=24*8,summary_all8_logical_bytes=24*8*8,qualification='Illustrative FP32 value+INT32 global-ID pair. Actual HCCL packet/alignment/protocol and NaN/mask metadata are not priced from bytes. Per-rank and full tensors are not summed as network wire bytes.'),
      original_C8_all16_history=dict(head_coverage=oldhead['coverage'],head_summary=oldhead['summary'],seed_coverage=oldseed['coverage'],seed_summary=oldseed['summary'],scope=oldhead['scope'],qualification='Separate original C8 112 local periods /16 ranks, not the two mirrored-profile windows. Old free-only summaries are ceilings containing mandatory work, not paid/request predictions. Head-end through seed max0.608161% period preserves older stopped child scope; joint context remains larger.'),
      decisions=['Keep old fixed-projection independent copy/cast/argmax/count and fixed-summary small recipes stopped under their recorded budget; no unchanged native fixture is justified by this source review.','The larger full-vocab/Markov/next-generation parent is retained for joint owner/layout/kernel design. Do not transfer the small fixed-projection child ceiling to that parent.','Candidate incremental price keeps original local GEMM, output Fill, rejection kernel, accepted count/seed and all downstream D2H/executor/Engine/scheduler work. Credit only this matched removable native family; no host gap, CPU copy, KV retirement or request credit.'],
      request_credit=0,missing_decisive_parameters=['Actual eligible population of reader-closed all-greedy standard batches on original C8 and each ordinary load.','Native identical argmax tie/NaN/-inf/global-ID/padding behavior with original GEMM output, and the complete primitive output/capture/alias contract.','Paid new local reduction + small HCCL + global merge latency at the same16rank/DP arrivals; asynchronous producer/consumer lifetime and fallback transition.','Full request exposure, including original D2H/client and next-generation dependencies; no new request experiment here.'],
      limits=['Two old representative EP0/8 mirrored C8 windows only, M24/B4/D20/V129280; 24 complete source-matched head/reject/seed chains. No sixordinary/full16rank timing inference.','Removable sum is a conditional native task budget, not an attained speedup or an upper bound on every larger layout/mathematical transformation. New post math and communication are charged explicitly.','Unconditional bonus -1 returns full scores; removing them is valid only inside a larger replacement whose actual score outputs are dead under proved outer guards. Never use IDs-only to replace standalone bonus Sampler.forward.','Original GEMM shape/dtype/reduction and all live output packet/count/logprobs/NaNs/RNG/ownership remain obligations. Source/protocol controls do not establish native bits.'],
      inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,service_requests=0,new_profile=0,remote_writes=[],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0)
    (R/'target-output-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(mean_window_pct=out['mean_window_pct'],paid_scenarios=[p['scenario_totals'] for p in profiles],request_credit=0)))
if __name__=='__main__':main()
