"""Aggregate current C32 diagnostic; distinguish profiler views and real timing."""
import collections,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(relative):return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
def dist(values):return {'min':min(values),'P50':statistics.median(values),'max':max(values),'ranks':len(values)}
def main():
    source='results/C32-D-two-step-evidence-v1.json';evidence=json.loads((ROOT/source).read_text());assert evidence['diagnostic_evidence_pass'] and not evidence['performance_certificate_allowed']
    workers=evidence['workers'];assert len(workers)==32
    fractions={}
    for name in ['Computing','Communication(Not Overlapped)','Free']:
        fractions[name]=dist([100*sum(float(r[name]) for r in w['native_step_statistics_rows'])/sum(float(r['Stage']) for r in w['native_step_statistics_rows']) for w in workers])
    spans=collections.Counter();API={}
    for w in workers:
        for span in w['span_like_host_operator_aggregates']:
            if span['name'].startswith('step['):spans[(span['name'],span['count'])]+=1
    for name in ['aclrtSynchronizeStream','aclrtLaunchKernelWithHostArgs','hcom_alltoallv_','aclnnNonzeroV2']:
        rows=[next(x for x in w['top_native_API_time'] if x['name']==name and x['level']=='acl') for w in workers]
        API[name]={'calls':dist([x['count'] for x in rows]),'host_API_time_sum_us':dist([x['time_sum_us'] for x in rows])}
    result=dict(project_id='glm53',loop='L06-C32-D-critical-path',decision='ACCEPTED_DIAGNOSTIC',evidence_level='E3_CURRENT_C32',
                requested_profile_steps=2,actual_D_ranks=32,profile_collection_duration_seconds=evidence['profile_collection_duration_seconds'],
                primary_performance_measurement=False,source_qualified_cache_prefixes=32,raw_D_storage_nulls_preserved=1,
                raw_stop_HTTP_pass=False,native_all32_completion_independently_verified=True,
                native_stage_fraction_percent=fractions,
                stage_fraction_scope='Ratios from original native Computing/Communication(Not Overlapped)/Free divided by Stage. Units cancel. Per-rank medians; not physical utilization or a whole-run causal budget.',
                actual_step_span_rank_distribution=[{'name':name,'calls_per_rank':count,'ranks':ranks} for (name,count),ranks in spans.items()],
                selected_ACL_API_statistics=API,
                API_scope='Each named ACL API is reported separately. Node-level launch counters overlap ACL launch counters and are not added to kernel launch counts; API wait times overlap device work.',
                mechanism_evidence=['Target verification dominates recorded host spans; 26ranks have2VERIFYbs2/10DRAFTbs2,6ranks have oneVERIFYbs1 plus oneVERIFYbs2 and5draft calls per size.','All32ranks record480hcom_alltoallv calls and roughly556aclrtSynchronizeStream calls in two steps. Current data-dependent host metadata boundary and generic AllToAll remain a material candidate.','Computation is approximately9percent of the sampled native stage; prioritize communication/host boundaries over isolated arithmetic kernels.','Current CPU43 delivery edges are submillisecond at the median; internal D buffering remains outside that edge measurement.'],
                next_candidate='Native installed DeepEP ops low-latency dispatch/combine, graphstilldisabled/MTP5/unchangedweights/precision/caches/parallelism; source/actualcorrectness required before performance.',
                source_evidence_sha256={p:digest(p) for p in [source,'results/sglang-C32-L04-control-v3-measurement-v1.json','results/sglang-C32-control-delivery-timing-review-v1.json','runs/20261017T1260-deepep-native-contract-v1/source-34.json','runs/20261017T1050-sglang-failure-source-v1/source-34.json','runs/20261017T1822-native-profile-control-source-v2/sglang.json']},
                limitations=['Two steps are not a long-run temporal stability study.','Profiler overhead and admission skew can change absolute durations; full-clock unprofiled measurement remains authoritative.','Changing an entire backend tests that component, without quantifying an isolated contribution from every internal primitive.','No optimization gain has yet been demonstrated by this diagnostic.'])
    p=ROOT/'results/C32-D-critical-path-mechanism-review-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'decision':result['decision'],'stage_percent':fractions,'span_rank_distribution':result['actual_step_span_rank_distribution']},indent=2))
if __name__=='__main__':main()
