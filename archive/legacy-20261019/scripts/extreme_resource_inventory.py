#!/usr/bin/env python3
"""Source/evidence-backed resource inputs for an eventual rank/stream DAG."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path('/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json')
MODEL=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py')
BLOCKS=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/layer/attention/layer.py')
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def item(value,evidence,status='observed'):return dict(value=value,status=status,evidence=evidence)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 cfg=json.loads(CONFIG.read_text());ratios=cfg['compress_ratios'][:cfg['num_hidden_layers']]
 assert len(ratios)==43 and collections.Counter(ratios)=={4:21,128:20,0:2}
 s115=read('evidence/20260925_loop039_gmm/run115/shape_summary.json')['rank_records'][0]['target_96x6_shape']
 assert s115['w1'][0]['shape']==[32,4096,512] and s115['w2'][0]['shape']==[32,2048,512]
 r146=read('evidence/20260925_loop044_target/run146/gmm_bound_audit.json')
 r148=read('evidence/20260925_loop044_target/run148/counter_analysis.json')
 r150=read('evidence/20260925_loop044_target/run150/counter_analysis.json')
 r152=read('evidence/20260925_loop044_target/run152/tp_collective_audit.json')
 r143=read('evidence/20260925_loop044_target/run143/union_audit.json')
 phase=read('evidence/20260926_loop059_boundary/run239/phase_analysis.json')
 active=r146['active_expert_layer_pairs_per_rank_cycle']['median'];routed=r146['routed_tokens_per_rank_cycle']['median']
 w1=r146['w1_packed_bytes_per_expert'];w2=r146['w2_packed_bytes_per_expert']
 bw1=r148['effective_read_GBps']['median'];bw2=r150['effective_read_GBps']['median']
 # W1 int32 packing represents eight 4-bit weights per element; this is an
 # operation count from recorded matrix shapes, not attainable FLOPs/s.
 w1_shape=s115['w1'][0]['shape'];w2_shape=s115['w2'][0]['shape']
 k1=w1_shape[1];n1=w1_shape[2]*8;k2=w2_shape[2]*8;n2=w2_shape[1]
 flops=routed*2*(k1*n1+k2*n2)
 packed=active*(w1+w2)
 conditional_ms=active*w1/(bw1*1e9)*1000+active*w2/(bw2*1e9)*1000
 pair_mean=statistics.median(sum(sum(y['duration_ms'] for y in z) for z in w['pairs'])/len(w['pairs']) for w in r152['windows'])
 prefill=[]
 for wave in phase['waves']:
  b=read(f"evidence/20260926_loop059_boundary/run239/boundary/rank0_cohort{wave['cohort']}.json")
  start=wave['marks_s']['client_start'];handoff=wave['marks_s']['handoff']
  shapes=[v['scheduled_tokens'] for v in b['calls'] if start-0.01<=v['t_ns']/1e9<handoff]
  prefill.append(dict(cohort=wave['cohort'],scheduled_token_counts=shapes,
                      prefill_to_handoff_s=wave['phases']['first_execute_to_handoff_s']))
 out=dict(schema_version=1,contract='DeepSeek V4 Flash W4A8; 8x910B3 DP1TP8 DSpark7; 48x32K->1024 c12',
          source_sha256={str(CONFIG):sha(CONFIG),str(MODEL):sha(MODEL),str(BLOCKS):sha(BLOCKS)},
          target=dict(layers=item(43,'config.json num_hidden_layers'),
                      compression_ratio_counts=item(dict(collections.Counter(ratios)),'config.json compress_ratios[:43]'),
                      target_rows_per_cycle=item(96,'Run115/Run98 fixed c12 graph','observed_fixed_shape'),
                      gmm=dict(w1_shape=item(w1_shape,'Run115'),w2_shape=item(w2_shape,'Run115'),
                               routed_tokens_per_rank_cycle=item(routed,'Run146 distinct diagnostic cycles'),
                               active_expert_layer_pairs_per_rank_cycle=item(active,'Run146'),
                               packed_active_weight_bytes_per_rank_cycle=item(packed,'Run146'),
                               packed_scale_bytes_if_once_per_active_expert=item(active*2*4096*8,'Run115 shapes; conditional read','conditional'),
                               logical_matmul_flops_per_rank_cycle=item(flops,'Run115 packed shapes and Run146 routed tokens','conditional'),
                               one_card_counter_bw_GBps=item(dict(w1=bw1,w2=bw2),'Runs148/150 synthetic one-card','conditional'),
                               one_card_packed_read_time_ms=item(conditional_ms,'Run146 active bytes divided by Runs148/150 counter bandwidth','conditional_not_e2e_floor'),
                               profiled_kernel_sum_ms=item(9.96625,'Run107 synchronized 15 target windows','profiled_not_exposed')),
                      other_kernel_families=item({k:v['kernel_sum_ms']['median'] for k,v in r143['families'].items() if k not in ['gmm1','gmm2','communication']},'Run143 task sums; overlap and profiler perturbation','diagnostic_only'),
                      cache_specs=item(dict(c4=dict(layers=21,state_dim=2048,dtype='float32',effective_block_size_at_service_block32=2,page_size_padded=4160),
                                            c128=dict(layers=20,state_dim=1024,dtype='float32',effective_block_size_at_service_block32=8,page_size_padded=32768),
                                            swa=dict(head_dim=512,dtype='bfloat16',window=128,block_size_at_service_block32=32),
                                            indexer=dict(head_dim=128,dtype='int8_on_910B3',compression_ratio=4,enabled_at_runtime='UNKNOWN')),
                                       'deepseek_v4.py and DSV4_BLOCK_SIZES; service --block-size 32','source_spec_not_traffic'),
                      dependencies=item(['43 ordered target layers','within routed MoE GMM1 -> GMM2','compressor -> cache scatter -> attention for c4/c128 (Run197)','collectives wait for all participating ranks'],
                                        'model source/Run197/Run145','partial_order_only')),
          communication=dict(profiled_adjacent_gather_pairs_per_target_window=item(r152['adjacent_allgather_pairs_per_window'],'Run152'),
                             profiled_adjacent_pair_task_ms_median=item(pair_mean,'Run152 per-window tasks','profiled_not_link_floor'),
                             remaining_hccl_task_sum_ms=item(5.2050285,'Run145 excludes first arrival-heavy reduce-scatter','profiled_not_link_floor'),
                             actual_collective_network_bytes='UNKNOWN',graph_path_capacity='UNKNOWN; Run237 eager API invalid for product calibration'),
          prefill=dict(observed_run239_wave_shapes=item(prefill,'Run239 rank0 scheduled token records','separate_diagnostic_pass'),
                       required_HBM_bytes='UNKNOWN',per_shape_FLOPs='UNKNOWN',KV_read_write_bytes='UNKNOWN'),
          unknowns=['Exact per-layer KV/cache read/write bytes, page padding and reuse under c12 real sequence lengths',
                    'Per-node non-GMM HBM and FLOP counters at target real shapes',
                    'Graph-path TP8 network bytes/service-time capacity without peer-arrival wait',
                    'Resource/stream dependencies and overlap under uninstrumented graph execution',
                    'Prefill source-level work and phase capacity across the observed changing shapes',
                    'Cycle acceptance/route response to candidate architecture changes'],
          interpretation='Only W4A8 GMM has a quantitative conditional capacity screen. No complete target, prefill or E2E achievable upper bound follows from this inventory.')
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(active_GB=packed/1e9,gmm_GFLOP=flops/1e9,conditional_one_card_read_ms=conditional_ms,
                       profiled_gmm_ms=9.96625,prefill_shape_counts=[len(x['scheduled_token_counts']) for x in prefill],
                       unresolved=len(out['unknowns'])),indent=2))
if __name__=='__main__':main()
