#!/usr/bin/env python3
"""Hash-check and calculate local interface accounts, never physical ceilings."""
import hashlib
import json
from pathlib import Path
import statistics

W=Path(__file__).resolve().parent;R=W.parent
read=lambda p:json.loads(p.read_text())
config=read(R/'evidence/model-config.json');text=config['text_config'];vision=config['vision_config']
d=vision['hidden_size'];i=vision['intermediate_size'];l=vision['num_hidden_layers'];h=text['hidden_size'];r=vision['downsample_ratio'];p=vision['patch_size']
vit_patch=d*(3*p*p)+d
vit_block=4*d*d+6*d+3*d*i
vit_params=vit_patch+l*vit_block+d
aligner_params=(d*r*r)*h+h+h*h+h
special_params=3*h
hc=text['hc_mult'];max_tokens=8192;tp=8;rows=24
assert (d,i,l,h,r,p,hc)==(1024,2816,32,5120,3,14,4)
accounts={'status':'CONDITIONAL_LOGICAL_ACCOUNTS_NOT_PHYSICAL_MEMORY_OR_COMMUNICATION_MEASUREMENTS',
    'vision':{'vit_parameters':vit_params,'aligner_parameters':aligner_params,'special_vector_parameters':special_params,
        'vit_aligner_bytes_if_bf16':2*(vit_params+aligner_params),'special_bytes_if_declared_fp32':4*special_params,
        'replication':'vision.py uses ordinary nn.Linear/ModuleList; no TP/DP sharding in these modules',
        'resident_device_allocation_measured':False},
    'mtp_buffer':{'elements_max_capacity':max_tokens*hc*h,'bytes_if_bf16':2*max_tokens*hc*h,
        'max_capacity_mib_if_bf16':2*max_tokens*hc*h/2**20,'tp':tp,'observed_compatible_full_rows':rows,
        'copy_payload_bytes_for_24rows_if_bf16':rows*hc*h*2,'physical_copy_identity_measured':False},
    'tail_gather_payload':{'old_logical_bytes_per_full_token_row':hc*h*2+hc*4,
        'candidate_logical_bytes_per_full_token_row':h*2,
        'premises':'full HC buffer has no required consumer in frozen DSpark domain; preserve rowwise FP32 collapse and BF16 rounding; same gather ordering/padding',
        'physical_link_bytes_or_speedup':None}}

exports=read(W/'tail-state-exports.json')
old=read(R/'diagnostic-20261029/official-analysis/operators.json')
hashes={x['source_dir']:x['sha256']['kernel_details.csv'] for x in old['rows']}
families=[]
for label,typ,shape,dtype in [
    ('full_hc_copy_compatible','TensorMove','"24,20480"','DT_BF16'),
    ('full_hc_cast_compatible','Cast','"24,4,5120"','DT_BF16'),
    ('full_hc_mul_compatible','Mul','"24,4,1;24,4,5120"','FLOAT;FLOAT'),
    ('full_hc_reduce_compatible','ReduceSum','"24,4,5120;1"','FLOAT;INT64'),
]:
    ranks=[]
    for row in exports['rows']:
        assert row['csv_sha256']==hashes[row['source_dir']]
        selected=[x for x in row['groups'] if x['type']==typ and x['input_shapes']==shape and x['input_dtypes']==dtype]
        assert len(selected)==1,(label,row['rank'],len(selected))
        group=selected[0];durations=[int(x) for x,n in group['duration_histogram_ns'].items() for _ in range(n)]
        assert len(durations)==group['count']==12 and sum(durations)==group['sum_ns']
        assert statistics.median(durations)==group['median_ns']
        ranks.append({'rank':row['rank'],'count':group['count'],'median_ns':group['median_ns']})
    assert {x['rank'] for x in ranks}==set(range(16))
    families.append({'label':label,'type':typ,'input_shapes':shape,'input_dtypes':dtype,'per_rank':ranks,
        'rank_median_of_call_medians_ns':statistics.median(x['median_ns'] for x in ranks)})
audit=read(W/'mtp-buffer-symbol-audit.json')
matches=[name+'/'+x['source']+':'+str(x['line']) for name,repo in audit['repositories'].items() for x in repo['matches']]
accounts['same_capture_families']=families
accounts['symbol_search_match_count']=len(matches)
accounts['symbol_search_does_not_close_reflection_or_arbitrary_observers']=True
accounts['config_sha256']=hashlib.sha256((R/'evidence/model-config.json').read_bytes()).hexdigest()
(W/'initialization-tail-evidence.json').write_text(json.dumps(accounts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'vision_vit_params':vit_params,'vision_aligner_params':aligner_params,'mtp_buffer_mib_if_bf16':accounts['mtp_buffer']['max_capacity_mib_if_bf16'],
    'rank_family_median_us':{x['label']:x['rank_median_of_call_medians_ns']/1000 for x in families},
    'symbol_matches':len(matches),'physical_traffic_or_gain_proven':False},ensure_ascii=False))
