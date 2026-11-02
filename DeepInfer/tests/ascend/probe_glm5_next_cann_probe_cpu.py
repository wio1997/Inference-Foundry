# SPDX-License-Identifier: Apache-2.0
"""Validate future PKI fixtures on actual config, CPU-only, without dispatch."""
import argparse,datetime,hashlib,json,os,platform
from pathlib import Path
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.glm5_next_cann_probe import pki_probe_cases,plan_pki_case,expected_pki_tokens,GEOMETRY
from deepinfer.model_executor.models.glm5_next.cann_indexer import CANNIndexerContract,prepare_nonempty_pki_cpu,pack_cann_token_indices_cpu

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);c=CANNIndexerContract.from_config(Glm5NextConfig.from_checkpoint(a.checkpoint));assert all(getattr(c,k)==v for k,v in GEOMETRY.items())
src=Path(__file__).resolve().parents[2]
files=['deepinfer/hugging_face/glm5_next_cann_probe.py','tests/ascend/probe_glm5_next_cann_indexer_npu.py','tests/ascend/test_glm5_next_cann_probe.py','tests/ascend/probe_glm5_next_cann_probe_cpu.py']
report={'status':'in_progress','NPU_execution':False,'native_imports':False,'ABI_validated':False,'model_generation':False,'weights_read_bytes':0,
        'architecture':platform.machine(),'torch':torch.__version__,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'checks':{},'cases':[]}
def check(name,condition):
    if not condition:raise AssertionError(name)
    report['checks'][name]={'passed':True}
try:
    for case in pki_probe_cases():
        plan=plan_pki_case(case);t=case['tokens'];oracle=expected_pki_tokens(case);name=case['name']
        cache=torch.zeros(plan['blocks'],32,1,128,dtype=torch.bfloat16)
        for rows in plan['pool_rows']:
            for block,row,ordinal in rows:
                cache[block,row,0,0]=1+ordinal//128;cache[block,row,0,1]=(ordinal%128)/128
        indices=torch.full((t,2048),-999,dtype=torch.int32);previous=0
        # Independent FP64 score-sort from physically addressed BF16 payload,
        # then scalar pool expansion; it does not call the token-set oracle.
        scalar=[]
        for request,(end,count,tail) in enumerate(zip(case['ends'],case['lengths'],case['tails'],strict=True)):
            values=cache[torch.tensor(plan['table'][request])[:max(1,(count+31)//32)]].reshape(-1,128)[:count].double().sum(-1)*32
            if count>1:check(name+f'/request{request}_strict_scores',bool((values[1:]>values[:-1]).all()))
            if end>previous:
                chosen=sorted(range(count),key=lambda j:float(values[j]),reverse=True)[:512]
                expanded=[j*4+k for j in chosen for k in range(4)];indices[previous,:len(expanded)]=torch.tensor(expanded,dtype=torch.int32)
                scalar.append(sorted(expanded+list(range(count*4,count*4+tail))))
            previous=end
        check(name+'/independent_physical_score_oracle',scalar==[oracle[row] for row in range(plan['live_tokens'])])
        ends=torch.tensor(case['ends'],dtype=torch.int32);lengths=torch.tensor(case['lengths'],dtype=torch.int32);tails=torch.tensor(case['tails'],dtype=torch.int32)
        table=torch.tensor(plan['table'],dtype=torch.int32).reshape(len(case['ends']),plan['columns'])
        inputs=prepare_nonempty_pki_cpu(torch.ones(t,32,128,dtype=torch.bfloat16),torch.ones(t,32,dtype=torch.bfloat16),ends,lengths,tails,table,c,num_kernel_blocks=plan['blocks'])
        check(name+'/sentinel_metadata',inputs[2].tolist()==case['ends']+[t+1] and inputs[3].tolist()==[max(1,x) for x in case['lengths']]+[1] and inputs[5][-1].tolist()==[0]*plan['columns'])
        packed=pack_cann_token_indices_cpu(indices,torch.tensor(case['positions'],dtype=torch.int64),ends,c,index_namespace='expanded_token')
        check(name+'/shape_dtype',tuple(packed.shape)==(t,1,2051) and packed.dtype==torch.int32)
        check(name+'/ranked_history_tail_and_padding',all(sorted([v for v in packed[row,0].tolist() if v>=0])==oracle[row] for row in range(t)))
        check(name+'/unused_columns_sentinel',all(packed[row,0,len(oracle[row]):].tolist()==[-1]*(2051-len(oracle[row])) for row in range(t)))
        report['cases'].append({'name':name,'tokens':t,'requests':len(case['ends']),'cache_bytes':plan['cache_bytes'],'selected_counts':[len(v) for v in oracle]})
    # Dropping the excluded-lowest pool is the differentiating top-k condition.
    mixed=pki_probe_cases()[1];wanted=expected_pki_tokens(mixed)[2]
    check('ranking_excludes_lowest_pool',wanted==list(range(4,2052)) and len(wanted)==2048)
    report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'cann-probe-preparation-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
