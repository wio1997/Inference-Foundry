# SPDX-License-Identifier: Apache-2.0
"""Actual-config cache geometry; synthetic staged bytes, no model/native work."""
import argparse,datetime,hashlib,json,os,platform,sys
from pathlib import Path
from unittest.mock import patch
import torch
from test_glm5_next_paged_commit import oracle,raw_bytes
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
import deepinfer.model_executor.models.glm5_next.paged_commit as commit

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior-cache',type=Path,required=True)
p.add_argument('--prior-request',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);src=Path(__file__).resolve().parents[2];prior=[json.loads(f.read_bytes()) for f in (a.prior_cache,a.prior_request)]
config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
for old in prior:
    assert old['status']=='passed' and not old['NPU_execution'] and old['config_sha256']==config_sha
    for n,sha in old['source_sha256'].items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==sha
files=['deepinfer/model_executor/models/glm5_next/paged_commit.py','tests/ascend/test_glm5_next_paged_commit.py','tests/ascend/probe_glm5_next_paged_commit_cpu.py']
report={'status':'in_progress','native_round':27,'NPU_execution':False,'model_generation':False,'weights_read_bytes':0,
        'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reused_evidence_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.prior_cache,a.prior_request)},
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'checks':{},'cases':[]}


def check(name,value):
    assert value,name
    report['checks'][name]={'passed':True}


try:
    c=Glm5NextConfig.from_checkpoint(a.checkpoint)
    cases=[(128,[3,127],[5,7],3,False),(256,[255],[6],3,False),(128,[4],[4],0,False),(256,[127],[1],0,False),(128,[],[],3,True),(256,[123],[0],3,True)]
    for case,(block,starts,lengths,padding,no_backing) in enumerate(cases):
        layout=DSACacheLayout.from_config(c,logical_block_size=block);check(f'case{case}_actual_dims',(layout.latent_dim,layout.index_dim,layout.pool_size,layout.lookahead)==(512,128,4,0))
        g=CANNCacheGeometry(layout,(4+padding)*256*4);n=len(starts);pages,tail_pages=(0,0) if no_backing else (6,3)
        raw=[torch.full((pages*g.main_page_bytes,),0xa5,dtype=torch.uint8),torch.full((pages*g.index_page_bytes,),0xa5,dtype=torch.uint8),torch.full((tail_pages*g.tail_page_bytes,),0xa5,dtype=torch.uint8)]
        total=sum(lengths);complete=sum((s+l)//4-s//4 for s,l in zip(starts,lengths))
        table=torch.tensor([[2,1,0],[3,4,5]][:n],dtype=torch.int32) if n and not no_backing else torch.empty((n,0),dtype=torch.int32)
        args=dict(num_main_pages=pages,num_tail_pages=tail_pages,logical_table=table,starts=torch.tensor(starts,dtype=torch.int64),lengths=torch.tensor(lengths,dtype=torch.int64),tail_page_ids=torch.tensor([2,0][:n] if not no_backing else [-1]*n,dtype=torch.int64),
                  latent=(torch.arange(total*512).reshape(total,512).float()/1024).bfloat16(),compressed=(torch.arange(complete*128).reshape(complete,128).float()/256).bfloat16(),tail_state=torch.arange((1+2*n)*4*256).reshape(1+2*n,4,256).float()/512)
        expected=oracle(g,raw,args);before=[raw_bytes(t) for t in raw];batch=commit.stage_dsa_writes(*raw,g,**args)
        check(f'case{case}_stage_does_not_write',[raw_bytes(t) for t in raw]==before)
        summary=batch.commit();check(f'case{case}_all_raw_bytes',[raw_bytes(t) for t in raw]==expected)
        check(f'case{case}_all_payload_rows',summary['rows']['main']==total and summary['rows']['index']==complete and summary['rows']['tail']==sum((s+l)%4 if l else 0 for s,l in zip(starts,lengths)))
        check(f'case{case}_terminal_state',summary['state']=='committed' and not summary['native_execution_validated'])
        if no_backing:check(f'case{case}_no_backing_noop',summary['payload_bytes']==0 and sum(t.numel() for t in raw)==0)
        if case==0:
            # Include failure AFTER tail was written: all three before-images
            # must restore exactly, including untouched pages/padding.
            changed=dict(args)
            for name in ('latent','compressed','tail_state'):changed[name]=args[name]+2
            batch=commit.stage_dsa_writes(*raw,g,**changed);saved=[raw_bytes(t) for t in raw];calls=[]
            def fail_last(target,ids,values):
                target.index_copy_(0,ids,values);calls.append(len(ids))
                if len(calls)==3:raise RuntimeError('injected after tail write')
            with patch.object(commit,'_write_rows',side_effect=fail_last):
                try:batch.commit()
                except RuntimeError:pass
                else:raise AssertionError('late failure did not surface')
            check('all_three_written_then_rolled_back',len(calls)==3 and [raw_bytes(t) for t in raw]==saved and batch.summary()['state']=='aborted')
        report['cases'].append({'logical_block_size':block,'tail_page_rows':g.tail_page_rows,'starts':starts,'lengths':lengths,'backing_bytes':sum(t.numel() for t in raw),'commit':summary,'source_values':'synthetic BF16 latent/compressed and FP32 completed tail scratch'})
    check('native_imports_absent','torch_npu' not in sys.modules and 'fla_npu' not in sys.modules)
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'paged-commit-result.json').write_text(json.dumps(report,indent=2)+'\n')
