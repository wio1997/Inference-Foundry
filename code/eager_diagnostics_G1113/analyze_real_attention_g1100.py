"""CPU FP32 reference for actual bounded attention snapshots, on server only.
Outputs discrepancy measurements; does not assert full model quality.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,time
import torch

def analyze(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    assert d['probe']=='G1096_REAL_ATTENTION' and d['q'].device.type=='cpu'
    q=d['q'].float();out=d['result'].float();sink=d['sink'].float()
    assert q.ndim==3 and q.shape[-1]==512 and q.shape==out.shape
    om={int(s):i for i,s in enumerate(d['original_slots'].tolist())};cm={int(s):i for i,s in enumerate(d['compressed_slots'].tolist())}
    rows=[]
    for i in range(q.shape[0]):
        ov=[om[j] for j in d['original_indices'][i].tolist() if j>=0]
        cv=[cm[j] for j in d['compressed_indices'][i].tolist() if j>=0]
        k=torch.cat([d['original_values'][ov],d['compressed_values'][cv]],dim=0).float()
        scores=q[i]@k.T*float(d['scaling'])
        probs=torch.softmax(torch.cat([scores,sink[:,None]],dim=1),dim=1)
        expected=probs[:,:-1]@k
        rounded=expected.to(torch.bfloat16);diff=(out[i]-expected).abs();bdiff=(out[i]-rounded.float()).abs()
        ulp=(torch.nextafter(rounded,torch.full_like(rounded,float('inf'))).float()-rounded.float()).abs()
        # An empirical operator discrepancy guard, not a full-model gate.
        # BF16 two-ULP plus small absolute allowance for FP32 accumulation near 0.
        allowance=torch.maximum(2*ulp,2e-5+2e-5*expected.abs())
        rows.append({'row':int(d['q_rows'][i]),'original_valid':len(ov),'compressed_valid':len(cv),
                     'heads':q.shape[1],'max_abs_vs_FP32':diff.max().item(),'RMSE_vs_FP32':diff.square().mean().sqrt().item(),
                     'max_abs_vs_BF16_reference':bdiff.max().item(),'exact_BF16_elements':int((out[i]==rounded.float()).sum()),
                     'elements':out[i].numel(),'outside_two_ULP_or_FP32_nearzero_allowance':int((bdiff>allowance).sum()),
                     'max_normalized_excess':(bdiff/allowance).max().item()})
    return {'file':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'pid':d['pid'],'layer_id':d['layer_id'],'ratio':d['ratio'],'mode':d['mode'],'call_in_ratio':d['call_in_ratio'],
            'full_q_shape':d['full_q_shape'],'positions':d['positions'].tolist(),'input_ids':d['input_ids'].tolist(),
            'rows':rows,'finite':bool(torch.isfinite(q).all() and torch.isfinite(out).all()),
            'scope':'Actual operator inputs/outputs vs CPU FP32 shared-KV+sink reference. Does not validate upstream cache contents, indexer decisions, model quality, or performance.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert args.input.is_dir() and not args.output.exists();torch.set_num_threads(4);files=sorted(args.input.glob('*.pt'));assert 1<=len(files)<=288
    start=time.monotonic();results=[analyze(p) for p in files]
    value={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(results),'elapsed_s':time.monotonic()-start,
           'physical_rank_pids':sorted({x['pid'] for x in results}),'results':results,'full_model_quality_accepted':False}
    args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'files':len(results),'ranks':len(value['physical_rank_pids']),'elapsed_s':value['elapsed_s'],'outside_guard_elements':sum(r['outside_two_ULP_or_FP32_nearzero_allowance'] for v in results for r in v['rows'])}),flush=True)
if __name__=='__main__':main()
