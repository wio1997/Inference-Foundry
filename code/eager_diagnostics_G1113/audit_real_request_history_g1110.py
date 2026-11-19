"""Audit C1 real row0 after verifying exact prompt/decoded-ID alignment.
Reuses G1104 tensors; numeric attention reference results remain unchanged.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import torch

def ordered(d,kind):
    m={s:i for i,s in enumerate(d[kind+'_slots'].tolist())};ids=[x for x in d[kind+'_indices'][0].tolist() if x>=0]
    return d[kind+'_values'][[m[x] for x in ids]]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;output=r/'client_ds41_g503/CPU_attention_reference_g1100/actual_G1104_history_row0_g1110.json';assert not output.exists();torch.set_num_threads(4);start=time.monotonic()
    control=r/'client_ds41_g503/diagnostic_attention_armed_g1104_control';diag=r/'client_ds41_g503/diagnostic_attention_armed_g1104';cases={x['case']:x for x in json.loads((control/'cases.json').read_text())};results={x['case']:x['response'] for x in json.loads((diag/'result.json').read_text())};groups={}
    for p in sorted((r/'sg_pd_g1098/forward_probe/G1104').glob('*.pt')):
        d=torch.load(p,map_location='cpu',weights_only=True);assert d['q_rows'][0]==0;groups.setdefault((d['pid'],d['ratio'],d['layer_id']),[]).append((p,d))
    align=[];pairs=[];counts=[]
    for key,items in groups.items():
        current=None;previous=None;decode_idx=0
        for p,d in sorted(items,key=lambda x:x[1]['call_in_ratio']):
            pos=d['positions'].tolist();ids=d['input_ids'].tolist();position=int(pos[0]);error=[];extra=False
            if position==0 and max(pos)>0:
                n=max(pos)+1;matches=[name for name,c in cases.items() if ids[:n]==c['input_ids']]
                if len(matches)==1:current=matches[0]
                else:current=None;error.append('prefill IDs do not exactly match one recorded prompt')
                if pos[:n]!=list(range(n)) or any(pos[n:]):error.append('prefill real-prefix/right-padding positions differ')
                previous=None;decode_idx=0
            else:
                if current is None:error.append('decode without identified real request')
                else:
                    expected_pos=len(cases[current]['input_ids'])+decode_idx;out_ids=results[current]['output_ids']
                    if position!=expected_pos:error.append('decode position differs')
                    if decode_idx>=len(out_ids) or ids[0]!=out_ids[decode_idx]:error.append('decode input ID differs from actual generated-ID sequence')
                    else:extra=out_ids[decode_idx]==1
                if any(pos[1:]):error.append('other nonzero positions in C1 padded decode')
                decode_idx+=1
            align.append({'file':p.name,'pid':d['pid'],'ratio':d['ratio'],'case':current,'position':position,'input_id_row0':ids[0],'alignment_errors':error,'EOS_input_extra_forward':extra,'sampled_rows':d['q_rows'],'full_q_shape':d['full_q_shape']})
            ov=ordered(d,'original');cv=ordered(d,'compressed');counts.append({'file':p.name,'position':position,'ratio':d['ratio'],'original_valid':ov.shape[0],'compressed_valid':cv.shape[0]})
            if previous is not None and not error and previous[5]==current and 0<=previous[2]<position<128:
                oldfile,old,oldpos,oldov,oldcv,oldcase=previous;v={'old_file':oldfile,'new_file':p.name,'case':current,'pid':d['pid'],'ratio':d['ratio'],'positions':[oldpos,position],
                  'scope':'Knownrealrow0, exactAPIprompt/decoded-IDandpositionalignment; shortfull-candidateprefix. Nativeindexorder/recomputationsemanticsstillboundinference.'}
                for kind,a,b in [('original',oldov,ov),('compressed',oldcv,cv)]:
                    n=min(a.shape[0],b.shape[0]);diff=(a[:n].float()-b[:n].float()).abs();v[kind]={'compared_rows':n,'elements':diff.numel(),'equal_elements':int((diff==0).sum()),'max_abs':diff.max().item() if diff.numel() else 0.,'exact':bool(torch.equal(a[:n],b[:n]))}
                pairs.append(v)
            previous=(p.name,d,position,ov,cv,current) if not error else None
    out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_count':len(align),'elapsed_s':time.monotonic()-start,'alignment':align,'alignment_error_count':sum(bool(x['alignment_errors']) for x in align),'history_comparisons':pairs,'history_comparison_count':len(pairs),'history_mismatch_count':sum(not x['original']['exact'] or not x['compressed']['exact'] for x in pairs),'causal_counts':counts,'EOS_extra_forward_snapshot_count':sum(x['EOS_input_extra_forward'] for x in align),'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'CPUonlyactualrowalignmentandconditionalKVhistoryaudit; no NPUrequest or fullmodelquality/performanceacceptance.'};output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['snapshot_count','alignment_error_count','history_comparison_count','history_mismatch_count','EOS_extra_forward_snapshot_count','elapsed_s']}),flush=True)
if __name__=='__main__':main()
