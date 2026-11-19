"""CPU reference and conditional short-prefix KV history audit on real snapshots.
Run on the server CPU client; raw tensors remain in controlled task storage.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import torch
from analyze_real_attention_g1100 import analyze

def ordered(d,kind):
    slots=d[kind+'_slots'].tolist();values=d[kind+'_values'];mapping={s:i for i,s in enumerate(slots)}
    indices=d[kind+'_indices'][-1].tolist();valid=[s for s in indices if s>=0]
    return valid,values[[mapping[s] for s in valid]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();assert not args.output.exists();torch.set_num_threads(4)
    files=sorted(args.input.glob('*.pt'));assert 1<=len(files)<=288;start=time.monotonic();rows=[analyze(p) for p in files]
    groups={}
    for p in files:
        d=torch.load(p,map_location='cpu',weights_only=True);groups.setdefault((d['pid'],d['ratio'],d['layer_id']),[]).append((p,d))
    comparisons=[];counts=[]
    for key,items in groups.items():
        previous=None;segment=-1
        for p,d in sorted(items,key=lambda x:x[1]['call_in_ratio']):
            positions=d['positions'].tolist();pos=int(positions[d['q_rows'][-1]])
            if previous is None or pos<=previous[2]:segment+=1;previous=None
            oi,ov=ordered(d,'original');ci,cv=ordered(d,'compressed')
            counts.append({'file':p.name,'pid':d['pid'],'ratio':d['ratio'],'segment':segment,'position':pos,
                           'original_valid':len(oi),'compressed_valid':len(ci),
                           'short_prefix_expected_if_window128_and_all_candidates':{'original':min(pos+1,128),'compressed':min((pos+1)//d['ratio'],512) if d['ratio'] else 0}})
            if previous is not None:
                oldfile,old,oldpos,oldov,oldcv=previous
                if pos==oldpos+1 and 0<=oldpos<127:
                    value={'old_file':oldfile,'new_file':p.name,'pid':d['pid'],'ratio':d['ratio'],'segment':segment,'positions':[oldpos,pos],
                           'scope':'Conditional history stability: same request, short causal/full-candidate prefix in native index order. Recompute/reordering semantics must be checked before calling a mismatch a bug.'}
                    for kind,oldvalues,newvalues in [('original',oldov,ov),('compressed',oldcv,cv)]:
                        n=min(oldvalues.shape[0],newvalues.shape[0]);diff=(oldvalues[:n].float()-newvalues[:n].float()).abs()
                        value[kind]={'compared_rows':n,'old_rows':oldvalues.shape[0],'new_rows':newvalues.shape[0],
                                     'equal_elements':int((diff==0).sum()),'elements':diff.numel(),'max_abs':diff.max().item() if diff.numel() else 0.,
                                     'exact':bool(torch.equal(oldvalues[:n],newvalues[:n]))}
                    comparisons.append(value)
            previous=(p.name,d,pos,ov,cv)
    out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'elapsed_s':time.monotonic()-start,'reference_results':rows,
         'history_comparisons':comparisons,'causal_counts':counts,'physical_rank_pids':sorted({r['pid'] for r in rows}),
         'outside_numeric_guard_elements':sum(r['outside_two_ULP_or_FP32_nearzero_allowance'] for v in rows for r in v['rows']),
         'conditional_history_mismatches':sum(not x['original']['exact'] or not x['compressed']['exact'] for x in comparisons),
         'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'reference_source_SHA256':hashlib.sha256(Path(__file__).with_name('analyze_real_attention_g1100.py').read_bytes()).hexdigest(),
         'scope':'Actual operator FP32 reference and conditional cache-history observations; no fullmodel quality/performance acceptance.'}
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['files','elapsed_s','outside_numeric_guard_elements','conditional_history_mismatches']}),flush=True)
if __name__=='__main__':main()
