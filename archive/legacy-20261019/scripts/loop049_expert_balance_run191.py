#!/usr/bin/env python3
"""Offline fixed-placement cross-cycle screen; no model/service interaction."""
import json, random, math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'evidence/20260925_loop049_expert_balance/run191'
rows={cycle:[json.loads((root/f'evidence/20260925_loop039_gmm/run121/counts/rank{rank}_cycle{cycle}.json').read_text()) for rank in range(8)] for cycle in (64,65)}
results=[]
def objective(bits,place):
    counts=[0]*8
    for expert,rank in enumerate(place):
        counts[rank]+=bits[expert]
    return max(counts),counts
for layer in range(43,86):
    bits={cycle:[int(v>0) for rank in range(8) for v in rows[cycle][rank]['rows'][layer]['counts']] for cycle in (64,65)}
    assert all(len(bits[c])==256 for c in bits)
    base=[e//32 for e in range(256)]
    for train,test in ((64,65),(65,64)):
        rng=random.Random(191000+layer*100+train)
        place=base[:]
        best,_=objective(bits[train],place)
        # Pair-swap keeps 32 experts per rank, accepts only train improvement.
        for _ in range(8000):
            a,b=rng.sample(range(256),2)
            if place[a]==place[b]: continue
            place[a],place[b]=place[b],place[a]
            candidate,_=objective(bits[train],place)
            if candidate<best:
                best=candidate
            else:
                place[a],place[b]=place[b],place[a]
        test_score,_=objective(bits[test],place)
        base_train,_=objective(bits[train],base)
        base_test,_=objective(bits[test],base)
        results.append({'layer_ordinal':layer,'train_cycle':train,'test_cycle':test,'baseline_train_max':base_train,'candidate_train_max':best,'baseline_test_max':base_test,'candidate_test_max':test_score,'test_delta':base_test-test_score,'theoretical_test_min':math.ceil(sum(bits[test])/8)})
summary={}
for train,test in ((64,65),(65,64)):
    part=[x for x in results if x['train_cycle']==train]
    summary[f'{train}_to_{test}']={'baseline_train_sum':sum(x['baseline_train_max'] for x in part),'candidate_train_sum':sum(x['candidate_train_max'] for x in part),'baseline_test_sum':sum(x['baseline_test_max'] for x in part),'candidate_test_sum':sum(x['candidate_test_max'] for x in part),'cross_cycle_test_gain_reads':sum(x['test_delta'] for x in part),'cross_cycle_test_gain_ms_at_1tb_s':sum(x['test_delta'] for x in part)*12*1024*1024/1e12*1000,'test_layers_improved':sum(x['test_delta']>0 for x in part),'test_layers_worsened':sum(x['test_delta']<0 for x in part),'test_layers_equal':sum(x['test_delta']==0 for x in part),'ideal_test_min_sum':sum(x['theoretical_test_min'] for x in part)}
result={'run':'run191','method':'Deterministic, per-layer pair-swap hill climb with 8000 trials; train on one Run121 cycle and evaluate fixed 32-expert/rank placement on the other; objective sum of per-layer maximum active expert counts.','summary':summary,'layers':results,'limitations':['Only two consecutive captured cycles, not a representative held-out workload; learned placement may overfit.','Source Run121 counts are active flags rather than per-expert compute or bytes; max active count is a proxy, not runtime duration.','Static remap is currently unsafe because execution map and checkpoint weight loading map differ (Run190).','No device or product throughput measurement.']}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
