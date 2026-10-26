#!/usr/bin/env python3
"""Check units/path comparability of Run152 product HCCL and Run237 eager chain."""
import argparse,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);v=a.parse_args()
 r152=json.loads((ROOT/'evidence/20260925_loop044_target/run152/tp_collective_audit.json').read_text())
 r237=json.loads((ROOT/'evidence/20260926_loop058_bound/run237/result.json').read_text())
 windows=r152['windows']
 product=[]
 for w in windows:
  pairs=w['pairs']
  assert 33<=len(pairs)<=41
  total=sum(sum(z['duration_ms'] for z in p) for p in pairs)
  product.append(dict(rank=w['rank'],cycle=w['cycle'],pairs=len(pairs),
                      total_ms=total,mean_task_pair_ms=total/len(pairs)))
 med_product=statistics.median(x['mean_task_pair_ms'] for x in product)
 eager=r237['per_pair_latest_median_ms']
 out=dict(run152_window_pair_count=r152['adjacent_allgather_pairs_per_window'],
          run152_pair_duration_ms_per_window=r152['pair_duration_ms_per_window'],
          run152_product_task_pair_ms_median=med_product,
          run237_eager_latest_rank_pair_ms=eager,
          raw_ratio=eager/med_product,
          comparison='INVALID_FOR_PRODUCT_CAPACITY_OR_SAVINGS',
          reason='Run152 0.838ms is the SUM of 33-41 adjacent pairs per target window, not each pair. Run237 is eager torch.distributed all_gather_into_tensor with host dispatch and no graph/model interleaving. HCCL count may denote a different buffer extent. Neither ratio nor difference is a product link floor or E2E opportunity.',
          run152_per_window=product)
 Path(v.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ['run152_product_task_pair_ms_median','run237_eager_latest_rank_pair_ms','raw_ratio','comparison']},indent=2))
if __name__=='__main__':main()
