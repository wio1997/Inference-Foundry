#!/usr/bin/env python3
"""Small illustrative check only; general equivalence is proved under listed conditions."""
from itertools import product
count=0
for cand in product((-1,0,1,2),repeat=3):
 ordered=sorted(cand,reverse=True)
 for lo in range(3):
  for hi in range(lo+1,4):
   included=[c for c in ordered if lo <= c < hi]
   for q in range(lo,hi):
    out=min(1,min((abs(q-c) for c in included),default=float('inf')))
    bitmap_out=int(q not in set(c for c in cand if c>=0))
    assert out==bitmap_out,(cand,lo,hi,q,out,bitmap_out)
    count+=1
assert count==640
print('checked',count,'candidate/tile/block combinations')
