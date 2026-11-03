#!/usr/bin/env python3
"""CPU-only toy counterexample; not a historical runtime input."""
from decimal import Decimal as D
p=list(map(D,['0.2','0.45','0.35']))
d1=list(map(D,['0.4','0.5','0.1']))
d2=list(map(D,['0.4','0.1','0.5']))
noise=[D(1)]*3; u=D('0.8'); token=0
assert all(sum(x)==1 for x in [p,d1,d2])
assert d1[token]==d2[token] and u>p[token]/d1[token]
def recover(d):
 return max(range(3),key=lambda i:max(D(0),p[i]-d[i])/noise[i])
r1,r2=recover(d1),recover(d2)
assert (r1,r2)==(2,1)
print('same selected token probability:',d1[token]);print('recovered tokens:',r1,r2)
print('Conditional fallback only; no historical runtime or distribution-equivalence claim.')
