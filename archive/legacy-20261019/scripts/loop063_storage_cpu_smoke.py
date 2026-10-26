import torch
from runtime.storage_intervals import audit_disjoint
x=torch.empty(16)
y=torch.empty(16)
a=audit_disjoint({'x':x[:8]},{'y':y,'x_tail':x[8:]})
b=audit_disjoint({'x':x[4:12]},{'x_head':x[:8]})
assert a['pass_'] and not b['pass_']
print('storage interval smoke pass')
