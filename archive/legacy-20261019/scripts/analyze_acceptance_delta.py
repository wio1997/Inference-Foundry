import re,json,sys
from pathlib import Path
base=Path(sys.argv[1])
def metrics(p):
 out={}
 for line in Path(p).read_text().splitlines():
  if line.startswith('vllm:spec_decode_num_') and '_created' not in line:
   key=line.rsplit(' ',1)[0];out[key]=float(line.rsplit(' ',1)[1])
 return out
b=metrics(base/'metrics_before.txt');a=metrics(base/'metrics_after.txt');d={k:a[k]-b.get(k,0) for k in a if k in b}
drafts=sum(v for k,v in d.items() if 'spec_decode_num_drafts_total' in k)
draft_tokens=sum(v for k,v in d.items() if 'spec_decode_num_draft_tokens_total' in k)
accepted=sum(v for k,v in d.items() if 'spec_decode_num_accepted_tokens_total' in k and 'per_pos' not in k)
positions={re.search(r'position="([^"]+)"',k).group(1):v for k,v in d.items() if 'accepted_tokens_per_pos_total' in k}
out={'counter_delta':d,'drafts':drafts,'draft_tokens':draft_tokens,'accepted_tokens':accepted,'accepted_per_draft':accepted/drafts if drafts else None,'advanced_tokens_per_cycle':1+accepted/drafts if drafts else None,'acceptance_per_position':{k:v/drafts for k,v in positions.items()},'note':'advanced includes one target token plus accepted draft tokens'}
p=base.parent/'acceptance_delta.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
