"""Classify visible first-attention stream patterns, not their semantic causes."""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict,Counter
import bisect,csv,json
B=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis');R=B/'operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces';T=B/'operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
groups=defaultdict(list)
for x in json.loads((T/'eager-target-attribution-v2/calls.json').read_text()):groups[x['rank']].append(x)
total=Counter();ranks=[]
for rank,cs in sorted(groups.items()):
 streams=defaultdict(list)
 for y in csv.DictReader((R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if y['Model ID']!='4294967295':continue
  y['s']=int(Decimal(y['Start Time(us)'])*1000);streams[y['Stream ID']].append(y)
 for k in streams:streams[k].sort(key=lambda x:x['s'])
 starts={k:[y['s'] for y in v] for k,v in streams.items()};counts=Counter()
 for x in cs:
  if x['attention_state']!='NO_EXPORTED_ATTENTION_UNRESOLVED':continue
  v=streams[x['stream_id']];i=bisect.bisect_left(starts[x['stream_id']],x['window_start_ns']);part=[];sawpost=False
  for y in v[i:]:
   if y['s']>=x['window_end_ns']:break
   if y['Type']=='HcPost':sawpost=True;break
   part.append(y)
  assert sawpost
  z=[y for y in part if y['Type']=='ZerosLike' and y['Input Shapes'].strip(chr(34)).endswith(',64,512')]
  q=[y for y in part if y['Type']=='QuantBatchMatmulV3' and '32768' in y['Output Shapes']]
  counts[(len(z),len(q))]+=1;total[(len(z),len(q))]+=1
 ranks.append({'rank':rank,'groups':[{'zero_64_512_count':a,'matmul_output_32768_count':b,'windows':n} for (a,b),n in sorted(counts.items())]})
assert total=={(1,0):400,(0,0):320},total
x={'status':'VISIBLE_PATTERN_CLASSIFICATION_NOT_BRANCH_IDENTITY','windows':sum(total.values()),'groups':[{'zero_64_512_count':a,'matmul_output_32768_count':b,'windows':n} for (a,b),n in sorted(total.items())], 'ranks':ranks,'limits':['Only the same model and stream between first HcPre and first HcPost inside each no-attention bracket are scanned.','ZerosLike shape is output padding, not direct measurement of query length.','A missing exported 32768-wide QuantBatchMatmulV3 does not prove no query projection of any form.','No individual subwindow is assigned to an empty-query branch based only on these patterns.'],'no_device_work':True}
p=T/'evidence/eager-no-attention-patterns.json';assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'status':x['status'],'groups':x['groups']}))
