"""Inspect a two-millisecond host API tail, with no causal ownership claim."""
from pathlib import Path
from decimal import Decimal,InvalidOperation
from collections import Counter
import csv,json,re,sqlite3
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'host-gap-tail-inspection';O.mkdir(exist_ok=False)
def n(s):
 try:return int(Decimal(str(s)))
 except (InvalidOperation,ValueError,TypeError):return None
gaps=list(csv.DictReader((T/'inter-forward-gap-analysis-v2/rank_gaps.csv').open()));summaries=[]
for ep in [0,8]:
 choices=[r for r in gaps if '_ep'+str(ep)+'_' in r['rank'] and r['target_forward_ordinal']=='7'];assert len(choices)==1
 row=choices[0];end=int(row['next_target_layer0_start_ns']);begin=end-2_000_000
 m=re.search(r'_rank\d+_(\d+)_',row['rank']);assert m;pid=int(m.group(1));guessed_main_tid=(pid<<32)|pid
 db=next((ROOT/row['rank']/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
 bytype={}
 for table,filtertype in [('PYTORCH_API',50001),('CANN_API',None)]:
  sql='SELECT a.startNs,a.endNs,a.type,a.globalTid,s.value FROM '+table+' a LEFT JOIN STRING_IDS s ON a.name=s.id WHERE a.globalTid=?'
  selected=[]
  for x,y,typ,tid,name in c.execute(sql,(guessed_main_tid,)):
   x=n(x);y=n(y)
   if x is None or y is None or x>=end or y<=begin or (filtertype is not None and typ!=filtertype):continue
   selected.append({'name':name,'type':typ,'start_relative_to_layer0_us':(x-end)/1000,'duration_us':(y-x)/1000,'global_tid':tid})
  selected.sort(key=lambda z:z['start_relative_to_layer0_us']);bytype[table]={'count':len(selected),
   'last_40_by_start':selected[-40:],'top_names':Counter(x['name'] for x in selected).most_common(12)}
 c.close();summaries.append({'ep':ep,'rank':row['rank'],'rank_filename_pid':pid,'guessed_main_global_tid':guessed_main_tid,
                            'global_tid_bit_encoding_status':'HEURISTIC_FROM_FILENAME_AND_OBSERVED_IDS_NOT_INDEPENDENTLY_PROVEN',
                            'interval_relative_to_layer0_us':[-2000,0],'APIs':bytype})
doc={'status':'BOUNDED_HEURISTIC_MAIN_THREAD_API_TAIL','windows':summaries,
 'limitations':['Thread selection uses an unverified globalTid encoding heuristic.',
                'Profiler CPU/device clock correspondence is not independently calibrated.',
                'Time proximity or matching thread does not prove the API caused this replay or the inter-DP gap.',
                'A long-lived or nested CPU frame is not a device activity counter.'],
 'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
