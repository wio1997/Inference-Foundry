import json,gzip,pathlib,collections,statistics
from decimal import Decimal as D
P=pathlib.Path('/Users/wio/work/Inference-Foundry-glm5-3/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007/first_step_prefix')
def t(e):return D(str(e['ts']))
def end(e):return t(e)+D(str(e.get('dur',0)))
allr={}
for rank in [0,13,15]:
 x=json.load(gzip.open(P/f'rank{rank}_trace_events.json.gz','rt'));ev=x['events'];e=[a['event'] for a in ev]; bounds=[a for a in e if a.get('name')=='aclnnEmbedding_GatherV2AiCore_GatherV2'];lo=min(map(t,bounds));hi=min(t(a) for a in e if a.get('name')=='aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2')
 nodes={a['args']['connection_id']:a for a in e if a.get('name')=='Node@launch'};dq=[a for a in e if a.get('cat')=='dequeue'];eq={a.get('args',{}).get('correlation_id'):a for a in e if a.get('cat')=='enqueue'};cpu=[a for a in e if a.get('cat')=='cpu_op'];ix={id(a['event']):a['index'] for a in ev};out={}
 for a in e:
  if not a.get('name','').startswith('hcom_') or not lo<=t(a)<hi:continue
  node=nodes.get(a['args']['connection_id']);ds=[b for b in dq if node and b['tid']==node['tid'] and t(b)<=t(node)<end(b)]
  if len(ds)!=1:continue
  en=eq.get(ds[0].get('args',{}).get('correlation_id'))
  if not en:continue
  scopes=[b for b in cpu if b['tid']==en['tid'] and t(b)<=t(en)<=end(b)]
  out[a['name']]={'rank':rank,'comm_index':ix[id(a)],'node_index':ix[id(node)],'enqueue_index':ix[id(en)],'devstart':str(t(a)),'devend':str(end(a)),'hoststart':str(t(en)),'hostend':str(end(en)),'node_start':str(t(node)),'scope':[(b['name'],str(t(b)),str(end(b))) for b in sorted(scopes,key=lambda b:end(b)-t(b))]}
 allr[rank]=out
keys=set.intersection(*(set(x) for x in allr.values()));rows=[]
for key in keys:
 a=[allr[r][key] for r in [0,13,15]]; ds=max(a,key=lambda z:D(z['devstart']));hs=max(a,key=lambda z:D(z['hoststart']));rows.append({'name':key,'latestdev':ds['rank'],'latesthost':hs['rank'],'devskew':str(max(D(z['devstart']) for z in a)-min(D(z['devstart']) for z in a)),'lastdev_to_end':str(max(D(z['devend']) for z in a)-D(ds['devstart'])),'rows':a})
print('joined',len(keys),'perrank',[len(allr[r]) for r in [0,13,15]],'latest dev',collections.Counter(z['latestdev'] for z in rows),'same',sum(z['latestdev']==z['latesthost'] for z in rows))
print('median skew',statistics.median(D(z['devskew']) for z in rows),'medianlastend',statistics.median(D(z['lastdev_to_end']) for z in rows))
for z in sorted(rows,key=lambda z:D(z['devskew']),reverse=True)[:5]:print(json.dumps(z))
pathlib.Path('/private/tmp/challenger_collective_prefix.json').write_text(json.dumps(rows,indent=2))
