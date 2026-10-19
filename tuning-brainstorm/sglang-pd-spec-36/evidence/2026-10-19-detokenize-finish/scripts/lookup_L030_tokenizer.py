#!/usr/bin/env python3
"""Recover only the two exact historical tokenizer assets from a public pinned repo."""
import datetime,hashlib,json,pathlib,sys,urllib.request,urllib.error
R=pathlib.Path(__file__).resolve().parents[1];spec=json.loads((R/sys.argv[1]).read_text());O=R/spec['output'];O.mkdir(parents=True,exist_ok=False)
result={'status':'UNAVAILABLE','scope':'TOKENIZER_ASSET_LOOKUP_ONLY','native_executed':False,'requests':[],'assets':{}}
def fetch(url,limit):
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Inference-Foundry-M033-source-audit/1'}),timeout=60) as r:
   data=r.read(limit+1);assert len(data)<=limit,'asset size limit';result['requests'].append({'url':url,'http_status':r.status,'bytes':len(data)});return data
 except urllib.error.HTTPError as e:
  result['requests'].append({'url':url,'http_status':e.code});return None
 except Exception as e:
  result['requests'].append({'url':url,'error_type':type(e).__name__,'error':str(e)[:200]});return None
try:
 raw=fetch('https://huggingface.co/api/models/'+spec['repository'],4_000_000)
 if raw is not None:
  meta=json.loads(raw);sha=meta['sha'];assert len(sha)==40 and all(c in '0123456789abcdef' for c in sha)
  result['repository']={'id':meta.get('id'),'sha':sha,'private':meta.get('private'),'gated':meta.get('gated')}
  if not meta.get('private') and not meta.get('gated'):
   for name,expected in spec['sha256'].items():
    assert name in ('tokenizer.json','tokenizer_config.json')
    data=fetch('https://huggingface.co/'+spec['repository']+'/resolve/'+sha+'/'+name,80_000_000)
    if data is None:continue
    got=hashlib.sha256(data).hexdigest();matches=got==expected
    result['assets'][name]={'sha256':got,'expected_sha256':expected,'bytes':len(data),'matches_historical':matches}
    if matches:
     p=R/'fixtures/L030-exact-tokenizer'/name;p.parent.mkdir(parents=True,exist_ok=True)
     if p.exists():assert p.read_bytes()==data
     else:p.write_bytes(data)
   if len(result['assets'])==len(spec['sha256']) and all(v['matches_historical'] for v in result['assets'].values()):result['status']='PASS'
except Exception as e:result['error_type']=type(e).__name__;result['error']=str(e)[:300]
result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'assets':result['assets'],'repository':result.get('repository')}));sys.exit(0 if result['status']=='PASS' else 3)
