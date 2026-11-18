"""Prepare frozen exact-length retrieval probes, then run them under the parent's PD workload lock.

Preparation uses CPU tokenization only. Neither command launches/restarts model services.
The parent must verify actual ownership and an idle serving/client pair before either command.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,time,urllib.error,urllib.request,uuid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def post(base,route,body,timeout=120):
 req=urllib.request.Request(base.rstrip('/')+route,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as f:assert f.status==200;return f.read()
def get(url,timeout=30):
 with urllib.request.urlopen(url,timeout=timeout) as f:assert f.status==200;return f.read()
def tokenized(base,body):
 d=json.loads(post(base,'/tokenize',body));assert isinstance(d['tokens'],list) and all(isinstance(t,int) for t in d['tokens']);assert d['count']==len(d['tokens']);return d

def prepare(a):
 from tokenizers import Tokenizer
 root=a.root.resolve();root.mkdir(parents=True,exist_ok=False);os.chdir(root)
 modelmeta=json.loads(get(a.tokenize_url.rstrip('/')+'/v1/models'));model=modelmeta['data'][0]['id'];assert model=='ds41flash'
 tokpath=Path('/models/ds41/tokenizer.json');assert sha(tokpath)=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
 tok=Tokenizer.from_file(str(tokpath));dm=json.loads(a.dataset_manifest.read_text());reserved=set();rows_checked=0
 for item in dm['profiles']:
  p=Path(item['path']);assert sha(p)==item['sha256']
  for line in p.read_text().splitlines():
   row=json.loads(line);prefix=tuple(tok.encode(row['question'][:5000],add_special_tokens=False).ids[:128]);assert len(prefix)==128;reserved.add(prefix);rows_checked+=1
 assert rows_checked==len(reserved)==448
 nonce='DS41-LONG-QUALITY-G703-'+uuid.uuid4().hex;records=[]
 sentence='This background paragraph discusses memory, storage, communication, and ordinary scheduling. '
 for length in a.lengths:
  labels=[f'REC{length}{x}' for x in ['A','B','C']]
  values=['VAL'+hashlib.sha256((nonce+label).encode()).hexdigest()[:12].upper() for label in labels];expected='|'.join(values)
  def chat(repeats,padding):
   parts=[nonce+' LENGTH '+str(length)+'\nArchived document follows.\n'];j=0
   for i in range(8):
    parts.append(sentence*repeats+'\n')
    if i in [0,3,6]:parts.append(f'Archive marker {labels[j]} has recorded value {values[j]}.\n');j+=1
   parts.append(' x'*padding+'\nQuestion: Return the recorded values for markers '+', '.join(labels)+' in exactly that order, separated by |. Output only the three values and separators, without explanations.')
   return {'model':model,'messages':[{'role':'user','content':''.join(parts)}],'add_generation_prompt':True,'add_special_tokens':False,'chat_template_kwargs':{'thinking':False},'return_token_strs':False}
  small=tokenized(a.tokenize_url,chat(1,0));twice=tokenized(a.tokenize_url,chat(2,0));stride=twice['count']-small['count'];assert stride>0
  repeats=max(0,1+(length-small['count'])//stride);padding=0
  for _ in range(8):
   cb=chat(repeats,padding);td=tokenized(a.tokenize_url,cb);delta=length-td['count']
   if delta==0:break
   if delta<0 and padding+delta<0:repeats-=1;assert repeats>=0;padding=0
   else:padding+=delta;assert padding>=0
  assert td['count']==length,(length,td['count'])
  rendered=json.loads(post(a.tokenize_url,'/detokenize',{'model':model,'tokens':td['tokens']}))['prompt'];assert isinstance(rendered,str)
  roundtrip=tokenized(a.tokenize_url,{'model':model,'prompt':rendered,'add_special_tokens':False,'return_token_strs':False});assert roundtrip['tokens']==td['tokens'],'Rendered chat did not roundtrip to identical completion token IDs'
  cpu_ids=tok.encode(rendered,add_special_tokens=False).ids;assert cpu_ids==td['tokens'],'Actual model tokenizer and serving renderer disagree'
  assert tuple(cpu_ids[:128]) not in reserved,'Reserved benchmark prefix collision'
  request={'model':model,'prompt':rendered,'temperature':0,'seed':0,'max_tokens':128,'stream':False,'add_special_tokens':False}
  p=root/f'request_{length}.json';save(p,request);save(root/f'chat_reference_{length}.json',cb)
  records.append({'input_tokens':length,'expected':expected,'labels':labels,'values':values,'request_file':p.name,'request_sha256':sha(p),'token_ids_sha256':hashlib.sha256(json.dumps(cpu_ids,separators=(',',':')).encode()).hexdigest(),'filler_repeats_per_eighth':repeats,'padding_tokens':padding,'positions':'One marker after each of sections1,4,7 of8; exact finalcount includes official nativechat template','no_reserved_prefix_collision':True})
 manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':model,'nonce':nonce,'tokenizer_sha256':sha(tokpath),'records':records,'benchmark_prefixes_checked':448,'generation_requests_submitted':False,'prompt_contract':'NativeDeepSeekV41chat tokenize( thinkingFalse/addgenerationTrue/addspecialFalse ), detokenized to validUTF8; exactcompletion roundtrip IDs and mounted tokenizer IDs match. Same frozen completion payload for bothframeworks.','grading':'Strict stripped text equals allthree syntheticvalues joinedby|; no format/answer relaxation','scope':'Selected long-context retrieval/cache-coherence probes, not a statistical long-context quality benchmark'}
 save(root/'manifest.json',manifest);print(json.dumps({'prepared':True,'generation_submitted':False,'lengths':[x['input_tokens'] for x in records],'manifest_sha256':sha(root/'manifest.json')}),flush=True)

def run(a):
 assets=a.assets.resolve(strict=True);manifest=json.loads((assets/'manifest.json').read_text());gate=json.loads(a.gate.read_text());assert gate['engine']==a.engine and gate['ready'] and gate['pd_transfer_verified'];assert gate['model_id']==manifest['model'];assert gate['max_model_len']>=262144+128
 for item in manifest['records']:assert sha(assets/item['request_file'])==item['request_sha256']
 out=a.root.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(out);started=time.monotonic()
 def state(phase,**kw):
  d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,**kw};save(out/'state.json',d);print(json.dumps(d),flush=True)
 def alarm(*_):raise TimeoutError('Long-quality whole-job bound reached; no automatic retry')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(a.total_timeout)
 def snapshots(tag):
  for role,item in gate['roles'].items():
   assert json.loads(get(item['base_url'].rstrip('/')+'/v1/models'))['data'][0]['id']==manifest['model'];get(item['base_url'].rstrip('/')+'/health')
   if item.get('metrics_url'):(out/f'{tag}_{role}_metrics.txt').write_bytes(get(item['metrics_url']))
 results=[]
 try:
  assert json.loads(get(gate['proxy_url'].rstrip('/')+'/v1/models'))['data'][0]['id']==manifest['model']
  save(out/'manifest.json',{'engine':a.engine,'assets_manifest_sha256':sha(assets/'manifest.json'),'gate':gate,'grading':manifest['grading'],'scope':manifest['scope'],'attempts_per_request':1,'selected_lengths':a.lengths})
  for item in manifest['records']:
   if item['input_tokens'] not in a.lengths:continue
   length=item['input_tokens'];body=json.loads((assets/item['request_file']).read_text());prior=None
   for phase in ['fresh','replay']:
    tag=f'{phase}_{length}';save(out/(tag+'_request.json'),body);snapshots(tag+'_before');state('request_running',case=tag)
    try:raw=post(gate['proxy_url'],'/v1/completions',body,timeout=1800)
    except urllib.error.HTTPError as exc:(out/(tag+'_HTTP_error.txt')).write_bytes(exc.read());raise
    (out/(tag+'_response.json')).write_bytes(raw);r=json.loads(raw);c=r['choices'][0];text=c.get('text') or '';usage=r.get('usage') or {};snapshots(tag+'_after')
    result={'case':tag,'input_tokens':length,'usage':usage,'expected':item['expected'],'actual':text.strip(),'strict_answer_passed':text.strip()==item['expected'],'finish_reason':c.get('finish_reason'),'response_sha256':hashlib.sha256(raw).hexdigest(),'scope':manifest['scope']};save(out/(tag+'_result.json'),result);results.append(result)
    assert usage.get('prompt_tokens')==length and 0<usage.get('completion_tokens',0)<=128,(tag,usage)
    assert c.get('finish_reason')=='stop' and result['strict_answer_passed'],(tag,result)
    if phase=='replay':
     assert prior==text,'Same valid-text prompt changed exact retrieved values'
     hits={}
     for role,role_gate in gate['roles'].items():
      key=role_gate.get('prefix_hit_counter');assert key,'Reviewed actual prefix counter required for cache replay'
      pat=r'^'+re.escape(key)+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?$'
      values=[]
      for boundary in ['before','after']:
       s=(out/f'{tag}_{boundary}_{role}_metrics.txt').read_text();nums=[float(x) for x in re.findall(pat,s,re.M)];assert nums;values.append(sum(nums))
      hits[role]=values[1]-values[0];assert hits[role]>0,(tag,role,hits)
     save(out/(tag+'_cache_coherence.json'),{'exact_same_answer':True,'actual_local_prefix_hit_deltas':hits,'scope':'Endtoendselectedretrieval replay, not directallcomponentstatebytes'})
    prior=text;state('case_passed',case=tag)
  save(out/'result.json',{'passed':True,'cases':results,'scope':manifest['scope'],'full_model_quality_accepted':False});state('selected_long_quality_complete_requires_parent_review')
 except BaseException as exc:state('failed_requires_review',error=repr(exc));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--tokenize-url');p.add_argument('--dataset-manifest',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--gate',type=Path);p.add_argument('--engine',choices=['vllm','sglang']);p.add_argument('--lengths',nargs='+',type=int,choices=[32768,65536,131072,262144],default=[32768,65536,131072,262144]);p.add_argument('--total-timeout',type=int,default=14400);a=p.parse_args();assert a.root.is_absolute()
 if a.mode=='prepare':assert a.tokenize_url and a.dataset_manifest;prepare(a)
 else:assert a.assets and a.gate and a.engine;run(a)
