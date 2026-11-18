"""CPU-only new formal256 dataset, preserving every old campaign byte."""
from pathlib import Path
import ast,hashlib,json,os,time,traceback,uuid
from tokenizers import Tokenizer
R=Path('/task/client_glm53_g576');OLD=Path('/task/client_vllm_g42');DATA=R/'datasets';CFG=R/'configs';START=time.monotonic()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):
 p=R/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
try:
 save('assets_state_g584.json',{'phase':'CPU_assets_preparing','pid':os.getpid(),'at_unix':time.time(),'inference_sent':False});assert not DATA.exists() and not CFG.exists();DATA.mkdir();CFG.mkdir()
 m=json.loads((OLD/'dataset_manifest.json').read_text());path=Path('/models/glm53/tokenizer.json');assert sha(path)==m['tokenizer_sha256'];tok=Tokenizer.from_file(str(path));prior_prefixes=set();prior=[]
 for e in m['profiles']:
  p=OLD/'datasets'/(e['phase']+'_'+str(e['input_tokens'])+'.jsonl');assert sha(p)==e['sha256'];rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  assert len(rows)==e['requests'];prior_prefixes.update(tuple(tok.encode(x['question'][:8192],add_special_tokens=False).ids[:128]) for x in rows);prior.append({'phase':e['phase'],'length':e['input_tokens'],'sha256':e['sha256'],'requests':e['requests']})
 new=[]
 for phase in ['warmup','warmup2','warmup3','warmup4','warmup5']:
  p=OLD/'datasets'/(phase+'_262144.jsonl');e=next(x for x in m['profiles'] if x['phase']==phase and x['input_tokens']==262144);assert sha(p)==e['sha256'];out=DATA/p.name;out.write_bytes(p.read_bytes());rows=[json.loads(l) for l in out.read_text().splitlines() if l.strip()];assert len(rows)==16
  assert all(len(tok.encode(x['question'],add_special_tokens=False).ids)==262144 for x in rows)
  new.append({'phase':phase,'input_tokens':262144,'requests':16,'min_actual':262144,'max_actual':262144,'path':str(out),'sha256':sha(out),'role':'Original completedwarmbytes copiedforprovenance; no newwarmuprequests sent or fabricated'})
 source=OLD/'datasets/formal_262144.jsonl';e=next(x for x in m['profiles'] if x['phase']=='formal' and x['input_tokens']==262144);assert sha(source)==e['sha256'];rows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()];assert len(rows)==32
 output=[];fresh=set()
 for i,row in enumerate(rows):
  assert time.monotonic()-START<850
  nonce='glm-g565-first-repair-256-'+uuid.uuid4().hex;ids=tok.encode(nonce+'\n'+row['question'],add_special_tokens=False).ids
  for _ in range(8):
   text=tok.decode(ids[:262144],skip_special_tokens=False);actual=tok.encode(text,add_special_tokens=False).ids
   if len(actual)==262144:break
   ids=actual+ids[262144:262272]
  assert len(actual)==262144 and text.startswith(nonce)
  prefix=tuple(actual[:128]);assert prefix not in prior_prefixes and prefix not in fresh;fresh.add(prefix)
  output.append(dict(question=text,answer='',max_out_len=1024,input_tokens=262144,request_id=nonce))
 out=DATA/'formal_262144.jsonl';out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output));new.append({'phase':'formal','input_tokens':262144,'requests':32,'min_actual':262144,'max_actual':262144,'path':str(out),'sha256':sha(out),'source_original_formal_sha256':e['sha256'],'role':'Freshunsubmittednoncefirst128prefixes; originalfirstfailedretained; not byteidenticaltooriginalV4dataset'})
 for phase in ['warmup','warmup2','warmup3','warmup4','warmup5','formal']:
  p=Path('/task/client_configs_sg_g199')/('sglang_'+phase+'_262144.py');s=p.read_text().replace('/task/client_vllm_g42/datasets/',str(DATA)+'/');ast.parse(s);(CFG/p.name).write_text(s)
 save('dataset_manifest_g584.json',{'schema':1,'tokenizer_sha256':m['tokenizer_sha256'],'client_concurrency':16,'output_tokens':1024,'profiles':new,'previous_profiles':prior,'original_manifest_sha256':sha(OLD/'dataset_manifest.json'),'same_G316_compute_epoch':True,'router_epoch':'G565','warmup_mode':'Existing five actualcompleted80requestwarmproof notrerun. Must beexplicitlyreviewedwithsameP/Dcurrentcachebeforetrial; no warmcommands fabricated.','formal_first_status':'Freshdatasetprepared_not_submitted','old256_first_status':'18success14router503_failed_nohot_noreplay','V4_exact_newdataset_comparison':'Requiresnewcounterparttrial; not claimedhere'})
 save('new_first_prefix_audit_g584.json',{'verified':True,'prefix_tokens':128,'fresh_formal_prefixes':len(fresh),'no_collisions_with_all_previous_datasets':True,'previous_profile_count':len(prior),'all_new_formal_exact262144':True,'tokenizer_sha256':m['tokenizer_sha256'],'formal_sha256':sha(out),'inference_sent':False})
 save('assets_state_g584.json',{'phase':'complete','pid':os.getpid(),'elapsed_s':time.monotonic()-START,'newformal_rows':32,'copiedcompletedwarm_rows':80,'original_bytes_untouched':True,'inference_sent':False,'server_usage_verified_for_newdata':False,'run_controller_prepared':False,'scope':'CPUassets only; warmproof import/newusage-sourceandfirsttrial gates stillrequired'})
except BaseException as e:
 save('assets_state_g584.json',{'phase':'failed','error':repr(e),'traceback':traceback.format_exc(),'inference_sent':False});raise
