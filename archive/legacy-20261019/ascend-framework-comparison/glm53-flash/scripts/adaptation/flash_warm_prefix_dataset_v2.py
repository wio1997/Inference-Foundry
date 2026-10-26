# SPDX-License-Identifier: Apache-2.0
"""Exact Flash token-ID warm dataset, generated under its actual tokenizer.
Generation is not a model call or cache warmup. Raw IDs remain run-private.
"""
import argparse,hashlib,json,pathlib,string
PREFIX=74400;SUFFIX=5600;OUTPUT=600;VOCAB=154856;MODEL_VOCAB=154880

def ids_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def lcp(a,b):
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return i
 return min(len(a),len(b))
def valid(ids,special):
 if not ids or any(type(x) is not int or not 0<=x<VOCAB or x in special for x in ids):raise ValueError('Invalid Flash text token IDs')
def seeded_tokens(tokenizer,seed,count,special):
 for multiplier in [2,4,8]:
  text='Record '+hashlib.sha256(seed.encode()).hexdigest()+'\n'+hashlib.shake_256(seed.encode()).hexdigest(count*multiplier)
  ids=tokenizer.encode(text,add_special_tokens=False)
  if len(ids)>=count:
   ids=list(ids[:count]);valid(ids,special);return ids
 raise ValueError('Synthetic source insufficient: never silently shorten tokens')
def build(tokenizer,concurrency,namespace,output):
 if concurrency not in (1,16,32,64):raise ValueError('C1 functional only, formal C16/32/64')
 if not namespace.startswith('glm53-flash-') or len(namespace)>128:raise ValueError('Own frozen dataset namespace required')
 if len(tokenizer)!=VOCAB:raise ValueError('Must use actual Flash tokenizer154856 (model embedding154880 has24 padding entries)')
 output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=False);special=set(tokenizer.all_special_ids);markers=[]
 for text in string.ascii_uppercase+string.digits:
  ids=tokenizer.encode(text,add_special_tokens=False)
  if len(ids)==1 and ids[0] not in special and ids[0] not in markers:markers.append(ids[0])
 if len(markers)<4:raise ValueError('Cannot construct four first-token-divergent suffixes')
 instruction=tokenizer.encode('\nExplain the record structure clearly in continuous prose.\nAnswer:',add_special_tokens=False);valid(instruction,special)
 if len(instruction)>=SUFFIX-2:raise ValueError('Instruction length invalid')
 prefixes=[];rows=[];all_full_sha=set()
 for i in range(concurrency):
  prefix_id=f'C{concurrency}-prefix{i:03d}';seed=f'{namespace}/C{concurrency}/prefix{i}';prefix=seeded_tokens(tokenizer,seed,PREFIX,special);prefixes.append(prefix);variants=[]
  for j in range(4):
   suffix=[markers[j]]+seeded_tokens(tokenizer,seed+f'/fresh-suffix{j}',SUFFIX-len(instruction)-1,special)+instruction
   full=prefix+suffix;assert len(prefix)==PREFIX and len(suffix)==SUFFIX and len(full)==80000
   digest=ids_sha(full);assert digest not in all_full_sha;all_full_sha.add(digest);variants.append({'variant':j,'suffix_ids':suffix,'suffix_ids_sha256':ids_sha(suffix),'input_ids_sha256':digest,'actual_input_tokens':len(full),'required_output_tokens':OUTPUT})
  assert all(lcp(variants[a]['suffix_ids'],variants[b]['suffix_ids'])==0 for a in range(4) for b in range(a+1,4))
  private={'project_id':'glm53-flash','namespace':namespace,'logical_prefix_id':prefix_id,'prefix_ids':prefix,'prefix_ids_sha256':ids_sha(prefix),'warmup_input_tokens':PREFIX,'warmup_max_output_tokens':1,'variants':variants}
  (output/(prefix_id+'.json')).write_text(json.dumps(private,separators=(',',':'))+'\n');rows.append({k:private[k] for k in ['logical_prefix_id','prefix_ids_sha256','warmup_input_tokens','warmup_max_output_tokens'] }|{'private_file':prefix_id+'.json','variants':[{k:v for k,v in x.items() if k!='suffix_ids'} for x in variants]})
 cross=[lcp(prefixes[i],prefixes[j]) for i in range(concurrency) for j in range(i+1,concurrency)]
 if cross and max(cross)>256:raise ValueError('Unexpected long shared cross-prefix segment; no performance requests allowed')
 manifest={'project_id':'glm53-flash','actual_tokenizer_vocab':VOCAB,'model_embedding_vocab':MODEL_VOCAB,'namespace':namespace,'concurrency':concurrency,'functional_only':concurrency==1,'independent_prefixes':concurrency,'logical_prefix_tokens':PREFIX,'fresh_first_token_divergent_suffix_variants':4,'suffix_tokens':SUFFIX,'actual_full_input_tokens':80000,'required_output_tokens':OUTPUT,'formal_requests_planned':4*concurrency if concurrency!=1 else 0,'functional_requests_planned':4 if concurrency==1 else 0,'cross_prefix_pair_count':len(cross),'cross_prefix_LCP_max':max(cross) if cross else None,'cross_prefix_LCP_min':min(cross) if cross else None,'cross_prefix_LCP_histogram':{str(n):cross.count(n) for n in sorted(set(cross))},'ID_SHA_algorithm':'SHA256 compactJSON UTF8 array, separators comma/colon','same_ID_files_for_both_frameworks':True,'actual_cache_warmup_or_model_calls':0,'cache_hit_tokens':None,'formal_requests_issued':0,'rows':rows};(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--concurrency',type=int,required=True);ap.add_argument('--namespace',required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();p=a.model/'tokenizer_config.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc'
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained(str(a.model),trust_remote_code=True,local_files_only=True);m=build(t,a.concurrency,a.namespace,a.output);print(json.dumps({k:v for k,v in m.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
