"""CPU-only 80K chat-formatted synthetic retrieval guards, separate from formaldata."""
import hashlib,json,struct
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers import Tokenizer
ROOT=Path('/run/glm53');MODEL=Path('/data/wio/models/GLM-5.3-w8a8')
assert hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
plain=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
source=Path('/dataset/prefix-00.txt').read_text()
SYSTEM='You read synthetic log records. Answer the final lookup question using the exact VERIFIED_VALUE stored beside the requested Lookup-ID. Output FINAL_ANSWER: followed by that value, with no explanation.'
anchor='GLM53_CONTENT_ANCHOR_527631'
rendered=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':anchor}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
assert rendered.count(anchor)==1
head,tail=rendered.split(anchor)
def encode(s):return tok.encode(s,add_special_tokens=False)
def digest(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()
manifest={'project_id':'glm53','qualification':'SEMANTIC_GUARDS_ONLY_NOT_FORMAL_INPUTS_OR_PERFORMANCE','tokenizer_sha256':'19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d','input_tokens':80000,'logical_prefix_tokens':74400,'requested_output_tokens':600,'rendered_chat_template_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'enable_thinking':False,'cases':[]}
for index,(key,value,fraction) in enumerate([('GLM53_NEEDLE_ALPHA','731924680531',.30),('GLM53_NEEDLE_BETA','865103792468',.70)]):
 position=int(len(source)*fraction)
 body=source[:position]+'\nLookup-ID: '+key+'; VERIFIED_VALUE: '+value+'.\n'+source[position:]+source
 raw=head+'Synthetic records begin. Only one VERIFIED_VALUE matches the requested Lookup-ID.\n'+body
 prefix=encode(raw)[:74400];assert len(prefix)==74400
 question='\nEND OF RECORDS.\nQuestion: What VERIFIED_VALUE is stored beside Lookup-ID '+key+'? Respond with FINAL_ANSWER: and the exact value only.\n'
 ending=encode(question+tail);assert 0<len(ending)<5600
 padding=encode('\nAdditional irrelevant records:\n'+source)[:5600-len(ending)]
 ids=prefix+padding+ending;assert len(ids)==80000
 decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 assert decoded.count(value)==1 and key in decoded and 'Question:' in decoded
 assert plain.encode(raw,add_special_tokens=False).ids[:74400]==prefix,'HF/tokenizerJSON encoding disagree'
 needle_prefix=tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert value in needle_prefix
 record={'case':key,'input_ids':ids,'prefix_ids':prefix,'input_tokens':80000,'prefix_tokens':74400,'requested_output_tokens':600,'ids_sha256_le_u32':digest(ids),'prefix_sha256_le_u32':digest(prefix),'expected_answer':value,'expected_marker':'FINAL_ANSWER:','decoded_text_sha256':hashlib.sha256(decoded.encode()).hexdigest()}
 name=f'semantic-{index}.json';(ROOT/name).write_text(json.dumps(record)+'\n');(ROOT/name.replace('.json','.txt')).write_text(decoded)
 manifest['cases'].append({k:v for k,v in record.items() if k not in ['input_ids','prefix_ids']}|{'file':name})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('GLM53_SEMANTIC_INPUT_MANIFEST='+json.dumps(manifest),flush=True)
