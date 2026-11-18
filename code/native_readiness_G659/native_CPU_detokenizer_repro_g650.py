from pathlib import Path
import hashlib, importlib.metadata, json, os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
os.environ['TOKENIZERS_PARALLELISM']='false'
import tokenizers,tokenizers.decoders
from transformers import TokenizersBackend
R=Path('/task/vllm_pd_g630');r=R/'diagnostic_g648'
t=TokenizersBackend.from_pretrained('/models/ds41',trust_remote_code=True)
rows=[]
for tag,original in [('manual129a','tail_129'),('manual32768','native_32768_32')]:
 body=json.loads((R/'functional_PD_g644'/(original+'_request.json')).read_text())
 response=json.loads((r/(tag+'_response.json')).read_text());output=response['choices'][0];ids=output['token_ids'];prompt=body['prompt']
 variants={}
 for name,context in [('full_prompt',prompt),('last8_prompt',prompt[-8:]),('empty',[])]:
  stream=tokenizers.decoders.DecodeStream(ids=context,skip_special_tokens=True);pieces=[];errors=[]
  for tid in ids:
   try:piece=stream.step(t._tokenizer,tid)
   except Exception as exc:errors.append(repr(exc));break
   pieces.append(piece or '')
  text=''.join(pieces);variants[name]={'output_chars':len(text),'first_piece_chars':len(pieces[0]) if pieces else None,'equals_server_text':text==output['text'],'errors':errors,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'first160':text[:160]}
 rows.append({'case':tag,'input_ids':len(prompt),'generated_ids':len(ids),'server_text_chars':len(output['text']),'direct_generated_decode_chars':len(t.decode(ids,skip_special_tokens=True)),'variants':variants})
versions={}
for name in ['tokenizers','transformers','fastokens']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
print(json.dumps({'scope':'CPU tokenizer reproduction inside same native image, no model or NPU initialization; last8 is diagnosis only, not an accepted fix','versions':versions,'DecodeStream_type':str(tokenizers.decoders.DecodeStream),'rows':rows}),flush=True)
