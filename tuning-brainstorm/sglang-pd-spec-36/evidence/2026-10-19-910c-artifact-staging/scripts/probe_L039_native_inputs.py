"""Revalidate existing frozen inputs in the actual910C image; CPU tokenizer only."""
import hashlib,importlib.metadata,json,pathlib,subprocess,sys,traceback
v={'status':'STARTING','node':46,'execution_location':'910C server, CPU tokenizer inside exact serving image','model_loaded':False,'NPU_computation':False,'PD_MTP_serving_executed':False,'performance':'NOT_MEASURED','cases':[]}
try:
 root=pathlib.Path('/sgl-workspace/sglang');v['source_commit']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert v['source_commit']=='ce0a8fea5f4245839d174b2b8c55692be1810a7f'
 v['accelerator_devices']=sorted({str(p) for pat in ['davinci*','devmm_svm','hisi_hdc'] for p in pathlib.Path('/dev').glob(pat)});assert not v['accelerator_devices']
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
 tok=get_tokenizer('/model',tokenizer_mode='auto',tokenizer_backend='huggingface',trust_remote_code=False,local_files_only=True)
 v['tokenizer_class']=type(tok).__module__+'.'+type(tok).__name__;v['packages']={n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','jinja2']}
 v['loaded_template_sha256']=hashlib.sha256(tok.chat_template.encode()).hexdigest();assert v['loaded_template_sha256']=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
 v['EOS_token_id']=tok.eos_token_id;v['PAD_token_id']=tok.pad_token_id;v['additional_stop_token_ids']=sorted(getattr(tok,'additional_stop_token_ids',None) or [])
 assert len(FIXTURES)==5
 for f in FIXTURES:
  kw=dict(f['template_kwargs']);kw['return_dict']=False
  rendered=tok.apply_chat_template(f['messages'],tokenize=False,**kw);ids=list(tok.apply_chat_template(f['messages'],tokenize=True,**kw));plain=tok.encode(rendered,add_special_tokens=False)
  sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
  row={'label':f['label'],'input_tokens':len(ids),'input_ids_sha256':sha,'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'template_kwargs':f['template_kwargs'],'rendered_matches':rendered==f['rendered_text'],'ids_match':ids==f['input_ids'],'plain_roundtrip_matches':plain==ids,'IDs_in_model_vocab':all(0<=i<154880 for i in ids)}
  assert len(ids)==f['input_tokens'] and sha==f['input_ids_sha256'] and all(row[x] for x in ['rendered_matches','ids_match','plain_roundtrip_matches','IDs_in_model_vocab']),row
  v['cases'].append(row)
 assert [x['input_tokens'] for x in v['cases']]==[512,512,32768,32768,32768]
 assert 'torch_npu' not in sys.modules
 v['status']='PASS';v['scope']='Five originalM001-v1 synthetic inputs match same C8template and IDs in actual910C image. No output/capacity/lifecycle or performance inference.'
except BaseException as e:v['status']='FAIL';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print('L039_RESULT='+json.dumps(v));sys.exit(0 if v['status']=='PASS' else 1)
