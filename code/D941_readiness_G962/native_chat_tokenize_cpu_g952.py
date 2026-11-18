from pathlib import Path
import datetime,hashlib,inspect,json,os
os.sched_setaffinity(0,{176});os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
assert not list(Path('/dev').glob('davinci[0-9]*')) and not Path('/dev/davinci_manager').exists()
r=Path('/task/native_profile_CPU_g838/chat_contract_g952');body=json.loads((r/'request.json').read_text());out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Actualnative0.32tokenizerfactory CPU-onlychatserialization; noNPUdevices/modelgeneration/weightload','NPU_devices':[]}
try:
 from vllm.tokenizers import get_tokenizer
 tokenizer=get_tokenizer(str(r/'tokenizer'),trust_remote_code=True)
 value=tokenizer.apply_chat_template(body['messages'],add_generation_prompt=True,tokenize=True,**body['chat_template_kwargs'])
 ids=list(value);out.update(tokenizer_type=type(tokenizer).__name__,tokenizer_module=type(tokenizer).__module__,tokens=ids,count=len(ids),factory_source=inspect.getsource(get_tokenizer),factory_file=inspect.getsourcefile(get_tokenizer),factory_SHA256=hashlib.sha256(Path(inspect.getsourcefile(get_tokenizer)).read_bytes()).hexdigest())
except Exception as e:
 import traceback;out.update(error=repr(e),traceback=traceback.format_exc())
import torch
out['NPU_initialized']=getattr(getattr(torch,'npu',None),'is_initialized',lambda:False)();assert not out['NPU_initialized'];(r/'result.json').write_text(json.dumps(out,indent=2)+'\n');print('G952_NATIVE_CPU_CHAT_TOKENIZE_JSON '+json.dumps(out),flush=True)
