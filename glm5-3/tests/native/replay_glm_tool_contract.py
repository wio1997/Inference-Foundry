"""Server CPU regression with native vLLM and actual model tokenizer; no inference."""
import importlib.util,json,sys
from pathlib import Path
from vllm.tokenizers import get_tokenizer
from vllm.exceptions import VLLMValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3")
spec=importlib.util.spec_from_file_location("glm_contract_candidate",root/"runtime/glm_tool_contract.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Candidate=module.Glm47ContractToolParser
tokenizer=get_tokenizer(tokenizer_name="/data/tiankuan/wio/GLM-5.3-w8a8",trust_remote_code=True)
tools=[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
text='prefix <tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call> tail'
cases=[]
try:
 ChatCompletionRequest(model='glm-53',messages=[{'role':'user','content':'x'}],tools=[])
except VLLMValidationError:pass
else:raise AssertionError('native empty tools must reject before parser')
for name,provided,choice in [("no_tools",None,None),("no_tools_none",None,"none"),("tools_none",tools,"none"),("tools_auto",tools,"auto")]:
 request=ChatCompletionRequest(model="glm-53",messages=[{"role":"user","content":"x"}],tools=provided,tool_choice=choice)
 for cls in [Glm47MoeModelToolParser,Candidate]:
  parser=cls(tokenizer,tools=request.tools)
  full=parser.extract_tool_calls(text,request=request)
  if cls is Candidate:
   if not provided or choice=="none":assert not full.tools_called and full.content==text,(name,full)
   else:assert full.tools_called and full.tool_calls[0].function.name=="get_weather" and json.loads(full.tool_calls[0].function.arguments)=={"city":"Shanghai"}
  for token_mode in [False,True]:
   parser=cls(tokenizer,tools=request.tools);current="";ids=[];deltas=[]
   chunks=[text[i:i+7] for i in range(0,len(text),7)]
   if token_mode:
    tokenids=tokenizer.encode(text,add_special_tokens=False)
    chunks=[tokenizer.decode([i],skip_special_tokens=False) for i in tokenids]
   for i,part in enumerate(chunks):
    previous=current;current+=part
    delta_ids=[tokenids[i]] if token_mode else []
    d=parser.extract_tool_calls_streaming(previous,current,part,ids,ids+delta_ids,delta_ids,request)
    ids+=delta_ids
    if d:deltas.append(d)
   d=parser.finish_streaming()
   if d:deltas.append(d)
   content="".join(d.content or "" for d in deltas);calls=[c for d in deltas for c in d.tool_calls or []]
   if cls is Candidate:
    if not provided or choice=="none":assert not calls and content==current,(name,token_mode,repr(content),repr(current))
    else:assert calls and any(c.function and c.function.name=="get_weather" for c in calls),(name,token_mode,calls)
   cases.append({"case":name,"class":cls.__name__,"token_ids":token_mode,"nonstream_tools":full.tools_called,"stream_tool_deltas":len(calls),"content_chars":len(content)})
# Reproduce late large tool-name buffering without an NPU or inference.
huge="<tool_call>"+"x"*316479
req=ChatCompletionRequest(model="glm-53",messages=[{"role":"user","content":"x"}])
parser=Candidate(tokenizer)
result=parser.extract_tool_calls(huge,request=req)
assert not result.tools_called and result.content==huge
print(json.dumps({"passed":True,"cases":cases,"large_no_tools_chars":len(huge),"scope":"Native CPU parser replay with actual model tokenizer; no live model change, no throughput or E2E claim"}))
