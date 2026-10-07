"""Actual sample/bookkeeping/backup-token/greedy-sampling AST on CPU tensors.

Device queue timing, attention, MTP model and copy streams remain doubles.
"""
from __future__ import annotations
import ast,contextlib,itertools,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT.parent/'padded_mtp_order_sources'
def node(path,name):
    matches=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(matches)==1,(path,name,len(matches));return matches[0]
def method(n,ns):
    n.decorator_list=[]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,'actual_installed_AST','exec'),ns);return ns[n.name]

def run(patched,batch,rows,discard,pp=1,greedy=True,generators=False,fits=True,num_spec=1,scheduled=1):
    order=[];copies=[];generator=NS(offset=100);generator.get_offset=lambda:generator.offset;generator.set_offset=lambda v:setattr(generator,'offset',v)
    ns={'torch':torch,'np':np,'copy':__import__('copy').copy,'get_pp_group':lambda:NS(world_size=pp),'record_function_or_nullcontext':lambda *args:contextlib.nullcontext(),'ModelRunnerOutput':lambda **kw:NS(**kw)}
    parse=method(node(SOURCES/'vllm__vllm__v1__sample__rejection_sampler.py','parse_output'),{'torch':torch,'PLACEHOLDER_TOKEN_ID':-1})
    def parse_wait(*args,**kw):order.append('target_D2H_wait');return parse(*args,**kw)
    ns['RejectionSampler']=NS(parse_output=parse_wait)
    model=ROOT/('model_runner_v1.py' if patched else 'original_model_runner_v1.py')
    sample=method(node(model,'sample_tokens'),ns);bookkeeping=method(node(model,'_bookkeeping_sync'),ns);skip=method(node(model,'_skip_drafting'),ns)
    backup_ns={'torch':torch,'np':np,'DeviceOperator':NS(index_fill=lambda t,d,i,v:t.index_fill(d,i.to(torch.int64),v))}
    backup=method(node(SOURCES/'vllm-ascend__vllm_ascend__spec_decode__llm_base_proposer.py','prepare_next_token_ids_padded'),backup_ns)
    draft_sample=method(node(SOURCES/'vllm__vllm__v1__spec_decode__llm_base_proposer.py','_sample_from_logits'),{'torch':torch})
    get_token=method(node(SOURCES/'vllm__vllm__v1__worker__gpu_input_batch.py','get_token_id'),{})
    requests={};ids=['req%d'%i for i in range(batch)];token_cpu=np.zeros((batch,100),dtype=np.int64);num_tokens=np.full(batch,6,dtype=np.int32)
    for i,key in enumerate(ids):
        req=NS(prompt_token_ids=[20+i,21+i,22+i,23+i,24+i],num_prompt_tokens=5,output_token_ids=[25+i]);req.get_token_id=get_token.__get__(req);requests[key]=req;token_cpu[i,:6]=req.prompt_token_ids+req.output_token_ids
    metadata=NS(all_greedy=greedy,output_token_ids=[requests[k].output_token_ids for k in ids])
    input_batch=NS(num_reqs=batch,req_ids=ids,req_id_to_index={k:i for i,k in enumerate(ids)},num_tokens_no_spec=num_tokens.copy(),num_tokens=num_tokens.copy(),token_ids_cpu=token_cpu,is_token_ids=np.zeros((batch,100),dtype=bool),sampling_metadata=metadata,generators={i:generator for i in discard} if generators else {},vocab_size=128)
    sampled=torch.tensor(rows,dtype=torch.int64);sampler=NS(sampled_token_ids=sampled,logprobs_tensors=None)
    buffer=NS(np=np.zeros(batch,dtype=np.int64),gpu=torch.zeros(batch,dtype=torch.int64));buffer.copy_to_gpu=lambda n:buffer.gpu[:n].copy_(torch.from_numpy(buffer.np[:n]))
    drafter=NS(backup_next_token_ids=buffer,_enable_probabilistic_draft_probs=False)
    config=NS(method='mtp',disable_padded_drafter_batch=False,use_eagle=lambda:True,uses_draft_model=lambda:False,uses_extract_hidden_states=lambda:False,use_ngram_gpu=lambda:False)
    scheduler=NS(num_scheduled_tokens={k:2 for k in ids},total_num_scheduled_tokens=batch*2,num_spec_tokens_to_schedule=scheduled)
    runner=NS(ascend_config=NS(scheduler_config=NS(profiling_chunk_config=NS(enabled=False,need_timing=False))),kv_connector_output=None,execute_model_state=(scheduler,torch.zeros(batch,128),None,object(),torch.zeros(batch*2,4),None,None,object(),torch.arange(batch*2),None,None,NS(uniform=True)),speculative_config=config,input_batch=input_batch,requests=requests,need_accepted_tokens=False,use_async_scheduling=False,routed_experts_initialized=False,num_prompt_logprobs={},num_spec_tokens=num_spec,model_config=NS(hf_config=NS(model_type='glm_moe_dsa')),num_discarded_requests=len(discard),discard_request_indices=NS(np=np.array(discard,dtype=np.int64)),max_model_len=100,dynamic_eplb=False,parallel_config=NS(data_parallel_size=1),drafter=drafter,device='cpu',valid_sampled_token_count_event=None)
    runner._bookkeeping_sync=bookkeeping.__get__(runner);runner._skip_drafting=skip.__get__(runner);runner._drafter_runs_model_forward=lambda:True
    runner._sample=lambda *args:sampler;runner._input_fits_in_drafter=lambda *args:fits
    runner._get_prompt_logprobs_dict=lambda *args:{};runner._finalize_dump_data=lambda:None;runner.finalize_kv_connector=lambda:order.append('KV_finalize')
    def tolist(t):order.append('target_D2H_wait');return t.tolist()
    runner._to_list=tolist
    def propose(sampled_ids,*args):
        assert isinstance(sampled_ids,torch.Tensor)
        order.append('draft_prepare')
        next_ids,count=backup(drafter,sampled_ids,requests,input_batch,torch.tensor(discard,dtype=torch.int64),len(discard))
        # Actual draft greedy sampler consumes logits only, no CPU history.
        logits=torch.nn.functional.one_hot((next_ids+3)%128,128).float();draft_ids,_=draft_sample(drafter,logits,metadata)
        order.append('MTP_enqueue');return draft_ids[:,None].expand(batch,scheduled).clone()
    runner.propose_draft_token_ids=propose
    runner._copy_draft_token_ids_to_cpu=lambda *args,**kw:copies.append(runner._draft_token_ids.clone())
    out=sample(runner,None)
    assert len(copies)==1 and order.count('target_D2H_wait')==1
    state=dict(sampled_token_ids=out.sampled_token_ids,request_outputs={k:v.output_token_ids for k,v in requests.items()},num_tokens=input_batch.num_tokens.tolist(),num_tokens_no_spec=input_batch.num_tokens_no_spec.tolist(),token_cpu=input_batch.token_ids_cpu.tolist(),generator_offset=generator.offset,draft_ids=copies[0].tolist(),output_spec_ids=out.spec_token_ids)
    return state,order

def predicate(patched,settings):
    pp,async_,method_,greedy,gens,prompt,route,fits,spec,scheduled,model,disabled=settings
    path=ROOT/('model_runner_v1.py' if patched else 'original_model_runner_v1.py')
    assignment=next(n for n in ast.walk(node(path,'sample_tokens')) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==('early_padded_drafter' if patched else 'early_pp_padded_drafter') and not isinstance(n.value,ast.Constant))
    self=NS(use_async_scheduling=async_,speculative_config=NS(method=method_),model_config=NS(hf_config=NS(model_type=model)),input_batch=NS(sampling_metadata=NS(all_greedy=greedy),generators={0:object()} if gens else {}),num_prompt_logprobs={'req':2} if prompt else {},routed_experts_initialized=route,num_spec_tokens=spec)
    return eval(compile(ast.Expression(body=assignment.value),'actual_early_guard','eval'),{'self':self,'pp':NS(world_size=pp),'use_pp_spec_decode':pp>1,'use_padded_batch':not disabled,'input_fits_in_drafter':fits,'scheduler_output':NS(num_spec_tokens_to_schedule=scheduled)})

def main():
    comparisons=[]
    for batch in (1,2,3,4):
        patterns=[[[7,8] for _ in range(batch)],[[7,-1] if i%2 else [9,10] for i in range(batch)],[[-1,-1] if i%2 else [11,12] for i in range(batch)],[[13] for _ in range(batch)]]
        for rows in patterns:
            for discard in ([],[0],list(range(batch))):
                for pp,greedy,gens,fits,spec,scheduled in ((1,True,False,True,1,1),(1,False,False,True,1,1),(1,True,True,True,1,1),(1,True,False,False,1,1),(1,True,False,True,2,1),(1,True,False,True,1,0),(2,True,False,True,1,1)):
                    before,old=run(False,batch,rows,discard,pp,greedy,gens,fits,spec,scheduled);after,new=run(True,batch,rows,discard,pp,greedy,gens,fits,spec,scheduled);assert before==after,(batch,rows,discard,before,after)
                    eligible=pp==1 and greedy and not gens and fits and spec==1 and scheduled>0
                    if eligible:assert old.index('target_D2H_wait')<old.index('MTP_enqueue') and new.index('MTP_enqueue')<new.index('target_D2H_wait')
                    else:assert old==new,(old,new)
                    comparisons.append(eligible)
    changed=parity=0
    for settings in itertools.product((1,2),(False,True),('mtp','eagle'),(False,True),(False,True),(False,True),(False,True),(False,True),(1,2),(0,1),('glm_moe_dsa','other'),(False,True)):
        a,b=predicate(False,settings),predicate(True,settings)
        if a!=b:assert not a and b;changed+=1
        else:parity+=1
    assert changed==1 and parity==4095
    initialized=bool(hasattr(torch,'npu') and torch.npu.is_initialized())
    assert not initialized
    out=dict(passed=True,actual_sync_sample_bookkeeping_backup_cases=len(comparisons),early_MTP_order_cases=sum(comparisons),exact_state_and_output_parity=True,actual_guard_cases=changed+parity,unchanged_guard_cases=parity,new_guard_cases=changed,NPU_initialized=False,performance_claim=False,full_API=False,limits=['MTP model, attention, device timing and copy streams are doubles.','Complete synchronous sample/bookkeeping and CPU backup/greedy sampling methods use actual installed AST and CPU tensors; asynchronous guard only, no async runtime acceptance.'])
    (ROOT/'CPU_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
