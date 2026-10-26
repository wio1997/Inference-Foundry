"""CPU full-caller contract from exact baseline/candidate AST methods.

Executes complete forward_sparse bodies, not a copied validation fragment. It
avoids framework/native initialization by compiling the actual method + actual
_expand_dsa_sparse_indices under CPU fakes. This is NOT normal-module-import or
native qualification; root's full-model rank imports discharge that separately.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace
import torch

HELPER_NAME="sglang.srt.hardware_backend.npu.ops.flash_sparse_kv_length_cache_npu"
NOPE_NAME="sglang.srt.hardware_backend.npu.flash_sparse_nope_npu"


class Mode:
    def __init__(self,name):self.name=name
    def is_extend(self):return self.name=="extend"
    def is_decode(self):return self.name=="decode"
    def is_draft_extend_v2(self):return False
    def is_target_verify(self):return False


class Device:
    type="npu";index=0
    def __str__(self):return "npu:0"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_method(path,env):
    tree=ast.parse(path.read_text(encoding="utf-8-sig"))
    found=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=="forward_sparse"]
    expand=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_expand_dsa_sparse_indices"]
    assert len(found)==len(expand)==1,"exact full methods must be unique"
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),expand[0],found[0]],type_ignores=[])
    ast.fix_missing_locations(module);env={**env,"__file__":str(path)}
    exec(compile(module,str(path),"exec"),env)
    return env["forward_sparse"]


def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline-caller",type=Path,required=True);p.add_argument("--caller",type=Path,required=True);p.add_argument("--helper",type=Path,required=True)
    p.add_argument("--expected-caller-sha256",required=True);p.add_argument("--expected-helper-sha256",required=True);p.add_argument("--engine-commit",required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();assert sha(a.baseline_caller)=="b76deb848792925be0a5fd596c0b921f034fde90318d6122306c96810c9d8278";assert sha(a.caller)==a.expected_caller_sha256 and sha(a.helper)==a.expected_helper_sha256;assert not a.output_dir.exists();a.output_dir.mkdir();sys.dont_write_bytecode=True
    os.environ.pop("GLM53_EXECUTION_IDENTITY_DIR",None)
    spec=importlib.util.spec_from_file_location(HELPER_NAME,a.helper);helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    helper._current_stream_key=lambda d:(str(d),7)
    (a.output_dir/"pre-execution-source.json").write_text(json.dumps({"scope":"CPU full actual-method AST execution; no NPU/native or normal module import","engine_commit_declared":a.engine_commit,"baseline_sha256":sha(a.baseline_caller),"caller_sha256":sha(a.caller),"helper_sha256":sha(a.helper),"probe_sha256":sha(__file__)},indent=2)+"\n")
    results=[]
    def one(candidate,mode="decode",kind="seq_lens_cpu_int",enabled=True,mutate=False,new_generation=False,guard=None,other=False,save=False):
        trace=[];native=[];count=2;target=Device()
        q=SimpleNamespace(shape=(count,1,512),device=target,dtype=torch.bfloat16)
        qends=torch.tensor([1,2],dtype=torch.int32)
        source=torch.tensor([12,17],dtype=torch.int64)
        def watched(tensor,label):
            def to(*args,**kw):
                assert not args and set(kw)=={"device","dtype"} and str(kw["device"])=="npu:0" and kw["dtype"]==torch.int32
                trace.append(label);return tensor.detach().clone().to(device="cpu",dtype=kw["dtype"])
            tensor.to=to;return tensor
        watched(qends,"q")
        watched(source,"kv")
        metadata=SimpleNamespace(actual_seq_lengths_q=qends,actual_seq_lengths_kv=None,seq_lens_cpu_int=None,seq_lens=None,block_tables=torch.zeros((2,1),dtype=torch.int32))
        setattr(metadata,kind,source)
        if kind=="actual_seq_lengths_kv":metadata.seq_lens_cpu_int=torch.tensor([90,91]);metadata.seq_lens=torch.tensor([99,98])
        elif kind=="seq_lens_cpu_int":metadata.seq_lens=torch.tensor([99,98])
        backend=SimpleNamespace(forward_metadata=metadata,qk_rope_head_dim=128 if other else 0,kv_cache_dtype=torch.bfloat16,is_draft_worker=False,graph_mode=False,
            kv_lora_rank=512,token_to_kv_pool=SimpleNamespace(set_kv_buffer=lambda *args:trace.append("state_write"),get_kv_buffer=lambda _: (torch.zeros((1,128,1,512)),torch.zeros((1,128,1,64)) if other else None)),
            _pad_topk_indices=lambda value,_:value)
        batch=SimpleNamespace(forward_mode=Mode(mode),spec_info=None,can_run_tbo=False,forward_metadata_ready=False,attn_cp_metadata=None,extend_seq_lens=torch.tensor([1,1]),out_cache_loc=torch.tensor([0,1]))
        if guard=="graph":backend.graph_mode=True
        elif guard:setattr(batch,guard,True if guard!="spec_info" else object())
        def capture(qnope,cache,indices,table,qv,kv,scale):
            assert qnope is q and scale==.0625 and qv.dtype==kv.dtype==torch.int32
            native.append({"q":qv.tolist(),"kv":kv.tolist(),"kv_object":id(kv)})
            return torch.full((count,1,512),float(kv.sum().item()),dtype=torch.bfloat16)
        nope=ModuleType(NOPE_NAME);nope.flash_sparse_nope=capture;sys.modules[NOPE_NAME]=nope
        def generic(**kwargs):
            return capture(kwargs["query"],kwargs["key"],kwargs["sparse_indices"],kwargs["block_table"],kwargs["actual_seq_lengths_query"],kwargs["actual_seq_lengths_kv"],kwargs["scale_value"]),None,None
        env={"torch":torch,"torch_npu":SimpleNamespace(npu_sparse_flash_attention=generic),"get_parallel":lambda:SimpleNamespace(dcp_enabled=False),"is_dsa_enable_prefill_cp":lambda:False,"_SPARSE_KV_LENGTH_DEVICE_CACHE":enabled}
        fn=extract_method(a.caller if candidate else a.baseline_caller,env)
        layer=SimpleNamespace(layer_id=0,scaling=.0625,tp_k_head_num=1)
        out=[]
        for i in range(2):
            if i and mutate:source.data[0]=31
            if i and new_generation:
                old=backend.forward_metadata
                backend.forward_metadata=SimpleNamespace(actual_seq_lengths_q=old.actual_seq_lengths_q,actual_seq_lengths_kv=old.actual_seq_lengths_kv,seq_lens_cpu_int=old.seq_lens_cpu_int,seq_lens=old.seq_lens,block_tables=old.block_tables)
            out.append(fn(backend,q,torch.zeros((count,1,512)) if save else None,None,layer,batch,save_kv_cache=save,q_rope=torch.zeros(1) if other else None,topk_indices=torch.zeros((count,2051),dtype=torch.int32)))
        assert [x["kv"] for x in native]==([[12,17],[31,17]] if mutate else [[12,17],[12,17]])
        assert all(x["q"]==[1,2] for x in native)
        return trace,native,out
    def observe(name,fn):
        try:detail=fn();results.append({"name":name,"status":"PASS","detail":detail})
        except Exception as ex:results.append({"name":name,"status":"FAIL","exception":type(ex).__name__,"traceback":traceback.format_exc()})
    def priority(mode,kind):
        at,an,ao=one(False,mode,kind);bt,bn,bo=one(True,mode,kind)
        assert at==["q","kv","q","kv"] and bt==["q","kv","q"]
        assert all(torch.equal(x,y) for x,y in zip(ao,bo)) and bn[0]["kv_object"]==bn[1]["kv_object"]
        return {"baseline_order":at,"candidate_order":bt,"KV_values":[x["kv"] for x in bn],"q_conversion_unchanged":True}
    for mode in ("extend","decode"):
        for kind in ("actual_seq_lengths_kv","seq_lens_cpu_int","seq_lens"):
            observe(mode+"_priority_"+kind,lambda mode=mode,kind=kind:priority(mode,kind))
    def bypass(guard=None,off=False,other=False):
        at,_,ao=one(False,other=other);bt,_,bo=one(True,enabled=not off,guard=guard,other=other)
        assert at==bt==["q","kv","q","kv"] and all(torch.equal(x,y) for x,y in zip(ao,bo))
        return {"both_orders":bt}
    observe("featureoff_original_fullcaller",lambda:bypass(off=True))
    for guard in ("graph","spec_info","can_run_tbo","forward_metadata_ready"):
        observe("fullcaller_bypass_"+guard,lambda guard=guard:bypass(guard=guard))
    observe("other_rope_branch_unchanged",lambda:bypass(other=True))
    def stale(mutate=False,new_generation=False):
        at,_,ao=one(False,mutate=mutate,new_generation=new_generation);bt,_,bo=one(True,mutate=mutate,new_generation=new_generation)
        assert at==bt==["q","kv","q","kv"] and all(torch.equal(x,y) for x,y in zip(ao,bo))
        return {"both_orders":bt}
    observe("actualcaller_CPU_data_write_invalidates",lambda:stale(mutate=True))
    observe("actualcaller_fresh_forward_metadata_invalidates",lambda:stale(new_generation=True))
    def state_writes():
        at,_,ao=one(False,save=True);bt,_,bo=one(True,save=True)
        assert at==["state_write","q","kv","state_write","q","kv"] and bt==["state_write","q","kv","state_write","q"]
        assert all(torch.equal(x,y) for x,y in zip(ao,bo))
        return {"baseline_order":at,"candidate_order":bt,"state_writes_each":2}
    observe("actualcaller_state_write_still_every_layer",state_writes)
    output={"scope":"CPU complete actual AST method contracts; fake native attention returns metadata-dependent output. Not normal module import, device ABI or model precision.","passed":sum(x["status"]=="PASS" for x in results),"failed":sum(x["status"]=="FAIL" for x in results),"results":results}
    (a.output_dir/"results.json").write_text(json.dumps(output,indent=2)+"\n");print(json.dumps(output,indent=2));return int(output["failed"]!=0)


if __name__=="__main__":raise SystemExit(main())
