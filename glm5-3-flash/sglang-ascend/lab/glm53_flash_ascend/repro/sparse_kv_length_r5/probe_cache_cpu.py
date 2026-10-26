"""R5 CPU fake contracts; no NPU API, allocation or model execution.

Root precommits this probe/helper/caller before use inside the container. Fake
conversion preserves CPU integer values and emulates the target device/stream;
it does not establish native/device ABI or performance.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace
import torch


class Mode:
    def __init__(self,name="decode"):self.name=name
    def is_decode(self):return self.name=="decode"
    def is_extend(self):return self.name=="extend"
    def is_target_verify(self):return self.name=="verify"
    def is_draft_extend_v2(self):return self.name=="draft"


class Device:
    def __init__(self,index=0,kind="npu"):self.index=index;self.type=kind
    def __str__(self):return self.type+":"+str(self.index)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--helper",type=Path,required=True);p.add_argument("--caller",type=Path,required=True)
    p.add_argument("--expected-helper-sha256",required=True);p.add_argument("--engine-commit",required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();assert sha(a.helper)==a.expected_helper_sha256;assert not a.output_dir.exists();a.output_dir.mkdir()
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location("r5_sparse_kv_cpu_target",a.helper);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    (a.output_dir/"pre-execution-source.json").write_text(json.dumps({"scope":"CPU fake contracts only","engine_commit_declared":a.engine_commit,"helper":{"sha256":sha(a.helper),"bytes":a.helper.stat().st_size},"caller":{"sha256":sha(a.caller),"bytes":a.caller.stat().st_size},"probe_sha256":sha(__file__)},indent=2)+"\n")
    os.environ.pop("GLM53_EXECUTION_IDENTITY_DIR",None)
    stream={"id":7};calls=[]
    def fake_convert(source,device):
        calls.append({"values":source.tolist(),"target":str(device),"dtype":"torch.int32"})
        return source.clone().to(device="cpu",dtype=torch.int32)
    m._convert=fake_convert;m._current_stream_key=lambda device:(str(device),stream["id"])
    device=Device()
    def state(source=None,kind="seq_lens_cpu_int",mode="decode"):
        source=torch.tensor([1,4,2048,16384],dtype=torch.int32) if source is None else source
        md=SimpleNamespace(actual_seq_lengths_kv=None,seq_lens_cpu_int=None,seq_lens=None)
        setattr(md,kind,source)
        backend=SimpleNamespace(graph_mode=False,forward_metadata=md)
        batch=SimpleNamespace(forward_mode=Mode(mode),spec_info=None,can_run_tbo=False,forward_metadata_ready=False,attn_cp_metadata=None)
        return backend,batch,source
    def run(st,enabled=True,target=device,count=4):
        return m.sparse_kv_lengths_to_device(*st,target,count,enabled=enabled,caller_file=str(a.caller))
    results=[]
    def observe(name,fn):
        before=len(calls);stream["id"]=7
        try:detail=fn();results.append({"name":name,"status":"PASS","conversion_calls":len(calls)-before,"detail":detail})
        except Exception as ex:results.append({"name":name,"status":"FAIL","conversion_calls":len(calls)-before,"exception":type(ex).__name__,"traceback":traceback.format_exc()})
    def reuse():
        st=state();start=len(calls);out=[run(st) for _ in range(11)]
        assert len(calls)-start==1 and all(x is out[0] for x in out) and torch.equal(out[0],st[2].to(torch.int32))
        return {"layer_calls":11,"actual_fake_copies":1}
    observe("first_copy_then_ten_same_forward_reuses",reuse)
    def off():
        st=state();start=len(calls);run(st,False);run(st,False);assert len(calls)-start==2
    observe("defaultoff_original_conversion_each_call",off)
    def all_sources():
        details=[]
        for kind in ("actual_seq_lengths_kv","seq_lens_cpu_int","seq_lens"):
            st=state(kind=kind);start=len(calls);run(st);run(st);assert len(calls)-start==1;details.append(kind)
        return details
    observe("all_three_selected_source_kinds",all_sources)
    def wrong_source():
        st=state();st[0].forward_metadata.actual_seq_lengths_kv=torch.tensor([9,8,7,6]);start=len(calls);run(st);run(st);assert len(calls)-start==2
    observe("does_not_replace_original_priority_source",wrong_source)
    def generation():
        st=state();start=len(calls);run(st);md=st[0].forward_metadata
        st[0].forward_metadata=SimpleNamespace(actual_seq_lengths_kv=md.actual_seq_lengths_kv,seq_lens_cpu_int=md.seq_lens_cpu_int,seq_lens=md.seq_lens)
        run(st);assert len(calls)-start==2
    observe("fresh_metadata_invalidates_across_forward_chunk",generation)
    def replacement():
        st=state();start=len(calls);run(st);source=st[2].clone();st[0].forward_metadata.seq_lens_cpu_int=source;run((st[0],st[1],source));assert len(calls)-start==2
    observe("new_same_values_tensor_identity_invalidates",replacement)
    def mutation(data=False):
        st=state();start=len(calls);run(st)
        if data:st[2].data[0]=17
        else:st[2][0]=17
        value=run(st);assert value[0].item()==17 and len(calls)-start==2
    observe("normal_inplace_version_invalidation",lambda:mutation(False))
    observe("CPU_value_key_detects_data_write",lambda:mutation(True))
    def inference():
        with torch.inference_mode():
            st=state();start=len(calls);run(st);st[2][0]=29;value=run(st)
            assert value[0].item()==29 and len(calls)-start==2 and m._version(st[2]) is None
    observe("inference_CPU_value_key_invalidation",inference)
    def stream_change():
        st=state();start=len(calls);first=run(st);stream["id"]=8;second=run(st);stream["id"]=7;third=run(st)
        assert len(calls)-start==2 and first is third and second is not first
    observe("cross_stream_bypass_keeps_first_stream_alive",stream_change)
    def guards():
        details=[]
        for field,value in [("spec_info",object()),("can_run_tbo",True),("forward_metadata_ready",True),("attn_cp_metadata",object())]:
            st=state();setattr(st[1],field,value);start=len(calls);run(st);run(st);assert len(calls)-start==2;details.append(field)
        st=state();st[0].graph_mode=True;start=len(calls);run(st);run(st);assert len(calls)-start==2
        for mode in ("verify","draft","idle"):
            st=state(mode=mode);start=len(calls);run(st);run(st);assert len(calls)-start==2
        return details+["graph","verify","draft","idle"]
    observe("graph_spec_TBO_preplanned_CP_and_unknown_modes_bypass",guards)
    def layout_dtype_count():
        src=torch.arange(8,dtype=torch.int64)[::2];st=state(src);assert torch.equal(run(st),src.to(torch.int32))
        start=len(calls);run(st,count=5);assert len(calls)-start==1
        for source in (torch.tensor([1.,2.]),torch.tensor(4,dtype=torch.int32)):
            st=state(source);start=len(calls);run(st);run(st);assert len(calls)-start==2
        st=state();start=len(calls);run(st,target=Device(kind="cpu"));run(st,target=Device(kind="cpu"));assert len(calls)-start==2
    observe("strided_i64_values_and_dtype_rank_count_device_guards",layout_dtype_count)
    def output_mutation():
        st=state();start=len(calls);first=run(st);first.add_(1);second=run(st)
        assert len(calls)-start==2 and torch.equal(second,st[2])
    observe("normal_cached_output_version_mutation_invalidates",output_mutation)
    def marker():
        st=state();directory=a.output_dir/"isolated-marker";os.environ["GLM53_EXECUTION_IDENTITY_DIR"]=str(directory);os.environ["GLM53_CANDIDATE_COMMIT"]=a.engine_commit
        module=ModuleType("sglang.srt.runtime_context");module.get_parallel=lambda:SimpleNamespace(attn_tp_rank=3)
        sys.modules["sglang.srt.runtime_context"]=module;m._IDENTITY_RECORDED=False
        run(st);assert not directory.exists();run(st);run(st)
        target=directory/"sparse-kv-length-cache-rank-3.json";value=json.loads(target.read_text())
        assert value["event"]=="sparse_nope_kv_length_device_cache_first_hit" and value["cache_scope"]=="same_eager_forward" and value["rank"]==3
        assert value["caller_source"]["sha256"]==sha(a.caller) and value["helper_source"]["sha256"]==sha(a.helper)
        assert len(list(directory.iterdir()))==1
        os.environ.pop("GLM53_EXECUTION_IDENTITY_DIR",None)
        return {"marker_actual_JSON_parsed":True,"not_a_runtime_NPU_marker":True}
    observe("marker_only_after_real_helper_reuse_once",marker)
    output={"scope":"CPU fake conversion/stream contracts; no NPU/ABI/model/performance qualification","passed":sum(x["status"]=="PASS" for x in results),"failed":sum(x["status"]=="FAIL" for x in results),"results":results}
    (a.output_dir/"results.json").write_text(json.dumps(output,indent=2)+"\n");print(json.dumps(output,indent=2));return int(output["failed"]!=0)


if __name__=="__main__":raise SystemExit(main())
