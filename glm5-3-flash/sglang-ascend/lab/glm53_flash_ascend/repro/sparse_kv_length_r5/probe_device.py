"""Bounded one-device R5 metadata equality/stream probe.

Root must confirm one free logical NPU, commit/bind source, then run in container.
No model or native attention kernel is loaded; this tests actual .to/cache values,
identity, no-version CPU changes and stream bypass only. Does not certify native
attention read-only ABI or full-model accuracy/performance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import torch
import torch_npu


class Mode:
    def __init__(self,name="decode"):self.name=name
    def is_decode(self):return self.name=="decode"
    def is_extend(self):return self.name=="extend"
    def is_target_verify(self):return False
    def is_draft_extend_v2(self):return False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",type=int,required=True);p.add_argument("--helper",type=Path,required=True);p.add_argument("--caller",type=Path,required=True)
    p.add_argument("--expected-helper-sha256",required=True);p.add_argument("--engine-commit",required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();assert sha(a.helper)==a.expected_helper_sha256;assert not a.output_dir.exists();a.output_dir.mkdir();sys.dont_write_bytecode=True
    (a.output_dir/"pre-execution-source.json").write_text(json.dumps({"scope":"One confirmed-free device, metadata conversion only; no model/native-attention/precision claim","device_logical_index":a.device,"engine_commit_declared":a.engine_commit,"helper_sha256":sha(a.helper),"caller_sha256":sha(a.caller),"probe_sha256":sha(__file__)},indent=2)+"\n")
    os.environ.pop("GLM53_EXECUTION_IDENTITY_DIR",None)
    torch.npu.set_device(a.device);device=torch.device("npu",a.device)
    spec=importlib.util.spec_from_file_location("r5_sparse_kv_device_probe",a.helper);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    native_convert=m._convert;copies=[]
    def counted(source,target):copies.append(1);return native_convert(source,target)
    m._convert=counted
    def state(source,mode="decode"):
        md=SimpleNamespace(actual_seq_lengths_kv=None,seq_lens_cpu_int=source,seq_lens=None)
        return SimpleNamespace(graph_mode=False,forward_metadata=md),SimpleNamespace(forward_mode=Mode(mode),spec_info=None,can_run_tbo=False,forward_metadata_ready=False,attn_cp_metadata=None),source
    def run(st,count=4):return m.sparse_kv_lengths_to_device(*st,device,count,enabled=True,caller_file=str(a.caller))
    results=[]
    def observe(name,fn):
        start=len(copies)
        try:detail=fn();results.append({"name":name,"status":"PASS","conversion_calls":len(copies)-start,"detail":detail})
        except Exception as ex:results.append({"name":name,"status":"FAIL","conversion_calls":len(copies)-start,"exception":type(ex).__name__,"traceback":traceback.format_exc()})
    def equality(dtype,data=(1,2048,16384,18432),count=4,mode="decode"):
        source=torch.tensor(data,dtype=dtype,device="cpu");expected=source.to(device=device,dtype=torch.int32)
        st=state(source,mode);start=len(copies);values=[run(st,count) for _ in range(11)]
        assert len(copies)-start==1 and all(x is values[0] for x in values)
        assert values[0].device==device and values[0].dtype==torch.int32 and torch.equal(values[0].cpu(),expected.cpu())
        return {"bitwise_integer_values":True,"layer_calls":11,"actual_first_copy_calls":1,"stream_key":m._current_stream_key(device)}
    observe("i32_actual_device_equality_and_reuse",lambda:equality(torch.int32))
    observe("i64_actual_device_cast_equality_and_reuse",lambda:equality(torch.int64))
    observe("prefill_one_request_2048_query_count",lambda:equality(torch.int32,(16384,),2048,"extend"))
    def inference_change():
        with torch.inference_mode():
            source=torch.tensor([1,2,3,4],dtype=torch.int32,device="cpu");st=state(source);start=len(copies);run(st);source.data[0]=31;value=run(st)
            assert len(copies)-start==2 and value.cpu().tolist()==[31,2,3,4]
        return {"CPU_value_key_detected_unversioned_change":True}
    observe("unversioned_CPU_change_device_value_updated",inference_change)
    def generation():
        source=torch.tensor([2048,4096,6144,8192],dtype=torch.int32,device="cpu");st=state(source);start=len(copies);first=run(st)
        st[0].forward_metadata=SimpleNamespace(actual_seq_lengths_kv=None,seq_lens_cpu_int=source,seq_lens=None)
        second=run(st);assert len(copies)-start==2 and first is not second and torch.equal(first.cpu(),second.cpu())
    observe("new_forward_chunk_metadata_cannot_reuse_old_copy",generation)
    def stream_bypass():
        source=torch.tensor([1,2,3,4],dtype=torch.int32,device="cpu");st=state(source);start=len(copies);first=run(st)
        alternate=torch.npu.Stream(device=device)
        with torch.npu.stream(alternate):second=run(st)
        third=run(st)
        assert len(copies)-start==2 and third is first and second is not first and torch.equal(second.cpu(),first.cpu())
        return {"alternate_stream_uses_original_copy":True,"no_wait_or_event_added_by_helper":True}
    observe("alternate_stream_bypass_preserves_first_stream_cache",stream_bypass)
    def device_source():
        source=torch.tensor([1,2,3,4],dtype=torch.int32,device=device);st=state(source);start=len(copies)
        a1=run(st);a2=run(st);assert len(copies)-start==2 and a1 is source and a2 is source
        return {"already_device_source_bypassed_no_D2H_guard":True}
    observe("NPU_source_original_to_noop_bypass",device_source)
    torch.npu.synchronize(device)
    output={"scope":"One-NPU metadata conversion/value/stream probe; test .cpu/synchronize are diagnostic only, never production logger or formal timing","device_logical_index":a.device,"passed":sum(x["status"]=="PASS" for x in results),"failed":sum(x["status"]=="FAIL" for x in results),"results":results}
    (a.output_dir/"results.json").write_text(json.dumps(output,indent=2)+"\n");print(json.dumps(output,indent=2));return int(output["failed"]!=0)


if __name__=="__main__":raise SystemExit(main())
