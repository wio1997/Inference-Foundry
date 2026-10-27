#!/usr/bin/env python3
"""Collect CPU-visible parameter/buffer metadata and exact data_ptrs only.

Pass an already-loaded model explicitly. No tensor values are read or copied.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, re
from pathlib import Path

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

def _metadata_tensor(tensor):
    """Call metadata/pointer APIs only; no item/cpu/numpy or synchronization."""
    ptr = int(tensor.data_ptr())
    try: shape = [int(x) for x in tensor.shape]
    except Exception: shape = None
    try: stride = [int(x) for x in tensor.stride()]
    except Exception: stride = None
    try: storage_offset = int(tensor.storage_offset())
    except Exception: storage_offset = None
    try: dtype = str(tensor.dtype)
    except Exception: dtype = None
    try: device = str(tensor.device)
    except Exception: device = None
    storage_ptr = storage_nbytes = None
    try:
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr()) or None
        storage_nbytes = int(storage.nbytes())
    except Exception:
        pass
    return {"data_ptr_hex":hex(ptr) if ptr else None,"data_ptr_raw_hex":hex(ptr),"shape":shape,"stride":stride,
            "storage_offset":storage_offset,"dtype":dtype,"device":device,
            "storage_ptr_hex":hex(storage_ptr) if storage_ptr is not None else None,
            "storage_nbytes":storage_nbytes}

def _named(model, kind):
    fn = getattr(model, "named_parameters" if kind=="parameter" else "named_buffers")
    try: return fn(recurse=True, remove_duplicate=False)
    except TypeError: return fn(recurse=True)

def collect_model_catalog(model, model_id=None, expected_layers=64):
    """Return pointer catalog. Preserve aliases; never silently deduplicate."""
    entries=[]
    for kind in ("parameter","buffer"):
        for name,tensor in _named(model,kind):
            row={"kind":kind,"name":str(name),**_metadata_tensor(tensor)}
            match=LAYER_RE.search(str(name))
            row["layer_index"]=int(match.group(1)) if match else None
            row["layer_path_clear"]=bool(match)
            row["classification"]=("layer" if match and int(match.group(1))<expected_layers else "nonlayer" if not match else "out_of_expected_layer_range")
            entries.append(row)
    by_ptr={}
    for row in entries:
        if row["data_ptr_hex"] is not None and row.get("device") != "meta": by_ptr.setdefault(row["data_ptr_hex"],[]).append(row)
    by_storage={}
    for row in entries:
        sp=row.get("storage_ptr_hex")
        if sp and row.get("device") != "meta": by_storage.setdefault(sp,[]).append(row)
    pointer_rows=[]
    for ptr,aliases in sorted(by_ptr.items(),key=lambda kv:int(kv[0],16)):
        layers=sorted({r["layer_index"] for r in aliases if r["classification"]=="layer"})
        classes=sorted({r["classification"] for r in aliases})
        storage_keys={r.get("storage_ptr_hex") for r in aliases if r.get("storage_ptr_hex")}
        storage_aliases=[r for sk in storage_keys for r in by_storage.get(sk,[])]
        storage_layers=sorted({r["layer_index"] for r in storage_aliases if r["classification"]=="layer"})
        storage_classes={r["classification"] for r in storage_aliases}
        status="ambiguous_shared_storage_cross_layer" if len(storage_layers)>1 else "ambiguous_cross_layer_alias" if len(layers)>1 else "layer_and_nonlayer_alias" if layers and ("nonlayer" in classes or (storage_layers and "nonlayer" in storage_classes)) else "nonlayer" if not layers else "out_of_expected_layer_range" if "out_of_expected_layer_range" in classes else "unique_layer"
        pointer_rows.append({"data_ptr_hex":ptr,"source_count":len(aliases),"candidate_layers":layers,"storage_candidate_layers":storage_layers,"status":status,"sources":aliases,"storage_alias_sources":storage_aliases})
    layer_indices=sorted({r["layer_index"] for r in entries if r["classification"]=="layer"})
    payload={"schema_version":1,"model_id":model_id,"expected_layer_count":expected_layers,
             "layer_indices_present":layer_indices,"missing_layer_indices":sorted(set(range(expected_layers))-set(layer_indices)),
             "entry_count":len(entries),"unique_data_ptr_count":len(pointer_rows),"pointers":pointer_rows,
             "unusable_entries":[r for r in entries if r["data_ptr_hex"] is None or r.get("device")=="meta"],
             "safety":"metadata and address APIs only; tensor values are not read; no cpu/item/numpy/synchronize calls"}
    payload["catalog_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
    return payload

def write_catalog(payload,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");os.replace(tmp,p)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output",type=Path);ap.add_argument("--model-id");ap.add_argument("--expected-layers",type=int,default=64)
    ap.add_argument("--factory",required=True,help="module:function returning an already loaded model; factory is called once")
    args=ap.parse_args();mod,fn=args.factory.split(":",1);model=getattr(importlib.import_module(mod),fn)()
    write_catalog(collect_model_catalog(model,args.model_id,args.expected_layers),args.output)
if __name__=="__main__":main()
