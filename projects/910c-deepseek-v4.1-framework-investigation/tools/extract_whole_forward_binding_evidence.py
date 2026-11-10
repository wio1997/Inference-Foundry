#!/usr/bin/env python3
"""Exact bounded source excerpts for dispatch facts, not kernel-review coverage."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import ast,json
ROOT=Path('audit/execution-search')
def main():
    d=json.loads((ROOT/'whole-forward-dispatch.json').read_text());sources={r['runtime_path']:r for r in d['sources']};meta=d['prior_exact_SHA_metadata_reference'];sources[meta['runtime_path']]=meta;ranges=defaultdict(list)
    for r in d['resolutions']:ranges[r['runtime_path']].append([r['first'],r['last']])
    selected={'/models/__init__.py':[[38,58]],'/models/deepseek_v41/model.py':[[151,260],[450,561],[766,842],[963,988],[1505,1608]],'/models/deepseek_v41/vl_model.py':[[45,101],[179,209],[261,275]],'/worker/model_runner_v1.py':[[2052,2119],[3262,3270],[3714,3907],[6128,6139]],'/device/hardware_profile.py':[[243,282],[298,314]],'/device/device_op.py':[[50,73],[2032,2041]],'/attention/dsa_v41.py':[[55,64],[839,865],[879,927],[1161,1279]],'/worker/device_metadata.py':[[34,57],[74,155],[174,185]],'/ascend_config.py':[[665,677],[798,806],[1298,1314]],'/spec_decode/dspark_proposer.py':[[23,75]],'/spec_decode/llm_base_proposer.py':[[115,124],[239,305],[638,652],[1454,1646]]}
    funcs={'/vllm_ascend/utils.py':['normalize_deepseek_v41_config','is_deepseek_v41','enable_dsa_cp'],'/patch/platform/patch_speculative_config.py':['_normalize_deepseek_dspark_draft','_dspark_post_init']}
    selected['/vllm_ascend/spec_decode/llm_base_proposer.py']=selected.pop('/spec_decode/llm_base_proposer.py')
    for path,s in sources.items():
        for suffix,rr in selected.items():
            if path.endswith(suffix):ranges[path]+=rr
        text=Path(s['local_path']).read_text();tree=ast.parse(text)
        for suffix,names in funcs.items():
            if path.endswith(suffix):ranges[path]+=[[n.lineno,n.end_lineno] for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if path.endswith('/config/speculative.py'):
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpeculativeConfig');ranges[path]+=[[n.lineno,n.end_lineno] for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ['use_eagle','uses_draft_model','use_dspark']];ranges[path]+=[[1461,1462],[588,588]]
        if path.endswith('/transformers_utils/configs/deepseek_v41.py'):ranges[path]+=[[1,len(text.splitlines())]]
    excerpts=[];index=[]
    for path,rr in sorted(ranges.items()):
        s=sources[path];raw=Path(s['local_path']).read_bytes();assert sha256(raw).hexdigest()==s['sha256'];lines=raw.decode().splitlines();merged=[]
        for a,z in sorted(rr):
            assert 1<=a<=z<=len(lines),(path,a,z,len(lines))
            if merged and a<=merged[-1][1]+1:merged[-1][1]=max(merged[-1][1],z)
            else:merged.append([a,z])
        index.append(dict(runtime_path=path,sha256=s['sha256'],binding_read_ranges=merged,qualification='Scalar/class/caller/graph-policy evidence only. Whole native arithmetic and every branch/reader are not closed by these source lines.'))
        for a,z in merged:excerpts.append(dict(runtime_path=path,sha256=s['sha256'],first_line=a,last_line=z,lines=lines[a-1:z]))
    out=dict(status='BOUNDED_DISPATCH_AND_CONFIG_BINDING_SOURCE_EXCERPTS',sources=index,excerpts=excerpts,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'whole-forward-binding-source-excerpts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(index),ranges=len(excerpts))))
if __name__=='__main__':main()
