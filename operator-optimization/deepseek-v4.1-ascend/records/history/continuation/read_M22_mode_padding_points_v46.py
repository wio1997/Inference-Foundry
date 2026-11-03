"""Saved-profile-led DP/mode/bucket questions; immutable legacy receipt verification."""
import ast,hashlib
from pathlib import Path
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,W,read,write
from finalize_Engram_whole_frontier_v30 import packet


def main():
    assert not(O/'REVIEW-CHECKPOINT-v46.json').exists()
    write(O/'PROFILE-CODE-QUESTION-GATE-v46.json',dict(status='PROFILE_FIRST_DP_MODE_PADDING_CASE_BINDING',
        previous_turn_classification='PROGRESS:v45registry28guardreconciliation',
        profile_facts=['64phase-associated decode graphs:16scheduledB2/T12 and48B4/T24; recorded target physical extent24.',
            'Saved main Engine capture sizes[8,16,24],DP2TP8/spec5; parser list chronology kept separate.',
            'Existing uniform6 plus TP8 minimum24 is a conditional interface bound; not a proof every24rowexecution has same cause.'],
        question='Which worker/dispatcher branches select uniform/full/nonuniform/fallback and coordinate padded token count across DP, and which actual branch values are not in saved profile?',
        allowed_new_source=['Same-SHA runner _determine_batch_execution_and_padding108lines in bounded chunks and _check_and_update_cudagraph_mode67lines only.',
            'Reuse previously extracted _sync_metadata_across_dp and stage dispatcher dispatch/descriptor/bucket snippets; no whole dispatcher/runner/SDK audit.'],
        no_model_torch_NPU_service_health_work=True,whole_module_closed=False,overall_complete=False))
    p=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/worker/model_runner_v1.py';b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c'
    t=ast.parse(b);lines=b.decode().splitlines();cls=next(c for c in t.body if isinstance(c,ast.ClassDef)and c.name=='NPUModelRunner');funcs=[]
    for name in ['_determine_batch_execution_and_padding','_check_and_update_cudagraph_mode']:
        f=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name==name);assert f.end_lineno-f.lineno+1<=115
        chunks=[]
        for a in range(f.lineno,f.end_lineno+1,35):
            z=min(a+34,f.end_lineno);chunks.append(dict(start=a,end=z,source=[f'{i+1}: {lines[i]}'for i in range(a-1,z)]))
        funcs.append(dict(name=name,start=f.lineno,end=f.end_lineno,chunks=chunks))
    manifest=read(O/'checkpoint-manifest-v16.json')['artifacts'];receipts={}
    for label in ['embedding-dispatch166-v16','padding-buckets166-v16','padding-config166-v16','final-scoped166-v16']:
        ref=next(a for a in manifest if Path(a['path']).name==label+'.json');rb=Path(ref['path']).read_bytes();assert hashlib.sha256(rb).hexdigest()==ref['sha256']
        x=packet(label,False);receipts[label]=dict(receipt=str(N/(label+'.json')),receipt_sha256=ref['sha256'],full_result=x['full_result'],full_result_sha256=x['result_sha256'],payload=x,
            verification='Historical immutable receipt SHA verified; legacy stdout omits full-result timestamp/envelope,so full-result SHA is not recomputed from subset.')
    old=read(O/'profile-raised-source-contract-v16.json');worker=next(x for x in old['excerpts']if x['path'].endswith('/worker/model_runner_v1.py'))
    sync=next(r for r in worker['ranges']if r['start']==825)
    assert sync['end']==885
    utility=next(r for r in receipts['padding-config166-v16']['payload']['result']if r['path'].endswith('/vllm_ascend/utils.py'))
    up=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/utils.py';assert hashlib.sha256(up.read_bytes()).hexdigest()==utility['sha256']=='5f88fd83bcba539c3d24324ed104469914254989996342b27bb36a326e3f8458'
    write(O/'M22-mode-padding-point-source-v46.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),path=str(p),sha256=sha,functions=funcs,reused_DP_sync=sync,
        historical_receipts=receipts,utils_historical166_stage_hash_attested=utility['sha256'],not_fresh_runtime_object_or_backend_snapshot=True,no_model_or_torch_import=True))
    for f in funcs:
        for r in f['chunks']:print('\n'.join(r['source']))
    ds=next(r for r in receipts['embedding-dispatch166-v16']['payload']['result']if r['path'].endswith('/cudagraph_dispatcher.py'))
    for f in ds['functions']:print('\n'.join(f['source']))


if __name__=='__main__':main()
