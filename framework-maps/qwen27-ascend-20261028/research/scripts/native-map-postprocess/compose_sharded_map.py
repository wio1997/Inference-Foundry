#!/usr/bin/env python3
"""Compose a lazy native-shard index into a copy of the existing preview map.

Run only on a provisioned CPU host. This reads the manifest/index and one
compressed rank detail at a time; it never opens the multi-GB native bundle.
The command's normal expected identity count is exactly 204.
"""
from __future__ import annotations
import argparse, copy, gzip, hashlib, json, os, shutil
from collections import Counter
from pathlib import Path

SECTIONS=('layer_rows','step_native_rows','eager_audits','eager_layer_rows','profile_rows')
MAX_RANK=256*1024*1024
MAX_UNATTR=256*1024*1024
HEX=set('0123456789abcdef')

class ComposeError(RuntimeError):pass
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read_json(p):
    try:
        with Path(p).open('r',encoding='utf-8') as f:return json.load(f)
    except Exception as e:raise ComposeError(f'cannot parse JSON {p}: {type(e).__name__}: {e}') from e
def ident(r):
    if not isinstance(r,dict):raise ComposeError('identity entry must be an object')
    vals=tuple(r.get(k) for k in ('framework','label','rank_basename','rank_key'))
    if not all(isinstance(x,str) and x.strip() for x in vals):raise ComposeError('identity requires explicit nonempty framework/label/rank_basename/rank_key')
    return vals
def checked_receipt(r,where):
    if not isinstance(r,dict) or not isinstance(r.get('path'),str):raise ComposeError(f'{where}: missing shard path')
    for k in ('json_bytes','gzip_bytes'):
        if type(r.get(k)) is not int or r[k]<0:raise ComposeError(f'{where}: invalid {k}')
    for k in ('json_sha256','gzip_sha256'):
        v=r.get(k)
        if not isinstance(v,str) or len(v)!=64 or any(c not in HEX for c in v):raise ComposeError(f'{where}: invalid {k}')
    p=Path(r['path'])
    if p.is_absolute() or '..' in p.parts or len(p.parts)!=1:raise ComposeError(f'{where}: shard path must be a simple basename')
    return p
def verify_gzip(path,receipt,limit):
    """Check compressed and inflated hashes/sizes in bounded chunks."""
    path=Path(path)
    if not path.is_file() or path.is_symlink():raise ComposeError(f'missing or non-regular shard: {path}')
    if path.stat().st_size!=receipt['gzip_bytes'] or sha_file(path)!=receipt['gzip_sha256']:
        raise ComposeError(f'compressed shard size/SHA mismatch: {path.name}')
    h=hashlib.sha256();n=0
    try:
        with gzip.open(path,'rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                n+=len(b)
                if n>limit:raise ComposeError(f'inflated shard exceeds configured limit: {path.name}')
                h.update(b)
    except ComposeError:raise
    except Exception as e:raise ComposeError(f'gzip integrity failure {path.name}: {type(e).__name__}: {e}') from e
    if n!=receipt['json_bytes'] or h.hexdigest()!=receipt['json_sha256']:
        raise ComposeError(f'inflated shard size/SHA mismatch: {path.name}')
    return n
def validate_inputs(index,manifest,manifest_bytes,index_dir,expected=204):
    if not isinstance(index,dict) or index.get('schema_version')!=1 or index.get('status')!='sharded_native_evidence_pending_root_acceptance':raise ComposeError('native index schema/status is not the gen34 sharded evidence contract')
    if index.get('identity_count')!=expected or index.get('bundle_errors')!=0:raise ComposeError('index must report exact identity_count and bundle_errors=0')
    limits=index.get('limits')
    if not isinstance(limits,dict):raise ComposeError('shard index lacks explicit builder limits')
    rank_limit=limits.get('max_rank_detail_json_bytes');unattr_limit=limits.get('max_unattributed_json_bytes')
    if type(rank_limit) is not int or not 0<rank_limit<=MAX_RANK:raise ComposeError('rank detail limit must be positive and no larger than 256 MiB')
    if type(unattr_limit) is not int or not 0<unattr_limit<=MAX_UNATTR:raise ComposeError('unattributed detail limit must be positive and no larger than 256 MiB')
    ins=index.get('inputs',{});mi=ins.get('completion_manifest',{});bi=ins.get('native_bundle',{})
    msha=sha_bytes(manifest_bytes)
    if mi.get('sha256')!=msha or mi.get('bytes')!=len(manifest_bytes):raise ComposeError('index completion-manifest receipt does not match supplied manifest bytes/SHA')
    if not isinstance(bi.get('path'),str) or type(bi.get('bytes')) is not int or bi['bytes']<=0 or not isinstance(bi.get('sha256'),str) or len(bi['sha256'])!=64 or any(c not in HEX for c in bi['sha256']):raise ComposeError('index lacks a valid prior native-bundle source receipt')
    if not isinstance(manifest,dict) or manifest.get('schema_version')!=1 or manifest.get('status')!='completed' or manifest.get('expected_rank_count')!=expected or not isinstance(manifest.get('rows'),list) or len(manifest['rows'])!=expected:
        raise ComposeError(f'completion manifest must be completed schema-v1 with exactly {expected} rows')
    expected_by_key={};triples=set()
    for r in manifest['rows']:
        fw,label,base,key=ident(r);tri=(fw,label,base)
        if key in expected_by_key or tri in triples:raise ComposeError('completion manifest has duplicate rank_key or framework/label/basename tuple')
        expected_by_key[key]={'framework':fw,'label':label,'rank_basename':base,'rank_key':key};triples.add(tri)
    rows=index.get('per_rank')
    if not isinstance(rows,list) or len(rows)!=expected:raise ComposeError('index per_rank must contain exact manifest membership')
    seen=set();paths=set();ordered=[]
    for r in rows:
        fw,label,base,key=ident(r)
        if key in seen or key not in expected_by_key or expected_by_key[key]!={'framework':fw,'label':label,'rank_basename':base,'rank_key':key}:
            raise ComposeError(f'index has duplicate or nonmanifest identity: {key}')
        rec=r.get('detail');rp=checked_receipt(rec,key)
        if rp.name in paths:raise ComposeError(f'duplicate shard path: {rp.name}')
        paths.add(rp.name);seen.add(key);ordered.append((r,index_dir/rp))
    if seen!=set(expected_by_key):raise ComposeError('index identity set differs from completion manifest')
    u=index.get('unattributed_shard');up=checked_receipt(u,'unattributed')
    if up.name in paths:raise ComposeError('unattributed shard path collides with a rank shard')
    sections=index.get('sections')
    if not isinstance(sections,dict) or set(sections)!=set(SECTIONS):raise ComposeError('index section inventory does not match shard contract')
    return ordered,(u,index_dir/up),expected_by_key,msha,rank_limit,unattr_limit

def verify_shards(index,ordered,unattr,rank_limit,unattr_limit):
    joined=Counter();lineage=Counter();rank_summaries=[]
    for row,path in ordered:
        rec=row['detail']
        if rec['json_bytes']>rank_limit:raise ComposeError(f'rank detail exceeds builder-declared byte limit: {path.name}')
        verify_gzip(path,rec,rank_limit)
        try:
            with gzip.open(path,'rt',encoding='utf-8') as f:detail=json.load(f)
        except Exception as e:raise ComposeError(f'invalid detail JSON {path.name}: {type(e).__name__}: {e}') from e
        ident_expected={k:row[k] for k in ('framework','label','rank_basename','rank_key')}
        if detail.get('schema_version')!=1 or detail.get('identity')!=ident_expected:raise ComposeError(f'detail identity mismatch: {path.name}')
        sections=detail.get('sections')
        if not isinstance(sections,dict) or set(sections)!=set(SECTIONS):raise ComposeError(f'detail sections missing/extra: {path.name}')
        counts=row.get('row_counts')
        if not isinstance(counts,dict):raise ComposeError(f'index rank row_counts missing: {row["rank_key"]}')
        actual={}
        for s in SECTIONS:
            if not isinstance(sections[s],list):raise ComposeError(f'detail section not a row array: {path.name}:{s}')
            n=len(sections[s]);actual[s]=n;joined[s]+=n;reported=counts.get(s)
            if reported is not None and (type(reported) is not int or reported!=n):raise ComposeError(f'index/detail row count mismatch: {row["rank_key"]}:{s}')
            if reported is None and s not in counts:raise ComposeError(f'index missing explicit nullable row count: {row["rank_key"]}:{s}')
        audit=detail.get('native_rank_audit')
        if not isinstance(audit,dict):raise ComposeError(f'detail audit missing: {path.name}')
        ls=(audit.get('lineage_summary') or {}).get('status') if isinstance(audit.get('lineage_summary'),dict) else None
        ls=ls or row.get('lineage_status') or 'unknown'
        if row.get('lineage_status')!=ls:raise ComposeError(f'index/detail lineage status mismatch: {row["rank_key"]}')
        lineage[(row['framework'],str(ls))]+=1
        rank_summaries.append({'framework':row['framework'],'label':row['label'],'rank_basename':row['rank_basename'],'rank_key':row['rank_key'],'lineage_status':ls,'row_counts':actual,'detail_json_bytes':rec['json_bytes'],'detail_gzip_bytes':rec['gzip_bytes']})
        del detail,sections,audit
    unrec,unpath=unattr
    if unrec['json_bytes']>unattr_limit:raise ComposeError('unattributed shard exceeds builder-declared byte limit')
    verify_gzip(unpath,unrec,unattr_limit)
    try:
        with gzip.open(unpath,'rt',encoding='utf-8') as f:uobj=json.load(f)
    except Exception as e:raise ComposeError(f'invalid unattributed JSON: {type(e).__name__}: {e}') from e
    if uobj.get('schema_version')!=1 or uobj.get('kind')!='unattributed_native_rows' or not isinstance(uobj.get('sections'),dict) or set(uobj['sections'])!=set(SECTIONS):raise ComposeError('unattributed shard schema mismatch')
    uncounts={}
    for s in SECTIONS:
        if not isinstance(uobj['sections'][s],list):raise ComposeError(f'unattributed section not an array: {s}')
        uncounts[s]=len(uobj['sections'][s])
    for s in SECTIONS:
        summary=index_section=index.get('sections',{}).get(s)
        if not isinstance(summary,dict):raise ComposeError(f'missing section totals for {s}')
        joined_claim=summary.get('rank_joined_row_count');unclaimed=summary.get('unattributed_row_count');source=summary.get('source_row_count')
        if any(type(x) is not int or x<0 for x in (joined_claim,unclaimed,source)):raise ComposeError(f'noninteger section totals for {s}')
        if joined[s]!=joined_claim or uncounts[s]!=unclaimed or source!=joined[s]+uncounts[s]:raise ComposeError(f'row conservation mismatch for {s}')
    del uobj
    return {'joined_row_counts':dict(joined),'unattributed_row_counts':uncounts,'lineage_counts':{f'{fw}:{ls}':n for (fw,ls),n in sorted(lineage.items())},'per_rank':rank_summaries}

def array_digest(v):return sha_bytes(json.dumps(v,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode('utf-8'))
def compose_map(combined,index,index_sha,manifest_sha,coverage,detail_base='native-rank-details/'):
    if not isinstance(combined,dict) or not isinstance(combined.get('metadata'),dict) or not isinstance(combined.get('new_measurements'),dict):raise ComposeError('combined map lacks metadata/new_measurements')
    nm=combined['new_measurements'];meta=combined['metadata'];cpu=nm.get('cpu_lifecycle_map')
    if not isinstance(cpu,dict) or not isinstance(cpu.get('native_bundle'),dict) or cpu['native_bundle'].get('status')!='pending_input_not_supplied':raise ComposeError('expected prior CPU map native_bundle pending field; refusing overwrite')
    if 'native_sharded_map' in nm:raise ComposeError('native_sharded_map already exists; refusing duplicate compose')
    protected={k:array_digest(v) for k,v in nm.items() if k!='cpu_lifecycle_map'}
    protected['cpu_lifecycle_map_non_native_fields']=array_digest({k:v for k,v in cpu.items() if k!='native_bundle'})
    protected['previous_native_bundle_pending']=array_digest(cpu['native_bundle'])
    # Only update the prior pending native status and add a lazy index reference.
    cpu['native_bundle']={'status':'export_verified_204_reviewable_pending_root_acceptance','identity_count':index['identity_count'],'bundle_errors':0,'completion_manifest_sha256':manifest_sha,'native_bundle_source':copy.deepcopy(index['inputs']['native_bundle']),'lineage_status_counts':coverage['lineage_counts'],'device_critical_path':'unresolved','layer_flow_binding':'unresolved','quality_equivalence':'unresolved','adoption_status':'not_decided','scope':'Artifact membership and shard integrity verified; do not infer graph task/layer/flow completion from export verification.'}
    nm['native_sharded_map']={'schema_version':1,'status':'REVIEWABLE_NATIVE_EVIDENCE_PENDING_ROOT_ACCEPTANCE','preview_not_final':True,'index':index,'detail_base_href':detail_base}
    meta['status']='REVIEWABLE_NATIVE_EVIDENCE_PENDING_ROOT_ACCEPTANCE';meta['preview_not_final']=True
    note='Native export shard integrity and manifest membership are reviewable; root acceptance, quality equivalence, adoption, device critical path, and layer/flow binding remain unresolved.'
    meta['validity']=(str(meta.get('validity',''))+' '+note).strip()
    after={k:array_digest(v) for k,v in nm.items() if k not in ('cpu_lifecycle_map','native_sharded_map')}
    after['cpu_lifecycle_map_non_native_fields']=array_digest({k:v for k,v in cpu.items() if k!='native_bundle'})
    after['previous_native_bundle_pending']=protected['previous_native_bundle_pending']
    if protected!=after:raise ComposeError('existing map arrays changed during composition')
    return combined,protected

def markdown(index,coverage,index_sha,manifest_sha):
    lines=['# Native sharded map coverage (reviewable; root acceptance pending)','',f"- Exact manifest identities: {index['identity_count']}",f"- Bundle errors from validated index: {index['bundle_errors']}",f"- Index SHA-256: `{index_sha}`",f"- Completion-manifest SHA-256: `{manifest_sha}`",'- Existing canary/formal arrays are retained unchanged; native detail stays in one lazy shard per rank.','- Export verification does not establish device critical path, complete layer or flow attribution, quality equivalence, or adoption.','- Null and unknown lineage/row counts remain unknown; no zero values are fabricated.','','## Section row conservation','','| section | rank joined | unattributed | source total |','|---|---:|---:|---:|']
    for s in SECTIONS:
        z=index['sections'][s];lines.append(f"| {s} | {z['rank_joined_row_count']} | {z['unattributed_row_count']} | {z['source_row_count']} |")
    lines+=['','## Lineage status counts','']
    for k,v in sorted(coverage['lineage_counts'].items()):lines.append(f'- `{k}`: {v}')
    lines+=['','']
    return '\n'.join(lines)

def copy_immutable(src,dst):
    src=Path(src);dst=Path(dst)
    if not src.is_file() or src.is_symlink():raise ComposeError(f'cannot stage missing/nonregular detail: {src}')
    try:os.link(src,dst)
    except OSError:shutil.copyfile(src,dst)

def build(combined_path,index_path,manifest_path,output_dir,expected=204):
    combined_path=Path(combined_path).resolve();index_path=Path(index_path).resolve();manifest_path=Path(manifest_path).resolve();out=Path(output_dir).resolve()
    if out.exists():raise ComposeError(f'output directory must be fresh: {out}')
    map_bytes=combined_path.read_bytes();index_bytes=index_path.read_bytes();manifest_bytes=manifest_path.read_bytes()
    combined=read_json(combined_path);index=json.loads(index_bytes);manifest=json.loads(manifest_bytes)
    ordered,unattr,_,msha,rank_limit,unattr_limit=validate_inputs(index,manifest,manifest_bytes,index_path.parent,expected)
    coverage=verify_shards(index,ordered,unattr,rank_limit,unattr_limit)
    index_sha=sha_bytes(index_bytes);before=sha_bytes(map_bytes)
    combined,protected=compose_map(combined,index,index_sha,msha,coverage)
    out.mkdir(parents=True,exist_ok=False)
    try:
        details=out/'native-rank-details';details.mkdir()
        copy_immutable(index_path,details/'index.json')
        for row,path in ordered:copy_immutable(path,details/row['detail']['path'])
        copy_immutable(unattr[1],details/unattr[0]['path'])
        map_out=out/'combined-map.json';map_raw=(json.dumps(combined,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8');map_out.write_bytes(map_raw)
        md=markdown(index,coverage,index_sha,msha).encode('utf-8');(out/'native-coverage.md').write_bytes(md)
        # Recheck source map bytes and protected canary/formal arrays before receipts are finalized.
        if sha_file(combined_path)!=before:raise ComposeError('source combined map changed during composition')
        receipt={'schema_version':1,'status':'REVIEWABLE_NATIVE_EVIDENCE_PENDING_ROOT_ACCEPTANCE','preview_not_final':True,'source_combined_map':{'path':str(combined_path),'bytes':len(map_bytes),'sha256':before},'source_index':{'path':str(index_path),'bytes':len(index_bytes),'sha256':index_sha},'source_completion_manifest':{'path':str(manifest_path),'bytes':len(manifest_bytes),'sha256':msha,'row_count':expected},'output_combined_map':{'path':str(map_out),'bytes':len(map_raw),'sha256':sha_bytes(map_raw)},'coverage_markdown':{'path':str(out/'native-coverage.md'),'bytes':len(md),'sha256':sha_bytes(md)},'detail_base_href':'native-rank-details/','verified_identity_count':expected,'bundle_errors':0,'section_conservation':{'joined_row_counts':coverage['joined_row_counts'],'unattributed_row_counts':coverage['unattributed_row_counts']},'lineage_counts':coverage['lineage_counts'],'preserved_existing_array_digests':protected,'limitations':['No quality-equivalence or adoption claim.','Export membership/hash integrity does not establish device critical path, layer/flow attribution, or whole-task completion.','Unknown and nullable counts remain unknown.'],'shard_set_sha256':sha_bytes(json.dumps([r['detail'] for r,_ in ordered],sort_keys=True,separators=(',',':')).encode())}
        rb=(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode();(out/'composition-receipt.json').write_bytes(rb)
        return receipt
    except Exception:
        shutil.rmtree(out,ignore_errors=True);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--combined-map',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--completion-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.combined_map,a.index,a.completion_manifest,a.output_dir),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
