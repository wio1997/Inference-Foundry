#!/usr/bin/env python3
"""Reconcile an exact fresh workbench base; only our status aside changes HTML."""
from pathlib import Path
from hashlib import sha1,sha256
import json,re,zlib
from update_embedding_static_maps import TITLE,BODY,NEXT,ASIDE_PATTERN,banner
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
BASE=Path('/tmp/embedding-static-public-base-7cf')
def main():
    receipt=json.loads(Path('/tmp/embedding-static-public-base-manifest.json').read_text());files=[]
    def add(path,content,source):
        b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
    names=['embedding-entry-dependencies.json','embedding-dispatch-dependencies.json','embedding-registration-dependencies.json','embedding-method-dependencies.json','embedding-coordinate-dependencies.json','embedding-selected-launch.json','embedding-quant-description.json','embedding-entry-primary.json','embedding-static-source-review.json','embedding-static-source-excerpts.json','embedding-static-frontier.json','embedding-static-integrity.json','first-static-parent-receipt.json','first-static-math-receipt.json','first-static-parent-price.json','first-static-parent-price-a1-invalid.json','词嵌入实际派发与首层静态计算整段判断.md']
    for n in names:add('handoff/evidence/'+n,(ROOT/n).read_text(),str(ROOT/n))
    tools=['read_embedding_entry.py','read_embedding_dispatch_dependencies.py','read_embedding_registration.py','read_embedding_method_dependencies.py','read_embedding_coordinate_dependencies.py','read_embedding_quant_description.py','read_first_static_parent_windows.py','read_first_static_math_boundaries.py','price_first_static_parent.py','review_embedding_static_entry.py','update_embedding_static_maps.py','verify_embedding_static_evidence.py','prepare_embedding_static_publication.py','stage_embedding_static_blobs.py']
    for n in tools:add('tools/'+n,(Path('tools')/n).read_text(),'tools/'+n)
    review=json.loads((ROOT/'embedding-static-source-review.json').read_text());index=[]
    for s in review['sources']:
        path='handoff/evidence/embedding-static-source'+s['runtime_path'];add(path,Path(s['local_path']).read_text(),s['local_path']);index.append(dict(runtime_path=s['runtime_path'],path=PREFIX+path,sha256=s['runtime_sha256']))
    hf=review['prior_exact_SHA_references'][0];path='handoff/evidence/embedding-static-source/vllm-workspace/vllm/vllm/transformers_utils/configs/deepseek_v41.py';add(path,Path(hf['local_path']).read_text(),hf['local_path']);index.append(dict(runtime_path='/vllm-workspace/vllm/vllm/transformers_utils/configs/deepseek_v41.py',path=PREFIX+path,sha256=hf['sha256'],qualification='prior exactSHA, not new current acquisition'))
    mathraw=json.loads(zlib.decompress((ROOT/'first-static-math-raw.json.zlib').read_bytes()));rankindex=[]
    for rank in mathraw:
        path=f"handoff/evidence/first-static-native-ranks/rank{rank['rank']:02d}.json";text=json.dumps(rank,separators=(',',':'))+'\n';add(path,text,'All original TASK/shape/stream/model timestamps and new source endpoints');rankindex.append(dict(rank=rank['rank'],path=PREFIX+path,canonical_sha256=sha256(text.encode()).hexdigest(),windows=len(rank['windows'])))
    canonical=lambda x:json.dumps(x,separators=(',',':')).encode()
    mathtext=zlib.decompress((ROOT/'first-static-math-raw.json.zlib').read_bytes());assert canonical(mathraw)==mathtext
    drop=[n+'_'+suffix for n in ['target','draft'] for suffix in ['pre_rope_end_task','cache_norm_task','Qb_task','static_after_HC_ns']]+['math_boundary_qualified']
    original_parent=json.loads(zlib.decompress((ROOT/'first-static-parent-raw.json.zlib').read_bytes()))
    parent_metadata=[{k:v for k,v in r.items() if k!='windows'} for r in original_parent]
    parent=[dict(**meta,windows=rank['windows']) for meta,rank in zip(parent_metadata,json.loads(canonical(mathraw)))]
    for rank in parent:
        for w in rank['windows']:
            for k in drop:w.pop(k,None)
    parenttext=zlib.decompress((ROOT/'first-static-parent-raw.json.zlib').read_bytes());assert canonical(parent)==parenttext
    native=dict(ranks=rankindex,parent_rank_metadata=parent_metadata,rank_order=[r['rank'] for r in mathraw],math_canonical_SHA256=sha256(mathtext).hexdigest(),parent_canonical_SHA256=sha256(parenttext).hexdigest(),parent_drop_fields=drop,reconstruction='Load rank JSON in rank_order. json.dumps(array,separators=(comma,colon)) exactly reconstructs original decompressed math JSON bytes. Remove parent_drop_fields only inside windows, then rebuild each parent rank from its ordered parent_rank_metadata followed by windows to reconstruct original decompressed parent JSON bytes. Both full arrays checked here. Local compressed original receipts retain their distinct binary hashes.',all_rank_windows=3696,active_rank_windows=3624,physical_active_wave_groups=454,priced_active_request_groups=453,qualification='No omitted native rows from these3696windows; synthetic/inactive windows are retained even though8synthetic rank windows were not assigned active-request price.')
    add('handoff/evidence/first-static-native-ranks/manifest.json',json.dumps(native,ensure_ascii=False,indent=2)+'\n','Full original canonical native records verified, no trace resampling')
    front=json.loads((ROOT/'embedding-static-frontier.json').read_text());base_checks=[]
    for name,info in receipt['files'].items():
        b=(BASE/name).read_bytes();blob=lambda x:sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()
        if blob(b)!=info['sha'] and b.endswith(b'\n') and blob(b[:-1])==info['sha']:b=b[:-1]
        assert blob(b)==info['sha'],name
        base_checks.append(dict(path=name,git_blob_sha1=blob(b),sha256=sha256(b).hexdigest(),bytes=len(b)))
        s=b.decode()
        if name.endswith('.html'):
            old=s;s,n=re.subn(ASIDE_PATTERN,lambda _:banner('../词嵌入实际派发与首层静态计算整段判断.md'),s,count=1,flags=re.S);assert n==1
            assert re.sub(ASIDE_PATTERN,'',old,flags=re.S)==re.sub(ASIDE_PATTERN,'',s,flags=re.S)
        elif name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
            assert '## 当前断点 · '+TITLE not in s;i=s.index('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':
                s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：本段23当前Python源+1prior HF精确SHA、29原方法stub控制；原156位77受限源码绑定/79待追。eager/FULL lookup路径已分开，较大input/Engram交付仍值得深入，新原语/框架准入0。位数不是算子数、运行人口或全链路完成率。',s,count=1,flags=re.M)
                a=s.find('**2026-11-10 当前资格**')
                if a>=0:
                    z=s.index('\n\n',a);s=s[:a]+'**2026-11-10 当前资格**：每替换原语逐位一致，BF16末位差异失败。本轮原语/框架准入0，旧封闭CPU租约原型1保留。77源码约束/79待追；Run133903Z有界PASS/NA，Run150110Z PLANNED。大input/Engram交付25%实删暴露、额外0.1ms/波的原C8条件整批净价1.142482%，非实测；453活跃DP组不等于454物理组，不代填六普通负载。所有native位、最后读者/退役和packet完整bytes等具体缺口保留，尚未扫全。'+s[z:]
        else:
            d=json.loads(s);d.update(status=front['status'],next_action=NEXT,embedding_static_entry_review=front,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run133903Z bounded source/old-native-record review PASSNA;77sourcebindings79pending; Run150110Z PLANNED',previous_goal_turn_classification='PROGRESS:actual OOT TP8 FLOAT embedding/eagerFULL split and source static firstHC preRoPE boundary;23current sources+1prior HF,29 source controls,7 additional sitebindings;3624active native endpoints qualified; larger paid input/Engram parent remains pending; no newprimitive/framework admission.')
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh pinned workbench '+receipt['base_commit']+'; preserve concurrent token/dataflow/timeline and history')
    dest='handoff/evidence/embedding-static-publication-manifest.json'
    manifest=dict(base_commit=receipt['base_commit'],paths=[f['path'] for f in files]+[PREFIX+dest],sources=index,base_checks=base_checks,source_bindings=77,pending_original_sites=79,original_method_controls=29,primitive_bitwise_required=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,HTML_changes='Only our status aside; all other bytes identical to pinned fresh base',native_records=native,qualification='Full sources and full3696nativewindows retained; source controls/paid old-cohort sensitivities do not establish full chain coverage or native replacement bits. After mutation verify exact paths and all file contents including full large HTML/rank files; never interpret empty connector content as data loss.')
    p=ROOT/'embedding-static-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(dest,p.read_text(),str(p))
    assert len({f['path'] for f in files})==len(files)
    p=Path('/tmp/embedding-static-publication-payload.json');p.write_text(json.dumps(dict(base_commit=receipt['base_commit'],base_tree_sha='fdceb8831229e596804f4a3940049f131a26d5a3',files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),bytes=p.stat().st_size,max_file_bytes=max(f['bytes'] for f in files))))
if __name__=='__main__':main()
