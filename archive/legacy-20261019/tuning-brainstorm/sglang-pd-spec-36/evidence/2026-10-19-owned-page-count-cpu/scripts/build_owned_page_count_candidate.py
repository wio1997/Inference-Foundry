#!/usr/bin/env python3
"""Build producer-owned opt-in count propagation, never alter installed files."""
import ast,copy,datetime,difflib,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];p=json.loads((R/'specs/L025-owned-design.json').read_text());O=R/'runs/L025-R001-owned-design';O.mkdir(parents=True,exist_ok=False);C=R/'candidates/npu-owned-page-count';C.mkdir(parents=True,exist_ok=False);sha=lambda b:hashlib.sha256(b).hexdigest();originals={f:(R/'sources/installed'/f).read_text() for f in p['baseline_hashes']}
for f,s in originals.items():assert sha(s.encode())==p['baseline_hashes'][f]
def replace(s,old,new):assert s.count(old)==1,(old[:100],s.count(old));return s.replace(old,new)
def modify(s,name,replacements):
 t=ast.parse(s);n=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name);old=ast.get_source_segment(s,n);new=old
 for a,b in replacements:new=replace(new,a,b)
 return replace(s,old,new)
paths=p['paths'];out=dict(originals);npu,alloc,eagle,pd=[paths[k] for k in ['npu','allocation','eagle','PD']]
out[npu]+='''

def _num_new_pages_from_owned_cpu(allocator, prefix_lens_cpu, seq_lens_cpu):
    """Only producers owning these mirrors may opt into this count."""
    if (
        type(allocator) is not NPUPagedTokenToKVPoolAllocator
        or allocator.page_size == 1
        or hasattr(allocator, "c128_attn_allocator")
    ):
        return None
    return get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=allocator.page_size,
        prefix_lens=prefix_lens_cpu,
    )
'''
out[alloc]=modify(out[alloc],'alloc_paged_token_slots_extend',[("    batch=None,\n)","    batch=None,\n    num_new_pages: Optional[int] = None,\n)"),("    extra_alloc_kwargs = {}\n","    extra_alloc_kwargs = {}\n    if num_new_pages is not None:\n        extra_alloc_kwargs[\"num_new_pages\"] = num_new_pages\n")])
out[alloc]=modify(out[alloc],'alloc_for_extend',[("        # Paged allocation - build last_loc\n",'''        owned_num_new_pages = None
        if _is_npu:
            from sglang.srt.hardware_backend.npu.allocator_npu import (
                _num_new_pages_from_owned_cpu,
            )

            owned_num_new_pages = _num_new_pages_from_owned_cpu(
                batch.tree_cache.token_to_kv_pool_allocator,
                prefix_lens_cpu,
                batch.seq_lens_cpu,
            )
        # Paged allocation - build last_loc
'''),("            batch=batch,\n        )","            batch=batch,\n            num_new_pages=owned_num_new_pages,\n        )")])
out[alloc]=modify(out[alloc],'alloc_for_spec_decode',[("    batch: Optional[ScheduleBatch] = None,\n)","    batch: Optional[ScheduleBatch] = None,\n    num_new_pages: Optional[int] = None,\n)"),("            out_cache_loc = ALLOC_EXTEND_FUNCS[device_type](",'''            owned_count_kwargs = {}
            if num_new_pages is not None:
                owned_count_kwargs["num_new_pages"] = num_new_pages
            out_cache_loc = ALLOC_EXTEND_FUNCS[device_type]('''),("                batch=batch,\n            )","                batch=batch,\n                **owned_count_kwargs,\n            )")])
out[eagle]=modify(out[eagle],'eagle_prepare_for_decode',[("    alloc_for_spec_decode(\n",'''    owned_num_new_pages = None
    if _is_npu and num_needed_tokens > 0:
        from sglang.srt.hardware_backend.npu.allocator_npu import (
            _num_new_pages_from_owned_cpu,
        )

        owned_num_new_pages = _num_new_pages_from_owned_cpu(
            batch.tree_cache.token_to_kv_pool_allocator,
            cur_kv_lens_cpu,
            nxt_kv_lens_cpu,
        )
    alloc_for_spec_decode(
'''),("        batch=batch,\n    )","        batch=batch,\n        num_new_pages=owned_num_new_pages,\n    )")])
out[pd]=modify(out[pd],'alloc_for_decode_prealloc',[("        else:\n            kv_loc = allocator.alloc_extend(",'''        else:
            owned_prefix_lens_cpu = torch.tensor([total_prefix_len], dtype=torch.int64)
            owned_seq_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
            if _is_npu:
                from sglang.srt.hardware_backend.npu.allocator_npu import (
                    _num_new_pages_from_owned_cpu,
                )

                owned_num_new_pages = _num_new_pages_from_owned_cpu(
                    allocator, owned_prefix_lens_cpu, owned_seq_lens_cpu
                )
                if owned_num_new_pages is not None:
                    extra_kwargs["num_new_pages"] = owned_num_new_pages
            kv_loc = allocator.alloc_extend('''),("                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),","                prefix_lens_cpu=owned_prefix_lens_cpu,"),("                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),\n                last_loc=last_loc,\n                extend_num_tokens=delta_len,\n                **extra_kwargs,\n            )\n        if dsv4_unwrap_prealloc", "                seq_lens_cpu=owned_seq_lens_cpu,\n                last_loc=last_loc,\n                extend_num_tokens=delta_len,\n                **extra_kwargs,\n            )\n        if dsv4_unwrap_prealloc")])
allowed={npu:[],alloc:['alloc_paged_token_slots_extend','alloc_for_extend','alloc_for_spec_decode'],eagle:['eagle_prepare_for_decode'],pd:['alloc_for_decode_prealloc']};ast_checks={};patch=[]
for f,after in out.items():
 before=originals[f];a=ast.parse(before);b=ast.parse(after);compile(b,f,'exec');restored=copy.deepcopy(b)
 if f==npu:
  restored.body=[x for x in restored.body if not (isinstance(x,ast.FunctionDef) and x.name=='_num_new_pages_from_owned_cpu')]
 else:
  for name in allowed[f]:
   old=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name);i=next(i for i,x in enumerate(restored.body) if isinstance(x,ast.FunctionDef) and x.name==name);restored.body[i]=copy.deepcopy(old)
 assert ast.dump(a,include_attributes=False)==ast.dump(restored,include_attributes=False),'Unexpected change outside declared functions: '+f
 if f==npu:
  oldcls=next(x for x in a.body if isinstance(x,ast.ClassDef));newcls=next(x for x in b.body if isinstance(x,ast.ClassDef));assert ast.dump(oldcls,include_attributes=False)==ast.dump(newcls,include_attributes=False)
 ast_checks[f]={'outside_declared_functions_unchanged':True,'changed_existing_functions':allowed[f],'new_private_helper_only':f==npu};target=C/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(after);patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+f,tofile='b/'+f));assert (R/'sources/installed'/f).read_text()==before
(C/'candidate.patch').write_text(''.join(patch));manifest={**p,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_hashes':{f:sha(s.encode()) for f,s in out.items()},'patch_sha256':sha((C/'candidate.patch').read_bytes()),'ast_checks':ast_checks,'allocator_class_AST_unchanged':True,'syntax_only':'PASS','CPU_correctness_executed':False,'native_executed':False,'accepted':False,'production_applied':False};(C/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(O/'result.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(O/'receipt.json').write_text(json.dumps({'exit_code':0,'source_unchanged':True,'artifacts':{str(q.relative_to(R)):sha(q.read_bytes()) for q in [C/'candidate.patch',C/'manifest.json',O/'result.json']}},indent=2)+'\n');print(json.dumps({'candidate':str(C),'files':len(out),'allocator_class_AST_unchanged':True,'syntax_only':True}))
