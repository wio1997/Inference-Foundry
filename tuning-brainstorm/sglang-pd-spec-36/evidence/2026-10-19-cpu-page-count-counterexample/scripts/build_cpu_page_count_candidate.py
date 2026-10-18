#!/usr/bin/env python3
"""Build an isolated review artifact; never patch installed or production source."""
import ast,copy,datetime,difflib,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];spec=json.loads((R/'specs/L024-candidate-design.json').read_text());source=R/'sources/installed'/spec['source_path'];before=source.read_text();sha=lambda b:hashlib.sha256(b).hexdigest();assert sha(source.read_bytes())==spec['baseline_sha256'];O=R/'runs/L024-R001-candidate-design';O.mkdir(parents=True,exist_ok=False);C=R/'candidates/npu-cpu-page-count';C.mkdir(parents=True,exist_ok=False)
old='''        if num_new_pages is None:
            num_new_pages_tensor = (
                (seq_lens + self.roundup) // self.page_size
                - (prefix_lens + self.roundup) // self.page_size
            ).sum()
            num_new_pages_item = num_new_pages_tensor.item()
        else:
            num_new_pages_item = num_new_pages
'''
new='''        if num_new_pages is None:
            # Callers must own fresh CPU mirrors; these metadata checks do not
            # compare tensor values or introduce a device synchronization.
            if (
                isinstance(prefix_lens_cpu, torch.Tensor)
                and isinstance(seq_lens_cpu, torch.Tensor)
                and prefix_lens_cpu.device.type == "cpu"
                and seq_lens_cpu.device.type == "cpu"
                and prefix_lens_cpu.dtype == prefix_lens.dtype
                and seq_lens_cpu.dtype == seq_lens.dtype
                and prefix_lens_cpu.dtype in (torch.int32, torch.int64)
                and seq_lens_cpu.dtype in (torch.int32, torch.int64)
                and prefix_lens_cpu.ndim == seq_lens_cpu.ndim == 1
                and prefix_lens_cpu.shape == prefix_lens.shape
                and seq_lens_cpu.shape == seq_lens.shape
            ):
                num_new_pages_item = get_num_new_pages(
                    seq_lens=seq_lens_cpu,
                    page_size=self.page_size,
                    prefix_lens=prefix_lens_cpu,
                )
            else:
                num_new_pages_tensor = (
                    (seq_lens + self.roundup) // self.page_size
                    - (prefix_lens + self.roundup) // self.page_size
                ).sum()
                num_new_pages_item = num_new_pages_tensor.item()
        else:
            num_new_pages_item = num_new_pages
'''
assert before.count(old)==1;after=before.replace(old,new);a=ast.parse(before);b=ast.parse(after);compile(b,'isolated-candidate','exec')
def method(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='alloc_extend')
f,g=method(a),method(b);dump=lambda x:ast.dump(x,include_attributes=False);assert dump(f.args)==dump(g.args);assert dump(f.body[0])==dump(g.body[0]);assert dump(ast.Module(body=f.body[2:],type_ignores=[]))==dump(ast.Module(body=g.body[2:],type_ignores=[]));assert dump(ast.Module(body=g.body[1].body[0].orelse,type_ignores=[]))==dump(ast.Module(body=f.body[1].body,type_ignores=[]));assert dump(ast.Module(body=f.body[1].orelse,type_ignores=[]))==dump(ast.Module(body=g.body[1].orelse,type_ignores=[]))
restored=copy.deepcopy(b);target=method(restored);target.body=copy.deepcopy(f.body);assert dump(restored)==dump(a),'Unexpected changes outside count body'
(C/'allocator_npu.py').write_text(after);patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+spec['source_path'],tofile='b/'+spec['source_path']));(C/'candidate.patch').write_text(patch);assert source.read_text()==before
manifest={**spec,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_sha256':sha(after.encode()),'patch_sha256':sha(patch.encode()),'baseline_function_ast_sha256':sha(dump(f).encode()),'candidate_function_ast_sha256':sha(dump(g).encode()),'outside_function_unchanged':True,'post_count_and_explicit_branch_unchanged':True,'original_device_fallback_ast_unchanged':True,'compile_status':'SYNTAX_ONLY_PASS','CPU_correctness_executed':False,'native_executed':False,'production_applied':False,'accepted':False};(C/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(O/'result.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');receipt={'exit_code':0,'artifacts':{str(p.relative_to(R)):sha(p.read_bytes()) for p in [C/'allocator_npu.py',C/'candidate.patch',C/'manifest.json',O/'result.json']},'source_unchanged':True,'no_remote_service_or_model_execution':True};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'candidate':str(C),'syntax_only':True,'accepted':False,'production_applied':False}))
