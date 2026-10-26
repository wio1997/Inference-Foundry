#!/usr/bin/env python3
"""Materialize one controller-specified AST candidate; no tests or device work."""
import ast,copy,datetime,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L006-R000-materialize';O.mkdir(parents=True,exist_ok=False)
frozen=R/'runs/L004-R001-descriptor-cpu/candidate-method.py';source=R/'sources/installed/python/sglang/srt/disaggregation/ascend/conn.py'
t=ast.parse(frozen.read_text());range_node=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='range_metadata');old=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='candidate_sender')
src=ast.parse(source.read_text());cl=next(n for n in src.body if isinstance(n,ast.ClassDef) and n.name=='AscendKVManager');installed=copy.deepcopy(next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='send_kvcache'));installed.name='installed_sender'
new=copy.deepcopy(old);new.name='columns_sender';count=0
replacement=ast.parse('''
def process_layers(layers_params):
    src_addrs, dst_addrs, lengths = [], [], []
    for src_ptr, dst_ptr, item_len in layers_params:
        for src_start, dst_start, run_len in transfer_ranges:
            src_addrs.append(src_ptr + src_start * item_len)
            dst_addrs.append(dst_ptr + dst_start * item_len)
            lengths.append(item_len * run_len)
    if not src_addrs:
        return 0
    return self.engine.batch_transfer_sync(mooncake_session_id, src_addrs, dst_addrs, lengths)
''').body[0]
for i,n in enumerate(new.body):
 if isinstance(n,ast.FunctionDef) and n.name=='process_layers':new.body[i]=replacement;count+=1
guard=ast.If(test=ast.parse('self.pp_size != 1 or self.is_hybrid_mla_backend or self.enable_custom_mem_pool',mode='eval').body,body=[ast.Return(value=ast.Call(func=ast.Name(id='installed_sender',ctx=ast.Load()),args=[ast.Name(id=a.arg,ctx=ast.Load()) for a in [*new.args.posonlyargs,*new.args.args]],keywords=[]))],orelse=[])
new.body.insert(0,guard);count+=1;assert count==2
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(range_node),copy.deepcopy(old),installed,new],type_ignores=[]);ast.fix_missing_locations(mod)
out=R/'candidates/L006/direct_native_columns.py';out.parent.mkdir(parents=True,exist_ok=False);out.write_text('# Experimental CPU-only candidate; not applied to production source.\n'+ast.unparse(mod)+'\n')
v={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'inputs':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [frozen,source]},'output':str(out.relative_to(R)),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'ast_changes':count,'fallback':'PP!=1 or hybrid or custom memory pool delegates to original installed sender; only fallback dispatch is tested for untested modes','exit_code':0,'tests_executed':False,'production_changed':False}
(O/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'output':v['output'],'sha256':v['output_sha256'],'exit_code':0}))
