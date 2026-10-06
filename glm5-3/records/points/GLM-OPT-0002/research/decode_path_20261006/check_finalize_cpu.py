"""Actual CPU/Gloo collective and buffer-lifetime equivalence, no NPU import."""
import ast
import json
import pathlib
import sys
import tempfile
import types
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def method(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PrepareAndFinalizeWithAll2All')
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'finalize')
    mod = ast.Module(body=[node], type_ignores=[])
    ns = {'torch': torch, 'dist': dist}
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)
    return ns['finalize']


def worker(rank, world, init, oldpath, newpath, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=world)
    old, new = method(oldpath), method(newpath)
    checks = 0
    retained = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for num_tokens in (1, 4, 5, 8, 32):
            for width in (13, 6144):
                padded = ((num_tokens + world - 1) // world) * world
                local_rows = padded // world
                for noncontiguous in (False, True):
                    local = torch.full((local_rows, width * (2 if noncontiguous else 1)), rank + 1., dtype=dtype)
                    if noncontiguous: local = local[:, ::2]
                    before = local.clone()
                    state = types.SimpleNamespace(tp_size=world, num_tokens=num_tokens, replace_allreduce=False,
                                                   moe_config=types.SimpleNamespace(tp_group=types.SimpleNamespace(device_group=dist.group.WORLD)))
                    # The list API accepts noncontiguous input; the direct API
                    # explicitly normalizes it, preserving the same values.
                    expected = torch.cat([torch.full((local_rows, width), r+1., dtype=dtype) for r in range(world)])[:num_tokens]
                    baseline = old(state, local, True, torch.Size((padded, width)))
                    patched = new(state, local, True, torch.Size((padded, width)))
                    assert torch.equal(expected, baseline) and torch.equal(expected, patched)
                    assert torch.equal(local, before) and patched.data_ptr() != local.data_ptr()
                    for previous, snapshot in retained[-3:]: assert torch.equal(previous, snapshot)
                    retained.append((patched, patched.clone()))
                    checks += 1
    state.tp_size = 1
    assert new(state, local, True, local.shape).data_ptr() == local.data_ptr()
    state.tp_size = world; state.replace_allreduce = True
    assert new(state, local, True, local.shape).data_ptr() == local.data_ptr()
    pathlib.Path(destination, f'rank{rank}.json').write_text(json.dumps({'rank':rank,'actual_collective_cases':checks,'input_preserved':True,'fresh_output_lifetime':True,'TP1_and_replace_allreduce':True})+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    oldpath, newpath, dest = sys.argv[1:]
    pathlib.Path(dest).mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='glm53-gather-cpu-') as tmp:
        mp.spawn(worker,args=(4,'file://'+tmp+'/init',oldpath,newpath,dest),nprocs=4,join=True)
    rows=[json.loads(pathlib.Path(dest,f'rank{r}.json').read_text()) for r in range(4)]
    print(json.dumps({'CPU_only':True,'NPU_used':False,'ranks':rows,'case_count_per_rank':60}),flush=True)
