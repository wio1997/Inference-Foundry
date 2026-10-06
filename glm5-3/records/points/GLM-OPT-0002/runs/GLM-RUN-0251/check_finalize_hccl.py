"""Bounded actual TP16 HCCL byte/layout/lifetime checks; no model weights."""
import ast
import json
import os
import socket
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu


def method(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PrepareAndFinalizeWithAll2All')
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'finalize')
    ns = {'torch': torch, 'dist': dist}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    return ns['finalize']


def bits(x):
    return x.detach().cpu().contiguous().view(torch.uint8)


def worker(rank, port, oldpath, newpath, destination):
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', init_method='tcp://127.0.0.1:%d' % port, rank=rank, world_size=16)
    old, new = method(oldpath), method(newpath)
    rows = []; retained = []
    for num_tokens in (1, 3, 16, 17, 32):
        padded = ((num_tokens + 15)//16)*16; local_rows = padded//16
        for special in (False, True):
            for noncontiguous in (False, True):
                cpu = torch.full((local_rows, 6144), rank+1., dtype=torch.bfloat16)
                if special:
                    cpu[:, :4] = torch.tensor([float('nan'),float('inf'),0.,-0.], dtype=torch.bfloat16)
                if noncontiguous:
                    expanded = torch.empty((local_rows, 12288), dtype=torch.bfloat16, device='npu')
                    expanded[:, ::2] = cpu.to('npu'); local = expanded[:, ::2]
                else: local = cpu.to('npu')
                before = bits(local)
                state = types.SimpleNamespace(tp_size=16, num_tokens=padded, replace_allreduce=False,
                        moe_config=types.SimpleNamespace(tp_group=types.SimpleNamespace(device_group=dist.group.WORLD)))
                # Compare full gathered storage first; only then test unpadding.
                # HCCL's stock list API rejects strided input. Normalize its
                # reference input; the candidate performs that normalization.
                a = old(state, local.contiguous(), True, torch.Size((padded, 6144)))
                b = new(state, local, True, torch.Size((padded, 6144)))
                ab = bool(torch.equal(bits(a), bits(b)))
                preserved = bool(torch.equal(before, bits(local)))
                expected = torch.cat([torch.full((local_rows,6144),r+1.,dtype=torch.bfloat16) for r in range(16)])
                if special: expected[:, :4] = cpu[0,:4]
                exact_expected = bool(torch.equal(bits(b), expected.contiguous().view(torch.uint8)))
                unpad = bool(torch.equal(bits(a[:num_tokens]), bits(b[:num_tokens])))
                lifetime = all(torch.equal(bits(x), snapshot) for x, snapshot in retained[-2:])
                retained.append((b, bits(b)))
                ok = ab and preserved and exact_expected and unpad and lifetime
                flags = torch.tensor([int(ok)], dtype=torch.int32, device='npu')
                dist.all_reduce(flags, op=dist.ReduceOp.MIN)
                globally_ok = bool(flags.item())
                rows.append(dict(num_tokens=num_tokens, padded=padded, special_nan_inf_signed_zero=special,
                                 noncontiguous=noncontiguous, stock_reference_input_normalized=noncontiguous,
                                 input_format=torch_npu.get_npu_format(local),
                                 full_output_bytes_equal=ab, input_bytes_preserved=preserved,
                                 independent_expected_bytes=exact_expected, unpad_bytes_equal=unpad,
                                 retained_output_lifetime=lifetime, all_ranks_passed=globally_ok,
                                 old_equal_self_with_nan=bool(torch.equal(local,local))))
    torch.npu.synchronize()
    Path(destination, 'rank%d.json'%rank).write_text(json.dumps(dict(rank=rank,cases=rows,passed=all(x['all_ranks_passed'] for x in rows)),indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    oldpath, newpath, destination, ownerpath = sys.argv[1:]
    Path(destination).mkdir(parents=True,exist_ok=True)
    parts = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    Path(ownerpath).write_text(json.dumps(dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=parts[19],state=parts[0],ppid=int(parts[1])))+'\n')
    with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    mp.spawn(worker,args=(port,oldpath,newpath,destination),nprocs=16,join=True)
    rows=[json.loads(Path(destination,'rank%d.json'%r).read_text()) for r in range(16)]
    passed=all(x['passed'] for x in rows)
    Path(destination,'result.json').write_text(json.dumps(dict(passed=passed,ranks=16,cases_per_rank=20,
                       actual_model_loaded=False,old_torch_equal_gate_false_for_nan=all(not c['old_equal_self_with_nan'] for r in rows for c in r['cases'] if c['special_nan_inf_signed_zero'])),indent=2)+'\n')
    print(json.dumps(dict(passed=passed,ranks=16,cases_per_rank=20)),flush=True)
    sys.exit(0 if passed else 1)
