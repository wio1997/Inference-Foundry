"""Keep CANN's device compiler on GCC11 headers; use GCC13 for host C++20."""
import pathlib,subprocess,json,os
root=pathlib.Path('/work')
prefix=root/'bisheng-gcc11'
for rel,target in [('lib/gcc/aarch64-linux-gnu/11','/usr/lib/gcc/aarch64-linux-gnu/11'),('include/c++/11','/usr/include/c++/11'),('include/aarch64-linux-gnu/c++/11','/usr/include/aarch64-linux-gnu/c++/11')]:
 p=prefix/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.symlink_to(target,target_is_directory=True)
entry=pathlib.Path('/usr/local/Ascend/cann-9.2.0/bin/bisheng')
record=root/'compiler-entrypoint.json'
if record.exists():
 real=pathlib.Path(json.loads(record.read_text())['original_bisheng'])
else:
 real=entry.resolve()
 if not entry.is_symlink():
  original=real;real=original.with_name(original.name+'.deepinfer-original');original.rename(real)
probe=root/'compiler-pair-probe.asc';probe.write_text('#include <utility>\n')
cmd=[str(real),'--gcc-toolchain='+str(prefix),'-std=c++20','--npu-arch=dav-3510','-fsyntax-only',str(probe)]
subprocess.run(cmd,check=True)
if entry.is_symlink():entry.unlink()
wrapper = """#!/usr/local/python3.12.13/bin/python3
import os,sys
args=sys.argv[1:]
device=any(a.endswith(('.asc','.cce')) or a in ('-xasc','-xcce') for a in args)
device=device or any(a=='-x' and i+1<len(args) and args[i+1] in ('asc','cce') for i,a in enumerate(args))
if device:
    command=[REAL, '--gcc-toolchain='+PREFIX, *args]
elif any(a.endswith(('.cpp','.cc','.cxx')) for a in args):
    filtered=[]
    skip=False
    for a in args:
        if skip:
            skip=False
            continue
        if a=='-mllvm':
            skip=True
            continue
        if a.startswith('--npu-arch=') or a in ('--cce-res-usage','-Wno-unneeded-internal-declaration','-Wno-microsoft-template'):
            continue
        filtered.append(a)
    command=['/usr/bin/g++-13', *filtered]
else:
    command=[REAL, *args]
os.execv(command[0],command)
""".replace('REAL',repr(str(real))).replace('PREFIX',repr(str(prefix)))
entry.write_text(wrapper)
entry.chmod(0o755)
record.write_text(json.dumps({'original_bisheng':str(real),'device_compiler_gcc_headers':11,'host_compiler':'/usr/bin/g++-13','dispatch':'C++ host sources use GCC13; Ascend C sources use Bisheng with GCC11 headers','toolchain_prefix':str(prefix),'reason':'CANN Bisheng rejects constrained std::pair overloads from GCC13.4 device headers','probe_passed':True},indent=2))
print('DEVICE_HEADER_TOOLCHAIN_PROBE_OK')
