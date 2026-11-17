"""生成G133/G134完整启动命令；本程序不执行Docker，不启动NPU。"""
from pathlib import Path
import argparse,hashlib,json,shlex
ROOT=Path(__file__).resolve().parents[1];CURRENT=ROOT/'reproduction/20261116'
def command(engine,runtime,model,name=None):
    argv=json.loads((CURRENT/f'deployment/{engine}-docker.json').read_text())
    for i,value in enumerate(argv):
        if i and argv[i-1]=='--name' and name:argv[i]=name
        elif i and argv[i-1]=='-v':
            host,rest=value.split(':',1)
            if host.startswith('/data/sglang-dsv41'):host=str(runtime)+host.removeprefix('/data/sglang-dsv41')
            elif host.startswith('/data/models/'):host=str(model)
            argv[i]=host+':'+rest
    return argv
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',choices=['sglang','vllm'],required=True)
    p.add_argument('--runtime-root',type=Path,default=ROOT/'.work/current');p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--name');p.add_argument('--shell',action='store_true');p.add_argument('--check-files',action='store_true')
    p.add_argument('--allow-rebuilt-binaries',action='store_true',help='只允许已记录重建SHA的产物；不代表目标机功能验收通过')
    a=p.parse_args();runtime=a.runtime_root.resolve();model=a.model_path.resolve();argv=command(a.engine,runtime,model,a.name)
    if a.check_files:
        for name,item in json.loads((CURRENT/'model-control-hashes.json').read_text()).items():
            assert hashlib.sha256((model/name).read_bytes()).hexdigest()==item['expected'],name
        assert not any((runtime/'sglang_penalty133'/name).exists() for name in ['projection_hits/FROZEN.json','smallHC_hits/FROZEN.json']), 'Do not reuse old qualification/freeze state for a new model generation'
        if a.engine=='sglang':
            manifest=json.loads((CURRENT/'runtime-manifest.json').read_text())
            for item in manifest['files']:assert hashlib.sha256((runtime/item['path']).read_bytes()).hexdigest()==item['published_sha256'],item['path']
            rebuilt={item['path']:item for item in json.loads((runtime/'rebuilt-binary-hashes.json').read_text())} if a.allow_rebuilt_binaries else {}
            for item in json.loads((CURRENT/'binary-manifest.json').read_text()):
                digest=hashlib.sha256((runtime/item['path']).read_bytes()).hexdigest()
                assert digest==item['measured_sha256'] or (a.allow_rebuilt_binaries and rebuilt[item['path']]['rebuilt_sha256']==digest),item['path']
        for i,value in enumerate(argv):
            if i and argv[i-1]=='-v':assert Path(value.split(':',1)[0]).exists(),value
    print(shlex.join(argv) if a.shell else json.dumps(argv,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
