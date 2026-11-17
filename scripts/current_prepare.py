"""准备G133/G134复现目录；默认仅列计划。build只在aarch64目标机CPU容器执行。"""
from pathlib import Path
import argparse,hashlib,json,platform,shutil,subprocess,sys,uuid

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/'reproduction/20261116'
SG='sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517'
NATIVE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,default=ROOT/'.work/current')
    p.add_argument('--stage',action='store_true');p.add_argument('--skip-datasets',action='store_true',help='仅供离线源码/路径验档，不能用于压测')
    p.add_argument('--build',action='store_true');a=p.parse_args();out=a.runtime_root.resolve()
    manifest=load(CURRENT/'runtime-manifest.json')
    if not (a.stage or a.build):
        print(json.dumps({'runtime_root':str(out),'source_files':len(manifest['files']),'binary_files':19,
            'plan':['stage 268 source/metadata files +16 configs +8 exact datasets','prepare existing 3 ABI adapters and strict QLI/Sparse provider',
                    'extract pinned native host objects/SDK by docker cp without starting container','link isolated MoE host/proto; retain native API/kernel bytes',
                    'build hidden Torch2.13 MoE adapter with no NPU devices','record actual rebuilt hashes; require later target startup/functional gates'],
            'device_execution':False},ensure_ascii=False,indent=2));return
    if a.stage:
        assert not out.exists(),'Existing runtime needs review; choose a fresh task directory'
        out.mkdir(parents=True)
        for item in manifest['files']:
            src=CURRENT/'runtime'/item['path'];assert sha(src)==item['published_sha256']
            dest=out/item['path'];dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(src,dest)
        shutil.copytree(CURRENT/'configs',out/'configs')
        (out/'datasets').mkdir();shutil.copyfile(CURRENT/'dataset-manifest.json',out/'datasets/manifest.json')
        if not a.skip_datasets:
            subprocess.run([sys.executable,str(ROOT/'scripts/unpack_data.py')],cwd=ROOT,check=True)
            for item in load(CURRENT/'dataset-manifest.json')['profiles']:
                rel=item['path'].removeprefix('/task/');src=ROOT/rel;assert sha(src)==item['sha256']
                shutil.copyfile(src,out/rel)
        for engine in ['sglang','vllm']:
            for key,value in load(CURRENT/f'deployment/{engine}-env.json').items():
                if key in {'TMPDIR','HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR',
                           'PYTHONPYCACHEPREFIX','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH'} and value.startswith('/task/'):
                    (out/value.removeprefix('/task/')).mkdir(parents=True,exist_ok=True)
        (out/'stage.json').write_text(json.dumps({'source_files':268,'datasets_staged':not a.skip_datasets,'NPU_actions':False},indent=2)+'\n')
        print('Staged source/configuration only; binaries are not built and no model was started.')
    if not a.build:return
    assert platform.machine()=='aarch64','Build on the actual aarch64 target; no client-VM emulation'
    assert (out/'stage.json').is_file(),'Stage a fresh runtime first'
    def run(cmd,**kw):return subprocess.run(cmd,cwd=ROOT,check=True,**kw)
    for image in [SG,NATIVE]:
        info=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
        assert info['Id']==image and info['Architecture']=='arm64','Obtain exact archived images first'
    if not (ROOT/'compat/vendor_strict_host_g2/strict_host_manifest.json').is_file():
        run([sys.executable,str(ROOT/'scripts/prepare_dependencies.py'),'--execute'])
    for item in load(CURRENT/'binary-manifest.json'):
        name=item['path']
        if name.startswith('compat/'):
            source=ROOT/name;assert source.is_file(),name
            dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    # Reuse the pinned donor already extracted by the common dependency recipe.
    donor=ROOT/'compat/donor_vendor/vendors/custom_transformer';assert donor.is_dir()
    s=out/'continuation_G58_provider_isolation_20261115';inputs=s/'link_sources';inputs.mkdir(parents=True,exist_ok=False)
    object_meta=load(CURRENT/'build/provider-object-inputs.json')
    sdk_meta=load(CURRENT/'build/provider-sdk-inputs.json')
    cid=subprocess.check_output(['docker','create','--network','none','--label','archive.purpose=current-extraction','--entrypoint','true',NATIVE],text=True).strip()
    try:
        for sub,entries in [('',object_meta['objects']),('sdk',sdk_meta)]:
            target=inputs/sub;target.mkdir(exist_ok=True)
            for name,item in entries.items():
                dest=target/name;run(['docker','cp',cid+':'+item['origin'],str(dest)])
                assert sha(dest)==item['sha256'],name
    finally:
        obj=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        assert not obj['State']['Running'] and obj['Config']['Labels']['archive.purpose']=='current-extraction'
        run(['docker','rm',cid])
    # Build the exact host/proto recipes; original full API is intentionally retained.
    shell='''set -euo pipefail
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 - <<'PY'
from pathlib import Path
import json,subprocess,ctypes,os
assert not list(Path('/dev').glob('davinci*'))
data=json.loads(Path('/recipe/build/provider-measured-manifest.json').read_text())
for name in ['libcust_opmaster_rt2.0.so','libcust_opsproto_rt2.0.so']:
    argv=data['commands'][name];Path(argv[argv.index('-o')+1]).parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(argv,check=True)
    ctypes.CDLL(argv[argv.index('-o')+1],mode=os.RTLD_NOW|os.RTLD_LOCAL)
PY
python3 /task/continuation_G57_binding_isolation_20261115/candidate/build_binding.py
'''
    binding=out/'continuation_G57_binding_isolation_20261115/candidate';shutil.copytree(CURRENT/'build/binding',binding,dirs_exist_ok=True)
    api=out/'continuation_G55_fused_moe_20261114/operator_sources/custom_transformer/op_api/lib/libcust_opapi.so'
    api.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(donor/'op_api/lib/libcust_opapi.so',api)
    assert sha(api)=='5e7735b6b84626596a5eb7ac6e4b2a028fa227ee8f267f435ca6cd5f4928fbea'
    run(['docker','run','--rm','--network','none','--runtime','runc','--cpus','4','--memory','24g','--label','archive.purpose=current-cpu-build',
         '-v',str(out)+':/task','-v',str(CURRENT)+':/recipe:ro','-e','MAX_JOBS=1','-e','TMPDIR=/task/build-tmp',
         '--entrypoint','bash',SG,'-lc','mkdir -p /task/build-tmp\n'+shell],timeout=3600)
    for item in load(CURRENT/'binary-manifest.json'):
        name=item['path'];prefix='continuation_G58_provider_isolation_20261115/provider4/custom_transformer/'
        if not name.startswith(prefix):continue
        rel=name.removeprefix(prefix)
        if '/op_tiling/' in rel:source=s/'link_build7/libcust_opmaster_rt2.0.so'
        elif rel.endswith('libcust_opsproto_rt2.0.so'):source=s/'link_build7/libcust_opsproto_rt2.0.so'
        else:source=donor/rel
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    actual=[]
    for item in load(CURRENT/'binary-manifest.json'):
        value=sha(out/item['path']);actual.append({**item,'rebuilt_sha256':value,'matches_measured':value==item['measured_sha256']})
    (out/'rebuilt-binary-hashes.json').write_text(json.dumps(actual,indent=2)+'\n')
    print('CPU build complete; measured hash differences are recorded. Target startup/functionality and new performance remain unverified.')
if __name__=='__main__':main()
