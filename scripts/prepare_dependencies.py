"""在 aarch64 目标机 CPU 容器中重建依赖；不映射 NPU，不启动模型。默认仅打印计划。"""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,uuid
ROOT=Path(__file__).resolve().parents[1]
IMAGE='quay.io/ascend/sglang:cann9.2.0-a3-20261111'
DONOR='quay.io/ascend/vllm-ascend:v0.30.0rc2-a3'
IID='sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517'
DID='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    plan=[
      'Extract exact custom_transformer vendor from pinned donor image without starting its container',
      'Filter 20 payloads, rename isolated opapi SONAME, preserve original device/AICPU payloads',
      'Build host registration subset with CANN9.2 ABI; assemble strict vendor and audit 17 preserved payloads',
      'Build three Torch2.13 adapters (workspace-pool, core5 RoPE, router) on target aarch64 CPU',
    ]
    if not a.execute:print(json.dumps({'plan':plan,'execute':False},ensure_ascii=False,indent=2));return
    def call(argv,**kw):return subprocess.run(argv,check=True,cwd=ROOT,**kw)
    for tag,expected in [(IMAGE,IID),(DONOR,DID)]:
        q=subprocess.run(['docker','image','inspect',tag],capture_output=True,text=True,cwd=ROOT)
        if q.returncode:call(['docker','pull','--platform','linux/arm64',tag])
        d=json.loads(subprocess.check_output(['docker','image','inspect',tag],text=True))[0]
        assert d['Architecture']=='arm64' and d['Id']==expected,'Tag changed; obtain exact archived image, do not silently substitute'
    dest=ROOT/'compat/donor_vendor/vendors/custom_transformer'
    assert not dest.exists(),'Existing donor needs review; no automatic overwrite'
    dest.mkdir(parents=True)
    name='sglang-archive-extract-'+uuid.uuid4().hex[:8]
    cid=subprocess.check_output(['docker','create','--name',name,'--network','none','--label','archive.branch=sglang-dsv4.1flash1017-910c','--entrypoint','true',DID],text=True).strip()
    try:
        call(['docker','cp',cid+':/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/.',str(dest)])
    finally:
        d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert not d['State']['Running'] and d['Config']['Labels']['archive.branch']=='sglang-dsv4.1flash1017-910c'
        call(['docker','rm',cid])
    (ROOT/'compat/host_subset_build_g5').mkdir(exist_ok=True)
    build_shell='''set -euo pipefail
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cd /task
python3 compat/cache/vendor_filter.py --inventory compat/reference/donor_candidate/vendor_files.json --registration-root compat/reference/donor_candidate/registration --source-root compat/donor_vendor/vendors/custom_transformer --output-root compat/vendor_filter_a3_four_ops --rewrite-opapi-soname
python3 compat/host_subset_recipe_g5/build_host.py --source-root /task/compat/reference/donor_candidate --cann-root /usr/local/Ascend/cann-9.2.0 --build-dir /task/compat/host_subset_build_g5/build --jobs 2
python3 - <<'PY'
import hashlib,subprocess
from pathlib import Path
b=Path('/task/compat/host_subset_build_g5/build')
sha=lambda n:hashlib.sha256((b/n).read_bytes()).hexdigest()
subprocess.run(['python3','compat/host_subset_assembly_g4/vendor_assembly_g4.py','--filter-root','/task/compat/vendor_filter_a3_four_ops','--build-dir',str(b),'--source-root','/task/compat/reference/donor_candidate','--recipe-root','/task/compat/host_subset_recipe_g5','--proto-sha256',sha('libcust_opsproto_rt2.0.so'),'--tiling-sha256',sha('libcust_opmaster_rt2.0.so'),'--output-root','/task/compat/vendor_strict_host_g2'],check=True)
PY
'''
    builds=[('compat/engram/attention/candidate_extension_g8','compat/candidate_build_g8'),
            ('compat/engram/attention/rope/g3_core5/candidate_extension','compat/rope_build_core5_g1'),
            ('compat/cache/router/candidate_extension_g2','compat/router_build_current_g2')]
    for source,out in builds:
        build_shell+='cd /task/'+source+'\npython3 setup.py build_ext --build-lib /task/'+out+'/lib --build-temp /task/'+out+'/build\n'
    name='sglang-archive-build-'+uuid.uuid4().hex[:8]
    command=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','24g','--label','archive.branch=sglang-dsv4.1flash1017-910c','-v',str(ROOT)+':/task','-e','MAX_JOBS=1','-e','TMPDIR=/task/.work/tmp','-e','SGLANG_V41_CANN_ROOT=/usr/local/Ascend/cann-9.2.0','--entrypoint','bash',IID,'-lc',build_shell]
    (ROOT/'.work/tmp').mkdir(parents=True,exist_ok=True)
    call(command,timeout=3600)
    actual={}
    for _,out in builds:
        libs=list((ROOT/out/'lib').glob('*.so'));assert len(libs)==1
        actual[str(libs[0].relative_to(ROOT))]=hashlib.sha256(libs[0].read_bytes()).hexdigest()
    (ROOT/'.work/rebuilt-extension-hashes.json').write_text(json.dumps(actual,indent=2)+'\n')
    print('CPU build complete; rebuilt binaries require target functional validation before making new performance claims.')
if __name__=='__main__':main()
