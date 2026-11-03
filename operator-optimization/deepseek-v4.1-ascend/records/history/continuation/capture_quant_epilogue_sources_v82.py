"""Pin only the public QuantBatchMatmulV3 entry/per-token operator files."""
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from pathlib import Path
import base64,hashlib,json
from finalize_origin_coverage_v9 import N,O,write


def get(url):
    return json.load(urlopen(Request(url,headers={'User-Agent':'OperatorEpilogueEvidence/1.0'}),timeout=15))


def main():
    assert not(O/'quant-epilogue-public-source-index-v82.json').exists()
    api='https://api.gitcode.com/api/v5/repos/cann/ops-nn'
    branch=get(api+'/branches/master');commit=branch['commit']['id']
    out=N/'public-quant-epilogue-source-v82';out.mkdir(exist_ok=True)
    files=[]
    for name in ['quant_batch_matmul_v3.cpp','quant_batch_matmul_v3_pertoken_opt.h','quant_batch_matmul_v3_pertoken_basic.h']:
        path='matmul/quant_batch_matmul_v3/op_kernel/'+name
        url=api+'/contents/'+path+'?ref='+commit;r=get(url)
        assert r['encoding']=='base64';b=base64.b64decode(r['content']);assert len(b)==r['size']
        p=out/name;p.write_bytes(b);gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if r.get('sha'):assert gitsha==r['sha']
        files.append(dict(path=path,local_path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=gitsha,
            api_blob_sha=r.get('sha'),contents_API=url,source_url=f'https://gitcode.com/cann/ops-nn/blob/{commit}/{path}',
            repo_commit=commit,binding='Exactpublicprimarycommit+blob;notinstalledCANN9.2.0operator/binary/tilingkey attestation.'))
    write(O/'quant-epilogue-public-source-index-v82.json',dict(repo='cann/ops-nn',commit=commit,commit_date=branch['commit'].get('committed_date'),
        retrieved_UTC=datetime.now(timezone.utc).isoformat(),files=files,new_public_operator_source_files=3,new_target_source_files_obtained=0,
        no_model_Torch_NPU_or_source_execution=True,no_browser_used=True,
        lookup_history=[dict(url='https://gitcode.com/cann/ops-transformer/blob/master/mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16_opt.h',result='webpagefetchtimedout;rawroute returnedHTML shell. PubliccontentsAPIworks.'),
            dict(url='https://api.gitcode.com/api/v5/repos/cann/ops-transformer/contents/mc2/3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16_opt.h?ref=master',result='AnonymouspublicAPIreadworks,butthisthirdpartyheaderhas no perToken input andcommentnotusedbyMC2;notusedtoidentifycurrentnativeDown.'),
            dict(url='https://api.github.com/repos/Ascend/ops-nn',result='404;officialGitCodeAPIusedinstead.')],
        loaded_native_branch_attested=False,whole_candidate_closed=False,overall_complete=False))
    print(dict(public_operator_source_files=3,repo_commit=commit,new_target_source_files=0,loaded_native_branch_attested=False))


if __name__=='__main__':main()
