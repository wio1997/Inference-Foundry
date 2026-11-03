"""Primary operator/API evidence for the BF16-output owner question only."""
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import base64, hashlib, json
from capture_A3_datapath_evidence_v80 import Extract
from finalize_origin_coverage_v9 import N, O, read, write


def fetch(url):
    return urlopen(Request(url, headers={'User-Agent': 'OperatorEpilogueEvidence/1.0'}), timeout=15).read()


def main():
    output = O/'quant-epilogue-primary-details-v82.json'
    assert not output.exists()
    index = read(O/'quant-epilogue-public-source-index-v82.json')
    commit = index['commit']
    files = []
    for name in ['quant_batch_matmul_v3_pertoken.h', 'quant_batch_matmul_v3_basic_epilogue.h']:
        path = 'matmul/quant_batch_matmul_v3/op_kernel/'+name
        api = 'https://api.gitcode.com/api/v5/repos/cann/ops-nn/contents/'+path+'?ref='+commit
        record = json.loads(fetch(api))
        assert record['encoding'] == 'base64'
        blob = base64.b64decode(record['content'])
        assert len(blob) == record['size']
        gitsha = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        assert gitsha == record['sha']
        local = N/'public-quant-epilogue-source-v82'/name
        local.write_bytes(blob)
        files.append(dict(path=path, local_path=str(local), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
            git_blob_sha1=gitsha, contents_API=api, repo_commit=commit,
            source_url='https://gitcode.com/cann/ops-nn/blob/'+commit+'/'+path))
    docs = []
    for name, url in [
        ('SetAtomicAdd', 'https://asc.gitcode.com/api/SIMD-API/basic_api/atomic_operations/SetAtomicAdd.html'),
        ('FixpipeL0CtoGM', 'https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.html')]:
        blob = fetch(url)
        local = O/(name+'-primary-source-v82.html')
        local.write_bytes(blob)
        parser = Extract(); parser.feed(blob.decode())
        text_path = O/(name+'-primary-text-v82.txt')
        text_path.write_text('\n'.join(parser.parts))
        docs.append(dict(id=name, url=url, snapshot_path=str(local), snapshot_bytes=len(blob),
            snapshot_sha256=hashlib.sha256(blob).hexdigest(), text_path=str(text_path),
            version='Current public primary API preview, not installed CANN9.2.0 binary attestation.'))
    write(output, dict(repo_commit=commit, files=files, primary_documents=docs,
        retrieved_UTC=datetime.now(timezone.utc).isoformat(), additional_public_operator_files=2,
        total_public_operator_files=5, new_target_source_files=0, loaded_native_branch_attested=False,
        no_browser_used=True, no_remote_server_model_Torch_NPU_work=True, overall_complete=False))
    print(dict(public_operator_files=5, primary_API_pages=len(docs), native_branch_attested=False))


if __name__ == '__main__': main()
