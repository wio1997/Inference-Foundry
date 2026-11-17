"""Offline archive checks only; does not execute any NPU workload."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    index=json.loads((ROOT/'evidence/components/provenance.json').read_text())
    for item in index:
        path=ROOT/item['artifact']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['published_sha256'],path
    receipts={label:json.loads((ROOT/f'evidence/components/{label}/receipt.json').read_text())
              for label in ['G49_sglang','G49_vllm_import_failure','G50_vllm']}
    total_cases=total_graphs=total_replays=0
    for label in ['G49_sglang','G50_vllm']:
        receipt=receipts[label];results=receipt['results']
        assert results['all8_passed'] and results['worker_exit_codes']==[0]*8
        assert len(results['reports'])==8
        assert {r['rank'] for r in results['reports']}==set(range(8))
        for report in results['reports']:
            assert report['passed'] and report['failure'] is None
            assert report['device_name']=='Ascend910B1'
            assert len(report['cases'])==10 and len(report['graphs'])==2
            assert {(c['chunk_rows'],c['tokens']) for c in report['cases']}=={
                (chunk,tokens) for chunk in [1<<22,128] for tokens in [0,1,3,16,37]}
            for case in report['cases']:
                assert case['bitwise_bf16_match']
                assert case['global_vocab_start']==384000000
                assert case['ids_stride']==[16,1]
                assert case['owned_heads']==3 and case['pad_heads']==4
            for graph in report['graphs']:
                assert graph['bitwise_bf16_match']
                assert graph['changed_host_weights_and_NPU_ids_replays']==3
                assert len(graph['output_sha256'])==len(set(graph['output_sha256']))==3
                total_replays+=3
            total_cases+=len(report['cases']);total_graphs+=len(report['graphs'])
        state=receipt['state']
        assert state['phase']=='stopped_released' and state['failure'] is None
        assert state['cleanup']['all8_process_lists_clear']
        assert not state['cleanup']['container_state']['Running']
        assert not state['cleanup']['container_state']['OOMKilled']
    failed=receipts['G49_vllm_import_failure']
    assert not failed['results']['all8_passed']
    assert failed['results']['worker_exit_codes']==[1]*8
    assert failed['state']['phase']=='failed_released'
    assert failed['state']['cleanup']['all8_process_lists_clear']
    assert any('partially initialized module' in s for s in failed['worker_logs'].values())
    a=sorted(receipts['G49_sglang']['results']['reports'],key=lambda r:r['rank'])
    b=sorted(receipts['G50_vllm']['results']['reports'],key=lambda r:r['rank'])
    for sg,vl in zip(a,b):
        assert sg['native_flags']==2 and sg['HostMalloc_header_arity']==2
        assert vl['native_flags']==268435458
        assert sg['cases']==vl['cases']
        assert sg['graphs']==vl['graphs']
    links=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:continue
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        try:text=path.read_text()
        except UnicodeError:continue
        assert not re.search(r'(?:Hzzgkj|lifang)'+r'@',text),path
        assert not re.search(r'bms-lifang-000[1-4]|/data1?/tiankuan/|/home/'+'fn-ubuntu',text),path
        if path.suffix=='.md':
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
                if link.startswith(('https://','http://','#')):continue
                assert (path.parent/link.split('#',1)[0]).is_file(),(path,link)
                links+=1
    assert (total_cases,total_graphs,total_replays)==(160,32,96)
    print(json.dumps(dict(provenance_files=len(index),cases=total_cases,graph_captures=total_graphs,
                         changed_input_replays=total_replays,all16_devices_passed=True,
                         cross_framework_output_hashes_equal=True,initial_import_failure_preserved=True,
                         relative_links=links,privacy_scan=True,full_model_or_performance_acceptance=False)))


if __name__=='__main__':main()
