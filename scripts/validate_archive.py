"""只检查归档完整性，不启动 Docker、NPU、服务或性能测试。"""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);p=ROOT/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
    for p in ROOT.rglob('*.py'):
        if '.git' not in p.parts and '__pycache__' not in p.parts:ast.parse(p.read_text(),filename=str(p))
    subprocess.run([sys.executable,str(ROOT/'scripts/unpack_data.py'),'--with-results','--verify-only'],cwd=ROOT,check=True)
    a=json.loads((ROOT/'evidence/original_acceptance/final_acceptance.json').read_text())
    assert a['formal_success']==128 and a['warmup_success']==64
    assert [x['input_tokens'] for x in a['metrics']]==[32768,65536,131072,262144]
    for x in a['metrics']:
        assert x['successful_requests']==32 and x['output_tokens']==1024 and x['configured_concurrency']==16
        assert abs(32*1024/x['duration_s']-x['aggregate_output_tokens_per_s'])<1e-4
    g=json.loads((ROOT/'evidence/g37/sglang_G37_all4_acceptance.json').read_text())
    v=json.loads((ROOT/'evidence/g37/native_vllm_baseline_acceptance.json').read_text())
    assert g['all4_SGLang_comparison_complete'] and not g['outperformance_goal_achieved']
    assert [x['input_tokens'] for x in g['profiles']]==[32768,65536,131072,262144]
    for sg,native in zip(g['profiles'],v['metrics']):
        x=sg['metrics'];assert x['input_tokens']==native['input_tokens']
        assert x['successful_requests']==32 and x['output_tokens']==1024 and x['configured_concurrency']==16
        assert sg['SQLite_timestamp_contract']['observed_peak_concurrency']==16
        assert abs(32*1024/x['duration_s']-x['aggregate_output_tokens_per_s'])<1e-4
        assert x['aggregate_output_tokens_per_s']<native['aggregate_output_tokens_per_s']
    manifest=json.loads((ROOT/'evidence/data-archive-manifest.json').read_text())
    assert sum(x.get('rows',0) for x in manifest['results'])==1056
    provenance=json.loads((ROOT/'evidence/g37/publication-provenance.json').read_text())
    for ent in provenance['files']:
        assert hashlib.sha256((ROOT/ent['path']).read_bytes()).hexdigest()==ent['published_sha256'],ent['path']
    g42=json.loads((ROOT/'evidence/g42/sglang_G42_all4_acceptance.json').read_text())
    assert g42['all4_SGLang_comparison_complete'] and not g42['outperformance_goal_achieved']
    for entry in g42['profiles']:
        x=entry['metrics']
        assert x['successful_requests']==32 and x['output_tokens']==1024 and x['configured_concurrency']==16
        assert entry['SQLite_timestamp_contract']['observed_peak_concurrency']==16
        assert abs(32768/x['duration_s']-x['aggregate_output_tokens_per_s'])<1e-4
    g43=json.loads((ROOT/'evidence/g43/sglang_G43_retry2_screen_acceptance.json').read_text())
    assert not g43['all4_SGLang_comparison_complete'] and not g43['screen_repeatability_verified']
    assert g43['metrics']['successful_requests']==32 and g43['SQLite_timestamp_contract']['observed_peak_concurrency']==16
    for ent in json.loads((ROOT/'evidence/g43/publication-provenance.json').read_text())['files']:
        assert hashlib.sha256((ROOT/ent['path']).read_bytes()).hexdigest()==ent['published_sha256'],ent['path']
    print('Archive integrity, source syntax, exact raw reconstruction, and metric arithmetic passed. This is not a new hardware or quality test.')
if __name__=='__main__':main()
