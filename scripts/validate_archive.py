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
    print('Archive integrity, source syntax, exact raw reconstruction, and metric arithmetic passed. This is not a new hardware or quality test.')
if __name__=='__main__':main()
