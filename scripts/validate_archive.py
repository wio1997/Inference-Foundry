"""离线核验归档；不启动模型、容器、NPU或新性能测试。"""
from pathlib import Path
import ast,csv,hashlib,ipaddress,json,lzma,math,re

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def value(v):return float(str(v).split()[0])


def main():
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);p=ROOT/name
        assert p.is_file() and digest(p)==expected,name
    for p in ROOT.rglob('*.py'):
        if '.git' not in p.parts:ast.parse(p.read_text(),filename=str(p))
    provenance={x['artifact']:x for x in json.loads((ROOT/'provenance/artifacts.json').read_text())}
    for name,e in provenance.items():assert digest(ROOT/name)==e['published_sha256'],name
    links=0
    for p in ROOT.rglob('*.md'):
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('https://','http://','mailto:','#')):continue
            name=link.split('#',1)[0];assert (p.parent/name).is_file(),(p.relative_to(ROOT),link);links+=1
    runs=json.loads((ROOT/'records/runs.json').read_text());ids=[x['id'] for x in runs];assert len(ids)==len(set(ids))
    rows_total=0;phases=0;scored=0;db_refs=0
    for entry in runs:
        m=json.loads((ROOT/entry['path']).read_text());e=m.get('raw_evidence')
        if not e:continue
        a=m['record']['metrics'];h=hashlib.sha256();rows=[]
        with lzma.open(ROOT/e['details'],'rb') as f:
            for line in f:
                h.update(line)
                if line.strip():rows.append(json.loads(line))
        assert h.hexdigest()==provenance[e['details']]['uncompressed_sha256']
        assert len(rows)==a['requests'] and all(r['success'] is True for r in rows)
        assert all(r['input_tokens']==a['input_tokens'] and r['output_tokens']==1024 for r in rows)
        assert len({r['data_id'] for r in rows})==len(rows)
        db_refs+=sum(isinstance(r.get('time_points'),dict) and '__db_ref__' in r['time_points'] for r in rows)
        common=json.loads((ROOT/e['summary']).read_text());duration=value(common['Benchmark Duration']['total'])/1000
        assert math.isclose(duration,a['duration_s'],abs_tol=1e-8)
        assert value(common['Total Requests']['total'])==value(common['Success Requests']['total'])==len(rows)
        assert value(common['Failed Requests']['total'])==0 and value(common['Max Concurrency']['total'])==16
        assert value(common['Total Input Tokens']['total'])==len(rows)*a['input_tokens']
        assert value(common['Total Generated Tokens']['total'])==len(rows)*1024
        for name,metric,tokens in [('Input Token Throughput','input_tokens_per_s',len(rows)*a['input_tokens']),('Output Token Throughput','output_tokens_per_s',len(rows)*1024),('Total Token Throughput','total_tokens_per_s',len(rows)*(a['input_tokens']+1024))]:
            got=value(common[name]['total']);assert math.isclose(got,tokens/duration,rel_tol=1e-5,abs_tol=.01)
            assert math.isclose(got,a[metric],rel_tol=1e-9,abs_tol=.0001)
        with (ROOT/e['statistics']).open(newline='') as f:table=list(csv.DictReader(f))
        for label in ['TTFT','TPOT','E2EL']:
            row=next(r for r in table if r['Performance Parameters']==label and r['Stage']=='total')
            mean=next(k for k in row if k.lower() in ['average','mean'])
            assert math.isclose(value(row[mean]),a[label.lower()+'_mean_ms'],abs_tol=1e-8)
        rows_total+=len(rows);phases+=1;scored+=bool(m['record'].get('scored'))
    component=json.loads((ROOT/'evidence/sinkhorn_component_results.json').read_text())
    assert component['passed'] and len(component['cases'])==27 and len(component['graph'])==4 and len(component['adapter_cases'])==5
    for name,expected in component['source_sha256'].items():assert digest(ROOT/'code/sinkhorn'/name)==expected,name
    model=json.loads((ROOT/'evidence/model_PD_acceptance.json').read_text());assert model['seven_PD_semantic_length_cases_passed'] and model['PD_registration_all16_return0']
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or p.suffix=='.xz':continue
        text=p.read_text(errors='replace')
        # Build byte fragments so the scanner's own source does not contain a match.
        forbidden=['/home/'+'fn-ubuntu','bms-'+'lifang-', 'lifang'+'@','Hzzgkj'+'@','/data1/'+'tiankuan','/data/'+'tiankuan']
        assert not any(x in text for x in forbidden),str(p.relative_to(ROOT))
        for addr in re.findall(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',text):
            try:ip=ipaddress.ip_address(addr)
            except ValueError:continue
            assert not ip.is_private or addr in ['0.0.0.0','127.0.0.1'],(p.relative_to(ROOT),addr)
    print(json.dumps(dict(archive_integrity=True,source_syntax=True,relative_links=links,request_rows=rows_total,validated_phases=phases,scored_phases=scored,raw_count_and_throughput_arithmetic=True,CSV_latency_values_match_original_audits=True,component_record_consistency=True,private_connection_scan=True,limitations=['离线验档不是新的硬件/质量/优化收益验收',f'{db_refs}条请求使用SQLite时序引用，原SQLite未纳入首批归档，不能仅凭此分支独立重建逐请求时序与延迟分位数']),ensure_ascii=False))


if __name__=='__main__':main()
