"""离线核验归档；不启动模型、容器、NPU或新性能测试。"""
from pathlib import Path
import ast,csv,hashlib,ipaddress,json,lzma,math,re
import sys
sys.dont_write_bytecode=True
from timing_witness import phase_witness

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
    sg_profile=ROOT/'evidence/matched_SG_G181/profile_G240'
    acceptance=json.loads((sg_profile/'G258_matched_SG_16rank_profile_acceptance.json').read_text())
    assert acceptance['profile_dataset_SHA256']=='6e1da60b597d8fb978cc9c7549c720a9de2a8f0b9af956fc73bd9e110f4eaae8'
    for role in ['p','d']:
        base=sg_profile/f'CPU_{role}'/'review_g252'
        row=json.loads((base/'role.json').read_text())
        assert row['role_complete'] and row['raw_unchanged_after_analysis'] and len(row['reports'])==8
        stable=json.loads((base/'raw_stability.json').read_text())
        assert stable['two_SHA_scans_equal'] and len(stable['files'])>1000
        for rank in range(8):
            report=json.loads((base/f'rank_{rank}.json').read_text())
            assert report['physical_npu']==rank and report['device_activity_verified']
            assert report['hardware_X_events']>0 and report['explicit_model_X_events']>0 and report['cpu_X_events']>0
    provenance={x['artifact']:x for x in json.loads((ROOT/'provenance/artifacts.json').read_text())}
    for name,e in provenance.items():assert digest(ROOT/name)==e['published_sha256'],name
    links=0
    for p in ROOT.rglob('*.md'):
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('https://','http://','mailto:','#')):continue
            name=link.split('#',1)[0];assert (p.parent/name).is_file(),(p.relative_to(ROOT),link);links+=1
    runs=json.loads((ROOT/'records/runs.json').read_text());ids=[x['id'] for x in runs];assert len(ids)==len(set(ids))
    correction=json.loads((ROOT/'evidence/matched_SG_G316/128K/compile_root_correction/G487_G316_D_compile_acceptance_correction.json').read_text())
    assert correction['full_no_compilation_protocol_accepted'] is False
    assert correction['active_D_compilation_no_changes']=='UNVERIFIED'
    proof=json.loads((ROOT/'evidence/matched_SG_G316/128K/compile_root_correction/G483_actual_D_cache_2202_parsed.json').read_text())
    assert all(not item['same_path'] for item in proof['root_aliases'].values())
    for ident in ['GLM-RUN-0088','GLM-RUN-0089']:
        record=json.loads((ROOT/f'records/points/GLM-OPT-0001/runs/{ident}/manifest.json').read_text())['record']
        assert record['scored'] is True
        assert record['measurement_qualification']['full_no_compilation_protocol_accepted'] is False
        assert record['measurement_qualification']['active_D_compilation_no_changes']=='UNVERIFIED'
    rows_total=0;phases=0;scored=0;db_refs=0;latency_values=0
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
        request_witness,latency_stats=phase_witness(ROOT,m)
        assert len(request_witness)==len(rows)
        for label in ['TTFT','TPOT','E2EL']:
            row=next(r for r in table if r['Performance Parameters']==label and r['Stage']=='total')
            mean=next(k for k in row if k.lower() in ['average','mean'])
            assert math.isclose(value(row[mean]),a[label.lower()+'_mean_ms'],abs_tol=1e-8)
            for stat,computed in latency_stats[label].items():
                assert math.isclose(value(row[stat]),computed,abs_tol=.051),(entry['id'],label,stat,value(row[stat]),computed)
                latency_values+=1
        rows_total+=len(rows);phases+=1;scored+=bool(m['record'].get('scored'))
    component=json.loads((ROOT/'evidence/sinkhorn_component_results.json').read_text())
    assert component['passed'] and len(component['cases'])==27 and len(component['graph'])==4 and len(component['adapter_cases'])==5
    current_profile=ROOT/'evidence/G4_profile_G118'
    if current_profile.exists():
        p=json.loads((current_profile/'CPU_p/review/role.json').read_text())
        d=json.loads((current_profile/'CPU_d/review/role.json').read_text())
        assert p['role_complete'] and len(p['reports'])==8 and not p['failures']
        assert not d['role_complete'] and not d['reports'] and len(d['failures'])==8
        for rank in range(8):
            report=json.loads((current_profile/f'CPU_p/review/rank_{rank}.json').read_text())
            assert report['device_activity_verified'] and report['explicit_positive_model_device_events']
            assert report['analysis']['hardware_X_events']>0
        audit=json.loads((current_profile/'CPU_d/D_trace_scope_audit.json').read_text())
        assert len(audit)==8 and {x['rank'] for x in audit}==set(range(8))
        assert all(x['positive_model_events']==0 for x in audit)
        for role in ['p','d']:
            assert json.loads((current_profile/f'control/{role}_stop.json').read_text())['status']==200
        control=json.loads((current_profile/'control/state.json').read_text())
        assert control['phase']=='terminal' and control['workload_rc']==0 and not control['stop_errors'] and not control['error']
    repaired_profile=ROOT/'evidence/G4_profile_G138'
    if repaired_profile.exists():
        acceptance=json.loads((repaired_profile/'acceptance.json').read_text())
        assert acceptance['same_capture_epoch']=='G138' and acceptance['analysis_epoch']=='G148'
        assert acceptance['all16_actual_model_device_events']
        for role in ['p','d']:
            base=repaired_profile/f'CPU_{role}/review'
            report=json.loads((base/'role.json').read_text())
            assert report['role_complete'] and len(report['reports'])==8 and not report['failures']
            for rank in range(8):
                item=json.loads((base/f'rank_{rank}.json').read_text())
                assert item['rank']==rank and item['role']==role and item['device_activity_verified']
                assert item['explicit_positive_model_device_events'] and item['analysis']['hardware_X_events']>0
                identity=item['identity']['worker_mapping']
                assert {w['physical_npu'] for w in identity}==set(range(8))
    matched=ROOT/'evidence/matched_SG_G181'
    if matched.exists():
        for role in ['p','d']:
            base=matched/role;ready=json.loads((base/'readiness.json').read_text())
            assert ready['native_dispatch_ledger_count']==16 and ready['PD_registration_count']==8
            assert ready['container']['State']['Running']
            summaries=list(base.glob('final_*/weight_ledgers/*.summary.json'));assert len(summaries)==16
            for p in summaries:
                item=json.loads(p.read_text());assert item['status']=='dispatch_coverage_complete'
                assert item['missing_parameter_count']==item['unexpected_count']==0
            registrations=list(base.glob('final_*/weight_ledgers/PD_registration_g26.pid*.json'));assert len(registrations)==8
            for p in registrations:
                item=json.loads(p.read_text());assert item['returncode']==0 and item['registered_entries']>0
            args=ready['API']['/server_info']['body'];args=args.get('server_args',args)
            assert args['tp_size']==args['ep_size']==8 and args['pp_size']==1 and args['context_length']==263296
            assert args['max_running_requests']==16 and not args['disable_overlap_schedule'] and not args['disable_radix_cache']
            assert args['speculative_num_steps']==3 and args['speculative_num_draft_tokens']==4
            if role=='d':
                assert args['cuda_graph_config']['decode']['backend']=='full'
                assert len([l for l in ready['graph_log_evidence'] if 'Capture target verify NPU graph end.' in l])==8
                assert any('Capture draft decode CUDA graph end.' in l for l in ready['graph_log_evidence'])
                assert any('Capture draft extend CUDA graph end.' in l for l in ready['graph_log_evidence'])
        result=json.loads((matched/'p/functional_PD_g204/result.json').read_text())
        assert result['passed'] and result['original_G182_failed'] and len(result['original_seven'])==7
        assert result['original_failure']['error']['code']=='server_selection_failed'
        for item in result['original_seven']:
            assert item['audit']['passed']
            name='evidence/matched_SG_G181/p/functional_PD_g182/'+Path(item['path']).name
            assert provenance[name]['original_sha256']==item['original_sha256']
        usage=result['remaining_new_prefix']['usage']
        assert usage['prompt_tokens']==32768 and usage['completion_tokens']==32 and usage['num_retractions']==0
        assert usage['spec_verify_ct']>0 and usage['spec_accept_length']>=1
    for name,expected in component['source_sha256'].items():assert digest(ROOT/'code/sinkhorn'/name)==expected,name
    model=json.loads((ROOT/'evidence/model_PD_acceptance.json').read_text());assert model['seven_PD_semantic_length_cases_passed'] and model['PD_registration_all16_return0']
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or p.suffix in ['.xz','.db','.pyc']:continue
        text=p.read_text(errors='replace')
        # Build byte fragments so the scanner's own source does not contain a match.
        forbidden=['/home/'+'fn-ubuntu','bms-'+'lifang-', 'lifang'+'@','Hzzgkj'+'@','/data1/'+'tiankuan','/data/'+'tiankuan']
        assert not any(x in text for x in forbidden),str(p.relative_to(ROOT))
        for addr in re.findall(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',text):
            try:ip=ipaddress.ip_address(addr)
            except ValueError:continue
            assert not ip.is_private or addr in ['0.0.0.0','127.0.0.1'],(p.relative_to(ROOT),addr)
    print(json.dumps(dict(archive_integrity=True,source_syntax=True,relative_links=links,request_rows=rows_total,validated_phases=phases,scored_phases=scored,raw_count_and_throughput_arithmetic=True,CSV_latency_values_match_original_audits=True,per_request_latencies_rebuilt_from_original_SQLite=True,latency_statistics_checked=latency_values,component_record_consistency=True,private_connection_scan=True,limitations=['离线验档不是新的硬件/质量/优化收益验收','保留AISBench原始采样时钟与流式批量事件口径；事件数不等于生成token数']),ensure_ascii=False))


if __name__=='__main__':main()
