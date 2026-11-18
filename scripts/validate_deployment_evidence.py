"""Offline deployment record checks; does not start models or touch NPUs."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    entries=json.loads((ROOT/'evidence/deployment_provenance.json').read_text())
    for item in entries:
        p=ROOT/item['artifact'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256'],p
    pinned=json.loads((ROOT/'metadata/pinned_files.json').read_text())
    files={x['Path']:x for x in pinned['files']}
    assert len(files)==95 and sum(x['Size'] for x in files.values())==528805292902
    for port,role in [(2203,'p'),(2204,'d')]:
        base=ROOT/f'evidence/full_model_G65/{port}'
        weights=json.loads((base/'verified_weight_snapshot.json').read_text())
        assert weights['phase']=='complete' and not weights['failed']
        assert set(weights['verified'])==set(files)
        for name,value in weights['verified'].items():
            assert value['sha256']==files[name]['Sha256']
            assert value['bytes']==files[name]['Size']
            assert value['revision']==files[name]['Revision']
        state=json.loads((base/'state.json').read_text())
        assert state['role']==role and state['phase']=='failed_released'
        assert state['cleanup']['all8_process_lists_clear']
        assert not state['cleanup']['container_state']['Running']
        log=(base/'server.log').read_text()
        assert 'validate_deepseek_v41_features' in log
        assert 'requires static verify, Mooncake, NIXL' in log
        argv=json.loads((base/'container_argv.json').read_text())
        assert '--read-only' in argv and any(x.endswith(':/task:ro') for x in argv)
        assert any(x.endswith(':/models/ds41:ro') for x in argv)
    for role in ['p','d']:
        c=json.loads((ROOT/f'code/full_model_candidates/sg_pd_g68/candidate_{role}.json').read_text())
        a=c['argv'];value=lambda flag:a[a.index(flag)+1]
        assert value('--tp-size')==value('--ep-size')=='8'
        assert value('--pp-size')==value('--attn-cp-size')==value('--dcp-size')=='1'
        assert value('--speculative-algorithm')=='DSPARK' and value('--speculative-num-draft-tokens')=='6'
        assert value('--kv-cache-dtype')==value('--dtype')=='bfloat16'
        assert value('--context-length')=='263296' and value('--max-running-requests')=='16'
        assert c['environment']['SGLANG_RAGGED_VERIFY_MODE']=='static'
        assert c['environment']['ASCEND_MF_TRANSFER_PROTOCOL']=='device_rdma'
        if role=='d':assert json.loads(value('--cuda-graph-config'))['decode']['backend']=='full'
    for path in (ROOT/'code/full_model_candidates').rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for port in [2203,2204]:
        base=ROOT/f'evidence/full_model_G69/{port}'
        state=json.loads((base/'state.json').read_text())
        assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
        assert not state['cleanup']['container_state']['Running']
        log=(base/'server.log').read_text()
        assert 'FileExistsError' in log and 'multiprocessing/spawn.py' in log and 'server_help.txt' in log
    entry=ast.parse((ROOT/'code/full_model_candidates/sg_pd_g70/launch_pd.py').read_text())
    assert isinstance(entry.body[-1],ast.If)
    assert ast.unparse(entry.body[-1].test)=="__name__ == '__main__'"
    assert all(isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.If)) for x in entry.body)
    regression=json.loads((ROOT/'evidence/full_model_G70/spawn_import_regression.json').read_text())
    assert regression['actual_runpy_spawn_import_name']=='__mp_main__'
    assert regression['entry_imported_without_candidate_file'] and regression['no_file_writes']
    for gen in [70,72]:
        for port in [2203,2204]:
            base=ROOT/f'evidence/full_model_G{gen}/{port}'
            state=json.loads((base/'state.json').read_text())
            assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
            assert not state['cleanup']['container_state']['Running']
            assert 'Unknown quantization method: ascend' in (base/'server.log').read_text()
    mapping=json.loads((ROOT/'evidence/target_quant_name_contract_G74.json').read_text())
    assert mapping['target_weight_descriptors']==mapping['remapped_descriptors']==46637
    assert not mapping['different_format_collisions']
    assert not mapping['native_ModelSlim_DSV4_detection_hc_head_prefix_present']
    assert mapping['known_required_native_names_resolve_after_native_remap']['model.layers.0.mlp.experts.0.gate_proj.weight']=='W4A8_DYNAMIC'
    launch=ast.parse((ROOT/'code/full_model_candidates/sg_pd_g75/launch_pd.py').read_text())
    assert any(isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__mp_main__'" for n in launch.body)
    for port in [2203,2204]: # G75
        base=ROOT/f'evidence/full_model_G75/{port}'
        state=json.loads((base/'state.json').read_text())
        assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
        assert 'Missing ModelSlim MoE quantization description' in (base/'server.log').read_text()
        receipts=list(base.glob('embedded_draft_quant.pid*.json'))
        assert len(receipts)==8
        for p in receipts:
            v=json.loads(p.read_text())
            assert v['entries']==1226 and v['dtype_counts']=={'BF16':1204,'F32':22}
            assert v['logical_checkpoint_bytes']==32428105992 and v['stages']==[0,1,2]
            assert v['effective_draft_quantization'] is None and v['target_modelslim_unchanged']
        assert len(list(base.glob('spawn_bootstrap.pid*.json')))==9
    probe=json.loads((ROOT/'evidence/native_quant_precedence_probe_G78.json').read_text())
    assert probe['actual_resolver_returned_key_names']==['hf_config','packed_modules_mapping','quant_method']
    assert probe['actual_ModelSlim_file_descriptors_read'] is False
    for port in [2203,2204]:
        base=ROOT/f'evidence/full_model_G78/{port}'
        observation=json.loads((base/'loading_observation.json').read_text())
        assert observation['actual_controller_live'] and observation['state']['phase']=='model_starting'
        for field in ['embedded_draft_quant','target_quant_input','target_quant_aliases']:
            assert len(observation[field])==8
        for value in observation['target_quant_input'].values():
            assert value['input_keys']==['quant_method','packed_modules_mapping','hf_config']
            assert value['resolved_descriptor_count']==187081
        for value in observation['target_quant_aliases'].values():
            assert value['added_aliases']==185582 and value['only_parameter_names_mapped']
        assert 'Multi-thread loading shards:' in observation['log_tail']
    for port in [2203,2204]:
        base=ROOT/f'evidence/full_model_G78_terminal/{port}'
        state=json.loads((base/'state.json').read_text())
        assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
        targets=list((base/'weight_ledgers').glob('target.*.json'))
        assert len(targets)==8
        for p in targets:
            record=json.loads(p.read_text())
            assert record['status']=='passed' and record['parameter_count']==1384 and not record['missing']
        drafts=list((base/'weight_ledgers').glob('draft.*.json'))
        assert drafts
        for p in drafts:
            record=json.loads(p.read_text())
            assert record['status']=='failed'
            assert set(record['missing'])=={'markov_head.markov_w1.weight','markov_head.markov_w2.weight'}
    probe=json.loads((ROOT/'evidence/native_Markov_remap_probe_G82.json').read_text())
    assert {v['native_generic'] for v in probe['results'].values()}=={'markov_head.markov_w1.weight','markov_head.markov_w2.weight'}
    cpu=json.loads((ROOT/'evidence/CPU_client_G83/create_receipt.json').read_text())
    assert cpu['NPU_devices']==[] and cpu['cpuset']=='184-187' and cpu['cpus']==4
    assert cpu['memory_bytes']==34359738368
    for port in [2203,2204]:
        base=ROOT/f'evidence/full_model_G82/{port}'
        state=json.loads((base/'state.json').read_text())
        assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
        assert not state['cleanup']['container_state']['Running']
        for stage,count in [('target',1384),('draft',73)]:
            ledgers=list((base/'weight_ledgers').glob(stage+'.*.json'))
            assert len(ledgers)==8
            for p in ledgers:
                value=json.loads(p.read_text())
                assert value['status']=='passed' and value['parameter_count']==count and not value['missing']
        log=(base/'server.log').read_text()
        assert '1670.75' in log and 'out of memory' in log.lower()
    pricing=json.loads((ROOT/'evidence/KV_accounting_G94/native_pricing_replay.json').read_text())
    assert pricing['bytes_per_full_token_before']==1670.75
    assert pricing['bytes_per_full_token_after_accounting']==3131.375
    assert [x['actual_single_C2_tensor_bytes_from_native_shape'] for x in pricing['actual_OOM_shape_replay']]==[3088711680,3148349440]
    source=ROOT/'code/native_KV/G86_native_planner_sources/sglang/srt/model_executor/pool_configurator.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==pricing['native_source_sha256']
    manifest=json.loads((ROOT/'evidence/CPU_client_G83/dataset_manifest.json').read_text())
    profiles=manifest['profiles'];assert len(profiles)==24
    assert manifest['output_tokens']==1024 and manifest['client_concurrency']==16
    assert not manifest['server_usage_validated']
    assert {(p['phase'],p['input_tokens']) for p in profiles}=={(phase,n) for phase in ['formal','warmup1','warmup2','warmup3','warmup4','warmup5'] for n in [32768,65536,131072,262144]}
    for p in profiles:
        assert p['min_actual']==p['max_actual']==p['input_tokens']
        assert p['requests']==(32 if p['phase']=='formal' else 16)
    audit=json.loads((ROOT/'evidence/CPU_client_G83/cold_prefix_audit.json').read_text())
    assert audit['verified'] and audit['prefix_count']==sum(p['requests'] for p in profiles)==448
    assert audit['all_phases_lengths_request_prefixes_unique'] and audit['no_service_request_submitted']
    base=ROOT/'evidence/SFA_operator_G92'
    state=json.loads((base/'state.json').read_text())
    assert state['phase']=='failed_released' and state['cleanup']['all8_process_lists_clear']
    ranks=list((base/'evidence').glob('rank*.json'));assert len(ranks)==8
    for p in ranks:
        value=json.loads(p.read_text())
        assert value['phase']=='failed' and 'sinks is required' in value['error']
        assert value['cases']==[{'ratio':128,'phase':'execute'}]
    base=ROOT/'evidence/SFA_operator_G104'
    state=json.loads((base/'state.json').read_text())
    assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
    ranks=list((base/'evidence').glob('rank*.json'));assert len(ranks)==8
    for p in ranks:
        value=json.loads(p.read_text());assert value['phase']=='contracts_measured'
        cases={(c['API'],c['ratio']):c for c in value['cases']}
        assert set(cases)=={(api,ratio) for api in ['target_custom_device','draft_builtin_host'] for ratio in [128,4,1,2]}
        for ratio in [128,4,1,2]:
            c=cases['target_custom_device',ratio]
            assert c['phase']=='metadata_rejected' and 'not in libopapi.so' in c['error']
        for ratio in [1,2]:
            c=cases['draft_builtin_host',ratio]
            assert c['phase']=='metadata_rejected' and c['cmp_topk']==512
        for ratio in [4,128]:
            c=cases['draft_builtin_host',ratio]
            assert c['phase']=='passed' and c['graph_captures']==1 and c['changed_input_replays']==3
    base=ROOT/'evidence/SFA_operator_G120'
    state=json.loads((base/'state.json').read_text())
    assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
    ranks=list((base/'evidence').glob('rank*.json'));assert len(ranks)==8
    reference=None;max_error=0.
    for p in sorted(ranks):
        value=json.loads(p.read_text());assert value['phase']=='passed' and value['device']=='Ascend910B1'
        assert {(c['ratio'],c['mode']) for c in value['cases']}=={(ratio,mode) for ratio in [1,2] for mode in ['decode','verify','prefill']}
        hashes=[]
        for c in value['cases']:
            assert c['phase']=='passed' and c['graph_captures']==1 and c['changed_input_replays']==3
            assert c['rows']=={'decode':2,'verify':12,'prefill':20}[c['mode']]
            assert len(c['replay_errors'])==len(set(c['hashes']))==3
            for e in [c['eager'],*c['replay_errors']]:
                assert e['all_finite'];max_error=max(max_error,e['max_abs_error'])
            hashes.append(c['hashes'])
        if reference is None:reference=hashes
        else:assert hashes==reference
    assert max_error==0.00031238794326782227
    base=ROOT/'evidence/SFA_operator_G134'
    state=json.loads((base/'state.json').read_text())
    assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
    ranks=list((base/'evidence').glob('rank*.json'));assert len(ranks)==8
    reference=None;max_error=0.;timings={}
    for p in sorted(ranks):
        value=json.loads(p.read_text());assert value['phase']=='passed'
        assert {(c['ratio'],c['mode']) for c in value['cases']}=={(ratio,mode) for ratio in [1,2] for mode in ['decode','verify','prefill']}
        hashes=[]
        for c in value['cases']:
            assert c['phase']=='passed' and c['rows']=={'decode':16,'verify':96,'prefill':8192}[c['mode']]
            assert len(c['FP64_checked_rows'])==(32 if c['mode']=='prefill' else c['rows'])
            assert len(set(c['FP64_checked_rows']))==len(c['FP64_checked_rows'])
            assert c['graph_captures']==1 and c['changed_input_replays']==3 and len(set(c['hashes']))==3
            assert 0<c['second_warmed_call_s']<c['two_warmup_calls_elapsed_s']
            timings.setdefault(f"C{c['ratio']}_{c['mode']}",[]).append(c['second_warmed_call_s'])
            for e in [c['eager'],*c['replay_errors']]:
                assert e['all_finite'];max_error=max(max_error,e['max_abs_error'])
            hashes.append(c['hashes'])
        if reference is None:reference=hashes
        else:assert hashes==reference
    summary=json.loads((base/'component_summary.json').read_text())
    assert summary['all8_passed'] and not summary['full_model_or_E2E_accepted']
    assert summary['cases']==summary['graph_captures']==48 and summary['changed_input_replays']==144
    assert summary['max_abs_error']==max_error==0.00024413317441940308
    assert summary['warm_component_elapsed_s_ranges']=={name:[min(values),max(values)] for name,values in timings.items()}
    large=ROOT/'evidence/SFA_operator_G160'
    if large.exists():
        summary=json.loads((large/'summary.json').read_text())
        state=json.loads((large/'state.json').read_text())
        assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
        actual_cases=[]
        for rank in range(8):
            item=json.loads((large/f'evidence/rank{rank}.json').read_text())
            assert item['rank']==rank and item['phase']=='passed'
            assert item['pool_bytes_each']==4312007680 and item['physical_row_offset']==4194561
            assert item['initialization_synchronized_before_attention'] and item['pointer_tables_prepared_before_graph']
            assert len(item['cases'])==6
            for case in item['cases']:
                assert case['phase']=='passed' and case['graph_captures']==1 and case['changed_input_replays']==3
                assert len(set(case['hashes']))==3 and case['full_shape_finiteness_checked']
                assert len(case['FP64_checked_rows'])==(32 if case['mode']=='prefill' else case['rows'])
                actual_cases.append(case)
        assert len(actual_cases)==summary['cases']==summary['graphs']==48
        assert summary['changed_input_replays']==144
        assert max(e['max_abs_error'] for c in actual_cases for e in [c['eager'],*c['replay_errors']])==summary['max_abs_error']
        failed=ROOT/'evidence/SFA_operator_G152'
        for rank in range(8):assert json.loads((failed/f'evidence/rank{rank}.json').read_text())['phase']=='failed'
    native_math=ROOT/'evidence/native_math_G174'
    if native_math.exists():
        prior_pass=0;prior_fail=0
        for rank in range(8):
            original=json.loads((ROOT/f'evidence/native_math_G166/evidence/rank{rank}.json').read_text())
            assert len(original['cases'])==13
            prior_pass+=sum(c['phase']=='passed' for c in original['cases'])
            prior_fail+=sum(c['phase']=='failed' for c in original['cases'])
            failed=json.loads((ROOT/f'evidence/native_math_G170/evidence/rank{rank}.json').read_text())
            assert failed['phase']=='failed' and not failed['cases']
            fixed=json.loads((native_math/f'evidence/rank{rank}.json').read_text())
            assert fixed['phase']=='passed' and fixed['scale_boundary_cases']==509
            assert fixed['all_scale_boundaries_bitwise_equal'] and fixed['scale_mismatch_count']==0
            assert len(fixed['cases'])==3 and {c['rows'] for c in fixed['cases']}=={16,96,8192}
            for case in fixed['cases']:
                assert case['phase']=='passed' and case['all_BF16_outputs_bitwise_equal']
                assert case['graph_captures']==1 and case['changed_input_replays']==3 and len(set(case['hashes']))==3
        assert prior_pass==80 and prior_fail==24
        state=json.loads((native_math/'state.json').read_text())
        assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
    index_pool=ROOT/'evidence/native_index_pool_G196'
    if index_pool.exists():
        for rank in range(8):
            for failed_gen in [184,188,192]:
                prior=json.loads((ROOT/f'evidence/native_index_pool_G{failed_gen}/evidence/rank{rank}.json').read_text())
                assert prior['phase']=='failed'
            item=json.loads((index_pool/f'evidence/rank{rank}.json').read_text())
            assert item['phase']=='passed' and item['bytes_per_token']==68 and item['native_pool_class']=='DeepSeekV4IndexerPool'
            assert len(item['cases'])==3 and {c['rows'] for c in item['cases']}=={16,96,8192}
            for case in item['cases']:
                assert case['phase']=='passed' and case['all_BF16_outputs_bitwise_equal_CPU_native']
                assert case['other_layer_unchanged'] and case['null_slot_unchanged']
                assert case['graph_captures']==1 and case['changed_input_replays']==3 and len(set(case['hashes']))==3
        state=json.loads((index_pool/'state.json').read_text())
        assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
    replay=json.loads((ROOT/'evidence/native_lowratio_pool_G226/selected_geometry_CPU_replay.json').read_text())
    assert replay['all8_actual_region_geometry_equal'] and not replay['live_deployed']
    assert replay['selected_bytes_per_lowratio_row']=={'KV_BF16':1024,'native_FP4_index':68}
    assert replay['lowratio_bytes_per_FULL_token_before_native_spec_inflation']==2730.0
    assert replay['8192_full_tokens_lowratio_padding_total']==563456
    assert len(replay['reports'])==8
    for row in replay['reports']:
        assert row['actual_region_bytes']==row['integer_prediction_bytes']
        original=json.loads((ROOT/f"evidence/native_lowratio_pool_G226/evidence/rank{row['rank']}.json").read_text())
        assert row['actual_region_bytes']==original['topology']['region_bytes']
    pool_base=ROOT/'evidence/native_lowratio_pool_G226'
    native_pool=json.loads((pool_base/'acceptance.json').read_text())
    assert native_pool['cases']==16 and native_pool['graph_captures']==16 and native_pool['changed_replays']==48
    assert native_pool['all8_controller_cleanup_verified'] and native_pool['all8_native_source_SHA_verified']
    for generation, expected_state in [(222,'failed_released'),(226,'stopped_released')]:
        base=ROOT/f'evidence/native_lowratio_pool_G{generation}'
        state=json.loads((base/'state.json').read_text())
        assert state['phase']==expected_state and state['cleanup']['all8_process_lists_clear']
        rows=sorted((base/'evidence').glob('rank*.json'));assert len(rows)==8
        for path in rows:
            row=json.loads(path.read_text())
            if generation==222:
                assert row['phase']=='failed' and "KeyError('compress_ratios')" in row['error']
                continue
            assert row['phase']=='passed' and len(row['native_source_sha256_verified'])==2
            topology=row['topology']
            assert topology['main_regions']==8 and topology['item_bytes']==[131072,8704,65536,65536,65536,4352,4352,4352]
            assert topology['pair_state_buffers']==3 and topology['draft_layers']==3 and topology['C2_request_ring_size']==8
            assert topology['native_source_mapping'] and topology['no_INT8_side_indexer_buffers']
            assert [case['ratio'] for case in row['cases']]==[1,2]
            for case in row['cases']:
                assert case['phase']=='passed' and case['bitwise_CPU_FP4_readback']
                assert case['graph_captures']==1 and case['changed_input_replays']==3 and case['wire_pages_checked']==[1,2,3]
    backend_base=ROOT/'evidence/native_backend_G268'
    accepted=json.loads((backend_base/'acceptance.json').read_text())
    assert accepted['cases']==48 and accepted['graph_captures']==48 and accepted['changed_input_replays']==144
    assert not accepted['request_window_exercised'] and not accepted['full_model_PD_or_performance_accepted']
    assert accepted['code_byte_identical_to_actual_executed_files'] and accepted['all8_controller_cleanup_verified']
    terminal=json.loads((backend_base/'state.json').read_text())
    assert terminal['phase']=='stopped_released' and terminal['cleanup']['all8_process_lists_clear']
    rows=sorted((backend_base/'evidence').glob('rank*.json'));assert len(rows)==8
    for path in rows:
        row=json.loads(path.read_text());assert row['phase']=='passed' and not row['request_window']
        assert row['model_config']['context']==263296 and row['model_config']['hf_model_type']=='deepseek_v41'
        assert [(c['ratio'],c['rows']) for c in row['cases']]==[(0,16),(1,16),(2,16),(0,96),(1,96),(2,96)]
        for c in row['cases']:
            assert c['phase']=='passed' and c['graph_captures']==1 and c['changed_input_replays']==3
            assert c['metadata_positions_exact'] and c['independent_FP64_all_rows'] and c['max_abs_error']<.003
    sdk=json.loads((ROOT/'evidence/fixed_image_SDK_binary_inventory.json').read_text())['inventory']
    assert sdk['memcache-hybrid']['version']==sdk['memfabric-hybrid']['version']=='1.3.0'
    print(json.dumps({'deployment_provenance_files':len(entries),'pinned_files_per_host':95,'bytes_per_host':528805292902,
                      'G65_failure_and_cleanup_retained':True,'SDK_binary_inventory_retained':True,
                      'full_model_PD_or_performance_acceptance':False}))


if __name__=='__main__':main()
