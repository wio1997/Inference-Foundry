"""Independent native source/metadata/interval checks;no accelerator execution."""
from pathlib import Path
import argparse,ast,hashlib,json,struct,subprocess,sys,textwrap
from finalize_origin_coverage_v9 import N,O,read,write
from decode_memset_metadata_v111 import decode_record,inline_flowtable

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'pad-provider-inputs-v111.json')['inputs']:ck('input:'+r['path'],pinned(r))
    src=read(O/'pad-provider-source-contract-v111.json');functions={}
    for item in src['source']:
        ck('source:'+item['path'],pinned(item));t=Path(item['path']).read_text()
        for s in item['parts']:
            ck('section:'+s['name'],t.splitlines()[s['first_line']-1:s['last_line']]==s['source']);functions[(Path(item['path']).name,s['name'])]='\n'.join(s['source'])
    pad=Path(src['source'][1]['path']).read_text()
    ck('barrier_private_workspace_source',pad.count('block_barrier(self.sync_workspace)')==1 and "(self.core_nums * 4,)"in pad and 'is_atomic_add=True'in pad)
    ck('outer_non4_guard','self.tiling_key != Constant.MODE4'in functions[('pad_v3.py','pad_v3_d_compute_tiling')])
    ck('original_mask_valueTrue','value=True'in functions[('sequence_parallel.py','_ascend_sp_padding_mask_impl')])
    meta=read(O/'pad-provider-metadata-contract-v111.json');ck('16_variants',sorted(r['workspace_num']for r in meta['MemSet_variants'])==list(range(1,17)))
    for m in meta['MemSet_variants']:ck('memset_meta:'+str(m['workspace_num']),m['core_num']==48 and m['parameters']==m['workspace_num']+2 and m['opParaSize']==96*m['workspace_num']+8 and m['dfx_size']==8)
    for m in meta['PadV3_variants']:ck('pad_meta:'+m['kernelName'],m['workspace']==dict(num=1,size=[1536],type=[0])and m['parameters'][4]==dict(dtype='float32',init_value=0.0)and m['compiled_core_num']==48 and m['coreType']=='MIX')
    for r in meta['binary_controls']['operators']:ck('binary:'+r['binary_path'],r['binary_sha256']==r['metadata_embedded_binary_sha256'])
    native=read(O/'pad-provider-native-reconciliation-v111.json');ck('1272_native7',len(native['regimes'])==7 and native['executions']==1272 and native['pending_regimes_promoted']==0)
    ck('12_shape_cohorts',len(native['typed_shape_cohorts'])==12 and sum(r['executions']for r in native['typed_shape_cohorts'])==1272)
    model=read(O/'MemSet-default-zero-metadata-model-v111.json');layout=model['layout']
    ck('source_record_size',sum(r['bytes']for r in layout)==model['record_bytes']==96)
    # Recover storage offsets directly from the source's running-index assignments.
    f=ast.parse(textwrap.dedent(functions[('mem_set.py','get_tiling_data')])).body[0];env={};recovered=[]
    def evaluate(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return env[n.id]
        if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Add):return evaluate(n.left)+evaluate(n.right)
        raise ValueError('unsupported source index expression')
    for n in f.body:
        if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ['input_scalar_index','input_scalar_index_int64']:env[n.targets[0].id]=evaluate(n.value)
        elif isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='set_as':
            arg=n.value.args[0]
            if isinstance(arg,ast.Subscript)and isinstance(arg.value,ast.Name)and arg.value.id in ['tiling_ub','tiling_ub_int64']:
                recovered.append((8 if arg.value.id.endswith('int64')else 4)*evaluate(arg.slice))
    ck('18_source_derived_offsets',recovered==[r['byte_offset']for r in layout]and len(recovered)==18)
    examples={r['name']:r for r in model['examples']}
    for name,r in examples.items():ck('decoded:'+name,decode_record(bytes.fromhex(r['raw_hex']))==r['record'])
    ck('one_vs48writers_samebytes',examples['one_writer_1536']['padded_tail']['submitted_store_value_bytes']==examples['48writers_same1536']['padded_tail']['submitted_store_value_bytes']==1536 and len(examples['one_writer_1536']['padded_tail']['spans'])==1 and len(examples['48writers_same1536']['padded_tail']['spans'])==48)
    ck('tail_support_matters',examples['padded_tail_100']['padded_tail']['union']==[[0,100]]and examples['padded_tail_100']['aligned_tail']['union']==[[0,128]])
    ck('multi_chunk_union',examples['multiple_full_chunks']['padded_tail']['union']==[[0,195968]]and examples['multiple_full_chunks']['padded_tail']['submitted_store_value_bytes']==195968)
    ck('empty_store',examples['empty']['padded_tail']['submitted_store_value_bytes']==0)
    one=next(r for r in meta['MemSet_variants']if r['workspace_num']==1);packet=model['inline_packet_reference'];metadata=read(Path(one['path']))
    got=inline_flowtable(packet['words'],packet['args_address'],packet['args_size'],metadata);ck('bounded_CPU_packet',got==packet['extracted']and not got['device_memory_dereferenced'])
    try:inline_flowtable(packet['words'],packet['args_address'],packet['args_size']+8,metadata);raise AssertionError('must reject unknown packing')
    except ValueError:ck('reject_unbound_packing',True)
    bad=bytearray.fromhex(examples['one_writer_1536']['raw_hex']);bad[88:92]=struct.pack('<i',1)
    try:decode_record(bad);raise AssertionError('must reject otherdtype')
    except ValueError:ck('reject_other_dtype',True)
    plan=read(O/'native-pad-probe-plan-v111.json');probe=Path(plan['probe']['path']);ck('probe_hash',pinned(plan['probe']))
    desc=json.loads(subprocess.check_output([sys.executable,str(probe)],text=True));ck('probe_CPU_default',desc==plan['description']and desc['status']=='PREPARED_NOT_EXECUTED'and desc['own_application_input_output_bytes']==44 and desc['controlled_operation']['value']is True)
    text=probe.read_text();ck('native_op_not_memcpy_probe','torch.ops.aten.constant_pad_nd(bool_input,[0,4],True)'in text and 'graph.replay('not in text and 'ctypes.string_at'not in text)
    ck('no_top_accelerator_imports',all('torch'not in ast.unparse(n)for n in ast.parse(text).body if isinstance(n,(ast.Import,ast.ImportFrom))))
    ck('source_connection_closed',not(N/'ssh166-pad-v111.sock').exists())
    if a.preflight:manifest=read(O/'checkpoint-manifest-v110.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v111.json');ck('PROGRESSnotcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        bm=read(O/'module-phase-branch-matrix-v111.json');ck('353rows26modules',len(bm['modules'])==26 and sum(len(m['branches'])for m in bm['modules'])==353)
        old=read(O/'operator-analysis-coverage-ledger-v110.json');new=read(O/'operator-analysis-coverage-ledger-v111.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        cov=read(O/'candidate-relation-coverage-v111.json');ck('132_25_463',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(132,25,463))
        for r in read(O/'candidate-record-index-v111.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v111.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,MemSet_metadata_variants=16,PadV3_metadata_variants=8,binaries_verified=3,
        source_layout_fields=18,metadata_examples=5,original_model_regimes_promoted=0,new_device_work=False,overall_complete=False)
    if a.preflight:write(O/'pad-provider-preflight-v111.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
