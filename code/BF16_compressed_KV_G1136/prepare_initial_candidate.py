from pathlib import Path
import ast,hashlib,json,shutil
T=Path('.');old=T/'eager_attention_probe_preparation_g1098';new=T/'BF16_compressed_KV_preparation_g1132';assert not new.exists();shutil.copytree(old,new)
for p in new.rglob('*'):
 if p.is_file() and p.name!='deployment_source_sha256.json':
  s=p.read_text();s=s.replace('sg_pd_g1098','sg_pd_g1132').replace('g1098','g1132').replace('generation=1098','generation=1132');p.write_text(s)
p=new/'npu_backend.py';s=p.read_text();before="""    sparse.fake_quant_compressed_kv = lambda x: (
        fake_quant_compressed_kv(x) if x.device.type == 'npu' else prior_quant(x))"""
after="""    # A2 stock-native032 stores post-RoPE compressed values as BF16.
    # Change only attention compressed-KV rounding; FP4 index keys/queries,
    # W4A8 checkpoint, pooling, RoPE and BF16 physical cache stay unchanged.
    seen = False
    def compressed_kv_bf16(x):
        nonlocal seen
        if x.device.type != 'npu':
            return prior_quant(x)
        assert x.dtype == torch.bfloat16 and x.shape[-1] == 512
        if not seen:
            import os
            receipt = {'pid': os.getpid(), 'policy': 'A2_BF16_compressed_KV_no_fake_FP4',
                       'shape': list(x.shape), 'dtype': str(x.dtype), 'device': str(x.device),
                       'identity_return_same_storage': True, 'FP4_index_quantization_unchanged': True,
                       'W4A8_checkpoint_unchanged': True,
                       'source_SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'scope': 'First actual compressed-KV dispatch; quality/performance not yet accepted'}
            (root / ('compressed_KV_BF16_first_call.pid' + str(os.getpid()) + '.json')).write_text(
                json.dumps(receipt, indent=2) + '\\n')
            seen = True
        return x
    sparse.fake_quant_compressed_kv = compressed_kv_bf16"""
assert s.count(before)==1;s=s.replace(before,after);ast.parse(s);p.write_text(s)
# Scope labels only; argv/env differ from G1098 solely by task output paths.
for name in ['candidate.json','candidate_d.json','candidate_p.json']:
 p=new/name;d=json.loads(p.read_text());d['scope']='DIAGNOSTIC_CANDIDATE: stocknative032 A2-aligned BF16 compressed attention KV; FP4 index and W4A8 weights unchanged. standaloneNoSP/eager/C1 probe first; not final PD alignment.';d['diagnostic_change']='Only compressed attention KV fake-FP4 removed on NPU; no FP4 index change. Other changes are output epoch paths.';d['required_readiness']=['Actualcontroller/container/sourceidentity','All8strict1384targetloads','NoSP/eager/prefixON/TP8EP8','PhysicalFP32C2ring8/pricing unchanged','Actualall8first-call compressedKVpolicy receipts','Same16byte-identicalrequestcontrolledcomparison','Full1319quality stillrequired beforebenchmark'];d['compressed_KV_precision_candidate']={'baseline':'G1098 BF16 physical cache after per16 E4M3scale/FP4E2M1rounding','candidate':'BF16 post-RoPE identity before same BF16 cache store','fixed_native_reference':'G1131 stocknative032 attention/dsa_v41.py _prepare_compressed_source writes postRoPE latent via scatter_cache_sk; no packed cache on A2','status':'PREPARED_NOT_DEPLOYED_OR_ACCEPTED','hypothesis':'Additional compressedKV rounding contributes to semantic gap; candidate keeps input/weights/indexquant/otherops fixed','reject_if':'same16semantic gap persists; do not call fixed/fullqualitypassed'};p.write_text(json.dumps(d,indent=2)+'\n')
p=new/'candidate_status.json';d=json.loads(p.read_text());d['G1132_compressed_KV']='PREPARED_NOT_DEPLOYED_OR_ACCEPTED';p.write_text(json.dumps(d,indent=2)+'\n')
files={str(p.relative_to(new)):hashlib.sha256(p.read_bytes()).hexdigest() for p in new.rglob('*') if p.is_file() and p.name!='deployment_source_sha256.json'};(new/'deployment_source_sha256.json').write_text(json.dumps(files,indent=2)+'\n')
changes=[n for n in files if (old/n).read_bytes()!=(new/n).read_bytes()]
for p in new.rglob('*.py'):ast.parse(p.read_text())
a=json.loads((old/'candidate.json').read_text());b=json.loads((new/'candidate.json').read_text());assert a['argv']==b['argv'];assert {k:v.replace('sg_pd_g1098','sg_pd_g1132') for k,v in a['environment'].items()}==b['environment']
assert (old/'fp4_index_pool.py').read_bytes()==(new/'fp4_index_pool.py').read_bytes();assert (old/'lowratio_pool.py').read_bytes()==(new/'lowratio_pool.py').read_bytes()
(T/'G1132_candidate_preparation_review.json').write_text(json.dumps({'files':files,'changed_files':changes,'argv_identical':True,'environment_only_output_paths_changed':True,'FP4_index_and_BF16_pool_byteidentical':True,'runtime_math_intervention':'NPU compressed attention KV fake-FP4 to BF16identity, after nativeRoPE; no weightquant/FP4index change','native_source_reference':'G1131_native_attention_sources/attention/dsa_v41.py:550-618','status':'PREPARED_NOT_DEPLOYED_NOT_ACCEPTED'},indent=2)+'\n');print('files',len(files),'changes',changes)
