"""Prepare only; never launches a local or remote NPU process."""
from pathlib import Path
import ast, datetime, hashlib, json

T=Path(__file__).resolve().parent
OUT=T/'sg_pair_targetonly_ring8_preparation_g1038'
assert not OUT.exists()
contract=json.loads((T/'G1033_pair_ring8_CPU_contract_2203_parsed.json').read_text())
for value in contract['results'].values():
    assert value['returncode']==0 and value['contract']['status']=='CPU_CONTRACT_PASSED'
    assert value['contract']['NPU_initialized'] is False
helper=T/'targetonly_pair_ring8_candidate_g1033/targetonly_ring8.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==contract['paired_helper_SHA256']
sources={'p':T/'sg_pd_g832/p','d':T/'sg_D_targetonly_ring8_preparation_g1020'}
flags=['--speculative-algorithm','--speculative-draft-model-path','--speculative-num-steps',
       '--speculative-num-draft-tokens','--speculative-attention-mode','--speculative-moe-a2a-backend']
manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status':'PREPARED_NOT_DEPLOYED','epoch':1038,'roles':{},
          'scope':'PairednoSP/ring8/prefixdiagnostic only; nativePDvalidator unchanged. Notfinalalignment.',
          'CPU_contract':'G1033_pair_ring8_CPU_contract_2203_parsed.json',
          'before_deployment':['Actualfailed1024clientended/D1020released',
                               'P832idle/load0 + ownedrouter lifecycle audit',
                               'Releaseguard1026 thencontrolledownedP832stop/release',
                               'All16deviceownersreleased before newpair claims'],
          'before_generation':['Actualbothnewcontroller/container identities',
                               'Eighttargetstrictloads eachrole',
                               'D8targetfullgraphs',
                               'ActualbothspecNone/prefixON',
                               'Actualring8allocation/pricing/wire + equal nativePDlayout',
                               'Native readyrouter/noRetry; freshpairguard canonicalFD',
                               'SameC16workerAST andinputs; nootherNPUjob'],
          'remaining':'OriginalDSPARKsemanticfix/full1319quality/4SGAISBenchtiers/matchedprofiles/map/E2E'}
for role,src in sources.items():
    old='sg_pd_g832' if role=='p' else 'sg_pd_g1020'
    dst=OUT/role;dst.mkdir(parents=True)
    baseline={}
    for file in src.rglob('*'):
        if not file.is_file():continue
        rel=str(file.relative_to(src));raw=file.read_bytes();baseline[rel]=hashlib.sha256(raw).hexdigest()
        data=raw.decode().replace(old,'sg_pd_g1038')
        if file.name=='node_controller.py':
            data=data.replace("'-g832'","'-g1038'").replace("'-g1020'","'-g1038'")
            data=data.replace('generation=832','generation=1038').replace('generation=1020','generation=1038')
        if rel=='ds41_native_gate/__init__.py' and 'from targetonly_ring8 import' not in data:
            anchor="MODEL='/models/ds41'\n"
            assert data.count(anchor)==1
            data=data.replace(anchor,anchor+'from targetonly_ring8 import install as install_targetonly_ring8\ninstall_targetonly_ring8(ROOT)\n')
        target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(data)
    (dst/'targetonly_ring8.py').write_bytes(helper.read_bytes())
    for name in ['native_pool_sha256.json','KV_native_source_sha256.json']:
        if not (dst/name).exists():
            (dst/name).write_bytes((T/'targetonly_pair_ring8_candidate_g1033'/name).read_bytes())
    cfg=json.loads((dst/'candidate.json').read_text());argv=cfg['argv']
    for flag in flags:
        if flag in argv:
            i=argv.index(flag);del argv[i:i+2]
    assert '--disable-radix-cache' not in argv
    if role=='d':assert '--disaggregation-decode-enable-radix-cache' in argv
    cfg['scope']='PREPARED_NOT_DEPLOYED: pairednoSP/ring8/prefixdiagnostic1038; notfinalDSPARKalignment.'
    cfg['required_readiness']=manifest['before_generation']
    cfg['diagnostic_change']='BothP/DnoSP; physicalC2ring8/nativepricing; nativePDvalidator unchanged.'
    (dst/'candidate.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for aux in ['candidate_p.json','candidate_d.json']:
        if (dst/aux).exists():
            auxrole=aux.split('_')[1][0]
            other=json.loads((T/'targetonly_pair_ring8_candidate_g1033'/aux).read_text())
            other['scope']=cfg['scope']
            other['environment']={k:v.replace(old,'sg_pd_g1038') for k,v in other['environment'].items()}
            (dst/aux).write_text(json.dumps(other,indent=2)+'\n')
    pins={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*')
          if p.is_file() and p.name!='deployment_source_sha256.json'}
    (dst/'deployment_source_sha256.json').write_text(json.dumps(pins,indent=2)+'\n')
    for file in dst.rglob('*.py'):ast.parse(file.read_text())
    manifest['roles'][role]={'baseline_SHA256':baseline,'prepared_SHA256':pins,
                              'helper_SHA256':contract['paired_helper_SHA256'],
                              'requested_mem_fraction':argv[argv.index('--mem-fraction-static')+1],
                              'actual_runtime_accepted':False}
(OUT/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'prepared_only':True,'roles':list(manifest['roles']),
                  'files':{k:len(v['prepared_SHA256']) for k,v in manifest['roles'].items()},
                  'helper_CPU_accepted_both_roles':True,'new_pair_runtime_accepted':False}))
