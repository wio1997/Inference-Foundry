#!/usr/bin/env python3
"""Record tiered frame compatibility, actual callers and isolated CPU evidence."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path('audit/execution-search')
RANGES={
 'model_runner_v1.py':[[1440,1466],[1608,1620],[1848,1874],[2287,2302],[2487,2510],[3600,3676],[3714,3907]],
 'dsa_v41_cp.py':[[35,167],[344,442]],
 'dsa_cp.py':[[615,650],[812,846],[934,964],[1138,1241]],
 'dsa_v41.py':[[205,224],[839,842],[879,1010],[1160,1307],[1389,1424]],
 'device_metadata.py':[],
 'utils.py':[[253,375]],
 'dspark_proposer.py':[[217,357],[365,429]],
 'llm_base_proposer.py':[[1020,1085],[1219,1246],[2560,2651]],
 'dspark.py':[[78,143],[346,372]],
 'backend.py':[[382,483]],
 'parallel_state.py':[[1548,1550]],
}
def main():
    deps=json.loads((ROOT/'common-frame-key-dependencies.json').read_text());sources=[];excerpts=[]
    for r in deps['sources']:
        p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];l=b.decode().splitlines()
        rr=([[1,42]] if r['runtime_path'].endswith('attention/backends/utils.py') else RANGES[p.name])
        sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,reviewed_ranges=rr,qualification='Exactlisted ranges only. Legacypriorbody ranges reused with matchingSHA; executor identity-only/priorlifetime review. No allnative/branch coverageclaim.'))
        for a,z in rr:
            assert 1<=a<=z<=len(l);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=l[a-1:z]))
    native=json.loads((ROOT/'common-frame-torch-cpu.json').read_text());assert native['returncode']==0
    result=native['result'];assert result['bitwise_mismatches']==0 and result['CPU_cases']==1728 and result['bitwise_comparisons']==44928
    assert result['guards_before']['forbidden_NPU_libraries']==result['guards_after']['forbidden_NPU_libraries']==[]
    catalog=json.loads((ROOT/'native-workspace-catalog.json').read_text());torch_version=next(v['constants']['__version__'] for v in catalog['versions'] if '/torch/version.py' in v['path']);assert torch_version==result['torch_version']=='2.10.0+cpu'
    resolved=[
      'RawCPUhelper contract depends on sameclosedinvocation/inputlease, actualTPworld/rank, numReq/numInput, selectedCPUQSL/length immutablevalues/dtypes and causal/outmode. CPcall118 suppliesnooptionalout andCPUQSL; otherLegacycallers writepersistentoutputs/startpos andmustbypassmemo.',
      'Runner CMbase CPUQSL and optimisticCPUlength views are shallowshared across theKVloop; targetCP receives fresh batchdict, physicalgroupdict separate. CPUcorrection synchronizesacceptedcountcopy beforemetadata. Nevermove cache beforecorrection or outsideclosedinvocation; Tensorversion cannotcatchNumPy writes.',
      'Common.replace is dataclasses.replace, shallow tensorfields. CP updates seq_lens_cpu but inherits _seq_lens_cpu/globalupperbound andotherfields. Preserve originalfielddifferences/precedence, notcanonicalizeboth tolocalCPUlength. PostCPUqsl also depends onactualTokens/destdtype; deviceprojection onauthoritativeseqLens dependency.',
      'Saved16workerCPUparentIDs show eachof11runnerframes has26CP/helper and52basecalls, plusone separateCP/helper draftcall andtwo draftbasecalls/cycle. Parentcounts do not showtensorarg values; original2FT/body misses remaininFreecost.',
      'LLMBase prepares shared_dsa_draft_cache but currentV41build_for_drafting839–842 dropskwargs. TheexistingLegacyDSA_cpu_local cache isnot proof currentV41draft sharesit. InobservedC8 drafthasonecall perstep, so no26fold within-draft CPUcredit.',
      'NativeSMLA/QLI callbacks capture common.query_start_loc andmaxattrs evenwhenbase coordinates alreadybatchcached; _publish_task keepsfirstbuffer/id perkey. CP perbuilderstableQSL/copies, physicalslot/table/C2ring/descriptor andoldproducer-consumer protocol retained. Broadercanonicaldeviceframe needsactualreader/owner proof.',
      'Scalar originalAST1728cases/46656comparisons and8wrongkeyrecipes, then actualinstalledTorchCPU1728cases/44928bytecomparisons/optionaloutfallback0mismatch. Newprototype snapshotsfirstCPUinputs underexplicit immutablelease andretainsperresource operations. Thisis isolatedCPUprimitive, notserving/NPUintegration.',
      'Torch2.10CPUbuild matchesfrozencatalog. Backendautoloadoff/-S explicitsite; noTorchNPU/vllm/NPUlibraries loaded before/after12.60s privateprobe. Six26call pairedCPUbenchmarks median2.226–2.445ms saved/frame; no requestextrapolation. Privatefilesonlyownedmapped/data/wio root; service/profile/NPUactions0.',
      'New144callerboundscenarios retainrecordedfirsthelper wholebody andCPbody remainder withclockguards, basewholekept. h.5ms/cycle/exposure.5 gives1.964percenthelper-only,3.275percentwithCPself.25eligible innewdiagnosticwindow. Earlier1.995/3.622mean27 scenarios remainhistoricalsensitivities, supersededforactual26callerpricing.'
    ]
    pending=[
      'Liveframeworkintegration ofCPUprototype must attachfreshframe only toeligibleCPtargetcaller andpreserveimmutability/fallback/scalars/return aliases. FullfrozenC8/sixordinary end-to-end andbackendloaded service qualification notprovided byisolatedCPUtest.',
      'Larger shareddeviceprojection: trace everycommon.seq_lens/queryStartLoc reader throughfirstbatchcache miss, allnativecallbacks/slots/C2 andoriginal capturepointers; keepmainstreamfirstproducer or movealldependent eagerreaders. Oldasync commonwriter-only recipe remainsinvalid.',
      'OnecanonicalNPUlength projection withperbuilderstableQSLcopies couldremove repeatedintegerwork whilepreservingoutputaddresses; actualconditions/nativebit/lifetime/net are nextscope, notalreadyadmitted.',
      'Partialmask/empty/tail/draft/PCP resourcebranch population andwhole-request criticaltransfer remainseparate. No profiler-derived percentage addedtoCPUunitbenchmark or otherwindows.'
    ]
    review=dict(status='BOUNDED_FRAME_KEY_MUTATION_CALLER_AND_ISOLATED_CPU_PRIMITIVE',sources=sources,resolved=resolved,remaining=pending,
                precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',isolated_CPU_primitive_prototypes_passed=1,new_framework_replacement_admissions=0,new_primitive_admissions=0,
                whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,service_actions=0,
                remote_writes=[native['host_root']],remote_write_scope='PrivateCPUprobe scripts/spec/tmp/cache/receipts only underowned/data/tiankuan/wio mapping; no frameworksource/service/globalenv changes.',
                evidence=['common-frame-key-dependencies.json','common-frame-key-cpu.json','common-frame-caller-logs.json','common-frame-caller-price.json','common-frame-torch-cpu.json','common-frame-cpu-import-guard.json','common-frame-key-fetch-a1-invalid.json'])
    (ROOT/'common-frame-key-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'common-frame-key-source-excerpts.json').write_text(json.dumps(dict(scope='Listedboundedsource only; isolatedTorchCPU isnot completeframework/nativebit qualification.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
    frontier=dict(**review,candidates=[
        dict(scope='OriginalCPUhelper memo withinexplicitclosed immutable CPUinput lease; optionaloutfallback retained',classification='已确认可实现',result='LimitedisolatedCPUprimitiveprototype:1728cases/44928exactbytecomparisons, sixpaired26calllocalcostcases, allperresource/deviceoperations unchanged byprototype. Sourcecallerconditions mandatory; liveframeworkreplacement notadmitted.'),
        dict(scope='Larger CPU/post/device commonprojection and publication acrossactual26runnerCP calls',classification='待确认',result='CPUresult/key/caller nowbounded; actualdevice-input readers, stablecapturepointers, originalfirstproducer/mainstream schedule/nativebit andpaidrequesttransfer remain.'),
        dict(scope='Globalmethodmemo skipsoptionalout writes; merge_CPUmirror_fields; actualTokens/dtype/causal/TP omitted; Tensorversion-only cache acrossNumPywrites',classification='已停止',result='Source/rawCPU counterexamples or realTorchNumPyversion witness refutethesespecificrecipes, notproperfrozenlease.'),
        dict(scope='AssumeV41draftinheritsLegacysharedcache orclaim26draftduplicates',classification='已停止',result='Currentbuild_for_drafting dropskwargs; archivedparentIDs showone draftcall/frame. No repeatwithinthisobservedrolebudget.'),
        dict(scope='New liveframework/NPU replacement admitted',classification='待确认',result='0. IsolatedCPUprimitive prototype and localcost are notfullintegration/modelbit or actualrequestgain.')])
    (ROOT/'common-frame-key-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources),nonempty_reviewed=sum(bool(r['reviewed_ranges']) for r in sources),excerpts=len(excerpts),isolated_CPU_passed=1,new_framework_admissions=0)))
if __name__=='__main__':main()
