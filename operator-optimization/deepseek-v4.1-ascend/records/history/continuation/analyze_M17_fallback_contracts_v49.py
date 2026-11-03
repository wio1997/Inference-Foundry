"""Source contracts and exact rational counterexamples; never import or execute model/device code."""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet


def block_formulas(ratios, uniforms):
    """Compare the two recorded formulas over nonnegative finite real ratios."""
    triton_prefix = F(1)
    pytorch_prefix = F(1)
    uniform_prefix = F(1)
    t, p, u = [], [], []
    for ratio, uniform in zip(ratios, uniforms):
        triton_prefix = min(triton_prefix * ratio, F(1))
        pytorch_prefix *= min(ratio, F(1))
        uniform_prefix *= uniform
        t.append(triton_prefix)
        p.append(pytorch_prefix)
        u.append(uniform_prefix)
    lt = max([i for i, (x, y) in enumerate(zip(t, u)) if x >= y], default=-1)
    lp = max([i for i, (x, y) in enumerate(zip(p, u)) if x >= y], default=-1)
    return t, p, u, lt, lp


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v49.json').exists()
    for a in read(O / 'checkpoint-manifest-v48.json')['artifacts']:
        b = Path(a['path']).read_bytes()
        assert len(b) == a['bytes'] and hashlib.sha256(b).hexdigest() == a['sha256'], a['path']
    now = datetime.now(timezone.utc).isoformat()
    source = read(O / 'M17-pytorch-fallback-source-v49.json')
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    functions = {f['name']: f for f in source['functions']}
    assert len(functions) == 4 and sum(len(f['guard_sites']) for f in functions.values()) == 10
    kernel = packet('M17-kernel166-v33', False)
    caller = packet('M17-source166-v33-v2', False)
    baseline = read(O / 'M17-current-probability-and-guard-scope-v33.json')
    kblock = next(f for f in kernel['result']['functions'] if f['name'] == 'rejection_random_sample_block_verify_kernel')
    dispatch = next(f for f in kernel['result']['functions'] if f['name'] == 'sample_recovered_tokens')
    assert any('pi = min(pi * target_prob / draft_prob, 1.0)' in line for line in kblock['source'])
    assert any('pi = pi.clamp(max=1.0)' in line for line in functions['rejection_random_sample_block_verify_pytorch']['source'])
    assert any('pi = torch.cumprod(pi, dim=-1)' in line for line in functions['rejection_random_sample_block_verify_pytorch']['source'])
    root_recovery = next(s for s in caller['result']['selected_statements'] if s['start'] == 573)
    assert 'sample_recovered_tokens(' in '\n'.join(root_recovery['source'])
    assert 'use_block_verify' not in '\n'.join(root_recovery['source'])
    assert any('use_block_verify: bool = False' in line for line in dispatch['source'])
    refs = [str(O / 'M17-pytorch-fallback-source-v49.json'), str(N / 'M17-kernel166-v33.json'),
            str(N / 'M17-source166-v33-v2.json'), str(O / 'M17-current-probability-and-guard-scope-v33.json')]
    # Pin each reused immutable result against its historical manifest without
    # incorrectly hashing a displayed subset as if it were the remote full payload.
    manifests = [read(p) for p in O.glob('checkpoint-manifest-v*.json')]
    historical_inputs = []
    for p in [N / 'M17-kernel166-v33.json', N / 'M17-source166-v33-v2.json',
              O / 'M17-current-probability-and-guard-scope-v33.json',
              O / 'M17-probability-consumer-and-RNG-contract-v33.json',
              O / 'M17-probability-branch-ledger-v33.json']:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert any(a['path'] == str(p) and a['sha256'] == h
                   for m in manifests for a in m.get('artifacts', [])), str(p)
        historical_inputs.append(dict(path=str(p), sha256=h, historical_manifest_match=True))

    cases, differing_last, equal_low_ratio_cases = 0, 0, 0
    for length in [1, 2, 3]:
        for ratios in product([F(0), F(1, 4), F(1, 2), F(1), F(2), F(4)], repeat=length):
            for uniforms in product([F(1, 4), F(1, 2), F(3, 4)], repeat=length):
                t, p, u, lt, lp = block_formulas(ratios, uniforms)
                assert all(a <= b for a, b in zip(p, t)) and lp <= lt
                cases += 1
                differing_last += lt != lp
                if all(r <= 1 for r in ratios):
                    assert p == t and lp == lt
                    equal_low_ratio_cases += 1
    t, p, u, lt, lp = block_formulas([F(1, 2), F(2)], [F(9, 10), F(9, 10)])
    assert (lt, lp) == (1, -1)
    witness = dict(
        scope='DERIVED_EXACT_RATIONAL_SOURCE_FORMULAS;NO_TORCH_TRITON_OR_DEVICE_EXECUTION',
        flags=dict(non_greedy=True, entropy_verify=False, synthetic=False, placeholders=False),
        draft_probs=[['1/2', '1/2'], ['1/2', '1/2']],
        target_probs=[['1/4', '3/4'], ['0', '1']], draft_token_ids=[0, 1],
        selected_ratios=['1/2', '2'], uniform_probs=['9/10', '9/10'],
        triton_prefix=[str(x) for x in t], pytorch_prefix=[str(x) for x in p],
        cumulative_uniform=[str(x) for x in u],
        triton_last_accepted=lt, pytorch_last_accepted=lp,
        bonus_token_id=0, supplied_recovery_ids=[1, 1],
        recovery_consistency='For these distributions ordinary positive-residual recovery has only vocabulary1positive in either row;any finite positive request exponential values select1.',
        triton_logical_output=[0, 1, 0], pytorch_logical_output=[1, -1, -1],
        interpretation='Both inputs are normalized valid two-vocabulary distributions. A later Triton ratio>1can restore cumulative acceptance;clipping each factor first cannot. This rejects unconditional backend unification,not current workload correctness.',
        floating_native_execution_or_bitwise_equivalence_tested=False)
    proof = dict(generated_utc=now, evidence_grade='DERIVED_FROM_HASH_MATCHED_SOURCE_AND_EXACT_RATIONAL_ALGEBRA',
        evidence=refs, historical_inputs=historical_inputs,
        formulas=dict(Triton='b0=1;bi=min(b(i-1)*ri,1)',
                      PyTorch='ai=product(j<=i,min(rj,1))',
                      uniform='ci=product(j<=i,uj);last=max{i:prefix_i>=ci},or-1'),
        witness=witness, finite_checks=dict(cases=cases, differing_last_accepted=differing_last,
            all_ratios_at_most1_cases_equal=equal_low_ratio_cases,
            does_not_estimate_runtime_frequency=True),
        general_domain_proof='For finite nonnegative ratios,induction gives ai<=bi:ri<=1multiplies both;ri>1leaves ai unchanged while min(bprev*ri,1)>=bprev>=aprev. With the same thresholds,the PyTorch accepted-position set is a subset and its last position cannot exceed Triton. This is real arithmetic,not a floating-point reordering certificate.',
        sufficient_equality_domain='All ri<=1 gives identical real prefixes. In NO_DRAFT_PROBS mode q=1and normalized finite selected probabilities satisfy this;also require no placeholder and matching uniform/entropy values. F64/F32 and reduction-order distinctions remain.',
        placeholder_distinction='Triton placeholder=-1sets cumulative pi=0and skips that uniform update. PyTorch substitutes safe token0forgather,still multiplies its probability/uniform,and only masks that position. A later accepted position may publish the earlier placeholder as part of the prefix. This matters only if a placeholder occurs inside the declared valid span;no such live input is asserted.',
        exclusions=['No zero/negative/nonfinite draft denominator or probability reassociation is covered by the real-positive-domain identity.',
                    'Uniforms enter PyTorch asFLOAT64butTritoncaller castsFLOAT32;even otherwise equal real formulas need their own native numerical contract.',
                    'No current non-greedy/block/fallback runtime occurrence or defect inferred from this counterexample.'],
        implemented=False, whole_module_closed=False, overall_complete=False)
    write(O / 'M17-block-backend-formula-counterexample-v49.json', proof)

    facts = [
        ('ordinary_acceptance', '897-1054',
         'VectorizesB×Swithcu-derivedstart/counts and clamped flattened indices;safe placeholdercolumn0gathers before late masking. NGRAMusesq=1;else selecteddenseq. Ordinary/entropy decisions requireq>0andp/q>=uorh*u. Synthetic usesu<rate and bypassesentropydecision,butselectedP/QandallrecoveredIDgathers already happened. Firstvalidfailure getsrecovery;laterpositions retain prioroutput;allacceptedgetbonus atn. Greedyrequestsretainprioroutput.'),
        ('block_acceptance', '1190-1282',
         'Uses per-factor upper clamp then cumprod,not Triton running-prefix clamp. Multiplies all uniforms then optional per-position entropy threshold. legalmask addsq>0/valid/notplaceholder;lastlegalposition controls publicationofALLdraftpositions through it,thenrecoveryatlast+1orbonus. It is not ordinaryfirstfailure. Padded/greedyrows still participate in probability/entropy/gather work before publication masking.'),
        ('entropy_readers', '990-995;1251-1256',
         'Usesori_target_probswhenpresent,elseprocessedtarget_probs. GathersB×S×Vbefore per-row-H_epsilon reduction. Reducing unique flatNrows first then gathering small summaries is a conditional same-row transformation;retainepsilon,sourcegeneration,dtype/reduction behavior and any original/raw observers.'),
        ('ordinary_recovery', '1104-1187',
         'ReturnsifN=0. Maps eachflat token to request withN×Bintervalmask/argmaxandunmatchedfallback0. NGRAMclonesPthenzerosvaliddraftcolumn;placeholderdoesnotzeroanycolumn. Denseusesmax(P-D,0). Expandsrequest-sharedQtoN×V;Q==0orInfgetsfinalscore-1e10,NaNnotexplicitlysanitized. Fullrowtorch.argmaxpublishesallrecoveredIDs.'),
        ('blockwise_recovery', '1285-1378',
         'Separateexplicitoption. ReturnsifN=0;mapsrequestandposition;hostmax().item()sizesgrid. selectedp/qusesdraftdenominatorclampmin1e-10andzero-invalidguards;runningprefixclampsaftermultiply. p_iisEXCLUSIVEprefixbeforecurrentposition. NGRAMresidual=max(p_i*P_withoutdraft,0);dense=max(p_i*P-D,0). SameQzero/Infscoreprotectionandfullrowargmax. This residual is not genericordinaryP-D.'),
        ('actual_recovery_dispatch', 'savedcaller573;wrapper738-795',
         'Rootcaller573doesnotpassuse_block_verify,whichdefaultsFalse. HAS_TRITONselectsTritonrecoveryfirst;elseexplicituse_block_verifychoosesblockwisefallback;otherwiseordinaryfallback. Selectingblockverificationlaterdoesnotitselfactivateblockwiserecovery.'),
        ('RNG_and_precision', 'existingv33contract',
         'TwoacceptancebodiesdrawnoRNGthemselves. ExistinguniformgenerationprecedesQgeneration;Qdefaultandseededdrawsincludeexistingzero-draft/greedyeffects. Pruningrecoveryrowscanretainallexistingdraws;skippingdrawsrequiresanindependentstate-advancecontract. F64uniformfallbackandF32Tritonremainseparate.'),
        ('input_and_output_boundaries', 'all4selectedbodies',
         'Callerbounds/cumulativepartition/shapes/dtypes/device,validIDsandN/Sarepreconditions;latevalid/greedymasksdonotproveearlygatherssafe. Helpersdo notconstituteanempty-all-draftcallerproof. Outputismutatedinplaceandunwritten/greedypositionspreservepreviousbank. Noalias/capture/lifetimeclaimfromshape. Fullmodulecontrol/processor/logprobobserversareoutsidethesefourbodies.'),
    ]
    write(O / 'M17-pytorch-fallback-contract-v49.json', dict(
        generated_utc=now, source=str(O / 'M17-pytorch-fallback-source-v49.json'), evidence=refs,
        measured_scope=baseline['sampler_scope'],
        findings=[dict(id='FB49-%02d' % (i + 1), grade='DOCUMENTED_FROZEN_SOURCE', topic=t, lines=l, finding=f)
                  for i, (t, l, f) in enumerate(facts)],
        local_questions_answered=['Twofallbackacceptancebodies','Two directlydispatched recoverybodies',
            'Blockrecurrence/backendnonidentity','Recoveryoptiondispatchandoutputselectedposition'],
        local_body_review_complete=True, all_sampler_branches_or_consumers_closed=False,
        no_new_model_import_device_service_or_remote_work=True,
        implemented=False, critical_path_gain_us=None, whole_module_closed=False, overall_complete=False))
    guards = []
    for fn, f in functions.items():
        for g in f['guard_sites']:
            guards.append(dict(id='FB49-' + str(g['line']), function=fn, line=g['line'],
                condition=g['condition'], reviewed_arms=['BODY', 'ORELSE_OR_FALLTHROUGH'],
                state='LOCAL_FROZEN_SOURCE_SEMANTICS_REVIEWED;RUNTIME_OCCURRENCE_UNPROVED',
                evidence=['M17-pytorch-fallback-source-v49.json', 'M17-pytorch-fallback-contract-v49.json']))
    assert len(guards) == len({g['id'] for g in guards}) == 10
    write(O / 'M17-fallback-local-guard-cases-v49.json', dict(
        generated_utc=now, functions=4, if_sites=10, if_arms_reviewed=20,
        source_lines=sum(f['end'] - f['start'] + 1 for f in functions.values()), cases=guards,
        scope='Only the four named bodies;10static if sites/20arms are source-navigation coverage,not measured frequencies or exhaustive whole-module semantic branches.',
        whole_module_closed=False, overall_complete=False))

    variants = [
        dict(id='F35-v49-FALLBACK-PREPLAN-RECOVERY', parent='F35-v33-LAZY-RECOVERY',
            state='CONDITIONAL_BACKEND_SPECIFIC_REORGANIZATION',
            change='Computeordinary-first-rejectorbackend-specificblock-last-acceptplanbeforematerializingper-positionrecovery;produceonlytheusedrecoveryrowsandavoidB×SrecoveredIDgatherwherepublicationallows.',
            conditions=['Preserve eachbackendformula,placeholder/qzero/entropy/syntheticguardsandcurrentoutputbankretainsemantics.',
                        'Retainalluniform/Qdefaultandseededdrawswithsameorder,stateandrequestrowmapping;rowpruningisnotRNGdrawpruning.',
                        'Useordinaryorblockwiseresidualaccordingtoactualrecoverydispatch;verificationmodealoneisnotselector.',
                        'AllotherrecoveredIDreaders,alias/capturebanksandlast-reader/ordermustbeclosedbeforedeletingwholebuffers.'],
            actual_execution_measured=False, implemented=False, gain_us=None),
        dict(id='F35-v49-FALLBACK-ENTROPY-ROW-SUMMARY', parent='F35-v33-ENTROPY-SUMMARY',
            state='CONDITIONAL_FALLBACK_FUSION_OR_SPLIT',
            change='ReduceH_epsilononuniqueNflatprobabilityrowsandgatherB×SsummariesinsteadofB×S×Vprobabilities;composewithselectedp/q/acceptanceonlywhenallconsumerspermit.',
            conditions=['Sameori-versusprocessedsourceandper-replaygeneration,epsilon,dtypeandrequiredrounding/reductionbehavior.',
                        'Preservevalid/padded/greedyindexsemantics,outputbankretentionandalloriginalscore/rawreaders.',
                        'Anidealrowwiseidentitydoesnotprovenativebitwiseparityorreuseacrossdifferentparameter/inputgenerations.'],
            actual_execution_measured=False, implemented=False, gain_us=None),
        dict(id='H02-v49-FALLBACK-STRUCTURE-BUFFERS', parent='H02',
            state='CONDITIONAL_METADATA_REUSE_AND_HOST_DEVICE_SPLIT',
            change='Reuseconstant/shape-specificdevicepositionsandlength-to-requestplan;forseparateblockwiserecoveryconsiderpassingavalidknownboundinsteadofmax().item()underanexplicitnewinterface.',
            conditions=['Cachekeyincludesdevice,dtype,B,S,capturedaddress/lifetimeanddynamiclengthgeneration;constantaddressesarenotvalueidentity.',
                        'StandardacceptancehassixexplicitbaseCPU.to(device)sites;blockacceptancehasthree,plusconditionalNGRAMonesandentropyconstant. Theseare source sites,notmeasuredcallsorsavings.',
                        'Anewmaxboundmustcovereveryvalidpositionandpreservetheexclusiveprefix/paddingandallcurrentconsumers;extraallocatedworkmustbepriced.',
                        'No currentfallbackexecution,hostcopylatency,syncdurationorphysicalbenefitclaimed.'],
            actual_execution_measured=False, implemented=False, gain_us=None),
    ]
    write(O / 'M17-fallback-conditional-variants-v49.json', dict(
        generated_utc=now, variants=variants,
        evidence=['M17-pytorch-fallback-contract-v49.json', 'M17-block-backend-formula-counterexample-v49.json'],
        exclusions=['Do not unify Triton and PyTorch block formulas across all dense-draft distributions.',
                    'Do not infer blockwise recovery from block-verification mode.',
                    'Do not assign measured samplergreedy work to unobservedfallback orcountunexecutedconstant sitecostassavings.'],
        new_main_candidates=0, model_candidates=35, framework_candidates=7,
        whole_module_closed=False, overall_complete=False))
    branch = deepcopy(read(O / 'M17-probability-branch-ledger-v33.json'))
    old = next(row for row in branch['rows'] if row['branch'] == 'PyTorch_fallback')
    old.update(state='FOUR_DIRECT_BODIES_REVIEWED;CONDITIONAL_BACKEND_CONTRACTS',
        evidence='v49reviewsordinary/blockacceptanceandordinary/blockwiserecovery. Formula clamp-placement nonidentity,explicitrecoveryselector,F64uniforms,placeholderandQzero/Infhandlingnowdocumented. Notruntimevalidationorwholemoduleclosure.')
    block_row = next(row for row in branch['rows'] if row['branch'] == 'block')
    block_row.update(state='DOCUMENTED_BACKEND_SPECIFIC_BLOCK_CONTRACTS',
        evidence='Tritonrunning-prefixclampdiffersfromPyTorchper-factorclampthencumprod;bothpublishthroughlastacceptedposition. Positivevalidprobabilitycounterexamplein v49. Do not apply one formula to both backends.')
    ngram_row = next(row for row in branch['rows'] if row['branch'] == 'no_draft_probs')
    ngram_row.update(evidence=ngram_row['evidence'] + ' v49:normalizedfinitep<=1makesbothprefixformulas equal in real arithmetic with no placeholder and matched thresholds;F64/F32 and native reduction differences still prevent unconditional backend identity.')
    branch.update(generated_utc=now, parent=str(O / 'M17-probability-branch-ledger-v33.json'),
        detailed_guard_cases=str(O / 'M17-fallback-local-guard-cases-v49.json'), branch_universe_exhaustive=False)
    write(O / 'M17-probability-branch-ledger-v49.json', branch)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v49.json')
    gate.update(status='LOCAL_FALLBACK_BODY_QUESTIONS_ANSWERED;WHOLE_MODULE_OPEN',
        answered_by=['M17-pytorch-fallback-contract-v49.json', 'M17-block-backend-formula-counterexample-v49.json',
                     'M17-fallback-local-guard-cases-v49.json'],
        closed_scope='Four exact source bodies,10ifsites/20arms;sourceformula/backendandselectedrecoverydispatchquestions answered.',
        not_closed='AllM17conditions/consumers,actualflags/entryepochs/aliases/nativeprecision/performanceandoriginalrawaccess.',
        stop_rule='Do not reread these four bodies as wholly unknown or implement a backend unification. Advance remaining finite source/consumer coverage,using originalprofile when its route becomes accessible.')
    write(O / 'PROFILE-CODE-QUESTION-GATE-v49.json', gate)
    print(json.dumps(dict(local_functions=4, if_sites=10, if_arms=20, rational_cases=cases,
        differing_last_accept=differing_last, equality_domain_cases=equal_low_ratio_cases,
        source_formula_counterexample=True, new_main_candidates=0, overall_complete=False)))


if __name__ == '__main__':
    main()
