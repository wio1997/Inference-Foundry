"""Reconcile current fallback costs against an already documented fused C2 helper."""
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,W,read,write

def main():
    assert not (O/'metadata-existing-builder-reconciliation-v68.json').exists()
    kernel=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/triton/c2_ring_metadata.py'
    sha=read(O/'M21-C2-metadata-fill-and-existing-fusion-contract-v39.json')['source_pins']['/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/c2_ring_metadata.py']
    assert hashlib.sha256(kernel.read_bytes()).hexdigest()==sha
    cpath=O/'metadata-field-operator-contract-v68.json';c=read(cpath)
    assert len(c['source_files'])==2
    c['source_files'].append(dict(path=str(kernel),sha256=sha,original166_hash_binding=True,existing_v39_equivalence_contract_reused=True))
    c['local_clause_review']='OnlySWA/CP/C2slots/lengths/RoPElookup/publishedbankdtypeandconsumers. 2callerfiles+existingv39fusedkernelhash/equivalencereused,3previous166pins. No framework/API/lifecycleaudit.'
    c['existing_C2_fused_builder']=dict(path=str(kernel),sha256=sha,already_implemented=True,
        prior_equivalence='F31-C2-existing-fused-builder-equivalence-v39.json',current_legacy_profiles_use_fallback=True,
        key_constraints='Existinghelperownsall5headerplanes/complete/sourcepos/C-Salready. Legacy910CconditionalreuseguardrequiresvalidINT32bounds/nonnegativepositions/sharedcos-sininputandoutputstridesoradaptseparatestrides/allbanks/frontiers. Notanewmissingkernel.')
    write(cpath,c)
    vpath=O/'metadata-field-fusion-variants-v68.json';v=read(vpath);before=[]
    for row in v['variants']:
        if row['id']=='META-C2-RING-HEADER':
            before.append(dict(row));row.update(choice='Header-onlydirectwritealternativetoexistingall-fieldfusedbuilder',
                conditions='INT32sourcewrap/L-Qlen/used/starts/firstblock,normalBTshape/errors/alias/allreads-beforestores,stagefrontiers. Existingv39builderalreadyalsofusesheader;notnewmissingkernel.')
        if row['id']=='META-C2-TOKEN-CS-COMPOSE':
            before.append(dict(row));row.update(choice='Reuseexistingv39all-fieldC2builderoutsidepacked-onlyguardconditionally',
                work_IO_effect='Alreadyfuses5headerplanes/complete/sourcepos/C-S;newnativecostbindingspricecurrentfallback',
                conditions='ValidINT32bounds/nonnegativepositiondomain,sharedcos-sinTABLE/OUTstridesoradaptseparatestrides,fullpersistentABI/oldpool-cacheallreaders/frontiers/capture;noexecuted910Ccompilationormissing-kernelclaim.')
    assert len(before)==2;write(vpath,v)
    write(O/'metadata-existing-builder-reconciliation-v68.json',dict(before_wording=before,
        reused_existing_kernel=dict(path=str(kernel),sha256=sha),prior_contract='F31-C2-existing-fused-builder-equivalence-v39.json',
        finding='v68newtypedTASK/IOmodelspricecurrentlegacyfallbackandstandalonealternatives;fullC2fusionmechanismalreadyexistsunderpackedguard. Do notclaimnewmissingimplementation. Retainstride/nonnegative-domain/INT32/lifetime/frontierconditions.',
        numeric_models_changed=False,implementation=False,native_test=False,overall_complete=False))
    print(dict(existing_C2_kernel_reconciled=True,source_pins=3,wording_refined=2,numeric_models_unchanged=True))

if __name__=='__main__':main()
