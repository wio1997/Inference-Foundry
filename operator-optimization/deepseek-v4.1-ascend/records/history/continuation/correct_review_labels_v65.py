"""Audited repair of review labels/verifier literals, without changing analysis rows."""
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,read,write

def main():
    receipt=O/'heads-sampling-persistence-correction-v65.json'
    assert not receipt.exists()
    manifest=read(O/'checkpoint-manifest-v65.json')
    for a in manifest['artifacts']:
        p=Path(a['path']);b=p.read_bytes()
        assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'],p
    paths=[N/'verify_heads_sampling_v65.py',N/'finalize_heads_sampling_v65.py',
        O/'module-coverage-gaps-v65.json',O/'heads-sampling-ceiling-summary-v65.json',O/'REVIEW-CHECKPOINT-v65.json',
        N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']
    before={str(p):p.read_text()for p in paths}
    write(O/'heads-sampling-persistence-correction-before-v65.json',dict(files=before,
        purpose='ExactbeforebytesasUTF8forlabel/verifierliteralcorrection;underlyingregimes/candidates/branchesunchanged.'))
    write(O/'checkpoint-manifest-before-review-correction-v65.json',manifest)
    q=N/'verify_heads_sampling_v65.py';s=q.read_text();assert s.count('==2182')==2
    q.write_text(s.replace('==2182','==2082'))
    q=N/'finalize_heads_sampling_v65.py';s=q.read_text();assert s.count('allother20module')==1
    q.write_text(s.replace('allother20module','allother21module'))
    for q in [O/'module-coverage-gaps-v65.json',O/'heads-sampling-ceiling-summary-v65.json',O/'REVIEW-CHECKPOINT-v65.json']:
        s=q.read_text();assert s.count('allother20module')==1
        q.write_text(s.replace('allother20module','allother21module'))
    cp=read(O/'REVIEW-CHECKPOINT-v65.json')
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);d['latest_profile_first_review']=cp;write(q,d)
    changes=[dict(path=str(q),before_sha256=hashlib.sha256(before[str(q)].encode()).hexdigest(),
        after_sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in paths]
    write(receipt,dict(reason='M00-M25has26modules;5refinedleaves21protected. Initialpreflight20assertstoppedbeforemutations. Subsequentbroadliteraleditinverifierhadunintentionallychanged2082to2182;detectedbyread-onlyreviewBEFOREverificationexecuted. Correctedtwoexactverifierconstantsandremainingallother20labels,retainingbeforefiles/manifest.',
        changes=changes,analysis_row_counts_changed=False,branch_or_candidate_records_changed=False,
        framework_or_server_actions=False,whole_modules_closed=0,overall_complete=False))
    paths_all={Path(a['path'])for a in manifest['artifacts']}
    paths_all.update([Path(__file__),receipt,O/'heads-sampling-persistence-correction-before-v65.json',O/'checkpoint-manifest-before-review-correction-v65.json'])
    manifest['artifacts']=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in sorted(paths_all)]
    manifest['post_persistence_correction']=str(receipt)
    write(O/'checkpoint-manifest-v65.json',manifest)
    print(dict(corrected_labels_and_verifier_only=True,files=len(changes),manifest_artifacts=len(paths_all),operator_regimes=2082,protected_modules=21))

if __name__=='__main__':main()
