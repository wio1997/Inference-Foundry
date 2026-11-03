"""Audited correction: preserve legacy HC branches and add the intended review history."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt=O/'M06-review-history-correction-v64.json';assert not receipt.exists()
 manifest_path=O/'checkpoint-manifest-v64.json';manifest=read(manifest_path)
 for a in manifest['artifacts']:
  p=Path(a['path']);b=p.read_bytes();assert len(b)==a['bytes'] and sha(p)==a['sha256'],p
 matrix_path=O/'module-phase-branch-matrix-v64.json';prev=read(O/'module-phase-branch-matrix-v63.json');current=read(matrix_path);before=deepcopy(current)
 old=next(m for m in prev['modules'] if m['id']=='M06');m=next(m for m in current['modules'] if m['id']=='M06')
 assert m.get('branch_schema') is None and old.get('branch_schema') is None
 assert m['branches'][:-1]==old['branches'] and len(m['branches'])==len(old['branches'])+1
 added=m['branches'][-1];assert added['branch']=='Indexer Key/cache and draft context operator work, typed partitions and generation constraints'
 assert 'F27' in m['candidate_links'] and 'F27' not in old['candidate_links']
 matrix_backup=O/'module-phase-branch-matrix-before-M06-correction-v64.json';matrix_backup.write_bytes(matrix_path.read_bytes())
 manifest_backup=O/'checkpoint-manifest-before-M06-correction-v64.json';manifest_backup.write_bytes(manifest_path.read_bytes())
 before_matrix_sha=sha(matrix_path);before_manifest_sha=sha(manifest_path)
 m['branches']=deepcopy(old['branches']);m.setdefault('review_history',[]).append(dict(version='v64',title=added['branch'],finding=added['evidence'],evidence=['index-context-native-chains-v64.json','index-context-operator-contract-v64.json','index-context-fusion-costs-v64.json','index-context-fusion-variants-v64.json'],gap=added['gap'],correction='LegacyM06hasnoSEMANTIC_BRANCHES_V1schema;thisisreviewhistory,notanewlegacybranch. ExistingHcPrebranchesandnewF27cross-operatorlinkpreserved.'))
 assert m['branches']==old['branches'] and len(m['review_history'])==len(old.get('review_history',[]))+1
 for mid in ['M01','M21','M22','M25','M07','M11','M12','M08','M14','M10']:
  assert next(x for x in current['modules'] if x['id']==mid)==next(x for x in prev['modules'] if x['id']==mid),mid
 write(matrix_path,current)
 summary=O/'index-context-ceiling-summary-v64.json';s=read(summary);s['post_persistence_correction']=str(receipt);write(summary,s)
 write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),kind='LOCAL_ANALYSIS_LEDGER_CORRECTION',reason='PreflightassertedM06semantic-schema,butactuallegacyschemawasNone. Rootorchestrationprintednonzeroexecresultthenunconditionallycalledfinalizer;itsgenericbranchappendran. Thiswasasave-orchestrationerror,notoperator/model/serverfailure.',before_matrix_backup=str(matrix_backup),before_matrix_sha256=before_matrix_sha,after_matrix_sha256=sha(matrix_path),before_manifest_backup=str(manifest_backup),before_manifest_sha256=before_manifest_sha,changed_module='M06',original_legacy_branch_count=len(old['branches']),new_review_history_count=len(m['review_history']),new_cross_operator_F27_link_retained=True,operator_models_changed=False,coverage_counts_changed=False,source_or_model_changed=False,protection_check='M06legacybranchesidenticalv63;all10otherprotectedrowsunchanged;M09/M18newsemanticinventoriesretained.',prevention='Alllaterdependentmutationsmustexplicitlycheckexec.exit_codebeforecallingthem;readnonzerotoolresultsisnotexecutiongating.',overall_complete=False))
 rid='run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind','review','--command','Post-v64localledgercorrection: preservelegacyM06branchesandmovesharedinputfindingtoreviewhistory;source/coverageunchanged','--code-ref','Localanalysisledgeronly;noNPU/model/service','--working-directory',str(N))
 args=['finish-run','--run-dir',run,'--status','pass','--correctness','not-applicable','--summary','CorrectedM06legacybranchappendafterfailedschema-preflightwasnotexecution-gated;oldbranchesandnewF27associationpreserved,receipt/beforematrix/manifestretained. Operatorformulasandcoverageunchanged.']
 for p in [receipt,matrix_backup,manifest_backup,matrix_path,Path(__file__)]:args+=['--artifact',str(p)]
 ctl(*args)
 regpath=O/'task-run-registration-v64.json';reg=read(regpath);reg['runs'].append(dict(run_dir=run,status='pass',description='Post-v64auditedM06review-historycorrectionafterungatedpreflightfailure;models/coverageunchanged.'));reg['compact_output']=ctl('compact','--task-dir',str(T));reg['validation_output']=ctl('validate','--task-dir',str(T));write(regpath,reg)
 artifacts={Path(a['path']) for a in manifest['artifacts']};artifacts.update([receipt,matrix_backup,manifest_backup,Path(__file__)])
 manifest.update(post_persistence_correction=str(receipt),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(artifacts)],overall_complete=False)
 write(manifest_path,manifest)
 print(json.dumps(dict(corrected_M06_legacy_branches=len(m['branches']),review_history_records=len(m['review_history']),manifest_files=len(manifest['artifacts']),runs=len(reg['runs']),operator_models_or_coverage_changed=False,validation=reg['validation_output'],overall_complete=False)))
if __name__=='__main__':main()
