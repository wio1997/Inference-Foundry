#!/usr/bin/env python3
"""Freeze current larger Engram entry without reopening stopped small recipes."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path('audit/execution-search')
RANGES={'models/deepseek_v41/model.py':[[55,98],[1073,1084],[1197,1241],[1555,1592]],'engram/__init__.py':[[1,21]],'engram/common.py':[[1,72]],'engram/layer.py':[[1,57]],'engram/embedding.py':[[333,366]],'utils.py':[[1869,1879]]}
def main():
 x=json.loads((ROOT/'engram-joint-entry.json').read_text());sources=[];excerpts=[]
 for r in x['sources']:
  p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];l=b.decode().splitlines();rr=next((v for k,v in RANGES.items() if r['runtime_path'].endswith('/'+k)),[])
  sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,reviewed_ranges=rr,scope='Listed entry ranges only, empty identity-only. Not fullnative/bit or activeweight-format/key proof.'))
  for a,z in rr:
   assert 1<=a<=z<=len(l);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=l[a-1:z]))
 d=dict(status='PARTIAL_CURRENT_GLOBAL_ENGRAM_ROWS_WKV_ROTATED_GATE_ENTRY',sources=sources,
  reached=['CurrenttwoBF16FLOAT WKV checkpoints each[25600,6144],300MiB logicalweight. Layer1/14 distinct, payload0; optional.quarot rotation_map matchesexistingheader, but values/repeatedblock/load not reread.',
   'Producer embeds allglobal rows/headfeatures then modelwait/shards bySP; savedtargetbucket global24/local3. BF16 ReplicatedLinearWKV -> FP32inverseblock32rotation/RMS/dot/signbit/mask/value/residual -> BF16fourwaystate -> ownHCPre and laterHCPost, with DSparkauxbeforeEngram.',
   'Eachrank24complete WKV+37gatecall deliveries, allfree1.1687/1.1764%window, half+h5 .5716/.5753%; original WKVretainedallothersfree only.4513/.4590%. Shortwindow not originalrequestgain.',
   'AlreadyqualifiedoldC8 TTFT WKVfree.615615%, gatefree.489683%, fullWKVgatecopiesfree1.115388%; old1%independentchildrenremainstopped. Do not repeatoldinactivepassthroughbitfailure or unchangedoverlap/stagedprefix experiments.',
   'Newdistinctgeometry uses existingglobalcompleteK rows for NshardedWKV, preserves each oldM3panel padded16 and originalK/Fixpipe, reorders independentpanels forB reuse, then restoresoriginalrowKV bynewAllToAll. DirectSP-owner lookupremovesglobalbank, so creditsconflict unless newexchange paid. IllustratedL0C8x16x128x4=64KiB, not fullallocation/nativeproof.'],
  remaining=['SelectedoriginalMatMulV2 BF16 K/Fixpipe/tiling/Ntile and exactglobal8panel-Nowner resource/AllToAll/stride/capture contract; activeM3 numericorder cannot be inherited from dtype or stockM24 API.',
   'Whole globalrowowner/WKV/rotatedgate/HC-input joint paidparent includinglookup/readiness and alloriginalrequest/prefill/ordinary groups; do not develop lowvalue WKV-only/Gate-only siblings under oldbudget.',
   'Actualrotatedgate reduction/arithmetic+BF16boundaries to nextHC residualreader, trueglobal bank lifetime and selective tuple/weight-generation reuse if it can change largerparentdecision.'],
  precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'engram-joint-entry-review.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');(ROOT/'engram-joint-entry-source-excerpts.json').write_text(json.dumps(dict(qualification='Boundedcurrententry only, no nativeadmission.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(sources),nonempty_reviewed=sum(bool(s['reviewed_ranges']) for s in sources),excerpts=len(excerpts))))
if __name__=='__main__':main()
