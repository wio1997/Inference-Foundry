#!/usr/bin/env python3
"""Bind current metadata source entry to already collected disjoint self costs."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path('audit/execution-search')
RANGES={'model_runner_v1.py':[[3815,3883]],'dsa_v41_cp.py':[[35,167]],
        'dsa_cp.py':[[1146,1241]],'dsa_v41.py':[[929,1010]],'device_metadata.py':[]}
def main():
    data=json.loads((ROOT/'common-frame-measured-entry.json').read_text());sources=[];excerpts=[]
    for r in data['sources']:
        p=Path(r['local_path']);b=p.read_bytes();h=sha256(b).hexdigest();assert h==r['runtime_sha256'];l=b.decode().splitlines();rr=RANGES[p.name]
        sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,reviewed_ranges=rr,
                            qualification='Only listedentrybodies; deviceexecutor currentidentity only, priorcanonicalframe lifetime review retained. No nativebit/implementation acceptance.'))
        for a,z in rr:
            assert 1<=a<=z<=len(l);excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=l[a-1:z]))
    review=dict(status='BOUNDED_MEASURED_COMMON_FRAME_ENTRY',sources=sources,
        resolved=['Currentfive moduleSHAs match both sourcepins and diagnosticadapter. Newfunctionselffields remove markerboundary once and partition prepareFree; CP/base/localhelper are disjointself intervals, not inclusivesum.',
                  'CP wrapper invokesglobalbuilder then localbasebuilder; basealreadyshares batchmetadata afterentry, but CPhelper/qslcopy/suffix/devicewhere runbeforelocalbasecache. Doublebasecall is structural, countsapprox27/54 percycle are not percaller argument identity proof;2unmatchedFTbodyentries retained.',
                  'Runner resets samegeneration batchdict once and physicalresource shared dict perKVgroup. GDN queryoverride, DSpark pergroup table/slot publication and speccommon selection remainbranches, notwholeobject cache credit.',
                  'CPhelper currentargumentqsl CPU selectsTorchfallback, constructsintersection/cumsum/CPUlength fromoriginalvalues; laterCPdevice lengths remain authoritative. Noncausal, nonCPU/triton and optionalout mutations need separatekey/reader qualification; no replacement ofdevice lengths byCPUupperbound.',
                  'Allpaid144cases retain wholebasebuilder118.491ms. CP+localhelper disjointself envelope192.559ms/new1025.503mswindow; helpereligible/meanfirstretain/exposure.5/h.5ms percycle gives1.995percentwindow, CP.25eligible gives3.622percent. Conditionaldiagnosticprice, notrequest/nativegain.'],
        remaining=['Bind exactsamegeneration compatibility fields (role/config/numReq/qsl/CPUlength/deviceauthority/actualcounts/causal/rank) to allactualcallers, not shape-only or approximatecount cache.',
                   'Share firstproducer originalarithmetic/CPUimmutablefacts; keep perbuilderstabledeviceQSL publication, resource-specificslots/tables/C2ring/nativepacketattrs and allmutation/readers. Pricecopy/pin/assembly and graph/epochownership.',
                   'Apply existingcanonicalframe acquirebeforemainwrite, immutablepinnedCPUretirement, producerdone+consumerdone/failurejoin. Nativeeventgeneration and everyreplacementbit stillunqualified.',
                   'Fresh functionwindow isdiagnosticwithprobes; actualfirstproducercost, percallcommonfraction, unprofiledcriticalshift andfulloriginalC8/sixordinary exposure missing. No newmeasurements untilgapchanges go/nogo.'],
        prior_reader_lifetime='common-frame-lifetime-frontier.json',precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'common-frame-measured-entry-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'common-frame-measured-source-excerpts.json').write_text(json.dumps(dict(scope='Fourboundedentrybodies, fifthfileidentityonly; notallcaller/reader/nativecoverage.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources),bodies=sum(bool(r['reviewed_ranges']) for r in sources),excerpts=len(excerpts),new_admissions=0)))
if __name__=='__main__':main()
