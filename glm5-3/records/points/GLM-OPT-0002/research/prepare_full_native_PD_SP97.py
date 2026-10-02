from pathlib import Path
import json,ast,hashlib,subprocess
g=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3');p=g/'records/points/GLM-OPT-0002';base=p/'runs/GLM-RUN-0091';native=p/'runs/GLM-RUN-0096';r=p/'runs/GLM-RUN-0097'
state=json.loads((native/'state.json').read_text());assert state['status']=='completed'
proof=json.loads((native/'reduction_brief.json').read_text());assert proof['functional_acceptance']and proof['semantic_diagnostic']['semantic_pass_cases']==3
assert not r.exists();r.mkdir()
for name in ['owner_guard.py','adopted_model_identities.json','native_member_identities.json','canonical_81932.body.json','epoch_check.py']:(r/name).write_bytes((native/name).read_bytes())
owners=json.loads((r/'adopted_model_identities.json').read_text());groups=[dict(id=k+'-native-independent-EP16',epoch=hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest(),members=[k])for k,o in owners.items()];(r/'execution_groups.json').write_text(json.dumps(groups,indent=2)+'\n')
s=(base/'prepare.py').read_text().replace('GLM-RUN-0090/state.json','GLM-RUN-0096/state.json').replace('NATIVE-PD-CPU-20261002T1652Z','NATIVE-PD-VALIDATION-CPU-20261002T1710Z').replace('a3718ce6f9bab4f1640e8ff933ee77042642b79acf1a376688ea70ece4b54c15','3b92c18c90662ba0847cce6b2bd31ee32cbbaa842054eb97bd47af0c2e724411').replace('==30','==33').replace('sameP89D88/CPU30','sameP89D96/SP_D48_P6,12,24,48/CPU33')
(r/'prepare.py').write_text(s)
for name in ['responses.py','tools.py','full_api.py']:
 s=(base/name).read_text().replace('run91','run97').replace('GLM-RUN-0091','GLM-RUN-0097').replace('response_affinity_gateway_v8.py','response_affinity_gateway_v9.py')
 if name=='responses.py':s=s.replace('GLM_PD_PRODUCERS=json.dumps','GLM_PD_AUDIT_DIR=str(r/"native_PD_raw"),GLM_PD_PRODUCERS=json.dumps')
 (r/name).write_text(s);ast.parse(s)
sources=list(r.glob('*.py'))+[r/'execution_groups.json',r/'adopted_model_identities.json',r/'native_member_identities.json',native/'state.json',native/'reduction_brief.json',p/'jobs/NATIVE-PD-VALIDATION-CPU-20261002T1710Z/reduction.json']
sources+=[g/'runtime'/n for n in ['native_pd_transport_v2.py','response_affinity_gateway_v9.py','work_seconds_placement.py','prefill_work_placement_v2.py','request_estimates.py','response_affinity.py','placement.py','coupled_placement.py','persistent_coupled_placement.py','sse_observer.py','phase_runner.py','controller.py','issue_budget_scheduler_v3.py','glm_tool_contract.py','native_acl_lifecycle.py']]
pins=[];(r/'sources').mkdir()
for i,f in enumerate(sorted(set(sources))):
 b=f.read_bytes();snap=r/'sources'/('%02d_'%i+f.name);snap.write_bytes(b);pins.append(dict(path=str(f),snapshot=str(snap),sha256=hashlib.sha256(b).hexdigest()))
spec=dict(run_id=r.name,stages=[dict(id=n,argv=['/usr/bin/python3',str(r/(f+'.py'))],timeout_s=t,sources=pins)for n,f,t in [('prepare','prepare',360),('fullapi','full_api',2400)]])
(r/'controller_spec.json').write_text(json.dumps(spec,indent=2)+'\n');sha=hashlib.sha256((r/'controller_spec.json').read_bytes()).hexdigest()
contract='SameP89+D96nativeSP/DSACPOFF/independentEP16/API2NPU32/source/epochs/CPU4096t1024c1/Pserial3Dserial1/no modelops. V9PDv2 actualnativeCLI gateway/twoSTOREepochs/3NEWcoldPDhelpers: Responses11285→32 JSON+typedSSE/retrieve+Chat81932→64/noDbodychangesexceptKV; helpermax1/storeFalse/privateID/fullraw0600/internalcredits separate. NativecustomIDs/preRPCownerconflicts/previous/replay/errors/tools/randomlogprobs/nativebackground8192positiveRunning0HTTPleases→cleanGatewayrestart/retrieve/cancel/idle;384fixed+actual8tools outputs. Reuse91functionalpackage/new97IDs,96D3exact+96coldpilot thennewconfigtruePD/fullAPI; oneconfiguredDcapture48 vsP6,12,24,48. No guaranteedcapacity/SLO/gain/KEEP, no oldqueue.'
m=dict(run_id=r.name,point_id='GLM-OPT-0002',kind='diagnostic',status='prepared',valid=False,verdict='INCONCLUSIVE',contract=contract,source=dict(base_commit=subprocess.check_output(['git','-C',str(g.parent),'rev-parse','HEAD'],text=True).strip(),source_identities=pins),controller_spec_sha256=sha,reused='91V8publicPD/fullAPI589+3helper/cancel residual51;93V9PDv2fullhelperraw/actualnative400/10112 functionvalid;96newDSPstartup/3math+96cold outputs audited; SPchangesclass/capture—no qualitycompare/math/operators edits.')
(r/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(r/'summary.md').write_text('# '+r.name+'\n\n'+contract+'\n')
oldj=p/'jobs/FULL-NATIVE-PD-LAUNCH91-20261002T1655Z';j=p/'jobs/FULL-NATIVE-PD-SP-LAUNCH97-20261002T1803Z';j.mkdir()
s=(oldj/'execute.py').read_text().replace('GLM-RUN-0090/state.json','GLM-RUN-0096/state.json').replace('Run91','Run97').replace('V8publicPDChat/typedResponses/fullnativeAPI diagnostic','V9PDv2/newDnativeSP/typedResponses/fullnativeAPI diagnostic');ast.parse(s);(j/'execute.py').write_text(s)
job=json.loads((oldj/'job.json').read_text());job.update(job_id=j.name,parent=dict(point_id='GLM-OPT-0002',run_id=r.name),goal=contract,inputs=[dict(path=str(r))]);job['scope']['writes']=[str(r),'/data/tiankuan/wio/glm52-pd/controller-owner.json'];job['execution']['command']='/usr/bin/python3 '+str(j/'execute.py');job['result']['path']=str(j/'result.json');job['acceptance']=['96terminal/sourcepins/CPU33/uniquecontroller/sameAPI2NPU32/SPgeometry',contract];(j/'job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(dict(run=r.name,pins=len(pins),spec=sha,job=str(j/'job.json'))))
