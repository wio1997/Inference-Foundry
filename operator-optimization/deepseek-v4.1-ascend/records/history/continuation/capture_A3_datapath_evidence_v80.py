"""Primary architecture/API evidence;public HTTP only,no browser or server work."""
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from html.parser import HTMLParser
import hashlib
from finalize_origin_coverage_v9 import O,read,write


class Extract(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)


def main():
    assert not(O/'A3-datapath-primary-evidence-v80.json').exists()
    inputs=[
        ('arch2201','https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html'),
        ('migration2201to3510','https://asc.gitcode.com/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.html'),
        ('GetTensorC','https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetTensorC.html'),
        ('L0CtoL1','https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/cube_compute_ISASI/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%87%BA/DataCopy_L0CToL1.html'),
        ('SSBuffer','https://asc.gitcode.com/guide/programming_guide/advanced_programming/inter_core_communication/ssbuffer_inter_core_memory_feature.html'),
        ('L1toUB','https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/DataCopyL1ToUB.html')]
    docs=[];texts={}
    for name,url in inputs:
        try:
            blob=urlopen(Request(url,headers={'User-Agent':'OperatorCeilingEvidence/1.0'}),timeout=12).read()
            p=O/f'{name}-primary-source-v80.html';p.write_bytes(blob)
            e=Extract();e.feed(blob.decode());text=' '.join(e.parts);texts[name]=text
            docs.append(dict(id=name,url=url,status='READ',snapshot_path=str(p),snapshot_sha256=hashlib.sha256(blob).hexdigest(),snapshot_bytes=len(blob),
                retrieved_UTC=datetime.now(timezone.utc).isoformat(),version_limit='PrimaryHuaweiasc-devkit previewmaster documentation,not installedCANN9.2.0SDK/binary attestation. Physicalarchitecture/API-class reference only.'))
        except Exception as exc:docs.append(dict(id=name,url=url,status='LOOKUP_UNAVAILABLE',error_type=type(exc).__name__,message=str(exc)[:300]))
    assert '192KB'in texts['arch2201']and '256'in texts['arch2201']and '128'in texts['arch2201']
    assert '新增UB到L1'in ''.join(texts['migration2201to3510'].split())
    assert '新增L0CBuffer到UB'in ''.join(texts['migration2201to3510'].split())
    assert 'Workspace'in texts['GetTensorC']
    hardware=read(O/'saved-profile-local-proof-index-v48.json')['saved_hardware_evidence']
    assert hardware['Cube_per_device']==24 and hardware['Vector_per_device']==48
    facts=[
        dict(id='ARCH-SPLIT',source='arch2201',grade='DOCUMENTED_ARCHITECTURE',statement='A3/A2arch2201 has separateAIC/AIV at1:2;AIC↔AIV exchanges useGM. No sharedAIC/AIVUB.'),
        dict(id='ARCH-NO-NEWER-CV-CHANNEL',source='migration2201to3510',grade='DOCUMENTED_ARCHITECTURE',statement='3510 addsUB→L1 andL0C→UB directpaths. Do notimportthosepaths into2201A3. Sameguide says2201GM→L0A/Bdirectpaths exist;thereforeL1-onlyingressis an explicit implementationdomain,not allalgorithms.'),
        dict(id='ARCH-CAPACITIES',source='arch2201',grade='DOCUMENTED_ARCHITECTURE',statement='PerAIVUBphysical192KiB;perAICL1 512KiB,L0A64KiB,L0B64KiB,L0C128KiB. SDKusableUBmayreserve256B/8KiB;currentcompileflagsnotknown.'),
        dict(id='ARCH-INGRESS-RATES',source='arch2201',grade='DOCUMENTED_ARCHITECTURE',statement='L1→L0A256B/cycle,L1→L0B128B/cycle perAIC. Useindependentbottleneckconstraints/max;sameMTE1labeldoesnotprovefullyserializedsum.'),
        dict(id='GETTENSORC-NOT-DIRECT-WIRE',source='GetTensorC',grade='DOCUMENTED_API',statement='GetTensorCcanexposeUBoutput,butasynchronousmodeusesSetWorkspaceGMresultcache;APIoutputtypealoneisnot evidenceofdirectL0C→AIVUB.'),
        dict(id='CUBE-CUBE-PATH',source='L0CtoL1',grade='DOCUMENTED_API_IF_READ',statement='L0C→L1matrixmoveout supportsA3 andcanpermitCube→Cubeintermediateslocallywithdtype/layout/owner/synccontracts. ItdoesnotcreateVectoraccess toAICL1.'),
        dict(id='SSBUFFER-GENERATION',source='SSBuffer',grade='DOCUMENTED_API_IF_READ',statement='SSBuffernewarchmessages/directCVexamples mustbecheckedproductwise;do notassumeA3supportfromagenericpage.'),
        dict(id='NO-L1-UB-BYPASS',source='L1toUB',grade='DOCUMENTED_API_IF_READ',statement='DataCopyL1ToUBsupporttable explicitlyexcludesA3/A2. L0C→L1doesnotthereforecreateanA3indirectCube→Vectoronchipbypass.'),
    ]
    write(O/'A3-datapath-primary-evidence-v80.json',dict(primary_sources=docs,facts=facts,saved_hardware=hardware,
        captured_frequency_MHz=1800,captured_frequency_is_proven_hardware_maximum=False,
        typed_Vector_Exp_Ln_Sqrt_Sort_INT64_numeric_capacity=None,FP32_or_HF32_numeric_capacity=None,actual_HBM_L2_GM_bandwidth_upper=None,
        specific_lookup_limits=[dict(url='https://gitcode.com/cann/runtime/blob/master/src/platform/platform_config/cann/Ascend910_9391.ini',result='webtoolinaccessible;candidateSKUwasnotanidentifiedcapturedSOCandisnotused'),
            dict(url='https://gitcode.com/cann/runtime/blob/master/src/platform/platform_config/cann/Ascend910_9362.ini',result='Primarysearchsnippet20Cube40Vector1500MHz doesnotmatchsaved24/48/1800;notusedasactualcapacity orvectorwidth/rate.')],
        no_browser_used=True,no_new_remote_server_or_NPU_work=True,overall_complete=False))
    print(dict(primary_pages_read=sum(r['status']=='READ'for r in docs),lookup_unavailable=sum(r['status']!='READ'for r in docs),
        documented_L1_L0_rates_B_per_cycle=[256,128],actual_peak_frequency_proved=False,Vector_numeric_capacity=None))


if __name__=='__main__':main()
