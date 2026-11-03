"""Compile/run the CPU-only copy core and replay a declared placeholder fixture."""
from pathlib import Path
import ctypes,hashlib,json,platform,re,struct,subprocess,sys
from finalize_origin_coverage_v9 import N,O,read,write
from decode_memset_metadata_v111 import decode_record,write_spans


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v126.json').exists()
    d=N/'eager-host-launch-source-v126';build=d/'cpu-build';build.mkdir(exist_ok=True)
    exe=build/'selftest';lib=build/('snapshot.dylib'if sys.platform=='darwin'else'snapshot.so')
    common=['/usr/bin/cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-I',str(d)]
    commands=[common+[str(d/'host_packet_snapshot.c'),str(d/'selftest_snapshot.c'),'-o',str(exe)],
        common+(['-dynamiclib']if sys.platform=='darwin'else['-shared','-fPIC'])+[str(d/'host_packet_snapshot.c'),'-o',str(lib)]]
    sanitized=build/'selftest-sanitized'
    commands.append(common+['-fsanitize=address,undefined','-fno-omit-frame-pointer',str(d/'host_packet_snapshot.c'),str(d/'selftest_snapshot.c'),'-o',str(sanitized)])
    for cmd in commands:subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=30)
    p=subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=10)
    selftest=json.loads(p.stdout)
    sanitizer_run=subprocess.run([str(sanitized)],check=True,capture_output=True,text=True,timeout=10)
    assert json.loads(sanitizer_run.stdout)['status']=='PASS'and not sanitizer_run.stderr
    class PH(ctypes.Structure):_fields_=[('addrOffset',ctypes.c_uint32),('dataOffset',ctypes.c_uint32)]
    class Snapshot(ctypes.Structure):_fields_=[('args_size',ctypes.c_size_t),('placeholder_count',ctypes.c_size_t),('args',ctypes.c_uint8*4096),('placeholders',PH*32)]
    core=ctypes.CDLL(str(lib));f=core.wio_snapshot_host_args
    f.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(PH),ctypes.c_size_t,ctypes.POINTER(Snapshot)];f.restype=ctypes.c_int
    resolve=core.wio_inline_data;resolve.argtypes=[ctypes.POINTER(Snapshot),ctypes.c_uint32,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t)];resolve.restype=ctypes.c_int
    old=read(O/'native-pad-packet-and-store-contract-v120.json')
    dump=read(N/'native-pad-probe-results-v120b/own-native-pad-graph.json')
    event=next(e for e in dump if e['name'].startswith('MemSet_'))
    words=[int(x,16)for x in re.findall(r'0x[0-9a-fA-F]+',event['args']['Kernel Args'])]
    assert len(words)==old['rendered_uint64_words']
    packet=b''.join(struct.pack('<Q',w)for w in words)
    assert len(packet)==152
    # This is an intentionally constructed host fixture,not a recovered host launch.
    host=bytearray(packet);host[8:16]=b'\0'*8
    buffer=ctypes.create_string_buffer(bytes(host));ph=PH(8,24);snapshot=Snapshot();off=ctypes.c_size_t()
    assert f(1,1,buffer,len(host),ctypes.byref(ph),1,ctypes.byref(snapshot))==0
    assert bytes(buffer.raw[:len(host)])==bytes(host)
    assert resolve(ctypes.byref(snapshot),8,96,ctypes.byref(off))==0 and off.value==24
    copied=bytes(snapshot.args[:snapshot.args_size]);record=decode_record(copied[off.value:off.value+96]);span=write_spans(record,False)
    assert record==old['decoded_source_record']and span['union']==[[0,32]]
    # Reject ambiguity and never treat copied pointer words as host addresses.
    snapshot.placeholder_count=2;snapshot.placeholders[1]=snapshot.placeholders[0]
    assert resolve(ctypes.byref(snapshot),8,96,ctypes.byref(off))==5
    result=dict(status='PASS',platform=dict(system=platform.system(),machine=platform.machine()),commands=commands,
        build_artifacts=[pin(exe),pin(lib),pin(sanitized)],core_selftest=selftest,ASan_UBSan='PASS',ctypes_snapshot_bytes=ctypes.sizeof(Snapshot),placeholder_ABI_bytes=ctypes.sizeof(PH),
        fixture=dict(source='Own v120 post-relocation dump;constructed zero-placeholder host packet plus explicit offsets8->24.',
            actual_new_host_packet_observed=False,args_bytes=152,declared_placeholder=dict(addrOffset=8,dataOffset=24),
            decoded_record=record,source_span_union=span['union'],input_unchanged=True,ambiguous_placeholder_rejected=True),
        CANN_library_loaded=False,new_NPU_context=False,device_or_tensor_pointer_dereferenced=False,
        installed_adapter_or_live_native_interception_tested=False,original_pending_operands_recovered=0,overall_complete=False)
    write(O/'host-packet-CPU-core-result-v126.json',result)
    print(json.dumps({k:result[k]for k in ['status','platform','ctypes_snapshot_bytes','CANN_library_loaded','new_NPU_context','installed_adapter_or_live_native_interception_tested']}))


if __name__=='__main__':main()
