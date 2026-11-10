#!/usr/bin/env python3
"""Bound actual frame read/overwrite paths and distinguish host/device retirement."""
from pathlib import Path
from hashlib import sha256
import ast,json

ROOT=Path('audit/execution-search')
RANGES={
 'worker/model_runner_v1.py':[[1418,1476],[1580,1638],[2004,2116],[2538,2611],[2878,2918],[3192,3275],[6120,6150]],
 'worker/device_metadata.py':[[1,73],[74,155],[146,172]],
 'ascend_forward_context.py':[[95,157]],
 'attention/dsa_v41.py':[[91,180],[290,420],[497,565],[828,878],[1290,1444]],
 'attention/context_parallel/dsa_v41_cp.py':[[185,230]],
 'spec_decode/dspark_proposer.py':[[226,357]],
 'spec_decode/llm_base_proposer.py':[[730,754],[950,1114],[1345,1398],[1960,2035],[2389,2488],[2560,2649]],
 'attention/utils.py':[[254,391]],
 'vllm/v1/attention/backends/utils.py':[[1,72]],
 'vllm/v1/attention/backend.py':[[385,502]],
 'vllm/forward_context.py':[[132,205],[269,300],[348,385]],
}
FIELDS={'query_start_loc','query_start_loc_cpu','seq_lens','seq_lens_cpu','_seq_lens_cpu','positions','positions_cpu','max_query_len','max_seq_len','num_actual_tokens','num_input_tokens','num_reqs','causal','block_table_tensor','slot_mapping','c2_ring_metadata','smla_metadata','qli_metadata','global_metadata'}
ROOTS={'cad','common','common_attn_metadata','old_common_metadata','metadata','query_metadata','cache_metadata','state_metadata'}
def main():
    dep_path=ROOT/'common-frame-lifetime-dependencies.json';dep=json.loads(dep_path.read_text());assert len(dep['sources'])==15
    sources=[];accesses=[];excerpts=[]
    for r in dep['sources']:
        raw=Path(r['local_path']).read_bytes();assert sha256(raw).hexdigest()==r['runtime_sha256']
        name=r['runtime_path'].removeprefix('/vllm-workspace/vllm-ascend/vllm_ascend/').removeprefix('/vllm-workspace/vllm/')
        ranges=RANGES.get(name,[]);lines=raw.decode().splitlines()
        for a,b in ranges:assert 1<=a<=b<=len(lines)
        sources.append(dict(**r,reviewed_ranges=ranges))
        excerpts.append(dict(runtime_path=r['runtime_path'],full_source_sha256=r['runtime_sha256'],reviewed_ranges=ranges,excerpts=[dict(first=a,last=b,text='\n'.join(lines[a-1:b])) for a,b in ranges]))
        tree=ast.parse(raw.decode())
        class Visitor(ast.NodeVisitor):
            def __init__(self):self.stack=[]
            def visit_ClassDef(self,n):
                self.stack.append(n.name);self.generic_visit(n);self.stack.pop()
            def visit_FunctionDef(self,n):
                self.stack.append(n.name);self.generic_visit(n);self.stack.pop()
            def visit_Attribute(self,n):
                base=n.value
                while isinstance(base,(ast.Attribute,ast.Subscript)):base=base.value
                if isinstance(base,ast.Name) and base.id in ROOTS and n.attr in FIELDS:
                    accesses.append(dict(runtime_path=r['runtime_path'],function='.'.join(self.stack),line=n.lineno,root=base.id,field=n.attr,context=type(n.ctx).__name__,text=lines[n.lineno-1].strip()))
                self.generic_visit(n)
        Visitor().visit(tree)
    def write(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    write('common-frame-literal-accesses.json',dict(status='LITERAL_NAVIGATION_ONLY_NOT_COMPLETE_DEF_USE',accesses=accesses,qualification='Selected attribute/root names in15 ASTs only. Does not resolve aliases, dynamic dispatch/reflection, graph capture or native observers. Counts do not prove all readers.',whole_chain_complete=False))
    write('common-frame-lifetime-source-excerpts.json',dict(sources=excerpts,qualification='Only explicitly reviewed ranges; []identity-only. Complete raw sources and SHA retained locally.'))
    result=dict(status='BOUNDED_SOURCE_AND_PARTIAL_ORDER_FRAME_RETIREMENT_REVIEW',whole_chain_complete=False,precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',sources=sources,
      resolved=[
        'V1 submit invokes each Python task.run synchronously before returning. Native scalar attrs are evaluated during those calls; later DSpark field rebinding does not rerun already-issued Python closures. Proposed delayed callbacks/capture still need explicit scalar binding.',
        'CommonAttentionMetadata.replace is dataclasses.replace; Ascend unpadded returns new object with many tensor views/shared tables, not deep storage snapshot. CPU mirrors may be optimistic by declared contract.',
        'DSpark set_inputs_first_pass replaces QSL/device lengths and clonesCPU lengths; prepare_inputs_padded builds a new object retaining source views. First no-spec path can pass originalcommon object, but callbacks already issued. Target/draft and post-sliding-window group coordinates must remain distinct.',
        'Current first persistent input overwrites include runner1435-1462 QSL/CPUlength and1591-1606 positions/device lengths, before _build_attention_metadata. A fence only atCPbuilder entry cannot protect globalframe source aliases already changed upstream.',
        'Local zero-query still updates global caches when cache rows exist, including C2 compressor stage wait. It skips local SMLA/QLI. Emptycache indexer may also skip wait. Waiting a consumed earlystage is not allproducer tail completion.',
        'V1 release marks consumer-side main reuse; nextproducer waits that fence. This does not order nextmain input writes after old unconsumedproducer reads. Finite DAG20cases/1033orders distinguishes deviceinputs, outputs and CPU pinned source retirement; not a runtime race/bug proof.',
        'cp_global already has globalQSL/device length/positions and C2length inputs. Newhead8 querypacket is missing, not all globalcoordinates. Original packet cost must be replaced or explicitly added, and compressed causal mask still needs full validation.',
      ],
      owner_protocol=dict(
        before_first_write='Acquire epoch/role owner before runner persistent QSL/positions/length writes; device stream must wait prior producer_done AND all actual consumer completion before alias reuse. Local-only CP buffers may use later acquire only if no global aliases.',
        host_snapshot='Use immutable private CPU/pinned snapshot until actual DMA completion observed, or fresh retained snapshot per generation. Enqueuing devicewait does not stop host overwrite. A bounded pool may wait when full; price that wait, never silently reuse busyhost storage.',
        normal_submit='Keep inputs_ready and existing COMPRESSOR/ATTENTION/INDEXER order/id frontiers. Record producer_done on metadata stream after all queued tasks; consumer_done must include joined auxiliaries, not just Python model return.',
        device_compute='Either compute common device facts once on main before eager resource readers, or move every dependent batchcopy/C2length/slot computation with a new predecessor on producer stream. Moving only commonwriter while eager _build_batch_metadata still reads it is invalid.',
        full_graph='Retain original stage/group event set and stable metadata output addresses; new packet mode requires route-specific capture/descriptor or otherwise coherent eager fallback. An extra DeviceMetadataTask frontier cannot be inserted into an already frozen descriptor unnoticed.',
        errors='Before-flight validation errors have no submitted work. After _submission_in_flight=True, task.run can raise before producer_done is recorded and before _model_forward finally exists. Retain/quarantine owners and establish terminal device completion or mark context unusable; never infer safe reuse from cleared host state.'),
      candidates=[dict(scope='Same-generation host/commondevice frame with per-resource assembly and full ownership',status='待确认',result='Large existing scoped budget still supports development. Concrete firstwrite/host-retire/tail-fence protocol now specified; native event generation, primitivebit and paidrequest cost remain.'),dict(scope='Frozen replacement primitive',status='已确认可实现',result='本轮0；finite order model only.'),dict(scope='release-only or devicewait-only pinned snapshot reuse',status='已停止',result='Source/DAG lacks required producer->mainwrite and CPU retirement edges; does not stop explicitly owned protocol.'),dict(scope='Move common producer asynchronous but leave eager dependent resource reads unchanged',status='已停止',result='Eager batchcopy/C2length/slots may read stale outputs before common task executes.')],
      remaining=['Native event/capture generation and exception/abort terminal protocol for concrete implementation','Actual alias/observer equivalence outside selectedliteral inventory and packet complete/compressed causal semantics','Perreplacement bitwise CPU/device/native packet/consumer inputs across actual ordinary/tail/empty/draft roles','Paid percaller unprofiled net and complete allrank/request critical exposure'],
      evidence=['metadata-frame-order-cpu.json','common-frame-literal-accesses.json','global-query-packet-price.json','common-metadata-delivery-price.json','metadata-reader-frontier.json'],
      errors_retained=[dict(kind='LOCAL_READER_ERROR',reason='Initial assumption frozen snapshot heldattention/utils.py raisedFileNotFoundError. Source obtained read-only fromcurrentcontainer, no native test ran.')],
      new_NPU_work=0,model_loads=0,new_profile=0,service_requests=0,remote_writes=[],new_primitive_admissions=0)
    write('common-frame-lifetime-frontier.json',result)
    print(json.dumps(dict(pins=len(sources),reviewed_source_files=sum(bool(r['reviewed_ranges']) for r in sources),literal_accesses=len(accesses),replacement_admissions=0,whole_chain_complete=False)))
if __name__=='__main__':main()
