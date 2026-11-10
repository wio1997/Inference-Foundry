#!/usr/bin/env python3
"""Finite partial-order models for shared frame storage, not a runtime race test."""
from pathlib import Path
from hashlib import sha256
import json

OUT=Path('audit/execution-search/metadata-frame-order-cpu.json')
def orders(nodes,edges):
    before={n:set() for n in nodes}
    for a,b in edges:before[b].add(a)
    def rec(prefix,remaining):
        if not remaining:
            yield prefix;return
        for n in sorted(remaining):
            if not (before[n]&remaining):yield from rec(prefix+[n],remaining-{n})
    return rec([],set(nodes))
def main():
    rows=[]
    for wait_qli in ['none','first','last']:
        for producer_fence in [False,True]:
            for aux_join in [False,True]:
                nodes=['write0','inputs_ready0','comp0','att0','qli_first0','qli_last0','producer_done0','model_end0','consumer_release0','write1','inputs_ready1','next_producer_write','aux_read0']
                edges=[('write0','inputs_ready0'),('inputs_ready0','comp0'),('comp0','att0'),('att0','qli_first0'),('qli_first0','qli_last0'),('qli_last0','producer_done0'),('att0','model_end0'),('model_end0','consumer_release0'),('consumer_release0','write1'),('write1','inputs_ready1'),('inputs_ready1','next_producer_write'),('producer_done0','next_producer_write'),('consumer_release0','next_producer_write'),('att0','aux_read0')]
                if wait_qli!='none':edges.append(('qli_'+wait_qli+'0','model_end0'))
                if producer_fence:edges.append(('producer_done0','write1'))
                if aux_join:edges.append(('aux_read0','consumer_release0'))
                count=input_bad=output_bad=0;example_input=example_output=None
                for order in orders(nodes,edges):
                    count+=1;pos={n:i for i,n in enumerate(order)}
                    if pos['write1']<pos['qli_last0']:
                        input_bad+=1
                        if example_input is None:example_input=order
                    if pos['next_producer_write']<pos['aux_read0']:
                        output_bad+=1
                        if example_output is None:example_output=order
                assert count>0
                assert (input_bad==0)==(producer_fence or wait_qli=='last')
                assert (output_bad==0)==aux_join
                rows.append(dict(wait_qli=wait_qli,producer_done_before_next_main_write=producer_fence,aux_consumer_join_before_release=aux_join,edges=edges,topological_orders=count,unsafe_input_overwrite_orders=input_bad,unsafe_output_overwrite_orders=output_bad,input_counterexample=example_input,output_counterexample=example_output))
    host_cases=[]
    for device_wait in [False,True]:
        for host_retire in [False,True]:
            for fresh_snapshot in [False,True]:
                nodes=['host_submit0','host_release_call','host_prepare1','cpu_write_snapshot','dma_read_old_snapshot','producer_done0','enqueue_device_wait','device_write_new']
                edges=[('host_submit0','host_release_call'),('host_release_call','host_prepare1'),('host_submit0','dma_read_old_snapshot'),('dma_read_old_snapshot','producer_done0'),('host_prepare1','cpu_write_snapshot'),('cpu_write_snapshot','enqueue_device_wait'),('enqueue_device_wait','device_write_new')]
                if device_wait:edges.append(('producer_done0','device_write_new'))
                if host_retire:edges.append(('producer_done0','cpu_write_snapshot'))
                total=bad=0;example=None
                for order in orders(nodes,edges):
                    total+=1
                    if not fresh_snapshot and order.index('cpu_write_snapshot')<order.index('dma_read_old_snapshot'):
                        bad+=1
                        if example is None:example=order
                assert (bad==0)==(host_retire or fresh_snapshot)
                host_cases.append(dict(device_wait=device_wait,host_retire_after_done=host_retire,fresh_snapshot=fresh_snapshot,edges=edges,topological_orders=total,unsafe_host_snapshot_overwrites=bad,counterexample=example))
    out=dict(status='PASS_FINITE_SOURCE_INSPIRED_ORDER_MODEL_NOT_RUNTIME_PROOF',cases=rows,host_snapshot_cases=host_cases,
      model='Main input-write/inputs-ready; serial COMPRESSOR->ATTENTION->two INDEXER tasks; consumer release on main; same producer stream and nextproducer waits release. Optional QLI consumer waits, explicit producer tail fence, auxiliary consumer join.',
      conclusions=['A nextproducer wait on prior consumer release does not order an earlier nextmainstream input write after an unconsumed oldproducer tail. Waiting firstQLI alone does not imply lastQLI done.','Producer_done before nextmainstream input write closes modeled deviceinput races; output buffer reuse still needs all auxiliary readers joined before consumerrelease.','A device stream wait does not block CPU writes to a pinned source snapshot. CPU reuse requires observed completion or a fresh retained snapshot, not merely enqueuing a devicewait.','A wait for truly last serialproducer can imply producercompletion in this model, but actual zero-query/emptycache/error branches need not issue it.'],
      limits=['Explicit finite event graph only; source effects/actual stream APIs/ExternalEvent generation/capture/error cleanup not executed.','Not an observation of an actual race, not proof of a frozen-service bug, deadlock freedom, hardware synchronization or primitive bit equality.','Early/abort submit that never records producer_done still needs an explicit terminal/quarantine protocol; modeled normal enqueue DAG does not prove it.'],
      new_NPU_work=0,new_profile=0,model_loads=0,service_requests=0,remote_writes=[],new_primitive_admissions=0,whole_chain_complete=False,probe_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(device_cases=len(rows),host_cases=len(host_cases),orders=sum(r['topological_orders'] for r in rows+host_cases),all_required_assertions_pass=True)))
if __name__=='__main__':main()
