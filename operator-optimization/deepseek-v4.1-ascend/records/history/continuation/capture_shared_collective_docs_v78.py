"""Capture two public primary API pages; no browser or remote server access."""
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter
import hashlib
from finalize_origin_coverage_v9 import O,read,write


class Extract(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)


def main():
    assert not(O/'shared-collective-API-evidence-v78.json').exists()
    docs=[]
    for name in ['ReduceScatter','AllReduce']:
        url=f'https://asc.gitcode.com/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/{name}.html'
        blob=urlopen(Request(url,headers={'User-Agent':'OperatorEvidence/1.0'}),timeout=25).read()
        p=O/f'{name}-API-source-v78.html';p.write_bytes(blob)
        parser=Extract();parser.feed(blob.decode());text=' '.join(parser.parts)
        assert 'Atlas A3'in text and'HCCL_DATA_TYPE_INT32'in text and'HCCL_REDUCE_MAX'in text and'63'in text
        docs.append(dict(name=name,url=url,snapshot_path=str(p),snapshot_sha256=hashlib.sha256(blob).hexdigest(),snapshot_bytes=len(blob),
            retrieved_UTC=datetime.now(timezone.utc).isoformat(),sections='Product support,datatype/reduceop parameters,and constraints',
            supports='Current public kernel-HCCL guide lists A3,INT32/FLOAT and SUM/MAX. It also limits total Prepare plus InterHcclGroupSync calls in one A3 communication domain to63.',
            version_limit='Not pinned to installed CANN9.2.0. Kernel-HCCL InitV2/SetCcTilingV2/submission-domain prerequisites;not an assertion about ordinary host-HCCL lifetime or current loaded binary.',
            establishes_throughput_or_installed_compatibility=False))
    ops=read(O/'communication-native-operator-population-v66.json')['operations']
    types=Counter((r['operator'],r['dtype'])for r in ops)
    assert sum(types.values())==14720 and not any('INT32'in dtype for _,dtype in types)
    write(O/'shared-collective-API-evidence-v78.json',dict(primary_sources=docs,
        saved_actual_collective_type_counts=[dict(operator=a,dtype=b,calls=n)for (a,b),n in sorted(types.items())],
        proposed_INT32_RS_observed_in_this_profile=False,proposed_MAX_reduction_kind_attested_by_typed_count_census=False,
        crosslayer_constraint=dict(domain='Only if a single kernel-HCCL submission/communication domain is retained across unrolledlayers,with the quoted current API limit applicable.',
            calls_per_layer_minimum_in_three_operation_graph=3,forty_layer_calls=120,limit=63,max_layers_without_any_other_calls=21,
            implication='Forty-layerone-domain3call/layerfusion exceeds63. Need legal separate domains/resets or other API/protocol. With extra tiled submissions/syncs,21is only an optimistic count cap.',
            not_a_limit_on_model_depth_or_ordinary_host_graph_capture=True,installed_version_applicability_proved=False),
        new_public_document_snapshots=2,new_target_source_files_obtained=0,no_browser_used=True,overall_complete=False))
    print(dict(public_documents=2,actual_collective_records=14720,proposed_INT32_RS_observed=False,conditional_kernel_domain_call_cap=63))


if __name__=='__main__':main()
