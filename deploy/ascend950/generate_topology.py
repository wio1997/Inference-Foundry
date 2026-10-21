#!/usr/bin/env python3
"""Derive a task-local 0..7 snapshot from read-only DCMI data; keep host files intact.

Packaging adaptation: replay against historical topology is checked separately;
each new server must run device communication checks before model deployment.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runtime-20261021/topology'
WHEEL_URL = 'https://files.pythonhosted.org/packages/7c/5f/82be79f738832074140e4af4bb580200ecc5411cfc1bdfaa90d86b079c47/unofficial_ascend_tools-0.0.11-py3-none-any.whl'
WHEEL_SHA = '42420d0c4fefe27e2f5d97940f322672968deefd3952704abdbc42d639d0d1c5'


def endpoints(rank, ranks, topo):
    result, seen = [], set()
    for level in rank['level_list']:
        layer = level['net_layer']
        for address in level['rank_addr_list']:
            eid = address['addr']
            if eid in seen:
                continue
            seen.add(eid)
            plane = address.get('plane_id', 'plane_0')
            if layer >= 1 and plane in ('plane_uboe', 'roce'):
                continue
            protocol = 'ub_ctp'
            if layer >= 1:
                for edge in topo.get('edge_list', []):
                    ports = set(edge.get('local_a_ports', []) + edge.get('local_b_ports', []))
                    if edge.get('topo_type') == 'CLOS' and edge.get('net_layer') == layer and ports.intersection(address.get('ports', [])):
                        options = edge.get('protocols', [])
                        protocol = 'ub_ctp' if 'UB_CTP' in options else 'ub_tp' if 'UB_TP' in options else 'ub_ctp'
                        break
            item = {'protocol': protocol, 'comm_id': eid, 'placement': 'device'}
            if layer >= 1 and plane:
                item['plane'] = plane
            if layer == 0:
                found = None
                for edge in topo.get('edge_list', []):
                    if edge.get('topo_type') != '1DMESH' or edge.get('link_type') != 'PEER2PEER' or edge.get('net_layer') != 0:
                        continue
                    for side, other in [('a', 'b'), ('b', 'a')]:
                        if edge['local_' + side] != rank['local_id'] or not set(edge.get('local_' + side + '_ports', [])).intersection(address.get('ports', [])):
                            continue
                        peer = next((r for r in ranks if r['local_id'] == edge['local_' + other]), None)
                        if peer is None:
                            continue
                        for peer_port in edge.get('local_' + other + '_ports', []):
                            for peer_level in peer['level_list']:
                                if peer_level['net_layer'] == 0:
                                    found = next((a['addr'] for a in peer_level['rank_addr_list'] if peer_port in a.get('ports', [])), None)
                                    if found:
                                        break
                            if found:
                                break
                    if found:
                        break
                if not found:
                    raise RuntimeError('Cannot derive a local peer endpoint; inspect topology, do not invent EIDs.')
                item['dst_eid'] = found
            result.append(item)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wheel = OUTPUT / 'unofficial_ascend_tools-0.0.11-py3-none-any.whl'
    if not wheel.exists():
        with urllib.request.urlopen(WHEEL_URL, timeout=60) as reply:
            wheel.write_bytes(reply.read())
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == WHEEL_SHA
    tools = OUTPUT / 'tools'
    tools.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(tools)
    sys.path.insert(0, str(tools))
    from unofficial_ascend_tools.rankinfo.main import main as query
    raw = OUTPUT / 'dcmi-rootinfo.json'
    query(['rootinfo', '--rank_count', '8', '--max_level', '2', '--output', str(raw)])
    data = json.loads(raw.read_text())
    data['rank_list'] = [r for r in data['rank_list'] if int(r['device_id']) in range(8)]
    assert {int(r['device_id']) for r in data['rank_list']} == set(range(8))
    data['rank_count'] = 8
    topo = json.loads(Path(data['topo_file_path']).read_text())
    (OUTPUT / 'hardware-topology.json').write_text(json.dumps(topo, indent=2))
    # Runtime bind-mount layout is fixed by runtime.sh.
    data['topo_file_path'] = '/work/runtime-20261021/topology/hardware-topology.json'
    ep = OUTPUT / 'hixlep'
    ep.mkdir(exist_ok=True)
    for rank in data['rank_list']:
        out = {'version': '1.3', 'net_instance_id': next((l.get('net_instance_id') for l in rank['level_list'] if l['net_layer'] == 1), None),
               'endpoint_list': endpoints(rank, data['rank_list'], topo)}
        (ep / ('ub_endpoint_npu_' + str(rank['device_id']) + '.json')).write_text(json.dumps(out, indent=2))
    (OUTPUT / 'hccl_rootinfo.json').write_text(json.dumps(data, indent=2))
    local = json.loads(json.dumps(data))
    for rank in local['rank_list']:
        rank['level_list'] = [level for level in rank['level_list'] if level['net_layer'] == 0]
        assert rank['level_list']
    (OUTPUT / 'hccl_rootinfo-local-only.json').write_text(json.dumps(local, indent=2))
    print('TASK_LOCAL_TOPOLOGY_READY', OUTPUT)


if __name__ == '__main__':
    main()
