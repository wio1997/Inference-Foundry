#!/usr/bin/env python3
"""Compare native replicas, real D cache and two independent post-D SDK destinations."""
import argparse
import hashlib
import json
from pathlib import Path
from analyse_post_consume_controls import archive, ranked, select


def private(rows, request_id):
    values=[r for r in rows if r['phase']=='pool_readback' and r['request_id'].startswith(request_id+'-')]
    assert len(values)==31 and all(r['tp_rank']==0 and r['sdk_result']==[0] for r in values)
    assert len({r['key'] for r in values})==31
    return {r['key']:r for r in values}


def sentinel(control, size):
    # sum(x)==128*n and sum(x*x)==128^2*n imply variance zero for unsigned bytes.
    return control[0]==128*size and control[3]==128*128*size


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--run-id',default='post-pool-state-control-v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    source,layouts,tests=archive(args.source,args.snapshot)
    target,_,_=archive(args.target,args.snapshot)
    run=tests[f'tests/{args.run_id}/result.json'];assert run['completed']
    native=ranked(select(target,run['D_native_control_id']))
    summaries=[];details=[]
    for wave in run['waves']:
        p=ranked(select(source,wave['P_native_id']))
        d=ranked(select(target,wave['D_local_id']))
        pp=private(source,wave['P_native_id']);dp=private(target,wave['D_local_id'])
        keys=[r['key'] for r in p[0]]
        assert set(pp)==set(dp)==set(keys)
        summary={'wave':wave['wave'],'output_matches_P_native':wave['pool_matches_P_native'],
                 'private_P_sdk_get_keys':31,'private_D_sdk_get_keys':31,'blocks':[]}
        for block,key in enumerate(keys):
            sr=p[0][block];dr=d[0][block];pr=pp[key];qr=dp[key]
            assert key==dr['key'] and sr['sizes']==dr['sizes']==pr['sizes']==qr['sizes']
            ps={json.dumps(r[block]['checksums']) for r in p.values()}
            ds={json.dumps(r[block]['checksums']) for r in d.values()}
            a,b,c,f=sr['checksums'],dr['checksums'],pr['checksums'],qr['checksums']
            if len(ps)!=1:classification='native_P_replicas_differ'
            elif len(ds)!=1:classification='real_D_replicas_differ'
            elif a==b==c==f:classification='all_match'
            elif c!=f:classification='private_read_destinations_or_times_disagree'
            elif c==f==b and a!=b:classification='common_pool_read_and_real_D_differ_from_P'
            elif c==f==a and b!=a:classification='earlier_D_differs_from_later_pool_reads'
            elif c==f and a!=c:classification='common_pool_read_differs_from_P_mixed_D'
            else:classification='mixed'
            changed=[]
            for i,(sc,dc,pc,qc,nc,size) in enumerate(zip(a,b,c,f,native[0][block]['checksums'],sr['sizes'])):
                if sc==dc==pc==qc:continue
                changed.append({'entry':i,'bytes':size,'P_native':sc,'real_D':dc,'private_P_pool':pc,'private_D_pool':qc,
                                'D_native':nc,'P_matches_D_native':sc==nc,
                                'real_D_zero':dc[0]==0,'private_P_zero':pc[0]==0,'private_D_zero':qc[0]==0,
                                'private_P_unexpected_sentinel':pc!=sc and sentinel(pc,size),
                                'private_D_unexpected_sentinel':qc!=sc and sentinel(qc,size),
                                'labels':layouts[0]['groups']['0'][i]['labels']})
            row={'block':block,'classification':classification,'P_replicas_identical':len(ps)==1,'D_replicas_identical':len(ds)==1,
                 'P_divergent_ranks':[rank for rank,r in p.items() if r[block]['checksums']!=a],
                 'D_divergent_ranks':[rank for rank,r in d.items() if r[block]['checksums']!=b],
                 'changed_entries':len(changed),'real_D_zero_changed_bytes':sum(x['bytes'] for x in changed if x['real_D_zero']),
                 'private_P_zero_changed_bytes':sum(x['bytes'] for x in changed if x['private_P_zero']),
                 'private_D_zero_changed_bytes':sum(x['bytes'] for x in changed if x['private_D_zero']),
                 'private_P_unexpected_sentinel_entries':[x['entry'] for x in changed if x['private_P_unexpected_sentinel']],
                 'private_D_unexpected_sentinel_entries':[x['entry'] for x in changed if x['private_D_unexpected_sentinel']]}
            summary['blocks'].append(row)
            if changed or len(ps)!=1 or len(ds)!=1:details.append({'wave':wave['wave'],'block':block,'key':key,'classification':classification,'entries':changed,'P_divergent_controls':{str(rank):r[block]['checksums'] for rank,r in p.items() if r[block]['checksums']!=a},'D_divergent_controls':{str(rank):r[block]['checksums'] for rank,r in d.items() if r[block]['checksums']!=b}})
        summary['all_views_match']=all(r['classification']=='all_match' for r in summary['blocks'])
        summaries.append(summary)
    result={'completed':True,'waves':summaries,'differences':details,
            'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.source,args.target]},
            'scope':'Only post-D128 observations. Independent private buffers initialized0x80, one SDK register/unregister per request, existing device_sdma GET. Native115-entry layout retained; four moments are not cryptographic identity. A later correct pool view cannot exclude late publication, earlier GET or target mutation. Common bad pool reads cannot alone distinguish stored DRAM bytes from common SDK read-source errors. No throughput inference or cross-host clock subtraction.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'waves':[{'wave':w['wave'],'output_matches':w['output_matches_P_native'],'all_views_match':w['all_views_match'],'bad_blocks':[r for r in w['blocks'] if r['classification']!='all_match']} for w in summaries]},indent=2))


if __name__=='__main__':main()
