"""Select explicit test-request replay pairs; preserve all other rows separately."""
import collections,json

def select_formal_replays(outer_rows,observer_rows,manifest):
    request_ids={r['request_id']for r in manifest['requests']}
    if not request_ids:raise ValueError('empty formal request manifest')
    by_external=collections.defaultdict(set)
    for row in observer_rows:
        if row.get('event')=='sg_request_id_link' and row.get('external_request_id')in request_ids:
            by_external[row['external_request_id']].add(row.get('rid'))
    if set(by_external)!=request_ids or any(len(v)!=1 or None in v for v in by_external.values()):
        raise ValueError('explicit client request ID to manager RID is not unique and complete')
    formal_rids={next(iter(v))for v in by_external.values()};selected=[];excluded=[]
    for row in outer_rows:
        work=row.get('work_key');work=json.loads(work)if isinstance(work,str)else work
        rids={r.get('rid')for r in (work or {}).get('requests',[])if r.get('rid')is not None}
        if rids & formal_rids:
            if not rids<=formal_rids:raise ValueError('formal replay mixed with nonformal requests; clean control invalid')
            selected.append(row)
        else:excluded.append(row)
    pairs=collections.defaultdict(list)
    for row in selected:pairs[(row['rank_id'],row['invocation'])].append(row['side'])
    if not pairs or any(sorted(v)!=['end','start']for v in pairs.values()):raise ValueError('formal replay pairs incomplete or duplicated')
    counts=collections.Counter(rank for rank,invocation in pairs)
    if set(counts)!={0,1,2,3} or len(set(counts.values()))!=1:raise ValueError('four-rank explicit pair counts differ')
    return selected,excluded,{'status':'exact_formal_request_selection','external_to_manager_rid':{k:next(iter(v))for k,v in by_external.items()},
        'formal_rows':len(selected),'excluded_nonformal_rows':len(excluded),'formal_pairs_per_rank':dict(counts),
        'scope':'Selection uses client manifest and explicit tokenizer request-ID association, not event order, timestamps or token-count inference.'}
