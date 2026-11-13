"""Host evidence-boundary checks only; no fabricated model experiment."""
import unittest
import hashlib
from analyze_cohort_c4 import position,validate_clocks,summarize_journal,formal_request_ids


def event(a,b,**kw):return dict(started_monotonic_ns=a,host_return_monotonic_ns=b,**kw)


class PhaseContracts(unittest.TestCase):
    def test_formal_boundary_partition_is_exhaustive(self):
        intervals=[(0,9),(10,20),(21,30),(5,15),(15,25),(5,25)]
        self.assertEqual([position(event(a,b),10,20) for a,b in intervals],
                         ['before','inside','after','crosses_start','crosses_end','crosses_both'])

    def test_clock_mismatch_and_unknown_identity_rejected(self):
        clock={'boot_id_sha256':'a'*64,'timens_offsets_sha256':'b'*64,
               'monotonic_implementation':'clock_gettime(CLOCK_MONOTONIC)',
               'perf_counter_implementation':'clock_gettime(CLOCK_MONOTONIC)'}
        validate_clocks(clock,clock)
        with self.assertRaises(ValueError):validate_clocks(clock,{**clock,'timens_offsets_sha256':'c'*64})
        with self.assertRaises(ValueError):validate_clocks({},clock)

    def test_warmup_and_boundary_calls_not_added_to_formal_counts(self):
        target=lambda a,b:event(a,b,mode='FULL',kind='decode',requests=4,scheduled_tokens=24,request_id_sha256=['a'*64])
        snap={'closed':True,'counts':{'target_invocations':3,'target_graph_wrapper_calls':1},
              'target_events':[target(0,5),target(11,12),target(19,21)],
              'graph_events':[event(11,12,scope='target',actual_replay=True,request_id_sha256=['a'*64])]}
        q=summarize_journal(snap,10,20,{'a'*64},{'a'*64})
        self.assertEqual(q['target_modes_inside'],{'FULL':1})
        self.assertEqual(q['window_partitions']['target'],{'before':1,'inside':1,'crosses_end':1})
        self.assertEqual(q['graph_routes_inside'],{'target/replay':1})

    def test_partial_or_dropped_journal_rejected(self):
        snap={'closed':True,'counts':{'target_invocations':1,'target_events_dropped':1},'target_events':[],'graph_events':[]}
        with self.assertRaises(ValueError):summarize_journal(snap,0,10,{'a'*64},{'a'*64})
        snap['counts'].pop('target_events_dropped')
        with self.assertRaises(ValueError):summarize_journal(snap,0,10,{'a'*64},{'a'*64})

    def test_delayed_warmup_inside_window_excluded_by_identity(self):
        records=[dict(request_id=str(i),input_tokens=1024,generated_token_ids=[0]*256,
                      driver_owner_released=True,disconnected=False) for i in range(38)]
        ids=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(38)]
        target=lambda a,b,keys:event(a,b,mode='FULL',kind='decode',requests=len(keys),scheduled_tokens=6*len(keys),request_id_sha256=keys)
        targets=[target(0,5,[rid]) for rid in ids[:6]]+[target(11,12,[rid]) for rid in ids[6:]]
        targets+=[target(13,14,[ids[0]]),target(15,16,[ids[0],ids[6]])]
        snap={'closed':True,'counts':{'target_invocations':len(targets)},'target_events':targets,'graph_events':[]}
        formal,all_ids=formal_request_ids(snap,list(reversed(records)),10,20)
        q=summarize_journal(snap,10,20,formal,all_ids)
        self.assertEqual(q['target_modes_inside'],{'FULL':32})
        self.assertEqual(q['inside_window_request_ownership']['target'],
                         {'formal_only':32,'nonformal_only':1,'mixed_formal_nonformal':1})
        # If first warmup work is delayed into the formal window, do not silently label it formal.
        snap['target_events'][0]=target(11,12,[ids[0]])
        with self.assertRaises(ValueError):formal_request_ids(snap,records,10,20)


if __name__=='__main__':unittest.main()
