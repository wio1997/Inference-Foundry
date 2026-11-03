"""CPU-only fake ports; no Docker, HTTP, SSH, accelerator or service control."""
from dataclasses import replace
import unittest

import isolated_execution_draft as d

RUN='a'*32


def base():
    return d.Baseline(d.BASE_ID,d.IMAGE_ID,True,200,0,0,frozenset({1}),frozenset({1}),True,True,True,
                      ('vllm','serve','--frozen'),(('serve.sh','hash'),),True,True)


class Fake:
    def __init__(self, failure=None, snapshots=None, recovery=True):
        self.events=[];self.failure=failure;self.snapshots=list(snapshots or [base(),base()]);self.recovery=recovery
    def call(self, name):
        self.events.append(name)
        if self.failure==name:raise TimeoutError(name)
    def acquire_same_owner_fence(self,path):assert path==d.OWNER_LOCK;self.call('lock')
    def assert_fence_live(self):self.call('check_lock')
    def start_watchdog_and_verify_fresh_lease(self,r):self.call('watchdog')
    def baseline_snapshot(self):self.call('snapshot');return self.snapshots.pop(0)
    def prepare_new_owned_runtime_and_container(self,r,b):
        self.call('prepare')
        return d.Diagnostic('b'*64,d.IMAGE_ID,'wio-engram-'+r,r,True,b.service_argv,b.config_hashes,True,True,True,True,True)
    def save_and_reread_two_identities(self,b,x):self.call('identities')
    def verify_fresh_watchdog_lease(self,r):self.call('lease')
    def stop_exact_baseline(self,b):assert b==d.BASE_ID;self.call('stop_baseline')
    def require_empirical_release_and_recheck(self):self.call('release_check')
    def start_exact_diagnostic(self,c):assert c=='b'*64;self.call('start_diag')
    def wait_diag_health_and_recheck_owner(self,x):self.call('diag_health')
    def record_runtime_installation_and_rank_coverage_or_unknown(self,x):self.call('coverage')
    def capture_no_profile_case(self,x):self.call('capture');return {'status':'simulated'}
    def recover_and_verify_exact_baseline(self,b,x,r):self.call('recover');return self.recovery
    def watchdog_done(self,r):self.call('watchdog_done')
    def release_fence(self):self.call('unlock')


class Client:
    def __init__(self,fail=None):self.events=[];self.fail=fail
    def verify_frozen_input_identity(self,seeds,requests):return self.fail!='input'
    def seed(self,r):return {'prompt_tokens':29696,'completion_tokens':1}
    def metrics(self):return {}
    def open_gate_atomic(self,r):
        self.events.append('open')
        if self.fail=='open':raise RuntimeError('ambiguous rename')
    def close_gate_and_record(self,r):self.events.append('close')
    def complete(self,r):
        if self.fail=='request' and r==3:raise RuntimeError('request error')
        return {'prompt_tokens':32768,'completion_tokens':1024}
    def summarize_non_sensitive(self,responses,before,after):
        return {'successful_requests':8,'prompt_tokens':262144,'generation_tokens':8192,
                'prefix_queries':262144,'prefix_hits':235904,'preemptions':0}


class DraftTests(unittest.TestCase):
    def test_success_and_shared_fence(self):
        p=Fake();self.assertEqual(d.draft_run(p,RUN),{'status':'simulated'})
        self.assertLess(p.events.index('watchdog'),p.events.index('stop_baseline'))
        self.assertLess(p.events.index('lease'),p.events.index('stop_baseline'))
        self.assertLess(p.events.index('release_check'),p.events.index('start_diag'))
        self.assertEqual(p.events[-3:],['recover','watchdog_done','unlock'])
    def test_other_process_at_second_check_never_stops(self):
        p=Fake(snapshots=[base(),replace(base(),npu_host_pids=frozenset({1,2}))])
        with self.assertRaises(d.Refusal):d.draft_run(p,RUN)
        self.assertNotIn('stop_baseline',p.events);self.assertIn('recover',p.events)
    def test_frozen_config_changed_never_stops(self):
        p=Fake(snapshots=[base(),replace(base(),service_argv=('changed',))])
        with self.assertRaises(d.Refusal):d.draft_run(p,RUN)
        self.assertNotIn('stop_baseline',p.events)
    def test_stale_lease_never_stops(self):
        p=Fake(failure='lease')
        with self.assertRaises(TimeoutError):d.draft_run(p,RUN)
        self.assertNotIn('stop_baseline',p.events);self.assertIn('recover',p.events)
    def test_stop_timeout_still_recovers(self):
        p=Fake(failure='stop_baseline')
        with self.assertRaises(TimeoutError):d.draft_run(p,RUN)
        self.assertIn('recover',p.events);self.assertNotIn('start_diag',p.events)
    def test_prepare_timeout_still_recovers_by_run_identity(self):
        p=Fake(failure='prepare')
        with self.assertRaises(TimeoutError):d.draft_run(p,RUN)
        self.assertIn('recover',p.events);self.assertNotIn('stop_baseline',p.events)
    def test_failed_release_never_starts_diag(self):
        p=Fake(failure='release_check')
        with self.assertRaises(TimeoutError):d.draft_run(p,RUN)
        self.assertNotIn('start_diag',p.events);self.assertIn('recover',p.events)
    def test_no_recovery_ack_keeps_watchdog_armed(self):
        p=Fake(recovery=False)
        with self.assertRaises(d.Refusal):d.draft_run(p,RUN)
        self.assertNotIn('watchdog_done',p.events)
    def test_frozen_inputs_and_case(self):
        c=Client();result=d.draft_case(c,RUN,list(range(8)),list(range(8)))
        self.assertEqual(result['prefix_hits'],235904);self.assertEqual(c.events,['open','close'])
    def test_gate_closes_on_ambiguous_open_or_request_failure(self):
        for failure in ('open','request'):
            c=Client(failure)
            with self.assertRaises(RuntimeError):d.draft_case(c,RUN,list(range(8)),list(range(8)))
            self.assertEqual(c.events,['open','close'])
    def test_wrong_frozen_input_never_opens_gate(self):
        c=Client('input')
        with self.assertRaises(d.Refusal):d.draft_case(c,RUN,list(range(8)),list(range(8)))
        self.assertEqual(c.events,[])


if __name__=='__main__':unittest.main()
