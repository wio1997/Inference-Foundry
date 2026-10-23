#!/usr/bin/env python3
"""Map the exported CANN RAW-based epoch axis to measured host wall time.

The affine rate comes solely from native collection metadata, never from the
diagnostic boundaries used to validate it. Original timestamps remain intact.
"""
from dataclasses import dataclass
import hashlib
import json


def nearest_ratio(value, numerator, denominator):
    assert denominator > 0
    if value < 0:
        return -nearest_ratio(-value, numerator, denominator)
    return (value * numerator + denominator // 2) // denominator


@dataclass(frozen=True)
class ClockMap:
    base_wall_ns: int
    wall_delta_ns: int
    raw_delta_ns: int
    evidence: dict

    @classmethod
    def from_profile(cls, profile):
        starts = list(profile.glob('PROF_*/host/start_info'))
        ends = list(profile.glob('PROF_*/host/end_info'))
        assert len(starts) == len(ends) == 1
        assert starts[0].parent == ends[0].parent
        start = json.loads(starts[0].read_text())
        end = json.loads(ends[0].read_text())
        base = int(start['collectionTimeBegin']) * 1000
        wall_delta = int(end['collectionTimeEnd']) * 1000 - base
        raw_delta = int(end['clockMonotonicRaw']) - int(start['clockMonotonicRaw'])
        assert wall_delta > 0 and raw_delta > 0
        evidence = dict(start_path=str(starts[0]), end_path=str(ends[0]),
                        start_sha256=hashlib.sha256(starts[0].read_bytes()).hexdigest(),
                        end_sha256=hashlib.sha256(ends[0].read_bytes()).hexdigest(),
                        start_metadata=start, end_metadata=end)
        return cls(base, wall_delta, raw_delta, evidence)

    def wall_ns(self, native_ns):
        return self.base_wall_ns + nearest_ratio(int(native_ns) - self.base_wall_ns,
                                                self.wall_delta_ns, self.raw_delta_ns)

    def native_ns(self, wall_ns):
        return self.base_wall_ns + nearest_ratio(int(wall_ns) - self.base_wall_ns,
                                                self.raw_delta_ns, self.wall_delta_ns)

    def duration_ns(self, native_duration_ns):
        return nearest_ratio(int(native_duration_ns), self.wall_delta_ns, self.raw_delta_ns)

    def as_dict(self):
        return dict(method='native collection metadata affine RAW-to-wall mapping; integer arithmetic',
                    formula='wall = begin_wall + round((exported_native - begin_wall) * wall_delta / raw_delta)',
                    base_wall_ns=self.base_wall_ns, wall_delta_ns=self.wall_delta_ns,
                    raw_delta_ns=self.raw_delta_ns,
                    rate_correction_ppm=(self.wall_delta_ns-self.raw_delta_ns)*1e6/self.raw_delta_ns,
                    evidence=self.evidence,
                    limits=['Collection wall timestamps have microsecond resolution.',
                            'Two endpoint observations do not prove absence of arbitrary nonlinear clock excursions.',
                            'Host clock validation does not by itself validate cross-device timestamp alignment or communication event boundary semantics.',
                            'Independent host/MSTX and CPU boundary checks are reported separately.'])


def audit_di_cpu_boundaries(connection, clock, spans, rank):
    """The diagnostic wrapper encloses the existing native record_function."""
    result = {}
    for label in ['schedule:cpp', 'schedule:clone', 'schedule:post']:
        native = connection.execute('''SELECT CAST(a.startNs AS INTEGER), CAST(a.endNs AS INTEGER)
            FROM PYTORCH_API a JOIN STRING_IDS s ON a.name=s.id
            WHERE s.value=? ORDER BY CAST(a.startNs AS INTEGER)''', ('deepinfer:'+label,)).fetchall()
        host = sorted((x for x in spans if x.get('dp_rank') == rank
                       and x.get('extra', {}).get('native_event_name') == label
                       and x.get('extra', {}).get('native_source') == 'deepinfer.cpu_timer.record'),
                      key=lambda x: x['wall_ns'])
        count_match = len(native) == len(host) and len(native) > 0
        failures = []
        raw_failures = 0
        margins = []
        if count_match:
            for index, ((start, end), outer) in enumerate(zip(native, host)):
                mapped_start, mapped_end = clock.wall_ns(start), clock.wall_ns(end)
                if not outer['wall_ns'] <= start <= end <= outer['wall_end_ns']:
                    raw_failures += 1
                before = mapped_start - outer['wall_ns']
                after = outer['wall_end_ns'] - mapped_end
                margins.append((before, after))
                if before < 0 or after < 0:
                    failures.append(dict(index=index, native_ns=[start,end],
                                         mapped_wall_ns=[mapped_start,mapped_end],
                                         outer_wall_ns=[outer['wall_ns'],outer['wall_end_ns']],
                                         margins_ns=[before,after]))
        result[label] = dict(native_count=len(native), host_count=len(host), count_match=count_match,
                             raw_enclosure_failures=raw_failures,
                             calibrated_enclosure_failures=len(failures), failures=failures,
                             min_start_margin_ns=min((m[0] for m in margins),default=None),
                             min_end_margin_ns=min((m[1] for m in margins),default=None),
                             passed=count_match and not failures)
    return result
