#!/usr/bin/env python3
"""Validate sanitized selected-event whole-span bins without accessing services."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'evidence/p_timeline_061/selected_p_full_span.json'
SHA='9c21d75ca1e8ee2d0a40a07b9183942545facbaeaad0b808e276810230f2fdce'
TRACES={'sglang':'92b0b3ec44036c268c23c4eb68844313e77406a2701b7ce629ae85f349f09922','vllm':'3b3dd739f60a1259511aeece38ac05a2c5cd6e54c00115611954e964fbe2f142'}


def need(v,m):
 if not v:raise ValueError(m)


def validate(d):
 need(d.get('schema')=='glm53-p-selected-events-two-independent-spans-v1','schema')
 caps=d['captures'];need(len(caps)==2 and {c['framework'] for c in caps}==set(TRACES),'engine coverage')
 for c in caps:
  need(c['rank']==0 and c['trace_sha256']==TRACES[c['framework']],'source identity')
  span=c['selected_event_span_ms'];width=c['bin_width_ms'];need(math.isfinite(span) and math.isfinite(width) and span>0 and width>=1000,'span')
  seen=set()
  for x in c['bins']:
   key=(x['bin'],x['category']);need(key not in seen and type(x['bin']) is int and x['bin']>=0 and x['bin']*width<span,'bin');seen.add(key);count=x['event_count'];value=x['duration_sum_ms_nonwallclock'];need((count is None and value is None) or (type(count) is int and count>=3 and isinstance(value,(int,float)) and math.isfinite(value) and value>=0),'small count or duration')
  need(bool(seen),'empty events')
 return True


def main():
 raw=PATH.read_bytes();need(hashlib.sha256(raw).hexdigest()==SHA,'source SHA');d=json.loads(raw);validate(d);print(json.dumps({'status':'pass','captures':len(d['captures']),'sha256':SHA}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: timeline input invalid')
