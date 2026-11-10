/* Interval capacity model. Scenario bounds, not executable-kernel replay. */
(function(root){'use strict';
const merge=xs=>{const out=[];for(const [a,b] of xs.slice().sort((x,y)=>x[0]-y[0])){if(b<=a)continue;if(out.length&&a<=out.at(-1)[1])out.at(-1)[1]=Math.max(out.at(-1)[1],b);else out.push([a,b]);}return out;};
const intersect=(a,b)=>{const out=[];let i=0,j=0;while(i<a.length&&j<b.length){const x=Math.max(a[i][0],b[j][0]),y=Math.min(a[i][1],b[j][1]);if(y>x)out.push([x,y]);if(a[i][1]<b[j][1])i++;else j++;}return out;};
const subtract=(a,b)=>{const out=[];let j=0;for(let [x,y]of a){while(j<b.length&&b[j][1]<=x)j++;let k=j;while(k<b.length&&b[k][0]<y){if(b[k][0]>x)out.push([x,Math.min(y,b[k][0])]);x=Math.max(x,b[k][1]);if(x>=y)break;k++;}if(x<y)out.push([x,y]);}return out;};
const duration=xs=>xs.reduce((s,[a,b])=>s+b-a,0);
function take(xs,amount,tail=false){let left=Math.max(0,amount),out=[];for(const[a,b]of(tail?xs.slice().reverse():xs)){const t=Math.min(left,b-a);if(t>1e-9)out.push(tail?[b-t,b]:[a,a+t]);left-=t;if(left<=1e-9)break;}return merge(out);}
function simulate(rank,opts={}){
 const o={early:0,eliminate:0,accelerate:0,release:0,overhead:0,planner:false,buffers:false,...opts};
 for(const k of ['early','eliminate','accelerate'])if(o[k]<0||o[k]>1)throw Error('fraction outside [0,1]');
 // Quantized profiler categories may overlap at sub-us boundaries. Keep
 // communication, then Free, and exclude ambiguous edges from compute supply.
 const rawC=merge(rank.activity.Computing),Q=merge(rank.activity['Communication(Not Overlapped)']),F=subtract(merge(rank.activity.Free),Q),C=subtract(rawC,merge(F.concat(Q)));
 const prep=merge(rank.host.filter(x=>x.name==='prepare input').map(x=>[Math.max(rank.start_ms,x.start_ms),Math.min(rank.end_ms,x.end_ms)]));
 const E=intersect(F,prep),computeRemoved=C.map(([a,b])=>[b-(b-a)*o.accelerate,b]).filter(([a,b])=>b>a);
 const eliminations=[],hidden=[],carriers=[],cycles=[];let h=0;
 for(const c of rank.cycles){
  const window=[[c.start_ms,c.end_ms]],e=intersect(E,window),elim=take(e,duration(e)*o.eliminate,true),remaining=subtract(e,elim);
  const major=remaining.find(([a,b])=>b-a>=1)||remaining[0];
  const deadline=major?major[0]:c.end_ms,release=Math.min(c.end_ms,c.start_ms+Math.max(0,o.release));
  const available=subtract(intersect(C,[[release,Math.max(release,deadline)]]),merge(prep.concat(computeRemoved)));
  const eligible=intersect(remaining,[[deadline,c.end_ms]]),budget=Math.min(duration(eligible)*o.early,duration(available));
  const amount=o.planner&&o.buffers?budget:0,shift=take(eligible,amount,true),carrier=take(available,amount);
  const active=o.eliminate>0||o.accelerate>0||o.early>0,extra=active?o.overhead:0;
  eliminations.push(...elim);hidden.push(...shift);carriers.push(...carrier);h+=extra;
  cycles.push({index:c.index,start_ms:c.start_ms,end_ms:c.end_ms,release_ms:release,deadline_ms:deadline,
    exposed_prepare_ms:duration(e),eliminated_ms:duration(elim),eligible_ms:duration(eligible)*o.early,capacity_ms:duration(available),covered_ms:amount,extra_ms:extra});
 }
 const removed=merge(eliminations.concat(hidden,computeRemoved)),base=rank.end_ms-rank.start_ms,newMs=base-duration(removed)+h;
 const transform=t=>{let delta=0;for(const[a,b]of removed){if(a>=t)break;delta+=Math.max(0,Math.min(t,b)-a);}let extra=0;for(const c of cycles)if(c.end_ms<=t+1e-8)extra+=c.extra_ms;return t-rank.start_ms-delta+extra;};
 const total={base_ms:base,new_ms:newMs,saved_ms:base-newMs,speedup:base/newMs,
  free_ms:duration(F),prepare_free_ms:duration(E),compute_ms:duration(C),comm_ms:duration(Q),
  already_prepare_compute_ms:duration(intersect(C,prep)),eliminated_ms:duration(eliminations),covered_ms:duration(hidden),compute_saved_ms:duration(computeRemoved),extra_ms:h,
  remaining_free_ms:duration(F)-duration(eliminations)-duration(hidden)+h};
 return {options:o,total,cycles,removed,eliminations:merge(eliminations),hidden:merge(hidden),carriers:merge(carriers),computeRemoved:merge(computeRemoved),transform,
  activity:{Computing:subtract(C,removed),'Communication(Not Overlapped)':subtract(Q,removed),Free:subtract(F,removed)}};
}
const api={merge,intersect,subtract,duration,take,simulate};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.OverlapModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
