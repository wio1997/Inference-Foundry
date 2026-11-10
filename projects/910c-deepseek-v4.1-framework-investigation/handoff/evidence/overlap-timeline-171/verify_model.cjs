const assert=require('node:assert/strict'),fs=require('node:fs');
const M=require('../balanced-profile-171/overlap-model.js'),D=JSON.parse(fs.readFileSync(__dirname+'/view-data.json'));
let checks=0;
for(const r of D.representatives){
 const base=M.simulate(r);assert(Math.abs(base.total.saved_ms)<1e-8);checks++;
 const blocked=M.simulate(r,{early:1});assert.equal(blocked.total.covered_ms,0);checks++;
 const unsafe=M.simulate(r,{early:1,planner:true});assert.equal(unsafe.total.covered_ms,0);checks++;
 const pass=M.simulate(r,{early:1,planner:true,buffers:true});assert(pass.total.covered_ms>0);assert(pass.total.covered_ms<=pass.total.prepare_free_ms+1e-7);checks++;
 const late=M.simulate(r,{early:1,planner:true,buffers:true,release:1000});assert.equal(late.total.covered_ms,0);checks++;
 const allDelete=M.simulate(r,{eliminate:1,early:1,planner:true,buffers:true});assert(allDelete.total.covered_ms<1e-6);assert(Math.abs(allDelete.total.eliminated_ms-allDelete.total.prepare_free_ms)<1e-6);checks++;
 const both=M.simulate(r,{early:1,accelerate:.5,planner:true,buffers:true});assert(both.total.covered_ms<=pass.total.covered_ms+1e-6);checks++;
 for(const opts of [{early:.5,planner:true,buffers:true,overhead:.2},{eliminate:.25,early:.5,accelerate:.25,planner:true,buffers:true,overhead:.2}]){
  const z=M.simulate(r,opts);assert(Math.abs(z.total.saved_ms-(z.total.eliminated_ms+z.total.covered_ms+z.total.compute_saved_ms-z.total.extra_ms))<1e-5);
  assert(Math.abs(z.transform(r.end_ms)-z.total.new_ms)<1e-5);
  assert(M.duration(M.intersect(z.eliminations,z.hidden))<1e-7);
  assert(M.duration(M.intersect(z.carriers,r.activity.Free))<1e-4);
  checks+=4;
 }
}
assert.equal(D.candidates.length,73);assert.equal(new Set(D.candidates.map(x=>x.id)).size,73);assert(D.candidates.every(r=>r.modules.length&&r.modules.every(m=>D.modules.some(x=>x.id===m))));checks+=3;
console.log(JSON.stringify({passed:true,checks,candidates:73,actual_capture_changed:false}));
