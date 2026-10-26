"""Targeted single-device FP32 sum/div diagnosis after retained BF16 ULP failure."""
import argparse,hashlib,itertools,json,os,pathlib,subprocess,traceback
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _division_probe(X,Y,Z,N:tl.constexpr,MODE:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 a=tl.load(X+i,i<N,other=0.);b=tl.load(Y+i,i<N,other=1.)
 if MODE==0:v=tl.div_rn(a,b)
 else:v=a/b
 tl.store(Z+i,v,i<N)

def difference(a,b):
 x,y=a.cpu().contiguous(),b.cpu().contiguous();d=(x-y).abs()
 return {'mismatches':int((x!=y).sum()),'elements':x.numel(),'max_abs':float(d.max()),'max_positive_fp32_ulp':int((x.view(torch.int32).long()-y.view(torch.int32).long()).abs().max())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 c=pathlib.Path(__file__).resolve().parents[4];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c,text=True).strip()
 torch.manual_seed(20261022);torch.npu.set_device(0);x=(torch.randn(4096,4,4,device='npu:0',dtype=torch.float32)*4).softmax(-1)+1e-6
 identity={'source_commit':head,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'device':'npu:0','seed':20261022,'target':str(triton.runtime.driver.active.get_current_target()),'input_sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()}
 (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
 rows=[]
 for axis in [1,2]:
  parts=x.unbind(axis);ref=x.sum(axis)
  for order in itertools.permutations(range(4)):
   q=[parts[i] for i in order]
   for mode,z in [('left',((q[0]+q[1])+q[2])+q[3]),('balanced',(q[0]+q[1])+(q[2]+q[3]))]:
    rows.append({'axis':axis,'order':order,'mode':mode,**difference(z,ref)})
 divs=[]
 for label,xx,yy in [('column-initial',x,x.sum(-2,keepdim=True).expand_as(x).contiguous()+1e-6),('random',x,torch.rand_like(x)*4+1e-6)]:
  ref=xx/yy
  for mode in [0,1]:
   out=torch.empty_like(xx);_division_probe[(triton.cdiv(xx.numel(),256),)](xx,yy,out,xx.numel(),mode,256,enable_fp_fusion=False,num_warps=4);torch.npu.synchronize()
   divs.append({'input':label,'mode':'tl.div_rn' if mode==0 else 'operator_div',**difference(out,ref)})
 result={'identity':identity,'sum_patterns':rows,'division':divs}
 (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'exact_sum_patterns':[r for r in rows if not r['mismatches']],'best_sum_patterns':sorted(rows,key=lambda r:r['mismatches'])[:4],'division':divs}))
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
