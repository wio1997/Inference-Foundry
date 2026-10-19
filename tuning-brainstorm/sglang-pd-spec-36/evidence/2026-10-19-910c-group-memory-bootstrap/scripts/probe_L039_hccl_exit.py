#!/usr/bin/env python3
"""16rank native HCCL exit control; no import or operation from DeepEP."""
import atexit,datetime,faulthandler,importlib.metadata,json,os,pathlib,sys,threading,time,traceback
rank=int(os.environ['RANK']); assert int(os.environ['WORLD_SIZE'])==16
root=pathlib.Path('/work'); out=root/f'rank-{rank:02}.json'
v={'rank':rank,'world':16,'status':'STARTING','stage':'before_torch','NPU_compute':False,'model_loaded':False,'PD_MTP_serving':False,'performance':'NOT_MEASURED','checks':[],'events':[]}
stack=(root/f'rank-{rank:02}.python-stack.log').open('w');faulthandler.enable(stack)
def save(event):
 v['events'].append({'event':event,'host_monotonic_ns':time.monotonic_ns()});v['threads']=[{'name':t.name,'daemon':t.daemon,'alive':t.is_alive()} for t in threading.enumerate()]
 out.write_text(json.dumps(v,indent=2)+'\n')
try:
 v['packages']={n:importlib.metadata.version(n) for n in ['torch','torch-npu']};v['HCCL_BUFFSIZE']=os.environ.get('HCCL_BUFFSIZE');assert v['HCCL_BUFFSIZE']=='2048'
 import torch,torch_npu
 import torch.distributed as dist
 v['stage']='init_hccl';save('before_init')
 torch.npu.set_device(rank);v['device_name']=torch.npu.get_device_name(rank);assert v['device_name']=='Ascend910_9392'
 dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=90))
 a=torch.tensor([rank],dtype=torch.int32,device=f'npu:{rank}');dist.all_reduce(a);torch.npu.synchronize();assert a.cpu().item()==120
 v['NPU_compute']=True;v['checks'].append({'name':'16rank HCCL exact rank sum','pass':True});v['stage']='destroy_group';save('before_destroy')
 dist.destroy_process_group();v['process_group_destroyed']=True;v['DeepEP_imported']=any(n=='deep_ep' or n.startswith('deep_ep.') for n in sys.modules);assert not v['DeepEP_imported']
 v['status']='PASS';v['stage']='before_sys_exit';save('after_destroy_before_sys_exit')
except BaseException as e:
 v['status']='FAIL';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};save('failure')
def final_marker():save('python_atexit_callback')
atexit.register(final_marker)
# Leave the watchdog active: an ordinary process exit cancels it; a hung exit emits own stacks.
faulthandler.dump_traceback_later(25,repeat=True,file=stack)
print('L039_RANK_RESULT='+json.dumps({'rank':rank,'status':v['status'],'stage':v['stage']}),flush=True)
sys.exit(0 if v['status']=='PASS' else 1)
