#!/usr/bin/env python3
"""Diagnose original msgspec array extension without changing source bodies."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import array,datetime,json,pathlib,struct,sys,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L031-R005-wire-diagnosis';O.mkdir(parents=True,exist_ok=False)
try:
 import L031_frontend_components as F
 observations=[]
 for count in [0,1,4,128]:
  obj=array.array('q',range(count));ext=F.ns['enc_hook'](obj);metadata,raw=F.ns['_unpack_buffer_ext'](memoryview(ext.data));record={'kind':'array-direct','count':count,'typecode':obj.typecode,'itemsize':obj.itemsize,'ext_code':ext.code,'ext_bytes':len(ext.data),'metadata':metadata,'raw_bytes':len(raw),'prefix_hex':bytes(ext.data[:8]).hex()}
  try:out=F.ns['ext_hook'](ext.code,memoryview(ext.data));record.update(status='PASS',equal=out==obj)
  except Exception as e:record.update(status='ERROR',error_type=type(e).__name__,error=str(e))
  observations.append(record)
 for bs in [1,2,3]:
  up=F.Upstream([str(x) for x in range(bs)],128);obj=up.next();F.ns['_USE_PICKLE_IPC']=False;packet=F.ns['msgpack_encode'](obj);seen=[]
  def raw_ext(code,data):
   metadata,raw=F.ns['_unpack_buffer_ext'](data);seen.append({'code':code,'ext_bytes':len(data),'metadata':metadata,'raw_bytes':len(raw),'prefix_hex':bytes(data[:8]).hex()});return F.msgspec.msgpack.Ext(code,bytes(data))
  F.msgspec.msgpack.decode(packet,ext_hook=raw_ext);record={'kind':'whole-BatchStrOutput','batch_size':bs,'packet_bytes':len(packet),'raw_extension_records':seen}
  try:out=F.ns['msgpack_decode'](packet);record.update(status='PASS',IDs_equal=[list(x) for x in out.output_ids]==[list(x) for x in obj.output_ids])
  except Exception as e:record.update(status='ERROR',error_type=type(e).__name__,error=str(e))
  observations.append(record)
 result={'status':'OBSERVATION_COMPLETE','observations':observations,'native_executed':False,'no_source_bodies_modified':True,'msgspec_version':F.msgspec.__version__,'performance':'NOT_MEASURED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except Exception as e:(O/'result.json').write_text(json.dumps({'status':'HARNESS_ERROR','error_type':type(e).__name__,'error':str(e)},indent=2)+'\n');traceback.print_exc();sys.exit(2)
