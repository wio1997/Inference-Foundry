"""CPU-only analysis on the owned remote four-core client; not a model workload."""
from pathlib import Path
import datetime,hashlib,json,math,os,sys,time
import torch
from safetensors.torch import load_file
assert not torch.cuda.is_initialized();torch.set_num_threads(1)
N=Path(sys.argv[1]);S=Path(sys.argv[2]);OUT=Path(sys.argv[3]);assert not OUT.exists();OUT.mkdir();start=time.monotonic();records=[]
def metric(a,b):
 assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
 af=a.float();bf=b.float();d=bf-af;norm=torch.linalg.vector_norm(af);cos=torch.nn.functional.cosine_similarity(af.flatten(),bf.flatten(),dim=0)
 return {'elements':a.numel(),'numeric_equal':int((a==b).sum()),'max_abs':float(d.abs().max()),'RMSE':float(d.square().mean().sqrt()),'relative_L2':float(torch.linalg.vector_norm(d)/norm) if norm else None,'cosine':float(cos),'native_L2':float(norm),'SG_L2':float(torch.linalg.vector_norm(bf))}
for layer in range(40):
 np=N/f'rank0_layer{layer:02d}.safetensors';sp=S/f'rank0_layer{layer:02d}.safetensors';nm=json.loads(np.with_suffix('.json').read_text());sm=json.loads(sp.with_suffix('.json').read_text());assert nm['input_ids']==sm['input_ids'] and nm['positions']==sm['positions']==list(range(30));assert hashlib.sha256(np.read_bytes()).hexdigest()==nm['file_SHA256'] and hashlib.sha256(sp.read_bytes()).hexdigest()==sm['file_SHA256']
 n=load_file(str(np),device='cpu');s=load_file(str(sp),device='cpu');assert torch.equal(n['positions'],s['positions'])
 row={'layer':layer,'input_ids_equal':True,'positions_equal':True,'native_file_SHA256':nm['file_SHA256'],'SG_file_SHA256':sm['file_SHA256'],'native_input_shape':nm['full_input_shape'],'SG_input_shape':sm['full_input_shape'],'native_TP_partial':nm['output_is_TP_partial'],'SG_TP_partial':sm['output_is_TP_partial'],'metrics':{}}
 for name in ['input','output']:
  if name=='output' and (nm['output_is_TP_partial'] or sm['output_is_TP_partial']):row['output_skipped']='TP partial needs explicit rank reduction before crossframework comparison';continue
  assert n[name].shape==s[name].shape==(30,5120)
  row['metrics'][name]={'all_real_rows':metric(n[name],s[name]),'BOS_row0':metric(n[name][0],s[name][0]),'last_real_row29':metric(n[name][-1],s[name][-1]),'row_relative_L2':[metric(n[name][i],s[name][i])['relative_L2'] for i in range(30)]}
 records.append(row)
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'torch':torch.__version__,'CPU_only':True,'NPU_initialized':torch.cuda.is_initialized(),'actual_layers':40,'compared_rank':0,'rows':records,'scope':'Same30teacherforcedtokens/positions, realrows; actualattentionnormalizedinput/projectedoutput. Rank0 numericalcomparison; other ranks requireintra-rank exactSHAchain. Environment/indexquant/CANN/batchpadding differences remain; no solecause, fullqualityorperformance acceptance.'};(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'result':str(OUT/'result.json'),'SHA256':hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),'layers':40,'elapsed_s':result['elapsed_s'],'first4layers_lastreal':[{k:r[k] for k in ['layer','metrics']} for r in records[:4]]}))
