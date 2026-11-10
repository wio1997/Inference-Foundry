#!/usr/bin/env python3
"""Finite/symbolic row-owner transport and resource accounting, no GEMM/NPU."""
from pathlib import Path
from hashlib import sha256
from array import array
import math,json
ROOT=Path('audit/execution-search');N=25600;K=6144;P=8
def partition(tile):
 assert N%tile==0
 q,r=divmod(N//tile,P);widths=[(q+(i<r))*tile for i in range(P)];starts=[];s=0
 for w in widths:starts.append(s);s+=w
 assert s==N
 return widths,starts,max(widths)
def pattern(row,col):return (row*257+col*17)&65535
def byte_oracle(carrier,tile):
 m=math.ceil(carrier/P);widths,starts,common=partition(tile);sends=[]
 for w,a in zip(widths,starts):
  buf=array('H')
  for row in range(P*m):
   buf.extend(pattern(row,col) for col in range(a,a+w));buf.extend([0xDEAD]*(common-w))
  sends.append(buf)
 checked=0;words=set()
 for owner in range(P):
  restored=array('H',[0])*(m*N)
  for src,(w,a) in enumerate(zip(widths,starts)):
   chunk=sends[src][owner*m*common:(owner+1)*m*common]
   for r in range(m):restored[r*N+a:r*N+a+w]=chunk[r*common:r*common+w]
  for r in range(m):
   for col in range(N):
    value=restored[r*N+col];assert value==pattern(owner*m+r,col);words.add(value);checked+=1
 return dict(carrier=carrier,padded_SP_rows=P*m,N_tile=tile,columns_by_source=widths,fixed_transport_columns=common,checked_original_output_words=checked,distinct_U16_patterns=len(words),bitword_mismatches=0,qualification='Fake original WKV output words inclpaddedrows, not numeric GEMM or native transport. U16 patterns include nonfinite encodings; no float arithmetic performed.')
def main():
 hwpath=ROOT/'o-projection-resource-ownership.json';hw=json.loads(hwpath.read_text())['hardware_reference'];assert hw['L0C_bytes']==131072
 cases=[]
 for carrier in [6,24,3328,3352,6528,6536,6540,6544]:
  m=math.ceil(carrier/P);mpad=math.ceil(m/16)*16
  for tile in [16,32,64,128,256,512,1024]:
   widths,starts,common=partition(tile);assert sum(widths)==N
   for a,w in zip(starts,widths):assert a%tile==w%tile==0
   assert all(a+w==b for a,w,b in zip(starts,widths,starts[1:]))
   original_padMAC=mpad*K*N;new_sumMAC=sum(P*mpad*K*w for w in widths);assert new_sumMAC==P*original_padMAC
   c0=P*16*tile*4
   cases.append(dict(global_source_rows=carrier,SP_rows_each=m,SP_padded_global=P*m,source_zero_pad_rows=P*m-carrier,original_rank_M_padded_reference=mpad,N_tile=tile,exact_N_widths=widths,fixed_transport_N_width=common,ragged_N_tiles_required=common!=min(widths),fixed_alltoall_N_padding_fraction=(P*common-N)/N,original_padMAC_per_rank=original_padMAC,new_padMAC_across_ranks=new_sumMAC,new_padMAC_total_ratio=1.0,original_rank_logical_weight_operand_bytes=K*N*2,rank_max_Nshard_weight_bytes=K*common*2,if_relaunch_whole_original_kernel_per_panel_max_weight_operand_bytes=P*K*common*2,if_Kouter_share_B_once_max_weight_operand_bytes=K*common*2,FP32_C0_single_buffer_eight_M16_panels_bytes=c0,FP32_C0_double_buffer_bytes=2*c0,eight_panels_single_buffer_fits_reference_L0C=c0<=hw['L0C_bytes'],rank_max_fixed_transport_bytes=P*m*common*2,rank_remote_fixed_transport_bytes=(P-1)*m*common*2,original_headAG_receive_bytes_reference=(P-1)*m*K*2))
 oracles=[byte_oracle(carrier,tile) for carrier,tile in [(24,128),(24,256),(6,128),(6,512)]]
 # A pure IEEE singleprecision formula witness; not original WKV accumulation identification.
 import struct
 f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
 left=f32(f32(2**24+1)-2**24);reassociated=f32(f32(2**24-2**24)+1);assert left==0 and reassociated==1
 result=dict(status='ENGRAM_ROW_N_OWNER_ADDRESS_AND_RESOURCE_SCREEN',geometry_cases=cases,word_oracles=oracles,
  rounded_sum_counterexample=dict(original_candidate_leftfold=left,new_partial_association=reassociated,terms=[2**24,1,-2**24],scope='Finite FP32 formula disproves unconditional splitK reassociation, not an observed WKV/device result or originalnativeK-tree proof.'),
  findings=['Original8 SP rowgroups and Npartition preserve totaluseful/paddedMAC only if originalrowpanels retained; no8xcomputecredit. M16 padding is an illustration, not activeoriginalMtile.','ForNtile256/512/1024, equal3200columns cutoriginalNtiles. Ragged ownerwidths or paddedfixed AllToAll cost must be paid. Ntile>=512 and eightFP32M16 accumulators exceed128KiBL0C in this oneAICschedule; smallerMgroups/reloads are distinct schedules.','Repeated wholeoriginalkernel perrowpanel rereadsB eighttimes, potentiallyerasing8xlogicalweight-ingresscredit. Kouter-eightpanels reusesB but needs originalK/Fixpipe plus accumulator/flag feasibility.','Currentfullweights remain until newloading/placement changes. LogicalNshard bytes are not physicalVRAM release or achievedHBMtraffic.','Existing globalfullK bank retained for Nowner, so existingheadAG is not removed. DirectSPowner insteadneeds a newglobalrowrestore or an unproven splitK reduction.','All originalrowpayload including paddedrows is restored before unchangedgate; no maskfalse passthrough, inactive-rank collective skip or zeroassumption made.'],
  limits=['56geometrycases and4finitewordoracles, not all legalbranch/nativeparameter coverage. No GEMM, FP32rotation, graphcapture or HCCL executed.','NewTP AllToAll ordering, correct tensorworkspace/address/owner generations and globalbank lastreader required; oldEDPexchange helper doesnot grant thisnewTPpath.','OriginalactiveMatMulV2 K/M/N/Fixpipe and exactprimitivebits stillunknown. No nativeprototype justified for alreadybelow-budget independentchildren.'],
  inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [hwpath,ROOT/'engram-joint-entry.json']],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'engram-row-owner-cpu.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(geometry_cases=len(cases),oracles=oracles,reference_smallN128=next(c for c in cases if c['global_source_rows']==24 and c['N_tile']==128)),ensure_ascii=False))
if __name__=='__main__':main()
