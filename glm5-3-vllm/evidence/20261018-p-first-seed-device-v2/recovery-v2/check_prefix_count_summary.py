from pathlib import Path
import copy,importlib.util,json
p=Path(__file__).with_name('summarize_prefix_count_fragment.py');spec=importlib.util.spec_from_file_location('parser',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture():
 cmp={'bitwise_equal':True,'different_elements':0,'max_abs':0.0}
 return {'completed':True,'rows':[{'N':n,'pattern':pattern,'eager_native_vs_prefix':dict(cmp),'routing_histogram_after_consumers_correct':True,'captured':[{'iteration':i//4,'mode':['native','prefix','prefix','native'][i%4],'comparison_to_current_native_eager':dict(cmp),'recovered_histogram_correct':True} for i in range(12)]} for n in [6,24] for pattern in ['scatter','head','tail','mixed']]}
a=fixture();assert m.summarize_prefix_count_fragment(a)['replays']==96;checks=['positive_real_schema_8cases_96replays']
a=fixture();a['rows'][0]['eager_native_vs_prefix']['max_abs']=0;assert m.summarize_prefix_count_fragment(a)['passed'];checks.append('integer_zero_is_finite_real')
def bad(name,mutator):
 x=fixture();x=mutator(x)
 try:m.summarize_prefix_count_fragment(x)
 except ValueError:checks.append(name);return
 raise AssertionError('accepted '+name)
def setrow(x,k,v):x['rows'][0][k]=v;return x
def setrec(x,k,v):x['rows'][0]['captured'][0][k]=v;return x
def setcmp(x,k,v):x['rows'][0]['eager_native_vs_prefix'][k]=v;return x
bad('not_mapping',lambda x:[]);bad('missing_completed',lambda x:{'rows':x['rows']});bad('completed_integer',lambda x:dict(x,completed=1));bad('seven_cases',lambda x:dict(x,rows=x['rows'][:7]));bad('N_bool',lambda x:setrow(x,'N',True));bad('N_float',lambda x:setrow(x,'N',6.0));bad('N_outside',lambda x:setrow(x,'N',12));bad('pattern_nonstring',lambda x:setrow(x,'pattern',[]));bad('duplicate_valid_pair',lambda x:setrow(x,'pattern','head'));bad('bitwise_integer',lambda x:setcmp(x,'bitwise_equal',1));bad('difference_bool',lambda x:setcmp(x,'different_elements',False));bad('difference_float',lambda x:setcmp(x,'different_elements',0.0));bad('different_payload',lambda x:setcmp(x,'different_elements',1));bad('max_abs_bool',lambda x:setcmp(x,'max_abs',False));bad('max_abs_nan',lambda x:setcmp(x,'max_abs',float('nan')));bad('max_abs_inf',lambda x:setcmp(x,'max_abs',float('inf')));bad('hist_unchecked',lambda x:setrow(x,'routing_histogram_after_consumers_correct',1));bad('partial_replay',lambda x:setrow(x,'captured',x['rows'][0]['captured'][:-1]));bad('iteration_float',lambda x:setrec(x,'iteration',0.0));bad('iteration_bool',lambda x:setrec(x,'iteration',False));bad('iteration_group_wrong',lambda x:setrec(x,'iteration',1));bad('replay_order_wrong',lambda x:setrec(x,'mode','prefix'));bad('replay_comparison_missing',lambda x:setrec(x,'comparison_to_current_native_eager',None));bad('replay_hist_not_proved',lambda x:setrec(x,'recovered_histogram_correct',False))
print(json.dumps({'completed':True,'passed':len(checks),'checks':checks,'scope':'Synthetic parser schema checks, reviewed ZCode proposal; no new device report exists yet.'}))
