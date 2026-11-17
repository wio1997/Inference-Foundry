"""Small independent CPU specification; no torch/DeepEP dependency."""
WORLD=8;EXPERTS=384;LOCAL_EXPERTS=48;TOPK=6;HIDDEN=5120

def route(source,t):
 return [((source+k)%WORLD)*LOCAL_EXPERTS+((t*5+source*7+k*3)%LOCAL_EXPERTS) for k in range(TOPK)]
def weights(source,t):return [(k+1+(source+t)%3)/128.0 for k in range(TOPK)]
def exponent(source,t):return -3-((source+2*t)%4)
def qrow(source,t,hidden=HIDDEN):
 uid=source*32+t;row=[(i*13+uid*7)%255-127 for i in range(hidden)]
 row[:4]=[source+1,t-16,127,-127];return row
def expected_counts(destination,tokens):
 c=[0]*LOCAL_EXPERTS
 for source in range(WORLD):
  for t in range(tokens):
   for e in route(source,t):
    if e//LOCAL_EXPERTS==destination:c[e%LOCAL_EXPERTS]+=1
 return c

def expected_payload_expert(source,t,destination):
 es=[e for e in route(source,t) if e//LOCAL_EXPERTS==destination];assert len(es)==1;return es[0]
