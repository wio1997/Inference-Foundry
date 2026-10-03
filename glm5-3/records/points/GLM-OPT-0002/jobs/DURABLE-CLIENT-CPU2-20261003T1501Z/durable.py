import sys,time
print('synthetic-BEGIN',flush=True)
print('synthetic-ERR',file=sys.stderr,flush=True)
time.sleep(.7)
print('synthetic-END',flush=True)
