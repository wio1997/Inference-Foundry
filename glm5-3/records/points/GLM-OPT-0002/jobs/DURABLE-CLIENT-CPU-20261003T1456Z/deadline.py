import signal,time
signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(0)))
try:
 print('synthetic-start',flush=True)
 time.sleep(8)
finally:print('synthetic-finally',flush=True)
