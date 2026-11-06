"""Isolated V5.4 worker; startup errors also produce authenticated receipts."""
from pathlib import Path
import os,sys,traceback
from wengua_course_v54 import runtime
from wengua_course import data
if __name__=='__main__':
 request=Path(sys.argv[1]).resolve()
 try:
  runtime.configure()
  from wengua_course_v54.process import worker
  code=worker(request)
 except BaseException as error:
  traceback.print_exc();code=130 if isinstance(error,KeyboardInterrupt) else 1
  if request.is_relative_to((runtime.ROOT/'local').resolve()) and request.parent.name=='worker_calls' and request.is_file():
   payload=data.read(request);data.write(request.with_suffix('.result.json'),{'schema':'wengua_source_course_v54_process','nonce':payload.get('nonce'),'request_sha256':data.digest(request.read_bytes()),'pid':os.getpid(),'ok':False,'error':{'type':type(error).__name__,'message':str(error),'cuda_oom':False}})
 raise SystemExit(code)
