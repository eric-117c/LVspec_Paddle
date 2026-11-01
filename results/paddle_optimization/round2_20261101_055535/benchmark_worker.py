import hashlib, importlib, json, os, pathlib, subprocess, sys, time, traceback
from datetime import datetime,timezone
ROOT=pathlib.Path('/home/wjy/workFiles/LVSpec');os.chdir(ROOT);sys.path.insert(0,str(ROOT))
OUT=pathlib.Path('/tmp/lvspec_round2_output.txt').read_text().strip();out=pathlib.Path(OUT)
os.environ['CUDA_VISIBLE_DEVICES']='1,3';os.environ['OMP_NUM_THREADS']='2'
metadata=dict(status='loading',pid=os.getpid(),physical_gpu_ids=[1,3],started_utc=datetime.now(timezone.utc).isoformat())
def status(**values):
 metadata.update(values);(out/'worker_status.json').write_text(json.dumps(metadata,indent=2)+'\n')
status()
from paddle_backend import checkpoint
models=checkpoint.load_model_pair('/home/wjy/models/Qwen2.5-VL-32B-Instruct','/home/wjy/models/Qwen2.5-VL-7B-Instruct',dtype='float16',target_devices=['gpu:0'],draft_devices=['gpu:1'])
checkpoint.load_model_pair=lambda *args,**kwargs: models
from paddle_backend.optimization import pack_draft_projections
pack_draft_projections(models[1])
from tools import benchmark_paddle
from paddle_backend import decoding, draft_graph
new_generate,new_graph=decoding.generate,draft_graph.DraftGraph
checkpoint_sources=[pathlib.Path('tools/benchmark_paddle.py'),*sorted(pathlib.Path('paddle_backend').glob('*.py'))]
(out/'source_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_sources},indent=2)+'\n')
(out/'resident_models.json').write_text(json.dumps({'target':'/home/wjy/models/Qwen2.5-VL-32B-Instruct','draft':'/home/wjy/models/Qwen2.5-VL-7B-Instruct','gpu_ids':'1,3','packing':'applied once after loading; benchmark packing function becomes a no-op'},indent=2)+'\n')
import paddle_backend.optimization as optimization
optimization.pack_draft_projections=lambda model:model
status(status='ready')
first=dict(name='five_cases',indices=list(range(5)),tokens=512,repeats=2,variant='new')
(out/'job_000.json').write_text(json.dumps(first))
job_index=0
while not (out/'stop').exists():
 job_path=out/f'job_{job_index:03}.json'
 if not job_path.exists():time.sleep(1);continue
 job=json.loads(job_path.read_text());directory=out/job['name'];directory.mkdir(exist_ok=True)
 status(status='benchmark_running',job=job['name'])
 if job.get('variant','new')=='previous':
  import importlib.util
  loaded=[]
  for name in ['draft_graph','decoding']:
   spec=importlib.util.spec_from_file_location('paddle_backend.previous_'+name,out/f'previous_{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);loaded.append(module)
  draft_graph.DraftGraph=loaded[0].DraftGraph;decoding.generate=loaded[1].generate
 else:draft_graph.DraftGraph=new_graph;decoding.generate=new_generate
 command=['tools/benchmark_paddle.py','--target','/home/wjy/models/Qwen2.5-VL-32B-Instruct','--draft','/home/wjy/models/Qwen2.5-VL-7B-Instruct','--inputs-npz',*[f'results/paddle_validation/ar_vs_lvspec_5cases/inputs_{i}.npz' for i in job['indices']],'--gpu-ids','1,3','--target-devices','0','--draft-devices','1','--dtype','float16','--max-new-tokens',str(job.get('tokens',512)),'--repeats',str(job.get('repeats',2)),'--methods','ar','lvspec','--interleave-methods','--pack-draft-projections','--output',str(directory)]
 (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n');(directory/'variant.json').write_text(json.dumps(job,indent=2)+'\n')
 monitor=subprocess.Popen([sys.executable,'tools/monitor_benchmark_gpus.py','--pid',str(os.getpid()),'--gpu-ids','1,3','--output',str(directory/'gpu_monitor.jsonl'),'--stop-file',str(directory/'finished')],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
 try:
  sys.argv=command
  benchmark_paddle.main()
  state={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat()}
 except Exception:
  traceback.print_exc();state={'status':'failed','completed_utc':datetime.now(timezone.utc).isoformat()}
 finally:
  (directory/'finished').touch();monitor.wait(timeout=15)
 (directory/'status.json').write_text(json.dumps(state,indent=2)+'\n');status(status='ready',last_job=job['name'],last_job_status=state['status']);job_index+=1
 print('JOB FINISHED',job['name'],state,flush=True)
status(status='stopped',stopped_utc=datetime.now(timezone.utc).isoformat())
