"""Record GPU process sharing while a benchmark PID is alive."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import time


def query(kind, fields):
    output = subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}",
                                      "--format=csv,noheader,nounits"], text=True)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(output)) if row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", type=Path, help="Finish when a resident worker completes this benchmark")
    args = parser.parse_args()
    selected = set(args.gpu_ids.split(","))
    foreign, samples = set(), 0
    with args.output.open("w") as file:
        while True:
            if args.stop_file and args.stop_file.exists():
                break
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError:
                break
            devices = {uuid: dict(index=index, memory_used_mb=memory, utilization_percent=utilization, processes=[])
                       for index, uuid, memory, utilization in query("gpu", "index,uuid,memory.used,utilization.gpu")
                       if index in selected}
            for uuid, pid, memory in query("compute-apps", "gpu_uuid,pid,used_memory"):
                if uuid in devices:
                    devices[uuid]["processes"].append(dict(pid=int(pid), memory_used_mb=memory))
                    if int(pid) != args.pid:
                        foreign.add(int(pid))
            file.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), devices=devices)) + "\n")
            file.flush()
            samples += 1
            time.sleep(5)
    summary = dict(benchmark_pid=args.pid, physical_gpu_ids=sorted(selected), snapshots=samples,
                   other_compute_pids=sorted(foreign), no_other_compute_process_observed=not foreign,
                   sampling_interval_seconds=5)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
