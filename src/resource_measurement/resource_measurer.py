import time
import psutil
from pydantic import BaseModel
import tracemalloc

class ResourcesUsage(BaseModel):
    cpu_time: float  # CPU time consumed
    peak_memory: float  # Peak memory used
    elapsed_time: float  # Time elapsed for the task
    state: str

def init_measurements():
    process = psutil.Process()
    init_time = time.time()
    init_cpu_time = sum(process.cpu_times()[:2])  # User + System CPU time
    tracemalloc.start()  # Start tracing memory allocations
    return ResourcesUsage(cpu_time=init_cpu_time, peak_memory=0, elapsed_time=init_time, state='init')

def finish_measurements(init_measurements):
    process = psutil.Process()
    finish_time = time.time()
    finish_cpu_time = sum(process.cpu_times()[:2])  # User + System CPU time
    _, peak_memory = tracemalloc.get_traced_memory()  # Get peak memory usage
    tracemalloc.stop()  # Stop tracing memory
    return ResourcesUsage(
        cpu_time=finish_cpu_time - init_measurements.cpu_time,
        peak_memory=peak_memory,  # Use peak memory instead of current
        elapsed_time=finish_time - init_measurements.elapsed_time,
        state='finish'
    )

def get_measurements(measurements):
    assert measurements.state == 'finish', 'The state of the measurements should be finish'
    return measurements.model_dump()