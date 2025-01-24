import os
print("Importing mp")
import multiprocessing as mp

print("Creating pool")
a = mp.Pool(2)

job_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
total_jobs = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))

print(f"Job {job_id} of {total_jobs}")
