import datetime

from simple_slurm import Slurm

n_mu = 301
n_zeeman = 61
n_moire = 301
total = n_mu * n_zeeman * n_moire

fps = 4 / 12
total_jobs = 10
num_cpus = 12

total_time = datetime.timedelta(seconds=total / fps / total_jobs / num_cpus)

slurm = Slurm(
    array=range(total_jobs),
    cpus_per_task=num_cpus,
    time=total_time,
)

script = slurm.arguments(shell="/bin/bash")
script += """
module load anaconda
source activate torch

python calc_conductance.py
"""

with open("run.sh", "w") as f:
    f.write(script)

print(script)
