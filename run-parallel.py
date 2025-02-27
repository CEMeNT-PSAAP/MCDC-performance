import argparse
import collections
import importlib.metadata
import numpy as np
import os
import platform as platform_
import yaml

from pathlib import Path


# Supported compute platforms and their parameters
PLATFORMS = ["dane", "lassen", "tioga", "tuolumne"]
#
JOB_SUBMISSION = {}
JOB_SUBMISSION["dane"] = 'sbatch'
JOB_SUBMISSION["lassen"] = 'bsub'
JOB_SUBMISSION["tioga"] = 'flux batch'
JOB_SUBMISSION["tuolumne"] = 'flux batch'
#
JOB_SCHEDULER = {}
JOB_SCHEDULER["dane"] = 'slurm'
JOB_SCHEDULER["lassen"] = 'lsf'
JOB_SCHEDULER["tioga"] = 'flux'
JOB_SCHEDULER["tuolumne"] = 'flux'
#
JOB_TIME = {}
JOB_TIME['dane'] = "XX:00:00"
JOB_TIME['lassen'] = "XX:00"
JOB_TIME['tioga'] = "XXh"
JOB_TIME['tuolumne'] = "XXh"
#
MAX_TIME = {}
MAX_TIME['dane'] = 24
MAX_TIME['lassen'] = 12
MAX_TIME['tioga'] = 12
MAX_TIME['tuolumne'] = 24
#
CPU_CORES_PER_NODE = {}
CPU_CORES_PER_NODE["dane"] = 112
CPU_CORES_PER_NODE["lassen"] = 44
CPU_CORES_PER_NODE["tioga"] = 64
CPU_CORES_PER_NODE["tuolumne"] = 96
#
GPUS_PER_NODE = {}
GPUS_PER_NODE["dane"] = 0
GPUS_PER_NODE["lassen"] = 4
GPUS_PER_NODE["tioga"] = 4
GPUS_PER_NODE["tuolumne"] = 4
#
MAX_NODES = {}
MAX_NODES["dane"] = 520
MAX_NODES["lassen"] = 256
MAX_NODES["tioga"] = 16 # Can be higher
MAX_NODES["tuolumne"] = 512 # Can be higher
#
MAX_NODES = {}
MAX_NODES["dane"] = 520
MAX_NODES["lassen"] = 256
MAX_NODES["tioga"] = 16 # Can be higher
MAX_NODES["tuolumne"] = 512 # Can be higher


# ======================================================================================
# Run options
# ======================================================================================

parser = argparse.ArgumentParser(description="MC/DC Performance Test Suite - Parallel")
parser.add_argument("--platform", type=str, required="True", choices=PLATFORMS)
args, unargs = parser.parse_known_args()

# Set platform parameters
platform = args.platform
job_submission = JOB_SUBMISSION[platform]
job_scheduler = JOB_SCHEDULER[platform]
job_time = JOB_TIME[platform]
cpu_cores_per_node = CPU_CORES_PER_NODE[platform]
gpus_per_node = GPUS_PER_NODE[platform]
max_nodes = MAX_NODES[platform]
max_time = MAX_TIME[platform]

# Get the PBS template
with open("template-%s.pbs"%job_scheduler, 'r') as f:
    pbs_template = f.read()

# ======================================================================================
# Preparation
# ======================================================================================

# Create version result to store the results
version = importlib.metadata.version("mcdc")
dir_version = "%s" % version
Path(dir_version).mkdir(parents=True, exist_ok=True)

# Create and get in to the folder
dir_serial = "%s/parallel/%s" % (version, platform)
Path(dir_serial).mkdir(parents=True, exist_ok=True)
os.chdir(dir_serial)

# Save the machine specification
# TODO: GPU specs
spec = {}
spec["architecture"] = str(platform_.architecture())
spec["machine"] = str(platform_.machine())
spec["node"] = str(platform_.node())
spec["platform"] = str(platform_.platform(aliased=True))
spec["processor"] = str(platform_.processor())
spec["release"] = str(platform_.release())
spec["system"] = str(platform_.system())
spec["version"] = str(platform_.version())
with open("machine_spec.yaml", "w") as f:
    yaml.dump(spec, f)

# Read the tasks
os.chdir("../../../")
with open("task-parallel.yaml", "r") as file:
    tasks = yaml.safe_load(file)

# ======================================================================================
# Run the tests
# ======================================================================================

# Loop over the test suite problems
os.chdir("test_suite")
for problem in tasks:
    # Get into the problem folder
    os.chdir(problem)

    # ==================================================================================
    # MC/DC
    # ==================================================================================

    os.chdir("mcdc")

    # Create and get into output folder
    Path("output").mkdir(parents=True, exist_ok=True)
    os.chdir("output")

    # Loop over methods
    for method in tasks[problem]["mcdc"]:
        # Loop over modes
        for mode in tasks[problem]["mcdc"][method]:
            # Only-CPU platform?
            if platform in ['dane'] and mode == 'gpu':
                continue

            # TODO: GPU mode
            if mode == 'gpu':
                continue

            # Run parameter
            N_base = tasks[problem]["mcdc"][method][mode]

            for N_node in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                N_rank = N_node * cpu_cores_per_node

                # Stop if exceeding maximum
                if N_node > max_nodes:
                    break

                # Create and get into sub output folder
                dir_output = "parallel-%s-%s-%s-node_%i" % (platform, method, mode, N_node)
                Path(dir_output).mkdir(parents=True, exist_ok=True)
                os.chdir(dir_output)

                # Copy necessary files
                os.system("cp ../../* . 2>/dev/null")

                def submit_case(case, the_time, powers):
                    # Exceed the time?
                    if the_time > max_time:
                        return

                    # Start building the PBS file
                    pbs_text = pbs_template[:]
                    pbs_text = pbs_text.replace('<N_NODE>', '%i' % N_node)
                    pbs_text = pbs_text.replace('<JOB_NAME>', 'mcdc-par-%s-%s-%s-%s' % (problem, method, mode, case))
                    pbs_text = pbs_text.replace('<TIME>', job_time.replace('XX', str(the_time)))
                    pbs_text = pbs_text.replace('<CASE>', "-"+case)

                    # Loop over runs
                    commands = ""
                    previous_output = None
                    for i in range(len(powers)):
                        power = powers[i]
                        N = int(2**power * N_node * N_base)

                        commands += (
                            "srun -n %i python input.py %s --mode=numba --N_particle=%i --output=output_%i --no-progress_bar --caching --runtime_output\n"
                            % (N_rank, method, N, power)
                        )

                        # Delete previous output (note that runtimes are saved)
                        if previous_output is not None:
                            commands += "rm %s.h5\n" % previous_output
                        previous_output = "output_%i" % power

                    # Finalize commands and PBS file
                    pbs_text = pbs_text.replace('<COMMANDS>', commands)
                    with open(f"submit-%s.pbs"%case, 'w') as f:
                        f.write(pbs_text)

                    # Submit job
                    os.system("%s submit-%s.pbs" % (job_submission, case))

                # Submit cases
                submit_case("case1", 3, [-4, -3, -2, -1, 0])
                submit_case("case2", 3, [1])
                submit_case("case3", 6, [2])
                submit_case("case4", 12, [3])
                submit_case("case5", 24, [4])

                os.chdir('..')

    os.chdir("../../..")
