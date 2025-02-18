import argparse
import collections
import importlib.metadata
import numpy as np
import os
import platform as platform_
import yaml

from pathlib import Path


# Supported compute platforms
PLATFORMS = ["dane"]
# TODO: Lassen, ...

# ======================================================================================
# Run options
# ======================================================================================

parser = argparse.ArgumentParser(description="MC/DC Performance Test Suite - Serial")
parser.add_argument("--platform", type=str, required="True", choices=["dane"])
parser.add_argument("--pbs", type=str, required="True")
args, unargs = parser.parse_known_args()

platform = args.platform

# Get the PBS template
with open(args.pbs, 'r') as f:
    pbs_template = f.read()


# ======================================================================================
# Preparation
# ======================================================================================

# Create version result to store the results
version = importlib.metadata.version("mcdc")
dir_version = "%s" % version
Path(dir_version).mkdir(parents=True, exist_ok=True)

# Create and get in to the folder
dir_serial = "%s/serial/%s" % (version, platform)
Path(dir_serial).mkdir(parents=True, exist_ok=True)
os.chdir(dir_serial)

# Save the machine specification
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
with open("task-serial.yaml", "r") as file:
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
            # Create and get into sub output folder
            dir_output = "serial-%s-%s-%s" % (platform, method, mode)
            Path(dir_output).mkdir(parents=True, exist_ok=True)
            os.chdir(dir_output)

            # Copy necessary files
            os.system("cp ../../* . 2>/dev/null")

            # Start building the PBS file
            pbs_text = pbs_template[:]
            pbs_text = pbs_text.replace('<N_NODE>', '1')
            pbs_text = pbs_text.replace('<JOB_NAME>', 'mcdc-ser-%s-%s-%s' % (problem, method, mode))
            pbs_text = pbs_text.replace('<TIME>', '12:00:00')

            # Run parameters
            task = tasks[problem]["mcdc"][method][mode]
            logN_min = task["logN_min"]
            logN_max = task["logN_max"]
            N_runs = task["N_runs"]

            # Loop over runs
            commands = ""
            previous_output = None
            for N in np.logspace(logN_min, logN_max, N_runs, dtype=int):
                output_name = "output_ser_%s_%s_%s_%i" % (
                    platform[:3],
                    method[:3],
                    mode[:3],
                    N,
                )
                commands += (
                    "python input.py %s --mode=%s --N_particle=%i --output=output_%i --no-progress_bar --caching --runtime_output\n"
                    % (method, mode, N, N)
                )

                # Delete previous output (note that runtimes are saved)
                if previous_output is not None:
                    commands += "rm %s.h5\n" % previous_output

                previous_output = "output_%i" % N

            # Finalize commands and PBS file
            pbs_text = pbs_text.replace('<COMMANDS>', commands)
            with open(f"submit.pbs", 'w') as f:
                f.write(pbs_text)

            # Submit job
            os.system("sbatch submit.pbs")

            os.chdir("..")
    os.chdir("../../..")

    # ==================================================================================
    # OpenMC
    # ==================================================================================
    # TODO
