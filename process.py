import argparse
import collections
import importlib.metadata
import matplotlib.pyplot as plt
import h5py, sys
import numpy as np
import os
import yaml


# Line styles
STYLE = {"python": "g^-", "numba": "bo--", "openmc": "rs:"}

# ======================================================================================
# Numba compile time
# ======================================================================================
# Temporary: manually determined by uncommenting [GET COMPILE TIME] below
# Will be replaced when better caching mechanics are implemented

nested_dict = lambda: collections.defaultdict(nested_dict)
COMPILE_TIME = nested_dict()
COMPILE_TIME["dane"]["azurv1"]["analog"] = 43.288844207
COMPILE_TIME["dane"]["kobayashi"]["analog"] = 55.095778819
COMPILE_TIME["dane"]["kobayashi"]["implicit_capture"] = 55.052752989
COMPILE_TIME["dane"]["kobayashi-coarse"]["analog"] = 43.325393946
COMPILE_TIME["dane"]["kobayashi-coarse"]["implicit_capture"] = 43.353021065
COMPILE_TIME["dane"]["shem361"]["analog"] = 46.255581683
COMPILE_TIME["dane"]["pincell"]["analog"] = 44.184755025

# TODO: Lassen, ...

# ======================================================================================
# Run options
# ======================================================================================

# Option parser
parser = argparse.ArgumentParser(description="MC/DC Performance Test Suite - Serial, Post Processor")
parser.add_argument("--platform", type=str, required="True", choices=["dane"])
args, unargs = parser.parse_known_args()

platform = args.platform

# ======================================================================================
# Preparation
# ======================================================================================

version = importlib.metadata.version("mcdc")

# Read the tasks
with open("task-serial.yaml", "r") as file:
    tasks = yaml.safe_load(file)


# ======================================================================================
# Process the test results
# ======================================================================================

# Records
record = {}

# Loop over the test suite problems
os.chdir("test_suite")
for problem in tasks:
    record[problem] = {}
    os.chdir(problem)

    # ==================================================================================
    # OpenMC (analog)
    # ==================================================================================
    # TODO

    record[problem]["OpenMC"] = {}

    # ==================================================================================
    # MC/DC
    # ==================================================================================

    os.chdir("mcdc")
    record[problem]["MC/DC"] = {}

    # Loop over methods
    for method in tasks[problem]["mcdc"]:
        record[problem]["MC/DC"][method] = {}

        # Set up the plot figures
        fig_runtime, ax_runtime = plt.subplots(1, 1, figsize=(4, 3))
        fig_simrate, ax_simrate = plt.subplots(1, 1, figsize=(4, 3))

        # Loop over modes
        for mode in tasks[problem]["mcdc"][method]:
            record[problem]["MC/DC"][method][mode] = {}

            # Output directory
            dir_output = "output/serial-%s-%s-%s" % (platform, method, mode)

            # Run parameters
            task = tasks[problem]["mcdc"][method][mode]
            logN_min = task["logN_min"]
            logN_max = task["logN_max"]
            N_runs = task["N_runs"]
            N_list = np.logspace(logN_min, logN_max, N_runs, dtype=int)

            # Set runtimes and simulation rates
            runtime = np.zeros(N_runs, dtype=float)
            simrate = np.zeros(N_runs, dtype=float)
            for i in range(N_runs):
                N = N_list[i]
                with h5py.File("%s/output_%i-runtime.h5" % (dir_output, N), "r") as f:
                    runtime[i] = f["simulation"][()]
                    simrate[i] = 10 * N / runtime[i] * 1e-3

                # [GET COMPILE TIME] uncomment below to predict compile time
                if mode == "numba":
                    print(problem, platform, method, N, runtime[i])

            # Record
            record[problem]["MC/DC"][method][mode]["tracking_rate"] = float(simrate[-1])
            if mode == "numba":
                compile_time = COMPILE_TIME[platform][problem][method]
                record[problem]["MC/DC"][method][mode]["compile_time"] = compile_time
                runtime_wo_compilation = runtime - compile_time
                simrate_wo_compilation = 10 * N_list / runtime_wo_compilation * 1e-3
                record[problem]["MC/DC"][method][mode]["tracking_rate"] = float(
                    simrate_wo_compilation[-1]
                )

            # Plot
            ax_runtime.plot(
                N_list * 10,
                runtime,
                STYLE[mode],
                fillstyle="none",
                label="MC/DC-%s" % mode,
            )
            ax_simrate.plot(
                N_list * 10,
                simrate,
                STYLE[mode],
                fillstyle="none",
                label="MC/DC-%s" % mode,
            )

            if mode == "numba":
                ax_runtime.plot(
                    N_list * 10,
                    runtime_wo_compilation,
                    ":ob",
                    fillstyle="none",
                    label="MC/DC-numba (w/o comp.)",
                )
                ax_simrate.plot(
                    N_list * 10,
                    simrate_wo_compilation,
                    ":ob",
                    fillstyle="none",
                    label="MC/DC-numba (w/o comp.)",
                )

        # Plot settings
        ax_runtime.set_xscale("log")
        ax_runtime.set_yscale("log")
        ax_runtime.set_xlabel("Number of source particles")
        ax_runtime.set_ylabel("Runtime [s]")
        ax_runtime.grid()
        ax_runtime.legend()
        ax_runtime.figure.savefig(
            "%s-%s-runtime.png" % (problem, method),
            bbox_inches="tight",
            pad_inches=0,
            dpi=600,
        )
        plt.close(ax_runtime.figure)

        # Plot settings
        ax_simrate.set_xscale("log")
        ax_simrate.set_xlabel("Number of source particles")
        ax_simrate.set_ylabel("Tracking rate [kparticles/s]")
        ax_simrate.grid()
        ax_simrate.legend()
        ax_simrate.ticklabel_format(axis="y", scilimits=(-2, 3))
        ax_simrate.figure.savefig(
            "%s-%s-tracking_rate.png" % (problem, method),
            bbox_inches="tight",
            pad_inches=0,
            dpi=600,
        )
        plt.close(ax_simrate.figure)

    os.chdir("../..")

# Save record
with open("../%s/serial/%s/record.yaml" % (version, platform), "w") as f:
    yaml.dump(record, f)
os.system("mv */mcdc/*png ../%s/serial/%s" % (version, platform))
