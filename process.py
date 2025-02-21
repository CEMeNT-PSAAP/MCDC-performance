import argparse
import collections
import importlib.metadata
import matplotlib.pyplot as plt
import h5py, sys
import numpy as np
import os
import yaml


# Supported compute platforms
PLATFORMS = ["dane", "lassen", "tioga", "tuolumne"]

# Line styles
STYLE = {"python": "g^-", "numba": "bo--", "openmc": "rs:"}

# ======================================================================================
# Numba compile time
# ======================================================================================
# Temporary: manually determined by uncommenting [GET COMPILE TIME] below
# Will be replaced when better caching mechanics are implemented

nested_dict = lambda: collections.defaultdict(nested_dict)
COMPILE_TIME = nested_dict()
#
COMPILE_TIME["dane"]["azurv1"]["analog"] = 43.07236623764038
COMPILE_TIME["dane"]["kobayashi-coarse"]["analog"] = 43.231773138046265
COMPILE_TIME["dane"]["kobayashi-coarse"]["implicit_capture"] = 43.23833656311035
COMPILE_TIME["dane"]["kobayashi"]["analog"] = 54.9150824546814
COMPILE_TIME["dane"]["kobayashi"]["implicit_capture"] = 54.820361614227295
COMPILE_TIME["dane"]["shem361"]["analog"] = 45.87299680709839
COMPILE_TIME["dane"]["pincell"]["analog"] = 44.184755025
#
COMPILE_TIME["lassen"]["azurv1"]["analog"] = 44.779712438583374
COMPILE_TIME["lassen"]["kobayashi-coarse"]["analog"] = 43.325393946
COMPILE_TIME["lassen"]["kobayashi-coarse"]["implicit_capture"] = 43.353021065
COMPILE_TIME["lassen"]["kobayashi"]["analog"] = 55.095778819
COMPILE_TIME["lassen"]["kobayashi"]["implicit_capture"] = 55.052752989
COMPILE_TIME["lassen"]["shem361"]["analog"] = 46.255581683
COMPILE_TIME["lassen"]["pincell"]["analog"] = 44.184755025
#
COMPILE_TIME["tioga"]["azurv1"]["analog"] = 107.16409755599989
COMPILE_TIME["tioga"]["kobayashi-coarse"]["analog"] = 107.08078032900016
COMPILE_TIME["tioga"]["kobayashi-coarse"]["implicit_capture"] = 107.5788309530003
COMPILE_TIME["tioga"]["kobayashi"]["analog"] = 125.50681239000005
COMPILE_TIME["tioga"]["kobayashi"]["implicit_capture"] = 125.72471901399967
COMPILE_TIME["tioga"]["shem361"]["analog"] = 139.97926422299997
COMPILE_TIME["tioga"]["pincell"]["analog"] = 120.22446619599987
#
COMPILE_TIME["tuolumne"]["azurv1"]["analog"] = 43.288844207
COMPILE_TIME["tuolumne"]["kobayashi-coarse"]["analog"] = 43.325393946
COMPILE_TIME["tuolumne"]["kobayashi-coarse"]["implicit_capture"] = 43.353021065
COMPILE_TIME["tuolumne"]["kobayashi"]["analog"] = 55.095778819
COMPILE_TIME["tuolumne"]["kobayashi"]["implicit_capture"] = 55.052752989
COMPILE_TIME["tuolumne"]["shem361"]["analog"] = 46.255581683
COMPILE_TIME["tuolumne"]["pincell"]["analog"] = 44.184755025

# TODO: Lassen, ...

# ======================================================================================
# Run options
# ======================================================================================

# Option parser
parser = argparse.ArgumentParser(description="MC/DC Performance Test Suite - Serial, Post Processor")
parser.add_argument("--platform", type=str, required="True", choices=PLATFORMS)
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
            imax = N_runs
            for i in range(N_runs):
                N = N_list[i]
                file_name = "%s/output_%i-runtime.h5" % (dir_output, N)
                if not os.path.isfile(file_name):
                    imax = i
                    break
                with h5py.File(file_name, "r") as f:
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
                    N_list[:imax] * 10,
                    runtime[:imax],
                STYLE[mode],
                fillstyle="none",
                label="MC/DC-%s" % mode,
            )
            ax_simrate.plot(
                N_list[:imax] * 10,
                simrate[:imax],
                STYLE[mode],
                fillstyle="none",
                label="MC/DC-%s" % mode,
            )

            if mode == "numba":
                ax_runtime.plot(
                    N_list[:imax] * 10,
                    runtime_wo_compilation[:imax],
                    ":ob",
                    fillstyle="none",
                    label="MC/DC-numba (w/o comp.)",
                )
                ax_simrate.plot(
                    N_list[:imax] * 10,
                    simrate_wo_compilation[:imax],
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
