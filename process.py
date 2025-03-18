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

    os.chdir("openmc")
    record[problem]["OpenMC"] = {}

    # Output directory
    dir_output = "output/serial"

    # Run parameters
    task = tasks[problem]["openmc"]
    logN_min = task["logN_min"]
    logN_max = task["logN_max"]
    N_runs = task["N_runs"]
    N_list = np.logspace(logN_min, logN_max, N_runs, dtype=int)

    # Set runtimes and simulation rates
    runtime_openmc = np.zeros(N_runs, dtype=float)
    simrate_openmc = np.zeros(N_runs, dtype=float)
    imax = N_runs
    for i in range(N_runs):
        N = N_list[i]
        file_name = "%s/output_%i-runtime.h5" % (dir_output, N)
        if not os.path.isfile(file_name):
            imax = i
            break
        with h5py.File(file_name, "r") as f:
            runtime_openmc[i] = f["runtime/simulation"][()]
            simrate_openmc[i] = 10 * N / runtime_openmc[i] * 1e-3

    # Record
    record[problem]["OpenMC"]["tracking_rate"] = float(simrate_openmc[imax - 1])

    # ==================================================================================
    # MC/DC
    # ==================================================================================

    os.chdir("../mcdc")
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

            # Record
            record[problem]["MC/DC"][method][mode]["tracking_rate"] = float(simrate[imax-1])
            if mode == "numba":
                compile_time = np.min(runtime[:imax])
                record[problem]["MC/DC"][method][mode]["compile_time"] = compile_time
                runtime_wo_compilation = runtime - compile_time
                simrate_wo_compilation = 10 * N_list / runtime_wo_compilation * 1e-3
                record[problem]["MC/DC"][method][mode]["tracking_rate"] = float(
                    simrate_wo_compilation[imax-1]
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

                # Plot OpenMC
                if method == 'analog':
                    ax_runtime.plot(
                        N_list[:imax] * 10,
                        runtime_openmc[:imax],
                        STYLE['openmc'],
                        fillstyle="none",
                        label="OpenMC",
                    )
                    ax_simrate.plot(
                        N_list[:imax] * 10,
                        simrate_openmc[:imax],
                        STYLE['openmc'],
                        fillstyle="none",
                        label="OpenMC",
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
