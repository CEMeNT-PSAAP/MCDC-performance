import h5py, sys
import numpy as np

output_name = sys.argv[1]
output_runtime_name = output_name[:-3] + "-runtime.h5"

with h5py.File(output_name, "r") as f1:
    with h5py.File(output_runtime_name, "w") as f2:
        for name in [
            "runtime/accumulating tallies",
            "runtime/active batches",
            "runtime/reading cross sections",
            "runtime/simulation",
            "runtime/total",
            "runtime/total initialization",
            "runtime/transport",
            "runtime/writing statepoints",
        ]:
            f2.create_dataset(
                name, data=f1[name]
            )
