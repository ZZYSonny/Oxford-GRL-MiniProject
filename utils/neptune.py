import neptune.new as neptune
import dataclasses
import inspect
import json
import os
import __main__

def init_tracker_once() -> neptune.Run:
    tracker = neptune.init_run(
        project="PlanarGNN/Mini",
        name="main", #os.environ.get("RUN_NAME", "local"),
        mode="async", # "debug",
        capture_stderr=False,
        capture_hardware_metrics=True
    )
    return tracker


def init_tracker() -> neptune.Run:
    for _ in range(3):
        try:
            return init_tracker_once()
        except:
            print("Neptune Failed. Retrying...")
    raise Exception("Failed to init Neptune")
