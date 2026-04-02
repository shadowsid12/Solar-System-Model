"""
experiments.py
--------------
Re-export shim. Each experiment now lives in its own module:

    experiment_1.py  —  Experiment 1: Orbital Periods
    experiment_2.py  —  Experiment 2: Energy Conservation
    experiment_3.py  —  Experiment 3: Satellite to Mars
"""

from experiment_1 import run_experiment_1
from experiment_2 import run_experiment_2
from experiment_3 import run_experiment_3, anim_best_trajectory

__all__ = [
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
    "anim_best_trajectory",
]
