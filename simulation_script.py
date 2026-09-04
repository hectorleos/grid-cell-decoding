import numpy as np
from math import pow
import matplotlib.pyplot as plt
plt.style.use('bmh')
import os
from utils import *
from tqdm import tqdm
from pprint import pprint

from models import DistanceCellModel, VectorCellModel, NestedModel
from simulation import run_load_simulation
OUTPUT_DIR = 'simulation_outputs'

# ================= MODEL & SIMULATION PARAMETERS =================
N_SIM_ROUND = 8
N_TRIALS = 100
ARENA_WIDTH = 100
STEP_SIZE = 1
CONV_THRESH = STEP_SIZE * 1.5
HEX_PROJ=True
model_names = ['nm', 'dcm', 'vcm']

no_distortion_params = {'undistorted-0': {'distortion': 'none-global', 'a': None, 'b':None}}
example_distortion_params = {'undistorted-0': {'distortion': 'none-global', 'a': None, 'b':None},
                     'shear1': {'distortion': 'shear', 'a': 0, 'b': 0.3},
                     'stretch1': {'distortion': 'stretch', 'a': 1, 'b': 0.67}}

# Global distortion parameters
shear_drift_global_params = get_param_dict({'a': [0], 'b': [0, 0.3, 0.6, 1, 2, 5], 'b_drift': [0.1]}, 
                                     'shear+drift-global', incl_undistorted=False)

shear_global_params = get_param_dict({'a': [0], 'b': [0, 1, 2, 5, 10]}, 
                                     'shear-global', incl_undistorted=False)
stretch_global_params = get_param_dict({'a': [1], 'b': [1, 0.7, 0.5, 0.3, 0.1]}, 
                                       'stretch-global', incl_undistorted=False)

# Local distortion parameters
shear_local_params = get_param_dict({'a': [0], 'b': [0, 0.05, 0.1, 0.3, 0.5]}, 
                                     'shear-local', incl_undistorted=False)
stretch_local_params = get_param_dict({'a': [1], 'b': [1, 0.99, 0.95, 0.9, 0.7]}, 
                                       'stretch-local', incl_undistorted=False)

# Modular distortion parameters
shear_modular_params = get_param_dict({'a': [0], 'b': [0, 1, 2, 5, 10]}, 
                                'shear-modular', incl_undistorted=False)
stretch_modular_params = get_param_dict({'a': [1], 'b': [0, 0.3, 0.5, 0.7, 0.9]},  # WARNING: Here b_final is 1 - b, so at b=0.5 we'd have b_M's = [0.99, 0.98, 0.97, 0.96, 0.93, 0.9 , 0.85, 0.78, 0.67, 0.5 ]
                                'stretch-modular', incl_undistorted=False)

stretch_simulations = [stretch_global_params, stretch_modular_params, stretch_local_params]
shear_simulations = [shear_global_params, shear_modular_params, shear_local_params, shear_drift_global_params]
# =================================================================

for curr_distortion_params in shear_simulations:
    for distortion_name in curr_distortion_params.keys():
        print(F'\n {"-+" * 30} RUNNING SIMULATIONS FOR DISTORTION={distortion_name} {"-+" * 30}')
        for model_name in model_names:
            x_histories, y_histories = run_load_simulation(model_name, 
                                                        arena_width=ARENA_WIDTH, 
                                                        step_size=STEP_SIZE,
                                                        convergence_threshold=CONV_THRESH,
                                                        n_trials=N_TRIALS, 
                                                        distortion_params=curr_distortion_params[distortion_name],
                                                        hexagonal_projection=HEX_PROJ,
                                                        n_sim_round=N_SIM_ROUND,
                                                        save_data=True)
    # Try catch
    try:
        plot_trajectories_all_models(curr_distortion_params, 
                            model_names, 
                            arena_width=ARENA_WIDTH, 
                            step_size=STEP_SIZE, 
                            convergence_threshold=CONV_THRESH, 
                            n_trials=N_TRIALS, 
                            n_sim_round= N_SIM_ROUND,
                            save_fig=True)
    except Exception as e:
        print(f"Error occurred while plotting trajectories for distortion={distortion_name}: {e}")
    