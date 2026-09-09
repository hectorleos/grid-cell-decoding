import numpy as np
from math import pow
import matplotlib.pyplot as plt
plt.style.use('bmh')
import pickle
from pathlib import Path
import os
from utils import *
from utils import real_to_grid_projection
from tqdm import tqdm
from models import DistanceCellModel, VectorCellModel, NestedModel

# Utility functions for simulation


def navigation_simulation(model, arena_width = 100, n_trials=5, step_size = 1, convergence_threshold = 2.5, conv_n_prev = 5, verbose=False, open_field=True):
    # Get evenly spaced start positions in real space
    border_offset = 0.05 * arena_width
    square_half_width = (arena_width / 2)
    offset_square_half_width = square_half_width - border_offset
    xs = np.linspace(-offset_square_half_width, offset_square_half_width, int(np.sqrt(n_trials)))
    ys = np.linspace(-offset_square_half_width, offset_square_half_width, int(np.sqrt(n_trials)))
    XS, YS = np.meshgrid(xs, ys)
    start_locs = np.array([XS.ravel(), YS.ravel()]).T.reshape(-1, 2) # Shape (n_trials, 2)
    print('OPEN FIELD:', open_field)
    # Get start positions in grid space

    # For each trial...
    x_histories = []
    y_histories = []
    for i in tqdm(range(n_trials), desc="Simulating trials"):

        # 1) Get start and goal positions 
        start_x, start_y = start_locs[i]
        goal_x, goal_y = [0, 0] # Goal is the same in real and grid space

      # print('-----Starting position:', (real_start_x, real_start_y), 'Grid position:', (grid_start_x, grid_start_y), 'Goal position:', (goal_x, goal_y))

        # 2) Conduct navigation simulation until convergence
        x_history = [start_x]
        y_history = [start_y]
        converged = False
        step_count = 0
        model.b_offset = 0
        while not converged:

            # Compute the population vector based on the current position and the goal
            curr_pop_vec_grid = model.forward(start_pos=[start_x, start_y], targ_pos=[goal_x, goal_y])
            curr_pop_vec_real = grid_to_real_projection(curr_pop_vec_grid)
            # Normalize
            curr_pop_vec_real /= np.linalg.norm(curr_pop_vec_real) + 1e-10 
            
          # print(step_count, 'Starting position:', (real_start_x, real_start_y), 'Grid position:', (grid_start_x, grid_start_y), 'Goal position:', (goal_x, goal_y))

            # Update position based on the population vector
            start_x = x_history[-1] + (curr_pop_vec_real[0] * step_size)
            start_y = y_history[-1] + (curr_pop_vec_real[1] * step_size)

            if not open_field:
                # Check if we have exited the arena
                x_exit = not (-square_half_width <= start_x <= square_half_width)
                y_exit = not (-square_half_width <= start_y <= square_half_width)

                # Cancel position update if we have exited the arena
                start_x = x_history[-1] if x_exit else start_x
                start_y = y_history[-1] if y_exit else start_y

          #  print(f'x_exit: {x_exit}, y_exit: {y_exit}, start_x: {start_x:.3f}, start_y: {start_y:.3f}')

            # Apply drift to the model's offset if required
            if model.b_drift is not None:
                model.b_offset += model.b_drift # dt = 1
         
         #   print('model.b_offset:',model.b_offset)
            if verbose:
                print(f"Step {step_count}: Position ({start_x:.3f}, {start_y:.3f})")
            x_history.append(start_x)
            y_history.append(start_y)

            # Update distorted cells (returns same coords if distortion is None)
           # gc_start_x, gc_start_y = gc_distortion(x_history[-1], y_history[-1], distortion_params=distortion_params)

            # Convergence happened if the distance moved in the last conv_n_prev steps is less than the convergence threshold
            if step_count > conv_n_prev:
                old_x = x_history[-conv_n_prev]
                old_y = y_history[-conv_n_prev]
                distance_moved = np.sqrt((start_x - old_x) ** 2 + (start_y - old_y) ** 2)
                if distance_moved < convergence_threshold: # or step_count > 1000:
                    converged = True
            step_count += 1
        x_histories.append(x_history)
        y_histories.append(y_history)
        
    return x_histories, y_histories

def run_load_simulation(model_name, arena_width, n_trials=400, step_size=0.1, convergence_threshold = 2.5, distortion_params = None, n_sim_round=1, save_data=True, N_dc=None, N_fvc=None):

    # Directories
    output_dir = Path('simulation_outputs', f'round-{n_sim_round}')
    distortion, a, b = distortion_params['distortion'], distortion_params['a'], distortion_params['b']
    distortion_type = distortion.split('-')[1]
    distortion_text = f'_{distortion}_a-{a}_b-{b}' if distortion is not None else '_undistorted'
    if '+drift' in distortion:
        distortion_text += f'_bdrift-{distortion_params["b_drift"]}'
    os.makedirs(Path(output_dir), exist_ok=True)
    os.makedirs(Path(output_dir) / Path(distortion_type), exist_ok=True)
    extra_param_str = f'_Ndc-{N_dc}_Nfvc-{N_fvc}' if (N_dc is not None) or (N_fvc is not None) else ''
    simulation_data_file = Path(output_dir) / Path(distortion_type) / f'{model_name}_trials-{n_trials}{distortion_text}{extra_param_str}.pkl'

    # If  new simulation and save results
    if not os.path.exists(simulation_data_file):

        #Initialize corresponding model
        if model_name == 'dcm':
            if N_dc is None:
                model = DistanceCellModel(distortion_params=distortion_params) # N_dc will be set to default value in models.py class
            else:
                model = DistanceCellModel(distortion_params=distortion_params, N_dc=N_dc)
        elif model_name == 'vcm':
            if N_fvc is None:
                model = VectorCellModel(distortion_params=distortion_params, arena_width = arena_width)  # N_fvc will be set to default value in models.py class
            else:
                model = VectorCellModel(distortion_params=distortion_params, arena_width = arena_width, N_fvc=N_fvc, N_cvc = N_fvc / 10)
        elif model_name == 'nm':
            model = NestedModel(distortion_params=distortion_params)
        else:
            raise ValueError("Invalid model name")

        print('='*70)
        print(f'Running new simulation for ***{model.long_name}*** with distortion={distortion} (a={a}, b={b})...')
        print('PARAMETERS:')
        print(f'\tM={model.M}, min_scale={model.s_M}, scale_ratio={model.scale_ratio}, m={model.m}, r_max={model.r_max}, poiss_time_w={model.poiss_time_w}')
        if model_name == 'dcm':
            print(f'\tN_dc={model.N_dc} (DCM)')
            print(f'\txdc_positions=[{model.xdc_positions[:3]},...,{model.xdc_positions[-3:]}] (DCM)')
        elif model_name == 'vcm':
            print(f'\tN_fvc={model.N_fvc} (VCM)')
            print(f'\tcvc_pos=[{model.cvc_pos[:3]},...,{model.cvc_pos[-3:]}] (VCM)')
            print(f'\tfvc_pos=[{model.fvc_pos[:3]},...,{model.fvc_pos[-3:]}] (VCM)')
        print(f'\tScales={model.scales}')

        x_histories, y_histories = navigation_simulation(model, 
                                                         arena_width=arena_width, 
                                                         n_trials=n_trials,
                                                         step_size=step_size,
                                                         convergence_threshold=convergence_threshold,
                                                         verbose=False)

        if save_data:
            with open(simulation_data_file, 'wb') as f:
                pickle.dump((x_histories, y_histories), f) 
                print(f'Saved simulation data to {simulation_data_file}')
        print('='*70,'\n')

    # Otherwise, load already simulated results
    else: 
    #    print(f'Loading simulation data from {simulation_data_file}')
        x_histories, y_histories = pickle.load(open(simulation_data_file, 'rb'))
    return [x_histories, y_histories]
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run navigation simulation.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (\'dcm\', \'vcm\', \'nm\')')
    parser.add_argument('--arena_width', type=float, default=50, help='Width of the arena')
    parser.add_argument('--n_trials', type=int, default=400, help='Number of trials')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step size for position updates')
    parser.add_argument('--distortion', type=str, default=None, help='Type of distortion (stretch, shear, symmetric)')
    parser.add_argument('--a', type=float, default=None, help='Parameter a for distortion')
    parser.add_argument('--b', type=float, default=None, help='Parameter b for distortion')
    parser.add_argument('--N_dc', type=int, default=None, help='Number of distance cells')
    parser.add_argument('--N_fvc', type=int, default=None, help='Number of field vector cells')
    args = parser.parse_args()

    run_load_simulation(args.model_name, args.arena_width, args.n_trials, args.step_size,
                        args.distortion, args.a, args.b, N_dc=args.N_dc, N_fvc=args.N_fvc)
    