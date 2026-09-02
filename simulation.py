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


def navigation_simulation(model, arena_width = 100, n_trials=5, step_size = 1, convergence_threshold = 2.5, conv_n_prev = 5, hexagonal_projection= False, verbose=False):
    print('hexagonal_projection:',hexagonal_projection)

    # Get evenly spaced start positions in real space
    border_offset = 0.05 * arena_width
    square_half_width = (arena_width / 2)
    offset_square_half_width = square_half_width - border_offset
    xs = np.linspace(-offset_square_half_width, offset_square_half_width, int(np.sqrt(n_trials)))
    ys = np.linspace(-offset_square_half_width, offset_square_half_width, int(np.sqrt(n_trials)))
    XS, YS = np.meshgrid(xs, ys)
    real_start_locs = np.array([XS.ravel(), YS.ravel()]).T.reshape(-1, 2)

    # Get start positions in grid space
    if hexagonal_projection:
        print('Projecting evenly spaced start locations onto hexagonal grid axes...')
        grid_start_locs = real_to_grid_projection(np.array([XS.ravel(), YS.ravel()]) , l1=1, l2=2).T  
    
    # For each trial...
    x_histories = []
    y_histories = []
    for i in tqdm(range(n_trials), desc="Simulating trials"):

        # 1) Get start and goal positions 
        real_start_x, real_start_y = real_start_locs[i]
        grid_start_x, grid_start_y = grid_start_locs[i] if hexagonal_projection else (real_start_x, real_start_y)
        goal_x, goal_y = [0, 0] # Goal is the same in real and grid space

      # print('-----Starting position:', (real_start_x, real_start_y), 'Grid position:', (grid_start_x, grid_start_y), 'Goal position:', (goal_x, goal_y))

        # 2) Conduct navigation simulation until convergence
        x_history = [real_start_x]
        y_history = [real_start_y]
        converged = False
        step_count = 0
        model.b_offset = 0
        while not converged:

            # Compute the population vector based on the current position and the goal
            grid_start_x, grid_start_y = real_to_grid_projection([real_start_x, real_start_y]) if hexagonal_projection else (real_start_x, real_start_y)
            curr_pop_vec = model.forward(start_pos=[grid_start_x, grid_start_y], targ_pos=[goal_x, goal_y])
            curr_pop_vec = grid_to_real_projection(curr_pop_vec) if hexagonal_projection else curr_pop_vec
            curr_pop_vec /= np.linalg.norm(curr_pop_vec) + 1e-10 # Normalize
          # print(step_count, 'Starting position:', (real_start_x, real_start_y), 'Grid position:', (grid_start_x, grid_start_y), 'Goal position:', (goal_x, goal_y))

            # Update position based on the population vector
            real_start_x = x_history[-1] + (curr_pop_vec[0] * step_size)
            real_start_y = y_history[-1] + (curr_pop_vec[1] * step_size)

            # Check if we have exited the arena
            x_exit = not (-square_half_width <= real_start_x <= square_half_width)
            y_exit = not (-square_half_width <= real_start_y <= square_half_width)

            # Cancel position update if we have exited the arena
            real_start_x = x_history[-1] if x_exit else real_start_x
            real_start_y = y_history[-1] if y_exit else real_start_y

          #  print(f'x_exit: {x_exit}, y_exit: {y_exit}, start_x: {start_x:.3f}, start_y: {start_y:.3f}')

            # Apply drift to the model's offset if required
            if model.b_drift is not None:
                model.b_offset += model.b_drift # dt = 1
         
         #   print('model.b_offset:',model.b_offset)
            if verbose:
                print(f"Step {step_count}: Position ({real_start_x:.3f}, {real_start_y:.3f})")
            x_history.append(real_start_x)
            y_history.append(real_start_y)

            # Update distorted cells (returns same coords if distortion is None)
           # gc_start_x, gc_start_y = gc_distortion(x_history[-1], y_history[-1], distortion_params=distortion_params)

            # Convergence happened if the distance moved in the last conv_n_prev steps is less than the convergence threshold
            if step_count > conv_n_prev:
                real_old_x = x_history[-conv_n_prev]
                real_old_y = y_history[-conv_n_prev]
                distance_moved = np.sqrt((real_start_x - real_old_x) ** 2 + (real_start_y - real_old_y) ** 2)
                if distance_moved < convergence_threshold: # or step_count > 1000:
                    converged = True
            step_count += 1
        x_histories.append(x_history)
        y_histories.append(y_history)
        
    return x_histories, y_histories

def run_load_simulation(model_name, arena_width, n_trials=400, step_size=0.1, convergence_threshold = 2.5, distortion_params = None, hexagonal_projection=False, n_sim_round=1, save_data=True):

    # Directories
    output_dir = Path('simulation_outputs', f'round-{n_sim_round}')
    distortion, a, b = distortion_params['distortion'], distortion_params['a'], distortion_params['b']
    distortion_type = distortion.split('-')[1]
    distortion_text = f'_{distortion}_a-{a}_b-{b}' if distortion is not None else '_undistorted'
    if '+drift' in distortion:
        distortion_text += f'_bdrift-{distortion_params["b_drift"]}'
    os.makedirs(Path(output_dir), exist_ok=True)
    os.makedirs(Path(output_dir) / Path(distortion_type), exist_ok=True)
    simulation_data_file = Path(output_dir) / Path(distortion_type) / f'{model_name}_trials-{n_trials}{distortion_text}.pkl'

    # If  new simulation and save results
    if not os.path.exists(simulation_data_file):

        #Initialize corresponding model
        if model_name == 'dcm':
            model = DistanceCellModel(distortion_params=distortion_params)
        elif model_name == 'vcm':
            model = VectorCellModel(distortion_params=distortion_params, arena_width = arena_width)
        elif model_name == 'nm':
            model = NestedModel(distortion_params=distortion_params)
        else:
            raise ValueError("Invalid model name")
        print(model.scales)

        print('='*70)
        print(f'Running new simulation for {model.long_name} with distortion={distortion} (a={a}, b={b})...')
        print(f'PARAMS: M={model.M}, min_scale={model.s_M}, scale_ratio={model.scale_ratio}, m={model.m}, r_max={model.r_max}, poiss_time_w={model.poiss_time_w}')
        x_histories, y_histories = navigation_simulation(model, 
                                                         arena_width=arena_width, 
                                                         n_trials=n_trials,
                                                         step_size=step_size,
                                                         convergence_threshold=convergence_threshold,
                                                         hexagonal_projection=hexagonal_projection,
                                                         verbose=False)

        if save_data:
            with open(simulation_data_file, 'wb') as f:
                pickle.dump((x_histories, y_histories), f) 
                print(f'Saved simulation data to {simulation_data_file}')
        print('='*70)

    # Otherwise, load already simulated results
    else: 
        print(f'Loading simulation data from {simulation_data_file}')
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
    args = parser.parse_args()

    run_load_simulation(args.model_name, args.arena_width, args.n_trials, args.step_size,
                        args.distortion, args.a, args.b)
    