import numpy as np
from math import pow
import matplotlib.pyplot as plt
plt.style.use('bmh')
import pickle
from pathlib import Path
import os
from utils import *
from tqdm import tqdm


from models import DistanceCellModel, VectorCellModel, NestedModel
OUTPUT_DIR = 'simulation_outputs'

# Utility functions for simulation

def sample_square_position(square_width = 100, hexagonal_projection=True):
    ''' Samples a random position within a square of given width. If hexagonal_projection is True, it 
        ensures that the sampled position lies within a hexagon inscribed in the square.'''
    square_half_width = square_width / 2
    if hexagonal_projection:
        while True:
            # Purposefully sampling from beyond the arena dimensions to cover for the whole range after projection
            x = np.random.uniform(-square_width, square_width)
            y = np.random.uniform(-square_width, square_width)
            hex_x, hex_y = project_onto_axis([x, y], 1), project_onto_axis([x, y], 2)
            if hex_x >= -square_half_width and hex_x <= square_half_width and hex_y >= -square_half_width and hex_y <= square_half_width:
                return [x, y]
    else:
        # Sample properly from within arena dimensions
        x = np.random.uniform(-square_half_width, square_half_width)
        y = np.random.uniform(-square_half_width, square_half_width)
    return [x, y]

def navigation_simulation(model, arena_width = 100, n_trials=5, step_size = 1, convergence_threshold = 2.5, conv_n_prev = 5, hexagonal_projection= False, verbose=False):
    print('hexagonal_projection:',hexagonal_projection)
    x_histories = []
    y_histories = []
    # For each trial...
    for i in tqdm(range(n_trials), desc="Simulating trials"):

        # 1) Sample a random start position and a goal position
 #       print(f"Trial {i+1}/{n_trials}")
        start_x, start_y = sample_square_position(arena_width, hexagonal_projection=hexagonal_projection)
        goal_x, goal_y = [0, 0] 

        # 2) Apply distortion if specified (returns same coords if distortion is None)
      #  gc_start_x, gc_start_y = gc_distortion(start_x, start_y, distortion_params=distortion_params)
      #  gc_goal_x, gc_goal_y = gc_distortion(goal_x, goal_y, distortion_params=distortion_params)

        # 3) Conduct navigation simulation until convergence
        x_history = [start_x]
        y_history = [start_y]
        converged = False
        step_count = 0
        while not converged:

            # Compute the population vector based on the current position and the goal
            curr_pop_vec = model.forward(start_pos=[start_x, start_y], targ_pos=[goal_x, goal_y])
            curr_pop_vec /= np.linalg.norm(curr_pop_vec) + 1e-10 # Normalize

            # Update position based on the population vector
            start_x = x_history[-1] + (curr_pop_vec[0] * step_size)
            start_y = y_history[-1] + (curr_pop_vec[1] * step_size)
            if verbose:
                print(f"Step {step_count}: Position ({start_x:.3f}, {start_y:.3f})")
            x_history.append(start_x)
            y_history.append(start_y)

            # Update distorted cells (returns same coords if distortion is None)
         #   gc_start_x, gc_start_y = gc_distortion(x_history[-1], y_history[-1], distortion_params=distortion_params)

            # Convergence happened if the distance moved in the last conv_n_prev steps is less than the convergence threshold
            if step_count > conv_n_prev:
                old_x = x_history[-conv_n_prev]
                old_y = y_history[-conv_n_prev]
                distance_moved = np.sqrt((start_x - old_x) ** 2 + (start_y - old_y) ** 2)
                if distance_moved < convergence_threshold or step_count > 1000:
                    converged = True

                    if start_x > 3 and False: # DElete at some point pls
                        print(f'------hmmm??? ({start_x},{start_y})')
                        print('distance_moved < convergence_threshold', distance_moved < convergence_threshold)
                        print('distance_moved:', distance_moved)
                        print('step_count > 1000', step_count > 1000)
            step_count += 1
        x_histories.append(x_history[:-conv_n_prev+1])
        y_histories.append(y_history[:-conv_n_prev+1])
    return x_histories, y_histories

def run_load_simulation(model_name, arena_width, n_trials=400, step_size=0.1, convergence_threshold = 2.5, distortion_params = None, hexagonal_projection=False, save_plot=True):

    # Directories
    distortion, a, b = distortion_params['distortion'], distortion_params['a'], distortion_params['b']
    distortion_type = distortion_params['distortion'].split('-')[1]
    distortion_text = f'_{distortion}_a-{a}_b-{b}' if distortion is not None else '_undistorted'
    os.makedirs(Path(OUTPUT_DIR), exist_ok=True)
    os.makedirs(Path(OUTPUT_DIR) / Path(distortion_type), exist_ok=True)
    simulation_data_file = Path(OUTPUT_DIR) / Path(distortion_type) / f'{model_name}_trials-{n_trials}{distortion_text}.pkl'

    # If  new simulation and save results
    if not os.path.exists(simulation_data_file):

        #Initialize corresponding model
        if model_name == 'dcm':
            model = DistanceCellModel(distortion_params=distortion_params)
        elif model_name == 'vcm':
            model = VectorCellModel(distortion_params=distortion_params,arena_width = arena_width)
        elif model_name == 'nm':
            model = NestedModel(distortion_params=distortion_params)
        else:
            raise ValueError("Invalid model name")

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

        if save_plot:
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
    