import numpy as np
import pickle
from pathlib import Path
import os
from utils import *
from tqdm import tqdm
from models import DistanceCellModel, VectorCellModel, NestedModel

def pop_vector_simulation(model, arena_width, n_trials):
    ''' Computes one population vector for each trial '''
    # For start location...
    start_locs = get_start_positions(arena_width, n_trials)
    pop_vecs = []
    for i in tqdm(range(n_trials), desc="Simulating trials"):
        # 1) Get start and goal positions 
        start_x, start_y = start_locs[i]
        goal_x, goal_y = [0, 0] # Goal is the same in real and grid space
        # 2) Conduct navigation trial
        model.b_offset = 0
        curr_pop_vec_grid = model.forward(start_pos=[start_x, start_y], targ_pos=[goal_x, goal_y])
        curr_pop_vec_real = grid_to_real_projection(curr_pop_vec_grid)
        pop_vecs.append(curr_pop_vec_real)
    return pop_vecs


def navigation_simulation(model, arena_width, n_trials, step_size, convergence_threshold, conv_n_prev, open_field, verbose=False):
    ''' Computes trajectory until convergence for each trial '''
    # For each trial...
    start_locs = get_start_positions(arena_width, n_trials)
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
            curr_pop_vec_real /= np.linalg.norm(curr_pop_vec_real) + 1e-10 
            
          # print(step_count, 'Starting position:', (real_start_x, real_start_y), 'Grid position:', (grid_start_x, grid_start_y), 'Goal position:', (goal_x, goal_y))

            # Update position based on the population vector
            start_x = x_history[-1] + (curr_pop_vec_real[0] * step_size)
            start_y = y_history[-1] + (curr_pop_vec_real[1] * step_size)

            if not open_field:
                # Check if we have exited the arena
                square_half_width = (arena_width / 2)
                x_exit = not (-square_half_width <= start_x <= square_half_width)
                y_exit = not (-square_half_width <= start_y <= square_half_width)

                # Cancel position update if we have exited the arena
                start_x = x_history[-1] if x_exit else start_x
                start_y = y_history[-1] if y_exit else start_y

            # Apply drift to the model's offset if required
            if model.b_drift is not None:
                model.b_offset += model.b_drift # dt = 1
         
         #   print('model.b_offset:',model.b_offset)
            if verbose:
                print(f"Step {step_count}: Position ({start_x:.3f}, {start_y:.3f})")
            x_history.append(start_x)
            y_history.append(start_y)

            # Convergence happened if the distance moved in the last conv_n_prev steps is less than the convergence threshold
            if step_count > conv_n_prev:
                old_x = x_history[-conv_n_prev]
                old_y = y_history[-conv_n_prev]
                distance_moved = np.sqrt((start_x - old_x) ** 2 + (start_y - old_y) ** 2)
                if distance_moved < convergence_threshold: 
                    converged = True
            step_count += 1
        x_histories.append(x_history)
        y_histories.append(y_history)
        
    return x_histories, y_histories

def run_load_simulation(model_name, arena_width, n_trials=400, step_size=0.1, convergence_threshold = 2.5, conv_n_prev=5, open_field=True, pop_vec_sim=False, distortion_params = None, modular_bs_selec='RS', N_dc=None, N_fvc=None, n_sim_round=-1, save_data=True):

    # Directories
    output_dir = Path('simulation_outputs', f'round-{n_sim_round}')
    distortion, a, b = distortion_params['distortion'], distortion_params['a'], distortion_params['b']
    distortion_type = distortion.split('-')[1]
    distortion_text = f'_{distortion}_a-{a}_b-{b}' if distortion is not None else '_undistorted'
    distortion_text += f'_bdrift-{distortion_params["b_drift"]}'  if '+drift' in distortion else ''
    extra_param_str = f'_Ndc-{N_dc}_Nfvc-{N_fvc}' if (N_dc is not None) or (N_fvc is not None) else ''
    os.makedirs(Path(output_dir), exist_ok=True)
    if pop_vec_sim:
        os.makedirs(Path(output_dir) / Path('pop_vecs'), exist_ok=True)
        simulation_data_file = Path(output_dir) / Path('pop_vecs') / f'{model_name}_trials-{n_trials}{distortion_text}{extra_param_str}.pkl'
    else:
        os.makedirs(Path(output_dir) / Path(distortion_type), exist_ok=True)
        distortion_text = distortion_text if 'modular' not in distortion else distortion_text.replace('modular', f'modular-{modular_bs_selec}')
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
        print(f'\tM={model.M}, min_scale={model.s_M}, scale_ratio={model.scale_ratio}, m={model.m}, r_max={model.r_max}, poiss_time_w={model.poiss_time_w}, modular_bs_selec={model.modular_bs_selec}')
        if model_name == 'dcm':
            print(f'\tN_dc={model.N_dc} (DCM)')
            print(f'\txdc_positions=[{model.xdc_positions[:3]},...,{model.xdc_positions[-3:]}] (DCM)')
        elif model_name == 'vcm':
            print(f'\tN_fvc={model.N_fvc} (VCM)')
            print(f'\tcvc_pos=[{model.cvc_pos[:3]},...,{model.cvc_pos[-3:]}] (VCM)')
            print(f'\tfvc_pos=[{model.fvc_pos[:3]},...,{model.fvc_pos[-3:]}] (VCM)')
        print(f'\tScales={model.scales}')

        if pop_vec_sim:
            pop_vecs = pop_vector_simulation(model, 
                                             arena_width=arena_width, 
                                             n_trials=n_trials)
            if save_data:
                with open(simulation_data_file, 'wb') as f:
                    pickle.dump(pop_vecs, f) 
                    print(f'Saved simulation data to {simulation_data_file}')
                    print('='*70,'\n')
            return pop_vecs

        else:
            x_histories, y_histories = navigation_simulation(model, 
                                                            arena_width=arena_width, 
                                                            n_trials=n_trials,
                                                            step_size=step_size,
                                                            convergence_threshold=convergence_threshold,
                                                            conv_n_prev=conv_n_prev,
                                                            open_field=open_field,
                                                            verbose=False)
            if save_data:
                with open(simulation_data_file, 'wb') as f:
                    pickle.dump((x_histories, y_histories), f) 
                    print(f'Saved simulation data to {simulation_data_file}')
                    print('='*70,'\n')
            return [x_histories, y_histories]

    # Otherwise, load already simulated results
    else: 
        print('Loading previously simulated results from', simulation_data_file)
        if pop_vec_sim:
            pop_vecs = pickle.load(open(simulation_data_file, 'rb'))
            return pop_vecs
        else:
            x_histories, y_histories = pickle.load(open(simulation_data_file, 'rb'))
            return [x_histories, y_histories]
        
