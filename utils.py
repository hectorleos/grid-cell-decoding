import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
plt.style.use('bmh')
import os
import pickle
from pathlib import Path

LONG_NAMES = {'nm': 'Nested Model', 'dcm': 'Distance Cell Model', 'vcm': 'Vector Cell Model'}

def get_param_dict(param_values, distortion_name, incl_undistorted=False):
    param_dict = {}
    if incl_undistorted:
        param_dict['undistorted-0'] = {'distortion': 'none-global', 'a': None, 'b': None}
    condition_idx = 0
    for a_value in param_values['a']:
        for b_value in param_values['b']:
            for b_drift_value in param_values.get('b_drift', [None]):
                param_dict[f'{distortion_name}-{condition_idx+1}'] = {'distortion': distortion_name, 'a': a_value, 'b': b_value, 'b_drift': b_drift_value}
                condition_idx += 1
    return param_dict

class GC_DistortionDELEEETE:
    def __init__(self, distortion=None, a=None, b=None):
        self.distortion = distortion
        self.a = a
        self.b = b

    def apply(self, X, Y):
        if self.distortion == 'stretch':
            dist_mat = np.array([[self.a, 0], [0, self.b]])
            inv_dist_mat = np.linalg.inv(dist_mat)
            X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
            Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
        elif self.distortion == 'shear':
            dist_mat = np.array([[1, self.a], [self.b, 1]])
            inv_dist_mat = np.linalg.inv(dist_mat)
            X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
            Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
        elif self.distortion == 'symmetric':
            X_new = X / (1 + self.a)
            Y_new = Y / (1 + (self.a * X / (1 + self.a)))  
        elif self.distortion == 'none'or self.distortion is None:
            return X, Y
        else:
            # print error message if distortion type is not recognized
            print(f"Warning: Distortion type '{self.distortion}' is not recognized. No distortion applied.")
            # throw error message
            
            return X, Y

        return X_new, Y_new

def gc_distortion(X, Y, distortion, a=None, b=None, reverse_distortion=False):
    '''Applied distortion to X and Y coordinates.'''
    if distortion == 'stretch':
        dist_mat = np.array([[a, 0], [0, b]])
        inv_dist_mat = np.linalg.inv(dist_mat)
        X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
        Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
        if reverse_distortion:
            X_new = dist_mat[0,0]*X + dist_mat[0,1]*Y
            Y_new = dist_mat[1,0]*X + dist_mat[1,1]*Y
    elif distortion == 'shear':
        dist_mat = np.array([[1, a], [b, 1]])
        inv_dist_mat = np.linalg.inv(dist_mat)
        X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
        Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
        if reverse_distortion:
            X_new = dist_mat[0,0]*X + dist_mat[0,1]*Y
            Y_new = dist_mat[1,0]*X + dist_mat[1,1]*Y
    elif distortion == 'symmetric':# we never use it
        X_new = X / (1 + a)
        Y_new = Y / (1 + (a * X / (1 + a)))  
    elif distortion is None or distortion == 'None':
        X_new = X
        Y_new = Y
    return X_new, Y_new


def grid_to_real_projection(pos_xy_mat, l1=1, l2=2):
    """ 
    Project matrix of 2D positions onto one of three axes (l = 1, 2, 3) that are 60 degrees apart.
    """
    phi_l1 = -(np.pi/6) + l1 * (np.pi/3)  
    phi_l2 = -(np.pi/6) + l2 * (np.pi/3)

    kl1 = np.array([np.cos(phi_l1), np.sin(phi_l1)])
    kl2 = np.array([np.cos(phi_l2), np.sin(phi_l2)])
    A = np.array([kl1, kl2])  # (2, 2)
    pos_xy_mat = np.array(pos_xy_mat)
    return A @ pos_xy_mat

def real_to_grid_projection(proj_xy_mat, l1=1, l2=2):
    """
    Inverse of the two-axis projection:
    Given coordinates (s1, s2) where s1 = k_l1 @ pos and s2 = k_l2 @ pos,
    recover the original 2D position pos.
    
    proj_xy_mat: (2, N) array where row 0 = projections on axis l1, row 1 = projections on axis l2
    """
    phi_l1 = -(np.pi/6) + l1 * (np.pi/3)
    phi_l2 = -(np.pi/6) + l2 * (np.pi/3)
    
    kl1 = np.array([np.cos(phi_l1), np.sin(phi_l1)])
    kl2 = np.array([np.cos(phi_l2), np.sin(phi_l2)])
    
    # The forward projection is: A @ pos = [s1, s2]
    # where A = [kl1; kl2]
    A = np.array([kl1, kl2])  # (2, 2)
    
    # Inverse: pos = A^{-1} @ [s1, s2]
    A_inv = np.linalg.inv(A)
    
    proj_xy_mat = np.array(proj_xy_mat)  # (2, N)
    return A_inv @ proj_xy_mat  # (2, N)

def sample_square_position(square_width = 100, hexagonal_projection=True):
    # FOR NOW WE'RE IGNORING THIS FUNCION
    ''' Samples a random position within a square of given width. If hexagonal_projection is True, it 
        ensures that the sampled position lies within a hexagon inscribed in the square.'''
    square_half_width = square_width / 2
    if hexagonal_projection:
        while True:
            # Purposefully sampling from beyond the arena dimensions to cover for the whole range after projection
            x = np.random.uniform(-square_width, square_width)
            y = np.random.uniform(-square_width, square_width)
            hex_x, hex_y = grid_to_real_projection([x, y], 1), grid_to_real_projection([x, y], 2)
            if hex_x >= -square_half_width and hex_x <= square_half_width and hex_y >= -square_half_width and hex_y <= square_half_width:
                return [x, y]
    else:
        # Sample properly from within arena dimensions
        x = np.random.uniform(-square_half_width, square_half_width)
        y = np.random.uniform(-square_half_width, square_half_width)
    return [x, y]




# Plotting functions

def plot_error(x_histories, y_histories, target_location = [0,0], plot=False):
    errors = []
    for x_history, y_history in zip(x_histories, y_histories):
        final_x = x_history[-1]
        final_y = y_history[-1]
        error = np.sqrt((final_x - target_location[0])**2 + (final_y - target_location[1])**2)
        errors.append(error)
    mean = np.mean(errors)
    median = np.median(errors)
    std = np.std(errors)
    if plot:
        plt.figure(figsize=(6, 2))
        plt.hist(errors, bins=30, alpha=0.7, color='gray', edgecolor='black')
    # plt.title('Distribution of Final Position Errors')
        plt.xlabel('Error (distance from target location)')
        plt.ylabel('Frequency')
    #  plt.xlim(0, 70)
        plt.grid(True)
        plt.show()
    print(f'Mean error: {mean:.2f}, Median error: {median:.2f}, Std error: {std:.2f}')


def plot_trajectories(model_name, x_histories, y_histories, arena_width, plt_to_return=None):

    n_trials = len(x_histories)
    if plt_to_return is None:
        # Create one-figure subplots for all trials
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        my_plt = axes
    else:
        my_plt = plt_to_return

    success_count = 0
    for trial in range(n_trials):
        last_x = x_histories[trial][-1]
        last_y = y_histories[trial][-1]
        success = np.sqrt(last_x**2 + last_y**2) <= 15
     #   color = 'blue' if success else 'red'


      # Use reverse cmap so darker colors will appear first in the plot (i.e., earlier steps will be darker, later steps lighter)
        color = 'crimson' if success else 'slategray'
       
        cmap_name = 'gist_heat_r' if success else 'bone_r'
        # Use only second half of the colormap to avoid very light colors that are hard to see
        cmap = plt.cm.get_cmap(cmap_name, 256)
        cmap = mcolors.ListedColormap(cmap(np.linspace(0.2, 1.0, 128)))

        success_count += success
        my_plt.scatter(x_histories[trial], y_histories[trial], s=1.3, cmap=cmap, c=range(len(x_histories[trial])), zorder=1)
                       #color=color, alpha=0.8)
        # Bring to the very front
        my_plt.scatter(x_histories[trial][-1], y_histories[trial][-1], color='gold', s=1, zorder=5)

    my_plt.set_xlim(-arena_width/2, arena_width/2)
    my_plt.set_ylim(-arena_width/2, arena_width/2)
    ax_ticks = [-50, -25, 0, 25, 50]
    my_plt.set_xticks(ax_ticks)
    my_plt.set_yticks(ax_ticks)
        
    if plt_to_return is None:
        my_plt.set_title(f'{LONG_NAMES[model_name]}')
        my_plt.set_xlabel('x (cm)')
        my_plt.set_ylabel('y (cm)')
        my_plt.axhline(0, color='gray', lw=0.5)
        my_plt.axvline(0, color='gray', lw=0.5)
        plt.show()
        print(f'Successful trials: {success_count}/{n_trials} ({(success_count/n_trials)*100:.1f}%)')
    else:
        return my_plt
#plot_trajectories(model_name, x_histories, y_histories, hexagonal_projection=True)

def load_simulation(model_name, arena_width, n_trials=400, step_size=0.1, convergence_threshold = 2.5, distortion_params = None, n_sim_round=1, save_data=True):

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
        print(f'ERROR: Simulation data file {simulation_data_file} does not exist. Please run the simulation first.')
        return None
    # Otherwise, load already simulated results
    else: 
      #  print(f'Loading simulation data from {simulation_data_file}')
        x_histories, y_histories = pickle.load(open(simulation_data_file, 'rb'))
    return [x_histories, y_histories]

def condition_parameter_name(cond, distortion_name, distortion_type):
    new_cond = cond
    if '-' in cond:
        new_cond = cond.replace('-', '=')
    if distortion_type == 'local' and distortion_name == 'shear':
        new_cond = new_cond.replace('b', 'b_{max}') if 'b' in new_cond else new_cond
    elif distortion_type == 'local' and distortion_name == 'stretch':
        new_cond = new_cond.replace('b', 'b_{min}') if 'b' in new_cond else new_cond
    elif distortion_type == 'modular': #and distortion_name == 'shear':
        new_cond = new_cond.replace('b', 'b_M') if 'b' in new_cond else new_cond
    elif 'drift' in distortion_name:
        new_cond = new_cond.split("_bdrift")[0]
    return new_cond

def plot_trajectories_all_models(distortion_params, model_names, arena_width, step_size, convergence_threshold, n_trials, n_sim_round, save_fig=False):
    n_models = len(model_names)
    n_conds = len(distortion_params)

    # Create subplots for each model
    fig, axes = plt.subplots(nrows=n_models, ncols=n_conds, figsize=(4 * n_conds, 4 * n_models), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, model_name in enumerate(model_names):
        for j, (cond_name, cond_params) in enumerate(distortion_params.items()):
            distortion, a, b = cond_params['distortion'], cond_params['a'], cond_params['b']  
            distortion_name = distortion.split('-')[0]
            distortion_type = distortion.split('-')[1]
            b_name = condition_parameter_name('b', distortion_name, distortion_type)
            b = round(1 - b,2) if distortion == 'stretch-modular' else b
            distortion_text = fr'$a={a}, {b_name}={b}$'
            ax = axes[i, j] if n_models > 1 else axes[j]
            plt.sca(ax)  # Set current axis
            # Run simulation for the current model and condition
            x_histories, y_histories = load_simulation(model_name, 
                                                            arena_width=arena_width, 
                                                            step_size=step_size,
                                                            convergence_threshold=convergence_threshold,
                                                            n_trials=n_trials, 
                                                            distortion_params=distortion_params[cond_name],
                                                            n_sim_round=n_sim_round,
                                                            save_data=False)
            ax = plot_trajectories(model_name, x_histories, y_histories, arena_width = arena_width, plt_to_return=ax)

            if j == 0:
                ax.set_ylabel(LONG_NAMES[model_name], fontsize=18)
            if i == 0:
                ax.set_title(distortion_text, fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'Trajectories for {distortion} distortion', fontsize=24)
    if save_fig:
        fig_dir = Path('simulation_outputs', f'round-{n_sim_round}', 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = Path(fig_dir, f'trajectories_{distortion}_ntrials-{n_trials}.png')
        fig_path = Path(fig_dir, f'trajectories_{distortion}_ntrials-{n_trials}.png')
        fig.savefig(fig_path, dpi=300)
        plt.close()
        print(f'Saved figure to {fig_path}')
    else:
        plt.show()


def calculate_error(pred_location, target_location):
    ''' Error is defined as the Euclidean distance between the final location and the target location. '''
    return np.sqrt((pred_location[0] - target_location[0])**2 + (pred_location[1] - target_location[1])**2)


def plot_error_bars(distortion_params, model_names, n_trials, n_sim_round, target_location = [0,0]):
    # Direc
    distortion = distortion_params[list(distortion_params.keys())[0]]['distortion']
    distortion_name = distortion.split('-')[0]
    distortion_type = distortion.split('-')[1]
    output_dir = Path(f'simulation_outputs') / Path(f'round-{n_sim_round}') / Path(distortion_type)

    # Create condition names based on the parameter values
    conditions = [f'b-{distortion_params[condition]["b"]}' for condition in distortion_params]
    if '+drift' in distortion:
        conditions = [f'b-{distortion_params[condition]["b"]}_bdrift-{distortion_params[condition]["b_drift"]}' for condition in distortion_params]

    # Calculate errors for each model x condition
    errors = {}
    for model_name in model_names:
        errors[model_name] = {}
        for curr_condition in conditions:
            errors[model_name][curr_condition] = []
            found_file = False
            output_dir = Path('simulation_outputs') / Path(f'round-{n_sim_round}') / Path(distortion_type)
            for f in os.listdir(output_dir):
                case_condition = curr_condition+'.pkl' in f and distortion_name in f
                if f'{model_name}_trials-{n_trials}' in f and case_condition:
                    found_file = True
             #       print('Reading file:', os.path.join(output_dir, f))
                    x_histories, y_histories = pickle.load(open(os.path.join(output_dir, f), 'rb'))
                    for x_history, y_history in zip(x_histories, y_histories):
                        final_x, final_y = x_history[-1], y_history[-1]
               #         print(f'({final_x:.2f}, {final_y:.2f}) with error {calculate_error([final_x, final_y], target_location):.2f}')
                        errors[model_name][curr_condition].append(calculate_error([final_x, final_y], target_location))
            if not found_file:
                print(f'ERROR: No file found for model={model_name}, condition={curr_condition}, trials={n_trials}.')
                return None

    # Plotting
    n_models = len(model_names)
    n_conditions = len(conditions)
    group_width = 0.8
    bar_width = group_width / n_conditions
    colors = plt.cm.Reds(np.linspace(0.5, 0.85, n_conditions))


    plt.figure(figsize=(2.5 * n_models + 2, 6))
    all_se = []
    for model_idx, model_name in enumerate(model_names):
        for cond_idx, condition in enumerate(conditions):
            error_list = errors[model_name][condition]
          #  print(f'Model: {model_name}, Condition: {condition}, Error mean: {np.mean(error_list)}')
            if len(error_list) == 0:
                continue

            x_pos = model_idx + (cond_idx - (n_conditions - 1) / 2) * bar_width
            color = colors[cond_idx]

            # Mean +/- standard error bars
            se = np.std(error_list)/np.sqrt(len(error_list))
            all_se.append(se)
            plt.errorbar(x_pos, np.mean(error_list), yerr=se,
                         fmt='o', color=color, markersize=8, capsize=4,
                         markeredgecolor='black', markeredgewidth=0.5, zorder=3)

            # Individual points, jittered within the bar's slot so they don't overlap the next condition
            jitter = np.random.uniform(-bar_width * 0.3, bar_width * 0.3, size=len(error_list))
            plt.scatter(x_pos + jitter, error_list, alpha=0.4, color=color, s=18, zorder=2)

    #plt.hlines(y=0, color='gray', xmin=-0.5, xmax=n_models - 0.5, zorder=1)

    plt.yscale('log')

    plt.xticks(range(n_models), model_names)
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title(f'Error bars for simulations with {distortion} perturbation (N={n_trials})')
   # plt.ylim(-0.5, np.max([5.0, np.mean(error_list) + 3 * np.max(all_se)]))  # Set y-limit to show all error bars
    # Legend maps color -> condition/intensity level, shown once (not per model)


    

    if distortion_type == 'local':
        if distortion_name == 'shear':
            legend_title = r'$b \sim U(0,b_{max})$'
        elif distortion_name == 'stretch':
            legend_title = r'$b \sim U(b_{min},1)$'
    elif distortion_type == 'modular':
        legend_title = r'$[b_1,...,b_M]$'
    elif 'drift' in distortion_name:
        b_drift_val = conditions[0].split("_bdrift-")[-1]
        legend_title = fr'$b_{{drift}}={b_drift_val}$'
    else:
        legend_title = ''

    for cond in conditions:

        conditions[conditions.index(cond)] = condition_parameter_name(cond, distortion_name, distortion_type)


    legend_elements = [
        Line2D([0], [0], marker='o', color='black', markerfacecolor=colors[i],
               markeredgecolor='black', markersize=8, label=rf'${conditions[i]}$')
        for i in range(n_conditions)
    ]
    plt.legend(handles=legend_elements, loc='lower right', title=legend_title, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.show()

