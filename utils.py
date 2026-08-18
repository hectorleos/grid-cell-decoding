import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('bmh')
import os
import pickle
from pathlib import Path


def get_param_dict(param_values, distortion_name, incl_undistorted=False):
    param_dict = {}
    if incl_undistorted:
        param_dict['undistorted-0'] = {'distortion': 'none-global', 'a': None, 'b': None}
    condition_idx = 0
    for a_value in param_values['a']:
        for b_value in param_values['b']:
            param_dict[f'{distortion_name}-{condition_idx+1}'] = {'distortion': distortion_name, 'a': a_value, 'b': b_value}
            condition_idx += 1
    return param_dict

class GC_Distortion:
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

def gc_distortion(X, Y, distortion, a=None, b=None):
    '''Applied distortion to X and Y coordinates.'''
    if distortion == 'stretch':
        dist_mat = np.array([[a, 0], [0, b]])
        inv_dist_mat = np.linalg.inv(dist_mat)
        X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
        Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
    elif distortion == 'shear':
        dist_mat = np.array([[1, a], [b, 1]])
        inv_dist_mat = np.linalg.inv(dist_mat)
        X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
        Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
    elif distortion == 'symmetric':
        X_new = X / (1 + a)
        Y_new = Y / (1 + (a * X / (1 + a)))  
    elif distortion is None or distortion == 'None':
        X_new = X
        Y_new = Y
    return X_new, Y_new


def project_onto_axis(pos_xy_mat, l):
    """ 
    Project matrix of 2D positions onto one of three axes (l = 1, 2, 3) that are 60 degrees apart.
    """
    phi_l = -(np.pi/6) + l * (np.pi/3)   # l = 1, 2, 3 -> axes 60 deg apart
    kl = np.array([np.cos(phi_l), np.sin(phi_l)])
    pos_xy_mat = np.array(pos_xy_mat)
    return kl @ pos_xy_mat

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


def plot_trajectories(model_name, x_histories, y_histories, arena_width = 50, hexagonal_projection=True):
    long_names = {'nm': 'Nested Model', 'dcm': 'Distance Cell Model', 'vcm': 'Vector Cell Model'}

    n_trials = len(x_histories)
    plt.figure(figsize=(6, 6))

    # Project the trajectories onto the hexagonal axes
    if hexagonal_projection:
        x_histories_proj = []
        y_histories_proj = []
        for trial in range(len(x_histories)):
            pos_xy_mat = np.array([x_histories[trial], y_histories[trial]])
            proj_X = project_onto_axis(pos_xy_mat, 1)
            proj_Y = project_onto_axis(pos_xy_mat, 2)
            x_histories_proj.append(proj_X)
            y_histories_proj.append(proj_Y)
        x_histories = x_histories_proj
        y_histories = y_histories_proj

    success_count = 0
    for trial in range(n_trials):
        last_x = x_histories[trial][-1]
        last_y = y_histories[trial][-1]
        success = np.sqrt(last_x**2 + last_y**2) <= 5
        color = 'blue' if success else 'red'
        success_count += success
        plt.scatter(x_histories[trial], y_histories[trial], color=color, s=1)
        plt.scatter(x_histories[trial][-1], y_histories[trial][-1], color='red', s=1)

    plt.xlim(-arena_width, arena_width)
    plt.ylim(-arena_width, arena_width)
  #plt.xticks([])
  #  plt.yticks([])
  #  plt.xlim(25,40)
  #  plt.ylim(-40,-20)
    plt.title(f'{long_names[model_name]}')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    plt.show()
    print(f'Successful trials: {success_count}/{n_trials} ({(success_count/n_trials)*100:.1f}%)')
#plot_trajectories(model_name, x_histories, y_histories, hexagonal_projection=True)

def calculate_error(pred_location, target_location):
    ''' Error is defined as the Euclidean distance between the final location and the target location. '''
    return np.sqrt((pred_location[0] - target_location[0])**2 + (pred_location[1] - target_location[1])**2)

def plot_error_bars(distortion, distortion_params, n_trials, target_location = [0,0], model_names = ['nm', 'dcm', 'vcm']):

    # Direc
    distortion_name = distortion.split('-')[0]
    distortion_type = distortion.split('-')[1]
    output_dir = Path('simulation_outputs') / Path(distortion_type)

    # Create condition names based on the parameter values
    conditions = [f'b-{distortion_params[condition]["b"]}' for condition in distortion_params]

    # Calculate errors for each model x condition
    errors = {}
    for model_name in model_names:
        errors[model_name] = {}
        for curr_condition in conditions:
            errors[model_name][curr_condition] = []
            found_file = False
            output_dir = Path('simulation_outputs') / Path('global') if 'None' in curr_condition else Path('simulation_outputs') / Path(distortion_type)
            for f in os.listdir(output_dir):
                case_condition = curr_condition in f and distortion_name in f
                case_global_undistorted = curr_condition in f and 'none-global' in f
                if f'{model_name}_trials-{n_trials}' in f and (case_condition or case_global_undistorted):
                    found_file = True
                  #  print('Reading file:', os.path.join(output_dir, f))
                    x_histories, y_histories = pickle.load(open(os.path.join(output_dir, f), 'rb'))
                    for x_history, y_history in zip(x_histories, y_histories):
                        final_x, final_y = x_history[-1], y_history[-1]
                        errors[model_name][curr_condition].append(calculate_error([final_x, final_y], target_location))
            if not found_file:
                print(f'ERROR: No file found for model={model_name}, condition={curr_condition}, trials={n_trials}.')
                return None

    # Plotting
    n_models = len(model_names)
    n_conditions = len(conditions)
    group_width = 0.8
    bar_width = group_width / n_conditions
    # Colormap grading from light (undistorted) to dark (highest intensity),
    if distortion_type == 'local':
        colors = plt.cm.Reds(np.linspace(0.5, 0.85, n_conditions))
    elif distortion_type == 'global':
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_conditions))

    plt.figure(figsize=(2.5 * n_models + 2, 6))
    all_se = []
    for model_idx, model_name in enumerate(model_names):
        for cond_idx, condition in enumerate(conditions):
            error_list = errors[model_name][condition]
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

    plt.hlines(y=0, color='gray', xmin=-0.5, xmax=n_models - 0.5, zorder=1)
    plt.xticks(range(n_models), model_names)
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title(f'Error bars for simulations with {distortion} perturbation (N={n_trials})')
    plt.ylim(-0.5, np.max([5.0, np.mean(error_list) + 3 * np.max(all_se)]))  # Set y-limit to show all error bars
    # Legend maps color -> condition/intensity level, shown once (not per model)


    

    if distortion_type == 'local':
        if distortion_name == 'shear':
            legend_title = r'$b \sim U(0,b_{max})$'
        elif distortion_name == 'stretch':
            legend_title = r'$b \sim U(b_{min},1)$'
    else:
        legend_title = 'Distortion Intensity'

    for cond in conditions:
        new_cond = cond
        if '-' in cond:
            new_cond = cond.replace('-', '=')
        if new_cond == 'b=None':
            new_cond = 'undistorted'
        if distortion_type == 'local' and distortion_name == 'shear':
            new_cond = new_cond.replace('b', r'$b_{max}$') if 'b' in new_cond else new_cond
        elif distortion_type == 'local' and distortion_name == 'stretch':
            new_cond = new_cond.replace('b', r'$b_{min}$') if 'b' in new_cond else new_cond

        conditions[conditions.index(cond)] = new_cond


    legend_elements = [
        Line2D([0], [0], marker='o', color='black', markerfacecolor=colors[i],
               markeredgecolor='black', markersize=8, label=conditions[i])
        for i in range(n_conditions)
    ]
    plt.legend(handles=legend_elements, loc='upper right', title=legend_title, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.show()

