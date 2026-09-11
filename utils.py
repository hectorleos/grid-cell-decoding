import numpy as np

def get_param_dict(distortion_name, param_values, incl_undistorted=False):
    ''' Used to load the distortion parameters in params.py with the required formatting. '''
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

def get_start_positions(arena_width, n_trials):
    ''' Returns evenly spaced start positions in real space for the given number of trials and arena width. '''
    border_offset = 0.05 * arena_width
    square_half_width = (arena_width / 2)
    offset_square_half_width = square_half_width - border_offset
    xs = np.linspace(-offset_square_half_width, offset_square_half_width, int(np.sqrt(n_trials)))
    ys = np.linspace(-offset_square_half_width, offset_square_half_width, int(np.sqrt(n_trials)))
    XS, YS = np.meshgrid(xs, ys)
    return np.array([XS.ravel(), YS.ravel()]).T.reshape(-1, 2)

def gc_distortion_deletemaybe(X, Y, distortion, a=None, b=None, reverse_distortion=False):
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

def real_to_grid_projection(proj_xy_mat, l1=1, l2=2):
    """ Project matrix of 2D positions onto one of three axes (l = 1, 2, 3) that are 60 degrees apart. """
    phi_l1 = -(np.pi/6) + l1 * (np.pi/3)
    phi_l2 = -(np.pi/6) + l2 * (np.pi/3)
    kl1 = np.array([np.cos(phi_l1), np.sin(phi_l1)])
    kl2 = np.array([np.cos(phi_l2), np.sin(phi_l2)])
    A = np.array([kl1, kl2])  # (2, 2)
    proj_xy_mat = np.array(proj_xy_mat)  # (2, N)
    return A @ proj_xy_mat  # (2, N)

def grid_to_real_projection(pos_xy_mat, l1=1, l2=2):
    """ Inverse of the two-axis projection in real_to_grid_projection. """
    phi_l1 = -(np.pi/6) + l1 * (np.pi/3)  
    phi_l2 = -(np.pi/6) + l2 * (np.pi/3)

    kl1 = np.array([np.cos(phi_l1), np.sin(phi_l1)])
    kl2 = np.array([np.cos(phi_l2), np.sin(phi_l2)])
    A = np.array([kl1, kl2])  # (2, 2)
    A = np.linalg.inv(A)
    pos_xy_mat = np.array(pos_xy_mat)
    return A @ pos_xy_mat


def calculate_error(pred_location, target_location):
    ''' Error is defined as the Euclidean distance between the final location and the target location. '''
    return np.sqrt((pred_location[0] - target_location[0])**2 + (pred_location[1] - target_location[1])**2)




