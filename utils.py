import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

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
