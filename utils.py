import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')


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
    return X_new, Y_new


def sample_disk_position(disk_radius = 50):
    sampled_angle = np.random.uniform(0, 2 * np.pi)
    sampled_radius = disk_radius * np.sqrt(np.random.uniform(0, 1)) 
    x = sampled_radius * np.cos(sampled_angle)
    y = sampled_radius * np.sin(sampled_angle)
    return [x, y]


def plot_trajectories(x_histories, y_histories, arena_width = 50):
    n_trials = len(x_histories)
    plt.figure(figsize=(6, 6))

    # Draw disk boundary
    circle = plt.Circle((0, 0), arena_width, color='black', alpha=0.5, linestyle='--', fill=False)
    plt.gca().add_patch(circle)

    success_count = 0
    for trial in range(n_trials):
        last_x = x_histories[trial][-1]
        last_y = y_histories[trial][-1]
        success = np.sqrt(last_x**2 + last_y**2) <= 5
        color = 'blue' if success else 'red'
        success_count += success
        plt.plot(x_histories[trial], y_histories[trial], color=color, markersize=0.001)
  #  plt.xlim(-1, 1)
  #
  #   plt.ylim(-1, 1)
  #  plt.xticks([])
  #  plt.yticks([])
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    plt.show()
    print(f'Successful trials: {success_count}/{n_trials} ({(success_count/n_trials)*100:.1f}%)')