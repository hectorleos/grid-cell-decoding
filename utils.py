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
    return X_new, Y_new



def plot_error(x_histories, y_histories, target_location = [0,0]):
    errors = []
    for x_history, y_history in zip(x_histories, y_histories):
        final_x = x_history[-1]
        final_y = y_history[-1]
        error = np.sqrt((final_x - target_location[0])**2 + (final_y - target_location[1])**2)
        errors.append(error)
    mean = np.mean(errors)
    median = np.median(errors)
    std = np.std(errors)
    plt.figure(figsize=(6, 2))
    plt.hist(errors, bins=30, alpha=0.7, color='gray', edgecolor='black')
   # plt.title('Distribution of Final Position Errors')
    plt.xlabel('Error (distance from target location)')
    plt.ylabel('Frequency')
  #  plt.xlim(0, 70)
    plt.grid(True)
    plt.show()
    print(f'Mean error: {mean:.2f}, Median error: {median:.2f}, Std error: {std:.2f}')