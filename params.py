# ================= MODEL PARAMETERS =================
N_MODULES = 10        # Number of GC modules (M)
N_GC_PER_MODULE = 20  # Number of GCs per module (m)
SCALE_RATIO = 1.5     # Scale ratio 
MIN_SCALE = 25        # Minimum grid scale (for geometric progression)
R_MAX = 30            # Maximum GC firing rate (r_max)
TIME_W = 0.1          # Time window in Poisson process (secs)
N_DC = 1000 #12500          # Number of distance cells
DC_RES = 0.5 #04         # Spatial resolution of distance cells (cm)
N_FVC = 1000         # Number of "fine-grained" vector cells per array (x or y, pos or neg)
E_MAX_EPS=0.01        # E%-max algorithm parameter
MODULAR_BS_SELEC='RS'   # Determines how beta values are selected in modular distortions (options: 'GP' for geometric progression or 'RS' for random sampling)

# ================= SIMULATION PARAMETERS =================
N_TRIALS = 100
ARENA_WIDTH = 100
STEP_SIZE = 1
CONV_THRESH = STEP_SIZE * 1.5
CONV_N_PREV = 5
OPEN_FIELD = True
MODEL_NAMES = ['nm', 'dcm', 'vcm']
N_DCS_LIST = [] #[12500, 15000, 17500, 20000, 22500, 25000] # If non-empty it will run the extra series of simulations
MAIN_SIM = len(N_DCS_LIST) == 0 

# ================= DISTORTION PARAMETERS =================
from utils import get_param_dict
#shear_drift_global_params = get_param_dict('shear+drift-global', {'a': [0], 'b': [0, 1, 2, 3, 5], 'b_drift': [0.1]})

# Global distortion
shear_global_params = get_param_dict('shear-global', {'a': [0], 'b': [0, 1, 2, 5, 10]})
stretch_global_params = get_param_dict('stretch-global', {'a': [1], 'b': [1, 0.7, 0.5, 0.3, 0.1]})

# Local distortion
shear_local_params = get_param_dict('shear-local', {'a': [0], 'b': [0, 0.05, 0.1, 0.3, 0.5]})
stretch_local_params = get_param_dict('stretch-local', {'a': [1], 'b': [1, 0.99, 0.95, 0.9, 0.7]})

# Modular distortion
shear_modular_params = get_param_dict('shear-modular', {'a': [0], 'b': [0, 1, 2, 5, 10]})
stretch_modular_params = get_param_dict('stretch-modular', {'a': [1], 'b': [1, 0.7, 0.5, 0.3, 0.1]})

stretch_simulations = [stretch_global_params, stretch_modular_params, stretch_local_params]
shear_simulations = [shear_global_params, shear_modular_params, shear_local_params]
all_simulations = shear_simulations + stretch_simulations