import numpy as np
from math import pow
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Global parameters across models
N_MODULES = 10        # Number of GC modules (M)
N_GC_PER_MODULE = 20    # Number of GCs per module (m)
SCALE_RATIO = 1.5       # Scale ratio 
MIN_SCALE = 25        # Minimum grid scale (for geometric progression)
R_MAX = 30              # Maximum GC firing rate (r_max)
TIME_W = 0.1            # Time window in Poisson process (secs)
N_DC = 12500    # Number of distance cells
DC_RES = 0.04    # Spatial resolution of distance cells (cm)
N_FVC = 12500    # Number of "fine-grained" vector cells per array (x or y, pos or neg)

class ParentNNClass():

    def __init__(self, M = N_MODULES, min_scale = MIN_SCALE, scale_ratio = SCALE_RATIO, m = N_GC_PER_MODULE, 
                 r_max = R_MAX, time_w = TIME_W, arena_width=None, distortion_params=None):
        # Grid cell parameters
        self.M = M                      # Number of GC modules
        self.s_M = min_scale            # Minimum grid scale (for geometric progression)
        self.scale_ratio = scale_ratio  # Common factor (for geometric progression)
        self.m = m          # Number of equally distributes spatial phases
        self.r_max = r_max  # Maximum GC firing rate (Hz)
        self.poiss_time_w = time_w # Time window in Poisson process (secs)
        self.arena_width = arena_width    # Maximum arena width (cm)

        # Distortion parameters
        if distortion_params is not None:
            self.distortion_name = distortion_params['distortion'].split('-')[0]
            self.distortion_type = distortion_params['distortion'] .split('-')[1]
            self.a = distortion_params['a']
            self.b = distortion_params['b']
            self.b_drift = None
            if '+' in self.distortion_name:
                if 'b_drift' in distortion_params:
                    self.distortion_name = self.distortion_name.split('+')[0]
                    self.b_drift = distortion_params['b_drift']
                    self.b_offset = 0
                else:
                    print('Error: Please provide b_drift parameter for distortion with drift.')
                    exit(1)
        else:
            print('Error: Please provide distortion parameters (None if no distortion is desired).')
            exit(1)

    # ----- Initialization functions -----
    def _create_scales(self):
        ''' Creates M scales according to geometric progression '''
        scales = []
        for i in np.arange(self.M, 0, -1):
            new_scale = round(self.s_M * (self.scale_ratio ** (self.M-i)),2)
            scales.append(new_scale)
        return scales

    def _create_phases(self):
        ''' Creates m equally spaced phases from 0 to 2*pi in radians'''
        return np.linspace(0, 2 * np.pi, self.m, endpoint=False)

    # ----- Neuron activation functions -----
    def _gc_rate(self, a, s_i, p_ij):
        ''' Calculates GC firing rate based on cosine tuned rate function '''
        num_bracket = (a - (s_i * (p_ij / (2 * np.pi)))) / s_i
        return self.r_max * (1 + np.cos(num_bracket * 2 * np.pi)) / 2

    def _emax_N_normalize(self, x, epsilon=0.01):
        ''' Applies E%-max algorithm and normalzation steps '''
        # E%-max algorithm: silence all values below (1-epsilon) of the maximum value
        threshold = x.max() * (1 - epsilon)
        dc_activations = np.where(x >= threshold, x, 0.0)

        # Normalize so total activity is constant across simulations
        total = dc_activations.sum()
        if total > 0:
            dc_activations = dc_activations / total
        return dc_activations

    # ----- Distortion functions -----
    def _localdist_sample_b(self):
        ''' Samples b_max or b_min for local distortion based on uniform distribution (local distortion) '''
        if self.distortion_name == 'shear':
            return np.random.uniform(0, self.b)
        elif self.distortion_name == 'stretch':
            return np.random.uniform(self.b, 1)

    def _modulardist_get_bs(self):
        ''' Obtaines M values for b following geometric progression (modular distortion) '''
        modular_bs = []
        if self.distortion_name == 'shear':
            for i in np.arange(self.M, 0, -1):
                modular_bs.append(round(self.b / (self.scale_ratio ** (self.M-i)),2))
            modular_bs = np.array(modular_bs)[::-1]
        elif self.distortion_name == 'stretch':
            for i in np.arange(self.M, 0, -1):
                modular_bs.append(1 - round(self.b / (self.scale_ratio ** (self.M-i)),2))
            modular_bs = np.array(modular_bs)[::-1]
        else:
            print(f"_modulardist_get_bs: Distortion type '{self.distortion_name}' is not recognized. No distortion applied.")
            exit(1)
        # IMPORTANT: modules in nested modules are reversed, so this array must also be reversed in that case
        modular_bs = modular_bs[::-1] if self.name == 'nm' else modular_bs
        return modular_bs


    def _apply_distortion(self, X, Y, a, b, goal_drift=False):
        ''' Applies parametric distortions following Edvarsen (2018)'''
        if self.distortion_name == 'stretch':
            dist_mat = np.array([[a, 0], [0, b]])
            inv_dist_mat = np.linalg.inv(dist_mat)
            X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
            Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
        elif self.distortion_name == 'shear':
            dist_mat = np.array([[1, a], [b, 1]])
            inv_dist_mat = np.linalg.inv(dist_mat)
            X_new = inv_dist_mat[0,0]*X + inv_dist_mat[0,1]*Y
            Y_new = inv_dist_mat[1,0]*X + inv_dist_mat[1,1]*Y
        elif self.distortion_name == 'symmetric':
            X_new = X / (1 + a)
            Y_new = Y / (1 + (a * X / (1 + a)))  
        elif self.distortion_name == 'none' or self.distortion_name is None:
            X_new = X
            Y_new = Y
        else:
            # print error message if distortion type is not recognized
            print(f"Warning: Distortion type '{self.distortion_name}' is not recognized. No distortion applied.")
            exit(1)
        if self.b_drift is not None and goal_drift: # b_drift is the same for x and y coords ONLY for goal
          X_new = X_new + self.b_offset
          Y_new = Y_new + self.b_offset

        return X_new, Y_new

class DistanceCellModel(ParentNNClass):
    
    # Call the parent class constructor to initialize the parameters
    def __init__(self, N_dc = N_DC, dc_res = DC_RES, **kwargs):
        # Original params  N_dc = 12500, dc_res = 0.04
        super().__init__(**kwargs)  # Call the parent class constructor
        self.name = 'dcm'
        self.long_name = 'Distance Cell Model'
        self.N_dc = N_dc        # Number of distance cells
        self.dc_res = dc_res    # Spatial resolution of distance cells (cm)

        # Initialize networks parameters
        self.scales = self._create_scales()
        self.phases = self._create_phases()
        self.xdc_positions = self._create_distance_cells()
        self.W_dcs = self._compute_weight_matrix()   # We use same W matrix for both x and y axes (for a total of 200 x 2 = 400 grid cells)
        self.W_pos_readout = self.xdc_positions / self.xdc_positions.max()  # Increases linearly with displacement along axis
        self.W_neg_readout = self.W_pos_readout.copy()[::-1]              # Decreases linearly (to encode opposite direction)


    def _create_distance_cells(self):
        ''' Creates N_dc distance cell according to spatial resolution '''
        return np.arange(0, self.N_dc * self.dc_res, self.dc_res) - (self.N_dc * self.dc_res / 2)  # Center the distance cells around 0
    
    def _compute_weight_matrix(self):
        ''' For each distance cell a, compute rate of cell j at scale i (i.e., r_ij) '''
        all_weights = []
        for curr_distance in self.xdc_positions:
            distance_weights = []
            for s_i in self.scales:
                for p_ij in self.phases:
                    r_ij = self._gc_rate(curr_distance, s_i, p_ij)
                    distance_weights.append(r_ij)
            all_weights.append(distance_weights)
        return np.array(all_weights)
    
    def _gc_spikes(self, curr_pos, goal_drift):
        ''' Given current 2D position, calculate GC spikes for all GCs in our network along axes x and y
            Returns a 2D array of shape (2, M*m) = (axes, modules * phases) with the number of spikes for each GC
        '''

        if self.distortion_type == 'global': # Apply same distortion to all cells
            curr_pos = self._apply_distortion(curr_pos[0], curr_pos[1], self.a, self.b, goal_drift=goal_drift)
        elif self.distortion_type == 'modular': # Apply different distortion to each module
            modules_bs = self._modulardist_get_bs()
        
        x_gc_spikes = []
        y_gc_spikes = []
        for i, s_i in enumerate(self.scales):
            if self.distortion_type == 'modular': 
                module_b = modules_bs[i]
                curr_pos = self._apply_distortion(curr_pos[0], curr_pos[1], self.a, module_b)

            for p_i in self.phases:
                if self.distortion_type == 'local': # Apply different distortion to each cell
                    sampled_b = self._localdist_sample_b()  # In this case, we sample b according to self.b
                    curr_pos = self._apply_distortion(curr_pos[0], curr_pos[1], self.a, sampled_b)

                r_ix = self._gc_rate(curr_pos[0], s_i, p_i) # FR at x-axis
                r_iy = self._gc_rate(curr_pos[1], s_i, p_i) # FR at y-axis
                spikes_ix = np.random.poisson(lam = self.poiss_time_w * r_ix)
                spikes_iy = np.random.poisson(lam = self.poiss_time_w * r_iy)
                #print(f"GC at phase {p_ij} with rate {np.round(r_ij)}: spikes: {spikes_ij}")
                x_gc_spikes.append(spikes_ix)
                y_gc_spikes.append(spikes_iy)
        return np.array([x_gc_spikes, y_gc_spikes])

    def forward(self, start_pos, targ_pos):
        ''' Given current 2D position a and target position b, apply forward pass by 
            1) calculating GC spikes
            2) multiplying it with the network weights
            3) applying E%-max algorithm & normalization step
        '''
        
        # 1. Calculate spikes for all GCs in our network start & goal positions
        start_gc_activations = self._gc_spikes(start_pos, goal_drift = False)
        goal_gc_activations = self._gc_spikes(targ_pos, goal_drift = True)

        # 2. Compute forward passes
        start_xdc_activations = self._emax_N_normalize(self.W_dcs @ start_gc_activations[0])
        start_ydc_activations = self._emax_N_normalize(self.W_dcs @ start_gc_activations[1])
        goal_xdc_activations = self._emax_N_normalize(self.W_dcs @ goal_gc_activations[0])
        goal_ydc_activations = self._emax_N_normalize(self.W_dcs @ goal_gc_activations[1])

        # 3. Read-out cells
        x_read_out_one = self.W_neg_readout @ start_xdc_activations + self.W_pos_readout @ goal_xdc_activations
        x_read_out_two = self.W_pos_readout @ start_xdc_activations + self.W_neg_readout @ goal_xdc_activations
        x_read_out_diff = round(x_read_out_one - x_read_out_two, 4)
        
        y_read_out_one = self.W_neg_readout @ start_ydc_activations + self.W_pos_readout @ goal_ydc_activations
        y_read_out_two = self.W_pos_readout @ start_ydc_activations + self.W_neg_readout @ goal_ydc_activations
        y_read_out_diff = round(y_read_out_one - y_read_out_two, 4)

        decoded_position = np.array([x_read_out_diff, y_read_out_diff]) * (self.xdc_positions.max()/2) # Convert back to cm - need division to account for four weights
        return decoded_position
    
  
class VectorCellModel(ParentNNClass):

    def __init__(self, N_fvc = N_FVC, N_cvc = N_FVC/10, **kwargs):
        # Original params N_fvc = 12500, N_cvc = 1250
        super().__init__(**kwargs)  
        self.name = 'vcm'
        self.long_name = 'Vector Cell Model'
        self.N_fvc = N_fvc    # Number of "fine-grained" vector cells per array (x or y, pos or neg)
        self.N_cvc = N_cvc    # Number of "course-grained" vector cells per array (x or y, pos or neg)
        self.max_range = N_DC * DC_RES  # Maximum range of VCM for fair comparison

        # Initialize networks parameters
        self.scales = self._create_scales()
        self.phases = self._create_phases()

        self.fvc_pos = self._create_fine_vector_cells()
        self.cvc_pos = self._create_coarse_vector_cells()

        self.W_phasediff = self._compute_phasediff_weight_matrix()
        self.W_GCtoFVC = self._compute_GCtoFVC_weight_matrix()
        self.W_FVCtoCVC = self._compute_FVCtoCVC_weight_matrix()

    def _create_fine_vector_cells(self):
        ''' "Fine-grained" linearly spaced cells, not mentioned in paper. '''
        if self.max_range is None:
            print('Error: max_range must be calculated to compare with DCM.')
            exit(1)
        return np.linspace(0, self.max_range, self.N_fvc)

    def _create_coarse_vector_cells(self):
        ''' Pseudo-exponential distribution of translation vectors encoded by vector cells. '''
        c_vcs = [0]
        for s_i in self.scales:
            segment = np.linspace(
                s_i * (self.max_range / (sum(self.scales) * 100)),
                s_i * self.max_range / sum(self.scales),
                round(self.N_cvc / len(self.scales))
            )
            c_vcs.extend(c_vcs[-1] + segment)
        return np.array(c_vcs)

    def _compute_phasediff_weight_matrix(self):
        Grid_Vec_w = np.zeros((self.m, self.m, self.m))
        for offset in range(self.m):
            for vec in range(self.m):
                syn = (offset + vec) % self.m  # circular shift (0-indexed, so no -1+1)
                Grid_Vec_w[offset, syn, vec] = 5e-4
        return Grid_Vec_w

    def _compute_GCtoFVC_weight_matrix(self):
        Wgc_fvc = np.zeros((self.m, self.N_fvc, len(self.scales)))
        for i, s_i in enumerate(self.scales):
            for offset in range(self.m):
                Wgc_fvc[offset, :, i] = (np.cos((np.mod(self.fvc_pos - (offset / self.m) * s_i, s_i) / s_i) * 2 * np.pi) + 1) / 2 * 5e-4
        return Wgc_fvc

    def _compute_FVCtoCVC_weight_matrix(self):
        Wfvc_cvc = np.zeros((self.N_fvc, len(self.cvc_pos)))
        for c in range(len(self.fvc_pos)):
            ind = np.argmin(np.abs(self.fvc_pos[c] - self.cvc_pos))
            Wfvc_cvc[c, ind] = 1
        return Wfvc_cvc

    def _gc_pair_spikes(self, start_pos, targ_pos, dim):
        '''
        Given 2-D start position and goal position, compute the multiplicative GC output for each scale along given axis (dim).
        Returns two arrays of shape (m, M) — the m phase-difference slots activated by the start x goal outer product, for each scale.
        '''

        multsyn_pos = np.zeros((self.m, self.M))  # positive direction
        multsyn_neg = np.zeros((self.m, self.M))  # negative direction

        if self.distortion_type == 'global': # Apply same distortion to all cells
            start_pos = self._apply_distortion(start_pos[0], start_pos[1], self.a, self.b, goal_drift = False)
            targ_pos = self._apply_distortion(targ_pos[0], targ_pos[1], self.a, self.b, goal_drift = True)
        elif self.distortion_type == 'modular': # Apply different distortion to each module
            modules_bs = self._modulardist_get_bs()

        for i, s_i in enumerate(self.scales):
            start_rates = []
            goal_rates = []

            if self.distortion_type == 'modular': # Apply different distortion to each module
                module_b = modules_bs[i]
                start_pos = self._apply_distortion(start_pos[0], start_pos[1], self.a, module_b)
                targ_pos = self._apply_distortion(targ_pos[0], targ_pos[1], self.a, module_b)

            for p in self.phases:
                if self.distortion_type == 'local': # Apply different distortion to each cell
                    sampled_b = self._localdist_sample_b() 
                    start_pos = self._apply_distortion(start_pos[0], start_pos[1], self.a, sampled_b)
                    targ_pos = self._apply_distortion(targ_pos[0], targ_pos[1], self.a, sampled_b)
                
                # Compute mean firing rates for each of the m GCs at start and goal, on each axis
                # Using index k directly as phase proxy (equivalent to Eq S6)
                start_rates.append(self._gc_rate(start_pos[dim], s_i, p) * self.poiss_time_w)
                goal_rates.append(self._gc_rate(targ_pos[dim], s_i, p) * self.poiss_time_w)
            start_rates = np.array(start_rates)
            goal_rates = np.array(goal_rates)

            # Sample Poisson spikes
            start_spikes = np.random.poisson(start_rates)  # shape (m,)
            goal_spikes  = np.random.poisson(goal_rates)

            # Compute multiplicative synapse output via Grid_Vec_w
            # einsum: outer product (m, m) contracted with Grid_Vec_w (m, m, m) -> (m,)
            multsyn_pos[:, i] = np.einsum('ij,ijv->v', np.outer(start_spikes, goal_spikes), self.W_phasediff)
            multsyn_neg[:, i] = np.einsum('ij,ijv->v', np.outer(goal_spikes, start_spikes), self.W_phasediff)

        return multsyn_pos, multsyn_neg
    


    
    def forward(self, start_pos, targ_pos):
        ''' Given current 2D position a and target position b, apply forward pass by 
            1) calculating GC spikes
            2) multiplying it with the network weights
            3) applying E%-max algorithm & normalization step
        '''

        # 1. Calculate spikes for all GCs in our network start & goal positions
        gc_x_pos, gc_x_neg = self._gc_pair_spikes(start_pos, targ_pos, dim=0)
        gc_y_pos, gc_y_neg = self._gc_pair_spikes(start_pos, targ_pos, dim=1)

        # 2. Project onto "fine-grained" vector cells
        fvc_x_pos, fvc_x_neg = np.zeros(self.N_fvc), np.zeros(self.N_fvc)
        fvc_y_pos, fvc_y_neg = np.zeros(self.N_fvc), np.zeros(self.N_fvc)
        for i in range(len(self.scales)):
            fvc_x_pos += gc_x_pos[:, i] @ self.W_GCtoFVC[:, :, i] 
            fvc_x_neg += gc_x_neg[:, i] @ self.W_GCtoFVC[:, :, i] 
            fvc_y_pos += gc_y_pos[:, i] @ self.W_GCtoFVC[:, :, i] 
            fvc_y_neg += gc_y_neg[:, i] @ self.W_GCtoFVC[:, :, i] 

        #3. Contatenate, conduct E-max + normalization step, and split back
        fvc_x = self._emax_N_normalize(np.concatenate([fvc_x_pos, fvc_x_neg]))
        fvc_y = self._emax_N_normalize(np.concatenate([fvc_y_pos, fvc_y_neg]))
        fvc_x_pos, fvc_x_neg = fvc_x[:self.N_fvc], fvc_x[self.N_fvc:]
        fvc_y_pos, fvc_y_neg = fvc_y[:self.N_fvc], fvc_y[self.N_fvc:]

       # 4. Project onto "coarse-grained" vector cells
        cvc_x_pos = fvc_x_pos @ self.W_FVCtoCVC  
        cvc_x_neg = fvc_x_neg @ self.W_FVCtoCVC
        cvc_y_pos = fvc_y_pos @ self.W_FVCtoCVC
        cvc_y_neg = fvc_y_neg @ self.W_FVCtoCVC

        # 5. Decode: weighted mean of active coarse VCs, pos direction minus neg
        x_decoded = np.nanmean(np.concatenate([self.cvc_pos[cvc_x_pos > 0],-self.cvc_pos[cvc_x_neg > 0]]))
        y_decoded = np.nanmean(np.concatenate([self.cvc_pos[cvc_y_pos > 0],-self.cvc_pos[cvc_y_neg > 0]]))

        return np.array([x_decoded, y_decoded])

class NestedModel(ParentNNClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.name = 'nm'
        self.long_name = 'Nested Model'
        self.decoding_v = 1

        # Initialize networks parameters
        self.scales = self._create_scales()[::-1]
        self.phases = self._create_phases()


    def _create_phases(self):
        ''' Creates m equally spaced phases for each scale'''
        return [np.linspace(0, scale, self.m, endpoint=False) for scale in self.scales]
    
    # --- Grid cell expected firing rate for 1D ---
    def _gc_rate(self, a, lambda_j,  c_j, kappa=1):
        ''' Calculates GC firing rate based on von Mises function for 1D case.
            a=current position
            c_j=spatial phase
            lambda_j=spatial period
            kappa=relative tuning width
           '''
        return self.r_max * np.exp(kappa * (np.cos(2 * np.pi * (a - c_j) / lambda_j) - 1))

    # --- Grid cell spiking function ---
    def _gc_spikes(self, curr_pos, goal_drift = False):
        ''' Computes GC spikes following Poisson process
        '''

        if self.distortion_type == 'global': # Apply same distortion to all cells
            curr_pos = self._apply_distortion(curr_pos[0], curr_pos[1], self.a, self.b, goal_drift=goal_drift)
        elif self.distortion_type == 'modular': # Apply different distortion to each module
            modules_bs = self._modulardist_get_bs()
            
        x_gc_spikes = []
        y_gc_spikes = []
        for lambda_i_idx, lambda_i in enumerate(self.scales):
            if self.distortion_type == 'modular': 
                module_b = modules_bs[lambda_i_idx]
                curr_pos = self._apply_distortion(curr_pos[0], curr_pos[1], self.a, module_b)
            x_scale_gc_spikes = []
            y_scale_gc_spikes = []
            scale_phases = self.phases[lambda_i_idx]

            for c_j in scale_phases:
                if self.distortion_type == 'local': # Apply different distortion to each cell
                    sampled_b = self._localdist_sample_b()
                    curr_pos = self._apply_distortion(curr_pos[0], curr_pos[1], self.a, sampled_b)
                r_jx = self._gc_rate(curr_pos[0], lambda_i, c_j) # FR at x-axis
                r_jy = self._gc_rate(curr_pos[1], lambda_i, c_j) # FR at y-axis
                spikes_jx = np.random.poisson(lam = self.poiss_time_w * r_jx)
                spikes_jy = np.random.poisson(lam = self.poiss_time_w * r_jy)
                x_scale_gc_spikes.append(spikes_jx)
                y_scale_gc_spikes.append(spikes_jy)
            x_gc_spikes.append(x_scale_gc_spikes)
            y_gc_spikes.append(y_scale_gc_spikes)
        return np.array([x_gc_spikes, y_gc_spikes])
    
    # --- Population vector functions ---
    def _phasor(self, c_j, lambda_i):
        ''' Returns phasor of given spatial phase '''
        theta = 2 * np.pi * c_j / lambda_i
        return np.cos(theta) + np.sin(theta) * 1j

    def sum_of_phasors(self, spikes, lambda_i, lambda_i_idx):
        return np.sum([spikes[lambda_i_idx][j] * self._phasor(c_j, lambda_i) for j,c_j in enumerate(self.phases[lambda_i_idx])])

    def population_vec_1d(self, gc_acts):
        ''' Computes population vector using nested approach according to eqs. 2 & 3 in Stemmler. et al (2015)
        '''
        lambda_i_idx = 0
        lambda_i = self.scales[lambda_i_idx]
        # Eq. 2
        curr_x_est = (lambda_i / (2 * np.pi)) * np.angle(self.sum_of_phasors(gc_acts, lambda_i, lambda_i_idx))
        cum_precision = self.m * lambda_i**-2  
        for lambda_i_idx in np.arange(1, len(self.scales)):
            lambda_i = self.scales[lambda_i_idx]
            precision_i = self.m * lambda_i**-2
            eta_tilde = precision_i / (cum_precision + precision_i)
            # Eq. 3
            phasor_angle = np.angle(self._phasor(-curr_x_est, lambda_i) * self.sum_of_phasors(gc_acts, lambda_i, lambda_i_idx))
            curr_x_est += eta_tilde * (lambda_i / (2 * np.pi)) * phasor_angle #= self.eq_3(gc_acts, lambda_i, lambda_i_idx, curr_x_est, eta_tilde)
            cum_precision += precision_i
        return curr_x_est

    def phase_diff_1d(self, start_gc_x, goal_gc_x):
        ''' Adapts eqs. 2 & 3 in Stemmler. et al (2015) to compute phase difference between start and goal location
        '''
        lambda_i_idx = 0
        lambda_i = self.scales[lambda_i_idx]

        # Eq. 2
        start_curr_sum = self.sum_of_phasors(start_gc_x, lambda_i, lambda_i_idx)
     #   curr_start_est = (lambda_i / (2 * np.pi)) * np.angle(start_curr_sum)
        goal_curr_sum =self.sum_of_phasors(goal_gc_x, lambda_i, lambda_i_idx)
       # curr_goal_est = (lambda_i / (2 * np.pi)) * np.angle(goal_curr_sum)
        curr_diff_est =  (lambda_i / (2 * np.pi)) * np.angle(np.conj(start_curr_sum) * goal_curr_sum)
        cum_precision = self.m * lambda_i**-2  
        for lambda_i_idx in np.arange(1, len(self.scales)):
            lambda_i = self.scales[lambda_i_idx]
     #       print('---lambda_i=',lambda_i)
            precision_i = self.m * lambda_i**-2
            eta_tilde = precision_i / (cum_precision + precision_i)

            # Eq. 3
            start_curr_sum = self.sum_of_phasors(start_gc_x, lambda_i, lambda_i_idx)
          #  curr_start_est += eta_tilde * (lambda_i / (2 * np.pi)) * np.angle(self._phasor(-curr_start_est, lambda_i) * start_curr_sum)
            goal_curr_sum = self.sum_of_phasors(goal_gc_x, lambda_i, lambda_i_idx)
          #  curr_goal_est += eta_tilde * (lambda_i / (2 * np.pi)) * np.angle(self._phasor(-curr_goal_est, lambda_i) * goal_curr_sum)           
            # Update difference estimate too
            curr_diff_est += eta_tilde * (lambda_i / (2 * np.pi)) * np.angle(self._phasor(-curr_diff_est, lambda_i) * np.conj(start_curr_sum) * goal_curr_sum)
            cum_precision += precision_i
        return curr_diff_est

    def phase_diff_1ddd(self, start_gc_x, goal_gc_x):
        lambda_i_idx = 0
        lambda_i = self.scales[lambda_i_idx]

        S_a = self.sum_of_phasors(start_gc_x, lambda_i, lambda_i_idx)
        S_b = self.sum_of_phasors(goal_gc_x, lambda_i, lambda_i_idx)
        D = S_b * np.conj(S_a)
        curr_diff_est = (lambda_i / (2 * np.pi)) * np.angle(D)

        cum_precision = self.m * lambda_i**-2
        for lambda_i_idx in np.arange(1, len(self.scales)):
            lambda_i = self.scales[lambda_i_idx]
            precision_i = self.m * lambda_i**-2
            eta_tilde = precision_i / (cum_precision + precision_i)

            S_a = self.sum_of_phasors(start_gc_x, lambda_i, lambda_i_idx)
            S_b = self.sum_of_phasors(goal_gc_x, lambda_i, lambda_i_idx)
            D = S_b * np.conj(S_a)

            curr_diff_est += eta_tilde * (lambda_i / (2 * np.pi)) * np.angle(
                self._phasor(-curr_diff_est, lambda_i) * D)
            cum_precision += precision_i

        return curr_diff_est
    
    def forward(self, start_pos, targ_pos):

        # 1. Calculate spikes for all GCs in our network start & goal positions
        start_gc_x, start_gc_y = self._gc_spikes(start_pos, goal_drift = False)
        goal_gc_x, goal_gc_y = self._gc_spikes(targ_pos, goal_drift = True)

        if self.decoding_v == 1:
 
            #2. Follow nested decoding method to obtain population vector
            start_x_hat = self.population_vec_1d(start_gc_x)
            start_y_hat = self.population_vec_1d(start_gc_y)
            goal_x_hat = self.population_vec_1d(goal_gc_x)
            goal_y_hat = self.population_vec_1d(goal_gc_y)
            self.phase_diff_1d(start_gc_x, goal_gc_x)

        # print('a',a, f'a_hat=({start_x_hat},{start_y_hat})')
        # print('b',b, f'b_hat=({goal_x_hat},{goal_y_hat})')
            x_decoded = goal_x_hat - start_x_hat
        # print('x_decoded',x_decoded)
            y_decoded = goal_y_hat - start_y_hat
       # print('y_decoded',y_decoded)

        elif self.decoding_v == 2:
            #2. Follow nested decoding method to obtain population vector
            x_decoded = self.phase_diff_1d(start_gc_x, goal_gc_x)
            y_decoded = self.phase_diff_1d(start_gc_y, goal_gc_y)

        return x_decoded, y_decoded

# Original VectorCellModel as described in paper

class VectorCellModelPaperVersion(ParentNNClass):

    def __init__(self, N_vc = 1250, arena_max = 50000, **kwargs):
        super().__init__(**kwargs)  # Call the parent class constructor
        self.N_vc = N_vc    # Number of vector cells per array (x or y, pos or neg)
        self.arena_max = arena_max  # cm

        # Initialize networks parameters
        self.scales = self._create_scales()
        self.phases = self._create_phases()
        self.xvc_pos = self._create_vector_cells(direction='positive')
        self.xvc_neg = self._create_vector_cells(direction='negative')
        self.xvcs = np.concatenate((self.xvc_pos, self.xvc_neg))
        self.yvc_pos = self._create_vector_cells(direction='positive')
        self.yvc_neg = self._create_vector_cells(direction='negative')
        self.yvcs = np.concatenate((self.yvc_pos, self.yvc_neg))
        self.Wpos = self._compute_weight_matrix(direction='positive')   
      #  self.Wneg = self._compute_weight_matrix(direction='negative')

    def _create_vector_cells(self, direction='positive'):
        ''' Pseudo-exponential distribution of translation vectors encoded by vector cells. '''
        linear = np.linspace(0, 1, self.N_vc)
        k = 5  # controls steepness - tune this
        xvc_positions = self.arena_max * (np.exp(k * linear) - 1) / (np.exp(k) - 1)
        if direction == 'negative':
            xvc_positions = -xvc_positions
        return xvc_positions
    
    def _compute_weight_matrix(self, direction):
        ''' Connectivity is based on unwrapped phase difference between two grid cell pairs. Activity only 
            propagates to vector cell if such phase difference matches the vector cell's preferred translation vector.
        '''
        
        # Initialize weights to zero
        all_weights = np.zeros((self.N_vc, self.M * self.m * self.m))

        # For each GC module, iterate over pairs of GCs...
        synapse_lookup = 2868
        synapse_index = -1
        for s_i in self.scales:
            for p_j in self.phases: # GC1
                for p_k in self.phases: # GC2
                    synapse_index += 1
                    if p_j == p_k:
                        continue  
                    else:
                        # Iterate over unwrapped phase differences to calculate displacements
                        if synapse_lookup is not None and synapse_index == synapse_lookup:
                            print('---Scale:', s_i, 'Phase 1:', p_j, 'Phase 2:', p_k, 'p_2 - p_1:', p_k - p_j, 'distance:', (p_k - p_j) * s_i / (2 * np.pi))
                            print('---Synapse index:', synapse_index)
                        n = 0
                        delta_p = (p_k - p_j) % (2 * np.pi) if direction == 'positive' else (p_j - p_k) % (2 * np.pi)
                        d = ((delta_p / (2 * np.pi)) + n) * s_i
                        n_synapses = 0
                        while (d < self.arena_max):
                         #   print(f'For n={n}, d={d}')
                            
                            # Compute the index of the vector cell that corresponds to this displacement
                            vc_index = np.argmin(np.abs(self.xvc_pos - d))
                            all_weights[vc_index, synapse_index] = 1 
                            n_synapses += 1

                            # Update variables
                            n += 1
                            delta_p = (p_k - p_j) % (2 * np.pi) if direction == 'positive' else (p_j - p_k) % (2 * np.pi)
                            d = ((delta_p / (2 * np.pi)) + n) * s_i
        # Normalize so no single GC pair dominates the weight matrix
      #  all_weights /= all_weights.sum(axis=1, keepdims=True) + 1e-10  # Add a small constant to avoid division by zero
      #  print(synapse_index)
        return np.array(all_weights)

    def _gc_pair_spikes(self, a, b):
        ''' Given current position a and target position b, calculate multiplied number of spikes for each GC pair
            Returns a 2D array of shape (2, M*m*m) = (axes, modules * phases * phases) 
        '''
        # For each GC module, iterate over pairs of GCs...
        x_gc_spikes = []
        y_gc_spikes = []
        synapse_index = -1
        for s_i in self.scales:
            for p_j in self.phases: # GC1
                for p_k in self.phases: # GC2
                    synapse_index += 1

                    # Obtain GC1 spikes for start position a
                    r_jx = self._gc_rate(a[0], s_i, p_j) 
                    r_jy = self._gc_rate(a[1], s_i, p_j)
                    spikes_jx = r_jx# np.random.poisson(lam = self.poiss_time_w * r_jx)
                    spikes_jy = r_jy# np.random.poisson(lam = self.poiss_time_w * r_jy)

                    # Obtain GC2 spikes for target position b
                    r_kx = self._gc_rate(b[0], s_i, p_k) 
                    r_ky = self._gc_rate(b[1], s_i, p_k)
                    spikes_kx = r_kx# np.random.poisson(lam = self.poiss_time_w * r_kx)
                    spikes_ky = r_ky# np.random.poisson(lam = self.poiss_time_w * r_ky)

                    # Multiply the spikes together
                    x_gc_spikes.append(spikes_jx * spikes_kx)
                    y_gc_spikes.append(spikes_jy * spikes_ky)

                    #if spikes_jx * spikes_kx > 800:
#                    print(f"S.I: {synapse_index} GC pair (s_i={s_i}, p_j dist={p_j * s_i / (2 * np.pi)}, p_k dist={p_k * s_i / (2 * np.pi)}): spikes_jx={spikes_jx}, spikes_kx={spikes_kx}, multiplied={spikes_jx * spikes_kx}")
#        print('Total number of GC pairs:', synapse_index)
        return np.array([x_gc_spikes, y_gc_spikes])
    
    def forward(self, a, b):
        ''' Given current 2D position a and target position b, apply forward pass by 
            1) calculating GC spikes
            2) multiplying it with the network weights
            3) applying E%-max algorithm & normalization step
        '''

        # 1. Calculate spikes for all GCs in our network start & goal positions
#        print(f'Calculating GC spikes for start position a={a} and goal position b={b}')
        gc_pos_activations = self._gc_pair_spikes(a, b)
       # gc_neg_activations = self._gc_pair_spikes(b, a) # Simply swap start and goal positions to get negative translation vectors

        # 2. Compute forward passes
#        print('gc_pos_activations[0]', gc_pos_activations[0])

        synapse_index = -1
        for s_i in self.scales:
            for p_j in self.phases: # GC1
                for p_k in self.phases: # GC2
                    synapse_index += 1
                    W_inds = np.where(self.Wpos[:,synapse_index]>0)
                    print('SI=', synapse_index,'GC act=', gc_pos_activations[0][synapse_index], 'with W pos inds=', W_inds, 'W=', self.Wpos[W_inds,synapse_index])
                    print('--+-Scale:', s_i, 'Phase 1:', p_j, 'Phase 2:', p_k, 'p_2 - p_1:', p_k - p_j, 'distance:', (p_k - p_j) * s_i / (2 * np.pi))


        xvc_pos_activations = self.Wpos @ gc_pos_activations[0]
       # xvc_neg_activations = self.Wneg @ gc_neg_activations[0]
        yvc_pos_activations = self.Wpos @ gc_pos_activations[1]
       # yvc_neg_activations = self.Wneg @ gc_neg_activations[1]

#        print('xvc_pos_activations:', xvc_pos_activations)
        xvc_pos_activations_normd = self._emax_N_normalize(xvc_pos_activations)
        yvc_pos_activations_normd = self._emax_N_normalize(yvc_pos_activations)
#        print('normd xvc_pos_activations_normd:', xvc_pos_activations_normd)
        for nomd_idx, nomd in enumerate(xvc_pos_activations_normd):
            print(f'For nomd_idx={nomd_idx}, encoding={self.xvc_pos[nomd_idx]}, orgl activation={xvc_pos_activations[nomd_idx]}, normd activation={nomd}')

     #   print(f'X VC pos activations: {xvc_pos_activations}')
        for pos in xvc_pos_activations:
            print(f'Activation: {pos}, Index: {np.where(xvc_pos_activations == pos)[0][0]}, VC Position: {self.xvc_pos[np.where(xvc_pos_activations == pos)[0][0]]}')
        xvc_activations = xvc_pos_activations_normd #self._emax_N_normalize(np.concatenate((xvc_pos_activations, xvc_neg_activations)))
        yvc_activations = yvc_pos_activations_normd #self._emax_N_normalize(np.concatenate((yvc_pos_activations, yvc_neg_activations)))

        # 3. Decode translation vector
        x_read_out = self.xvc_pos @ xvc_activations
        y_read_out = self.yvc_pos @ yvc_activations
        translation_vector = np.array([x_read_out, y_read_out])
        return translation_vector

