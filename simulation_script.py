import argparse
from simulations import run_load_simulation
from plotting import plot_trajectories, plot_trajectories_all_models, plot_popvecs_all_models

# Global parameters across models
import params
N_TRIALS = params.N_TRIALS
ARENA_WIDTH = params.ARENA_WIDTH
STEP_SIZE = params.STEP_SIZE
CONV_THRESH = params.CONV_THRESH
CONV_N_PREV = params.CONV_N_PREV
OPEN_FIELD = params.OPEN_FIELD
MODEL_NAMES = params.MODEL_NAMES
N_DCS_LIST = params.N_DCS_LIST # If empty MAIN_SIM = True
MAIN_SIM = params.MAIN_SIM
MODULAR_BS_SELEC=params.MODULAR_BS_SELEC
PLOT_SIM_TRAJECTORIES = False
SAVE_DATA = True
SAVE_PLOTS = False

def run_simulation_series(n_sim_round, 
                            distortion_params_list=params.stretch_simulations,
                            pop_vec_sim=False, 
                            n_trials=N_TRIALS,
                            arena_width=ARENA_WIDTH,
                            step_size=STEP_SIZE,
                            convergence_threshold=CONV_THRESH,
                            conv_n_prev=CONV_N_PREV,
                            open_field=OPEN_FIELD,
                            model_names=MODEL_NAMES,
                            n_dcs_list=N_DCS_LIST,
                            modular_bs_selec=MODULAR_BS_SELEC,
                            main_sim=MAIN_SIM,
                            plot_sim_trajectories=PLOT_SIM_TRAJECTORIES,
                            save_data=SAVE_DATA, 
                            save_plots=SAVE_PLOTS):
    if main_sim:
        for curr_distortion_params in distortion_params_list:
            for distortion_name in curr_distortion_params.keys():
                print(F'\n {"-+" * 30} RUNNING SIMULATIONS FOR DISTORTION={distortion_name} {"-+" * 30}')
                for model_name in model_names:
                    sim_output = run_load_simulation(model_name, 
                                        arena_width=arena_width, 
                                        n_trials=n_trials, 
                                        step_size=step_size,
                                        convergence_threshold=convergence_threshold,
                                        conv_n_prev=conv_n_prev,
                                        open_field=open_field,
                                        pop_vec_sim=pop_vec_sim,
                                        distortion_params=curr_distortion_params[distortion_name],
                                        modular_bs_selec=modular_bs_selec,
                                        n_sim_round=n_sim_round,
                                        save_data=save_data)
                    if plot_sim_trajectories:
                        x_histories, y_histories = sim_output
                        plot_trajectories(model_name, x_histories, y_histories, arena_width = arena_width)
            if save_plots:
                if pop_vec_sim:
                    plot_popvecs_all_models(curr_distortion_params, 
                                            model_names=model_names, 
                                            arena_width=arena_width, 
                                            n_trials=n_trials,
                                            n_sim_round=n_sim_round,
                                            save_fig=save_plots)
                else:
                    plot_trajectories_all_models(curr_distortion_params, 
                                        model_names=model_names, 
                                        arena_width=arena_width, 
                                        step_size=step_size, 
                                        convergence_threshold=convergence_threshold, 
                                        n_trials=n_trials, 
                                        n_sim_round=n_sim_round,
                                        save_fig=save_plots)

    else:
        for curr_distortion_params in distortion_params_list:
            for curr_N_dc in n_dcs_list:
                # Only run for largest distortion
                largest_distortion_name = list(curr_distortion_params)[-1]
                print(F'\n {"-+" * 30} RUNNING SIMULATIONS FOR DISTORTION={largest_distortion_name} {"-+" * 30}')
                run_load_simulation('vcm', 
                                    arena_width=arena_width, 
                                    n_trials=n_trials, 
                                    step_size=step_size,
                                    convergence_threshold=convergence_threshold,
                                    conv_n_prev=conv_n_prev,
                                    open_field=open_field,
                                    pop_vec_sim=pop_vec_sim, 
                                    distortion_params=curr_distortion_params[largest_distortion_name],
                                    modular_bs_selec=modular_bs_selec,
                                    n_sim_round=n_sim_round,
                                    N_fvc=curr_N_dc, #!!!
                                    N_dc=curr_N_dc,  #!!!
                                    save_data=save_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulations for different models and distortion parameters.')
    parser.add_argument('--n_sim_round', type=int, default=None, help='Simulation round number')
    parser.add_argument('--pop_vec_sim', action='store_true', help='Run population vector simulation')
    parser.add_argument('--plot_sim_trajectories', action='store_true', help='Plot simulation trajectories')
    parser.add_argument('--save_data', action='store_true', help='Save simulation data')
    parser.add_argument('--save_plots', action='store_true', help='Save plots of simulation results')
    args = parser.parse_args()

    # Run the simulation series with the provided arguments
    run_simulation_series(n_sim_round=args.n_sim_round, 
                          pop_vec_sim=args.pop_vec_sim,
                          plot_sim_trajectories=args.plot_sim_trajectories,
                          save_data=True, #args.save_data,
                          save_plots=args.save_plots)