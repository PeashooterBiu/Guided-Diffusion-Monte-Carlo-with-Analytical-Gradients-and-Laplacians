import numpy as np
import DMC_rs_print_xyz_lib as out
import DMC_rs_lib as lib
import Guided_DMC_lib as guided_lib
import time
import sys
import multiprocessing
import os
# import tensorflow as tf
# print(tf.__version__)
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# find all xyz filenames and names without extensions within input directory
def find_xyz_files(directory):
    if not os.path.exists(directory):
        return "Directory not found"
    xyz_files = [file for file in os.listdir(directory) if file.endswith('.xyz')]
    xyz_names = [file.split(".")[0] for file in os.listdir(directory) if file.endswith('.xyz')]
    return xyz_names, xyz_files


def process_batch(args):
    # Start timing for this batch
    start_time = time.time()
    # Reseed the random number generator
    np.random.seed(os.getpid())
    (guided, n_walkers, LMs, LM_ratio, num_molecules, name, dt, DMC_sim_length, wave_func_interval, equilibration_phase, dtype) = args  
    # Normalize the ratio of LMs
    LM_ratio /= np.sum(LM_ratio)
    # Create an empty walker array
    configs = np.zeros((n_walkers, num_molecules, 3, 3))

    # Fill the walker array with copies of the LMs
    walker_idx = 0
    for index, weight in enumerate(LM_ratio):
        num_copies = int(n_walkers * weight)
        configs[walker_idx:walker_idx+num_copies] = LMs[index]
        walker_idx += num_copies
    
    # Check if any walkers remain uninitialized and assign them to the first LM
    remaining_walkers = n_walkers - walker_idx
    if remaining_walkers > 0:
        configs[walker_idx:] = LMs[0]
        
    # print(np.unique(lib.total_pe(configs)))
    # print(lib.total_pe(configs))
    # DMC simulation
    if guided:
        DMC_configs = guided_lib.sim_loop(configs, DMC_sim_length, dt, wave_func_interval, equilibration_phase, dtype, fast_steps=equilibration_phase-5000, free_steps=20)
    else:
        DMC_configs = lib.sim_loop(configs, DMC_sim_length, dt, wave_func_interval, equilibration_phase)        
    # Stop timing and calculate the duration for this batch
    end_time = time.time()
    runtime = end_time - start_time

    # Return both the result and the runtime for this batch
    return DMC_configs, runtime
    

def parallel_processing(n_trials, args):
    args_list = [args for _ in range(n_trials)]
    start_time = time.time()  # Start time for all trials
    with multiprocessing.Pool() as pool:
        # Get both results and runtimes for all trials
        results_with_times = pool.map(process_batch, args_list)
    # Separate the results and runtimes
    results = [result for result, _ in results_with_times]
    runtimes = [runtime for _, runtime in results_with_times]
    # Calculate the average runtime across all trials
    average_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
    # Optionally, you can do more detailed analysis on runtimes like min, max, median, etc.
    return results, average_runtime



'''MMC + GD batch main call'''
if __name__ == "__main__":
    # Set basic parameters
    dic = {1: "monomer", 2: "dimer", 3:"trimer", 4:"tetramer", 5:"pentamer", 6:"hexamer", 8:"octamer", 12:"dodecamer", 20:"icosamer", 40:"tetracontamer"}
    # num_molecules = int(sys.argv[1])
    num_molecules = 6
    name = dic[num_molecules]
    # LM_index = int(sys.argv[1])
    # out_dir = f"LM_DMC_output/archive/LM_{LM_index}"
    # out_dir = "isomerized_LM2_output"
    # out_dir = f"formal_output/2e4walkers27500steps/LM_{LM_index}"
    # # out_dir = f"formal_output/GM_intraguided"
    # out_dir = f"intra_grand_isomer_fraction/2e5walkers32500steps_first_6LMs"
    # out_dir = f"intra_guided_different_LMs/LM{LM_index}"
    n_trials = 10  # Adjust based on your system's capabilities
    guided = True
    # if len(sys.argv) == 4:
    #     guided = True
    # if guided:
    #     out_dir = "formal_output/GM_intraguided"
    # else:
    #     out_dir = "formal_output/GM_unguided"
    out_dir = "intra_grand_isomer_fraction/2e5walkers32500steps_first_15LMs"
    # out_dir = "formal_output/2e5walkers32500steps_first_15LMs"
    dt = 10
    # n_walkers = int(5e4)
    np.save(f"{out_dir}/hi.npy", np.array([1]))
    
    n_walkers = int(2e5)
    equilibration_phase = int(7500) + int(5000)
    DMC_sim_length = equilibration_phase + int(2e4) 
    n_snapshots = 20
    wave_func_interval = int((DMC_sim_length-equilibration_phase)//n_snapshots)
    '''data dtype'''
    dtype = np.float64
    
    
    # extract all local minima
    xyz_names, xyz_filenames = find_xyz_files("isomers_xyz")
    xyz_filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    LMs = []
    for xyz_filename in xyz_filenames:    
        LM = out.gen_walker_array(f"isomers_xyz/{xyz_filename}", 1, 0, num_molecules)[0]/0.529
        LMs.append(LM)
    LMs = np.array(LMs)  
    
    # LM_2 = out.gen_walker_array(f"isomers_xyz/special/isomerized_LM2.xyz", 1, 0, num_molecules)[0]/0.529
    # LMs[2] = LM_2
    
    # LM_index = 0
    # LM_ratio = np.zeros(shape=len(xyz_filenames))
    # LM_ratio[:6] = 1
    LM_ratio = np.ones(shape=len(xyz_filenames))
    LM_ratio[15:] *= 0
    # print(LM_ratio)
    
    
    '''(optional) write out the reference LM'''
    # out.write_xyz(f"{out_dir}/{name}_reference.xyz", LMs[LM_index]*0.529)
    
    args = (guided, n_walkers, LMs, LM_ratio, num_molecules, name, dt, DMC_sim_length, wave_func_interval, equilibration_phase, dtype)
    
    if guided:
        string_guided = f"{name} Intramolecularly Guided DMC\ndtype: {dtype}\nn_walkers: {n_walkers}\nDMC_sim_length: {DMC_sim_length}\nequilibration_phase: {equilibration_phase}\nwave_func_interval: {wave_func_interval}\nn_snapshots: {n_snapshots}\n"
        print("start\n", string_guided)
    else:
        string_unguided = f"{name} Unguided DMC\ndtype: np.float64\nn_walkers: {n_walkers}\nDMC_sim_length: {DMC_sim_length}\nequilibration_phase: {equilibration_phase}\nwave_func_interval: {wave_func_interval}\nn_snapshots: {n_snapshots}\n"
        print("start\n", string_unguided)
    
    results, average_runtime = parallel_processing(n_trials, args)
    
    if guided:
        print("-------------Finished-------------\n", string_guided)
        print("average runtime per trial: ", average_runtime, " s")
    # else:
    #     print("-------------Finished-------------\n", string_unguided, f" LM_{LM_index}")
    #     print("average runtime per trial: ", average_runtime, " s")
    
    
    AVG_E0_collections = np.zeros(shape=n_trials)
    AVG_inter_collections = np.zeros(shape=n_trials)
    AVG_intra_collections = np.zeros(shape=n_trials)
    # save the results
    for index, data in enumerate(results):
        DMC_data = data
        final_walkers = DMC_data["final_walkers"]
        E0_estimations = DMC_data["E0"]
        E0_inters = DMC_data["E0_inter"]
        E0_intras = DMC_data["E0_intra"]
        walker_snapshots = DMC_data["walkers"]
        # if index == 0:
        #     print("snapshot length: ", len(walker_snapshots))
        '''save the results of MMC and DMC'''
        out.write_xyz(f"{out_dir}/{name}_DMC_final_snapshot_walkers{n_walkers}_{index}.xyz", final_walkers*0.529)
        # np.save(f"{out_dir}/{name}_DMC_final_snapshot_walkers{n_walkers}_{index}.npy", final_walkers)
        lengths = np.zeros(shape=len(walker_snapshots))
        for i, walker_snapshot in enumerate(walker_snapshots):
            lengths[i] = walker_snapshot.shape[0]
        walker_snapshots_concatenated = np.concatenate(walker_snapshots)
        np.save(f"{out_dir}/{name}_DMC_snapshots_walkers{n_walkers}_{index}.npy", walker_snapshots_concatenated)
        np.save(f"{out_dir}/{name}_DMC_snapshots_walkers{n_walkers}_{index}_ragged.npy", np.array(walker_snapshots, dtype=object))
        # out.write_xyz(f"{out_dir}/{name}_DMC_snapshots_walkers{n_walkers}_{index}.xyz", walker_snapshots_concatenated*0.529)
        np.save(f"{out_dir}/{name}_snapshot_lengths_walkers{n_walkers}_{index}.npy", lengths)
        np.save(f"{out_dir}/{name}_energies_walkers{n_walkers}_{index}.npy", E0_estimations)
        np.save(f"{out_dir}/{name}_inter_energies_walkers{n_walkers}_{index}.npy", E0_inters)
        np.save(f"{out_dir}/{name}_intra_energies_walkers{n_walkers}_{index}.npy", E0_intras)
        average_energy = np.average(E0_estimations[equilibration_phase:])*627.503
        average_inter = np.average(E0_inters[equilibration_phase:])*627.503
        average_intra = np.average(E0_intras[equilibration_phase:])*627.503
        AVG_E0_collections[index] = average_energy
        # AVG_inter_collections[index] = average_inter
        # AVG_intra_collections[index] = average_intra
        print(f"energy of trial {index}: ", average_energy, "kcal/mol")
        
    np.save(f"{out_dir}/{name}_trial_energies_walkers{n_walkers}_{index}.npy", AVG_E0_collections)
    avg = np.average(AVG_E0_collections)
    # avg_inter = np.average(AVG_inter_collections)
    # avg_intra = np.average(AVG_intra_collections)
    
    # print("averaged inter energy of all trials: ", avg_inter, "kcal/mol")   
    # print("averaged intra energy of all trials: ", avg_intra, "kcal/mol")   
    print("averaged energy of all trials: ", avg, "kcal/mol")    
    print("STD/AVG of energy of all trials:   ", np.std(AVG_E0_collections)/avg)   

    # save all the parameters 
    parameters_documentation = (
    f"guided? {guided}\n"
    f"dype: {dtype}\n"
    f"n_walkers: {n_walkers}\n"
    f"num_molecules: {num_molecules}\n"
    f"name: {name}\n"
    f"dt: {dt}\n"
    f"DMC_sim_length: {DMC_sim_length}\n"
    f"equilibration_phase: {equilibration_phase}\n"
    f"wave_func_interval: {wave_func_interval}\n"
    f"n_snapshots: {n_snapshots}\n"
    f"LM ratio in initialization: {LM_ratio.tolist()}\n"
    f"average runtime per trial: {average_runtime} s")
    f"average energy across all trials: {avg} kcal/mol"
    
    with open(f'{out_dir}/{name}_parameters.txt', 'w') as file:
        file.write(parameters_documentation)