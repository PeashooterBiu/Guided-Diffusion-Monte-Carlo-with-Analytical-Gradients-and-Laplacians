# Will Solow, Skye Rhomberg
# CS446 Spring 2021
# Diffusion Monte Carlo (DMC) Simulation w/ Descendent Weighting
# Script Style
# Last Updated 02/28/2021

# This is a library of the scientific constants and functions used in our DMC simulations
# Everything in here should be constant across all simulations which import this file

# Imports
#added to prevent 64 threads from being launched
import DMC_rs_print_xyz_lib as lib
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time as time
# import DMC_processor_lib as lib


###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23


# Chemsitry constants for intermolecular energy
# Input as equation to avoid rounding errors
# Rounding should be at least 15 decimals otherwise error in the Lennard Jones 
# Energy will be incorrect by a magnitude of at least 100 depending on the 
# distance between atoms
sigma = 3.165492 / 0.529177
epsilon = 0.1554252 * (4.184 / 2625.5)

# Coulombic charges
q_oxygen = -.84
q_hydrogen = .42

# Coulomb's Constant
# Removed as of 7/26/21 due to corresponce with Prof Madison
# See line 123
coulomb_const = 1.0 / (4.0*np.pi)



# Normalization constant
# Used in graphing the wave function. Can be found experimentally using the file
# dmc_rs_norm_constant.py. 
# Not calculated here given the time it takes each simulation
#N = 4.0303907719347185
# Norm constant 2
N = 4.033938699359097

# Number of coordinates
# Always 3, used for clarity
coord_const = 3



####################################################################################
# Simulation Loop Constants

# Set the dimensions of the 4D array of which the walkers, molecules, atoms, and positions 
# reside. Used for clarity in the simulation loop
walker_axis = 0
molecule_axis = 1
atom_axis = 2
coord_axis = 3


####################################################################################
# Molecule Model Constants


# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.99491461957
hydrogen_mass = 1.007825
HOH_bond_angle = 112.0



# Equilibrium length of OH Bond
# Input as equation to avoid rounding errors
eq_bond_length = 1.0 / 0.529177

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
# Input as equation to avoid rounding errors 
bohrToAng = 0.529177
kOH = 1059.162 * (bohrToAng **2) * (4.184/2625.5) #Has units of Hartree/Bohr^2

# Spring constant of the HOH bond angle
kA = 75.90 * (4.184 / 2625.5)



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)



# Calculate the reduced mass of the system
# Note that as the wave function is being graphed for an OH vector, we only consider the
# reduced mass of the OH vector system
reduced_mass = ((atomic_masses[0]+atomic_masses[1])*atomic_masses[2])/np.sum(atomic_masses)


# Returns an array of atomic charges based on the position of the atoms in the atomic_masses array
# This is used in the potential energy function and is broadcasted to an array 
# of distances to calculate the energy using Coulomb's Law. 
atomic_charges = np.array([q_oxygen, q_hydrogen, q_hydrogen])



#######################################################################################
# Simulation


# Create an array of the charges 
# Computes the product of the charges as the atom charges are multiplied 
# together in accordance with Coulomb's Law.
# Removed coulomb const as of 7/26/21
coulombic_charges = (np.transpose(atomic_charges[np.newaxis]) \
                    @ atomic_charges[np.newaxis])  #* coulomb_const


# Input: 4D Array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: 1D Array of intramolecular potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths and 
# bond angles from equilibrium
def intra_pe(x):
    # Return the two OH vectors
    # Used to calculate the bond lengths and angle in a molecule
    OH_vectors = x[:,:,1:] - x[:,:,np.newaxis,0]
	
    # Returns the lengths of each OH bond vector for each molecule 
    # in each walker. 
    lengths = np.linalg.norm(OH_vectors, axis=3)
	
    # Calculates the bond angle in the HOH bond
    # Computes the arccosine of the dot product between the two vectors, by normalizing the
    # vectors to magnitude of 1
    angle = np.arccos(np.sum(OH_vectors[:,:,0] * OH_vectors[:,:,1], axis=2) \
	        / np.prod(lengths, axis=2))
			
    # Calculates the potential energies based on the magnitude vector and bond angle
    pe_bond_lengths = .5 * kOH * (lengths - eq_bond_length)**2
    pe_bond_angle = .5 * kA * (angle - eq_bond_angle)**2
	
    # Sums the potential energy of the bond lengths with the bond angle to get potential energy
    # of one molecule, then summing to get potential energy of each walker
    return np.sum(np.sum(pe_bond_lengths, axis = 2) + pe_bond_angle, axis=1)



# The lambda function below changes all instances of -inf or inf in a numpy 
# array to 0 assuming that the -inf or inf values result from divisions by 0
inf_to_zero = lambda dist: np.where(np.abs(dist) == np.inf, 0, dist)
    

# Input: 4D Array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: Three 1D arrays for Intermolecular Potential Energy, Coulombic energy, 
#         and Leonard Jones energy
# Calculates the intermolecular potential energy of a walker based on the 
# distances of the atoms in each walker from one another
def inter_pe(x):
    
    # Returns the atom positions between two distinct pairs of molecules 
    # in each walker. This broadcasts from a 4D array of walkers with axis 
    # dimesions (num_walkers, num_molecules, num_atoms, coord_const) to two 
    # arrays with dimesions (num_walkers, num_distinct_molecule_pairs, num_atoms
    # , coord_const), with the result being the dimensions:
    # (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const).


    # Create indexing arrays for the distinct pairs of water molecules in the 
    # potential energy calculation. Based on the idea that there are num_molecules 
    # choose 2 distinct molecular pairs.

    molecule_index = lambda n_mol: list(zip(*it.combinations(range(n_mol),2)))

    # These arrays line up such that the corresponding pairs on the second 
    # dimension are the distinct pairs of molecules
    molecule_index_a, molecule_index_b = molecule_index(x.shape[1]) 

    pairs_a = x[:,molecule_index_a]
    pairs_b = x[:,molecule_index_b]
    
    
    
    # Returns the distances between two atoms in each molecule pair. The 
    # distance array is now of dimension (num_walkers, num_distinct_pairs, 
    # num_atoms, num_atoms) as each atom in the molecule has its distance 
    # computed with each atom in the other molecule in the distinct pair.
    # This line works similar to numpy's matrix multiplication by broadcasting 
    # the 4D array to a higher dimesion and then taking the elementwise 
    # difference before squarring and then summing along the positions axis to 
    # collapse the array into distances.
    distances = np.sqrt( np.sum( (pairs_a[...,None] \
            - pairs_b[:,:,np.newaxis,...].transpose(0,1,2,4,3) )**2, axis=3) )
   
   
   
    # Calculate the Coulombic energy using Coulomb's Law of every walker. 
    # Distances is a 4D array and this division broadcasts to a 4D array of 
    # Coulombic energies where each element is the Coulombic energy of an atom 
    # pair in a distinct pair of water molecules. 
    # Summing along the last three axis gives the Coulombic energy of each 
    # walker. Note that we account for any instance of divide by zero by calling 
    # inf_to_zero on the result of dividing coulombic charges by distance.
    coulombic_energy = np.sum( inf_to_zero(coulombic_charges / distances), axis=(1,2,3))
    
    
    

    # Calculate the quotient of sigma with the distances between pairs of oxygen 
    # molecules Given that the Lennard Jones energy is only calculated for O-O 
    # pairs. By the initialization assumption, the Oxygen atom is always in the 
    # first index, so the Oxygen pair is in the (0,0) index in the last two 
    # dimensions of the 4D array with dimension (num_walkers,
    # num_distinct_molecule_pairs, num_atoms, coord_const).
    sigma_dist = inf_to_zero( sigma / distances[:,:,0,0] )
    
    # Calculate the Lennard Jones energy in accordance with the given equation
    # Sum along the first axis to get the total LJ energy in one walker.
    lennard_jones_energy = np.sum( 4*epsilon*(sigma_dist**12 - sigma_dist**6), axis = 1)
    
    
    
    # Gives the intermolecular potential energy for each walker as it is the sum 
    # of the Coulombic Energy and the Leonard Jones Energy.
    intermolecular_potential_energy = coulombic_energy + lennard_jones_energy
    
    
    
    # Return all three calculated energys which are 1D arrays of energy values 
    # for each walker
    return intermolecular_potential_energy, coulombic_energy, lennard_jones_energy

    
    
# Input: 4D array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: 1D array of the sum of the intermolecular and intramolecular potential 
# energy of each walker
def total_pe(x):
    # print(f"total_pe: x shape = {x.shape}, x dtype = {x.dtype}")
    # Calculate the intramolecular potential energy of each walker
    intra_potential_energy = intra_pe(x)
    
    # Calculate the intermolecular potential energy of each walker
    # only if there is more than one molecule in the system
    inter_potential_energy = 0
    if x.shape[1] > 1:
        inter_potential_energy, coulombic, lennard_jones = inter_pe(x)
    
    # Return the total potential energy of the walker
    return intra_potential_energy + inter_potential_energy



#######################################################################################
# Simulation loop
# Iterates over the walkers array, propogating each walker. Deletes and replicates those 
# walkers based on their potential energies with respect to the calculated reference energy


# Input:
# walkers: 4D numpy array (n_walkers, n_molecules, n_atoms, coord_const)
# sim_length: int. number of iterations of the main simulation loop
# dt: float. time step for simulation
# dw_save: int. interval (in number of sim steps) after which to save a snapshot
#   If value == 0, no snapshots will be saved
# do_dw: bool. If true, keep track of ancestors, return a bincount at end of loop
# Output: dict of various outputs
#   'w': walkers. ndarray -- shape: (n_walkers,n_molecules,n_atoms,coord_const) 
#   'r': reference energy at each time step. 1d array
#   'n': num_walkers at each time step. 1d array
#   's': snapshots. python list of walker 4D arrays
#   'a': ancestor_weights of each walker at sim end. 1d array
'''optimized for memory allocation with in-place operations and minima temporary arrays'''
def sim_loop(walkers, sim_length, dt, wf_save, equilibration_phase, dtype = np.float64):
    start_time = time.time() 
    n_walkers, num_molecules, n_atoms, coord_const = walkers.shape
    wave_func_snapshots = []
    num_walkers = np.zeros(sim_length)
    reference_energy = np.zeros(sim_length)

    for i in range(sim_length):
        # print out runtime and progress
        if i % (sim_length // 10) == 0 and i > 0:
            elapsed_time = time.time() - start_time
            progress = i / sim_length
            projected_total_time = elapsed_time / progress
            time_left = projected_total_time - elapsed_time
            print(f"Simulation progress: {progress * 100:.0f}%. (projected) time left: {time_left:.2f} seconds")
        # Propagate each walke
        walkers = walkers + np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses,
                (walkers.shape[walker_axis], num_molecules, coord_const, 1)),
            (walker_axis, molecule_axis, coord_axis, atom_axis))))

        # Calculate potential energies after propagation
        potential_energies = total_pe(walkers)

        # Calculate the reference energy
        reference_energy[i] = np.mean(potential_energies) + (1.0 - (walkers.shape[walker_axis] / n_walkers)) / (2.0 * dt)
        num_walkers[i] = walkers.shape[walker_axis]

        # Save snapshots if needed
        if wf_save > 0 and i % wf_save == 0 and i >= equilibration_phase:
            wave_func_snapshots.append(walkers)

        # branching step: decide which walkers to replicate or delete
        thresholds = np.random.uniform(0, 1, size=walkers.shape[0])
        B = np.exp(-dt * (potential_energies - reference_energy[i]))
        prob_replicate = B - 1
        prob_remain = B
        walkers_to_replicate = walkers[prob_replicate > thresholds]
        walkers_to_remain = walkers[prob_remain > thresholds]
        walkers = np.concatenate([walkers_to_remain, walkers_to_replicate])

    return {"E0": reference_energy, "walkers": wave_func_snapshots, "final_walkers": walkers, "populations": num_walkers}



def DW_trace(walkers, sim_length, dt):
    n_walkers, num_molecules, n_atoms, coord_const = walkers.shape 
    # print(walkers.shape)
    # ancestor_indices = np.arange(1, n_walkers+1, step = 1)
    ancestor_indices = np.arange(n_walkers, step = 1)
    initial_indices = ancestor_indices.copy()
    reference_energy = np.zeros(sim_length)
    # n_descendants_list = []
    # ancestor_indices_list = []
    for i in range(sim_length):
        walkers = walkers + np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses,
                (walkers.shape[walker_axis], num_molecules, coord_const, 1)),
                (walker_axis, molecule_axis, coord_axis, atom_axis))))
        # Calculate potential energies after propagation
        potential_energies = total_pe(walkers)
        # Calculate the reference energy
        reference_energy[i] = np.mean(potential_energies) + (1.0 - (walkers.shape[walker_axis] / n_walkers)) / (2.0 * dt)
        # branching step
        thresholds = np.random.rand(walkers.shape[walker_axis])     
        B = np.exp( -dt* ( (potential_energies) - reference_energy[i]) )
        prob_replicate, prob_remain = B - 1, B
        # replicate walkers
        replicate_mask = prob_replicate > thresholds
        walkers_to_replicate = walkers[replicate_mask]
        indices_to_replicate = ancestor_indices[replicate_mask]
        # delete/remain walkers
        remain_mask = prob_remain > thresholds
        walkers_to_remain = walkers[remain_mask]
        indices_to_remain = ancestor_indices[remain_mask]
        walkers = np.concatenate([walkers_to_remain, walkers_to_replicate])
        # print("n_walkers: ", walkers.shape[0])
        ancestor_indices = np.concatenate([indices_to_remain, indices_to_replicate])
        
        if i == sim_length - 1:
            # count the number of descendants for each ancestor
            n_descendants = [np.count_nonzero(ancestor_indices == i) for i in initial_indices]
    # return the weights
    # return np.array(n_descendants_list), ancestor_indices_list
    return n_descendants