# Sijing (Paul) Zhu, Spring 2024
# Diffusion Monte Carlo (DMC) Simulation with intramolecular harmonic guiding function

import os
# os.environ["OMP_NUM_THREADS"] = "1"
# import tensorflow as tf
import numpy as np
import itertools as it
import time
import sys
from datetime import datetime, timedelta
# from numba import njit, vectorize
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import numdifftools as nd 
# import autograd as auto
# from autograd import grad
# np.random.seed(0)
# tf.random.set_seed(10)

#####################################
# choice for potential function
potential = "qSPC/Fw"
# choice for trial functioppy

#####################################


###################################################################################
# Scientific Constants
hbar = 1
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


# reduced mass of the HOH angle bend
p_OH = 1/eq_bond_length 
mu_H = 1/atomic_masses[1]
mu_O = 1/atomic_masses[0]
G = 2* p_OH**2 * mu_H + (2*p_OH**2 - 2*p_OH**2*np.cos(eq_bond_angle)) * mu_O
reduced_mass_HOH = 1/G


# Calculate the reduced mass of the system
# Note that as the wave function is being graphed for an OH vector, we only consider the
# reduced mass of the OH vector system
reduced_mass = ((atomic_masses[0]+atomic_masses[1])*atomic_masses[2])/np.sum(atomic_masses)
reduced_mass = 1/((mu_H*2 + mu_O)*2)
# print(reduced_mass)
reduced_mass_OH = 1/(mu_H + mu_O)
# print(reduced_mass)


# print("reduced mass OH:", reduced_mass)
# print("reduced mass HOH: ", reduced_mass_HOH)

# diffusion constant
D = hbar**2 / (2 * reduced_mass)

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
    # Calculate the intramolecular potential energy of each walker
    intra_potential_energy = intra_pe(x)
    # Calculate the intermolecular potential energy of each walker
    # only if there is more than one molecule in the system
    inter_potential_energy = 0
    if x.shape[1] > 1:
        inter_potential_energy, coulombic, lennard_jones = inter_pe(x)
    # Return the total potential energy of the walker
    return intra_potential_energy + inter_potential_energy


# the function that calls different types of potential function
def V(x):
    if potential == "qSPC/Fw":
        return total_pe(x)

# print("reduced mass: angle,  ", reduced_mass_HOH)
# Input: 4D array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: 1D array of the value of trial function at each walker position
# @njit(parallel = True)
def trial_function(walkers, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype = np.float32):
    walkers = np.array(walkers, dtype=dtype)
    # Return the two OH vectors
    # Used to calculate the bond lengths and angle in a molecule
    OH_vectors = walkers[:,:,1:] - walkers[:,:,np.newaxis,0]
    # Returns the lengths of each OH bond vector for each molecule 
    # in each walker. 
    # lengths = np.linalg.norm(OH_vectors, axis=3)
    lengths = np.sqrt(np.sum(OH_vectors**2, axis=3))
    # Calculate the dot product between the normalized OH vectors to obtain the bond angle
    normalized_OH_vectors = OH_vectors / lengths[:, :, :, np.newaxis]
    angle = np.arccos(np.sum(normalized_OH_vectors[:, :, 0] * normalized_OH_vectors[:, :, 1], axis=2))
    bond_lengths_modes = .5 * np.sqrt(kOH * reduced_mass_OH) * (lengths - eq_bond_length)**2
    bond_angle_modes = .5 * np.sqrt(kA * reduced_mass_HOH) * (angle - eq_bond_angle)**2
    return np.exp(-np.sum(np.sum(bond_lengths_modes, axis = 2) + bond_angle_modes, axis=1))


# repeat the given array n_coord_prime as element and form a new array. 
# new axis formed at the last position
def repeat_elements(arr, n_coord_prime):
    # Reshape the input array to have shape (n_walkers, n_coord, 1)
    reshaped_arr = arr[:, :, np.newaxis]
    # Repeat the elements along the n_coord axis by n_coord' times
    repeated_arr = np.repeat(reshaped_arr, n_coord_prime, axis=2)
    return repeated_arr


# Input: 4D array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: gradients of same shape as walker 
def trial_function_numerical_d1(walkers, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, epsilon = 1e-3, dtype = np.float64):
    walkers = np.array(walkers, dtype=dtype)
    '''allow smaller finite step for more accurate float carrier'''
    if dtype == np.float32:
        epsilon = 5e-3
    elif dtype == np.float64:
        epsilon = 1e-3
    n_walkers, n_molecules, n_atoms, n_coordinates = walkers.shape
    # walkers = np.array(walkers, dtype=np.float64)
    # now the flatten_walkers should have shape (n_walkers, n_molecules*n_atoms*n_coordinates)
    flattened_walkers = np.reshape(walkers, newshape=(n_walkers, n_molecules*n_atoms*n_coordinates))
    # for each walker, repeat the coordinates by n_walkers, n_molecules*n_atoms*n_coordinates times
    # now the walkers have shape (n_walkers, n_molecules*n_atoms*n_coordinates, rep = n_molecules*n_atoms*n_coordinates)
    # the copies are created along axis = 2
    rep = n_molecules*n_atoms*n_coordinates
    repeated_walkers = repeat_elements(flattened_walkers, rep)
    # apply pertubations only along the diagonal entries along the last two axes:
    perturbation = np.eye(rep) * epsilon
    walkers_plus = repeated_walkers + perturbation
    walkers_minus = repeated_walkers - perturbation
    # transpose the arrays by moving the rep axis to the front
    walkers_plus = np.moveaxis(walkers_plus, (0,1,2), (1,2,0))
    walkers_minus = np.moveaxis(walkers_minus, (0,1,2), (1,2,0))
    # walkers_plus = walkers_plus.transpose(1, 2, 0)
    # walkers_minus = walkers_minus.transpose(1, 2, 0)
    # reshape the arrays inorder to plug in intro trial_function()
    walkers_plus = np.reshape(walkers_plus, newshape = (rep * n_walkers, n_molecules, n_atoms, n_coordinates))
    walkers_minus = np.reshape(walkers_minus, newshape = (rep * n_walkers, n_molecules, n_atoms, n_coordinates))
    # calculate the values, those should have shape (rep*nwalkers)
    values_plus = trial_function(walkers_plus, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype=dtype)
    values_minus = trial_function(walkers_minus, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype = dtype)
    # reshape the values into shape (rep, nwalkers)
    values_plus = np.reshape(values_plus, newshape = (rep, n_walkers))
    values_minus = np.reshape(values_minus, newshape = (rep, n_walkers))
    # calculate the gradient, it should have shape (nwalkers, rep)
    gradients = (values_plus - values_minus) / (2*epsilon)
    # transpose the values, those should have shape (nwalkers, rep)
    gradients = np.transpose(gradients)
    # reshape the gradients back to (nwalkers, n_molecules, n_atoms, n_coordinates)
    gradients = np.reshape(gradients, newshape=(n_walkers, n_molecules, n_atoms, n_coordinates))
    return np.array(gradients)


# Input: 4D array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: gradients of same shape as walker 
def trial_function_numerical_d2(walkers, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, epsilon = 1e-3, dtype = np.float64):
    walkers = np.array(walkers, dtype=dtype)
    '''allow smaller finite step for more accurate float carrier'''
    if dtype == np.float32:
        epsilon = 5e-3
    elif dtype == np.float64:
        epsilon = 1e-3
    n_walkers, n_molecules, n_atoms, n_coordinates = walkers.shape
    # walkers = np.array(walkers, dtype=np.float64)
    # now the flatten_walkers should have shape (n_walkers, n_molecules*n_atoms*n_coordinates)
    flattened_walkers = np.reshape(walkers, newshape=(n_walkers, n_molecules*n_atoms*n_coordinates))
    # for each walker, repeat the coordinates by n_walkers, n_molecules*n_atoms*n_coordinates times
    # now the walkers have shape (n_walkers, n_molecules*n_atoms*n_coordinates, rep = n_molecules*n_atoms*n_coordinates)
    # the copies are created along axis = 2
    rep = n_molecules*n_atoms*n_coordinates
    repeated_walkers = repeat_elements(flattened_walkers, rep)
    # apply pertubations only along the diagonal entries along the last two axes:
    perturbation = np.eye(rep) * epsilon
    walkers_plus = repeated_walkers + perturbation
    walkers_minus = repeated_walkers - perturbation
    # transpose the arrays by moving the rep axis to the front
    walkers_plus = np.moveaxis(walkers_plus, (0,1,2), (1,2,0))
    walkers_minus = np.moveaxis(walkers_minus, (0,1,2), (1,2,0))
    # walkers_plus = walkers_plus.transpose(1, 2, 0)
    # walkers_minus = walkers_minus.transpose(1, 2, 0)
    # reshape the arrays inorder to plug in intro trial_function()
    walkers_plus = np.reshape(walkers_plus, newshape = (rep * n_walkers, n_molecules, n_atoms, n_coordinates))
    walkers_minus = np.reshape(walkers_minus, newshape = (rep * n_walkers, n_molecules, n_atoms, n_coordinates))
    # calculate the values, those should have shape (rep*nwalkers)
    values_plus = trial_function(walkers_plus, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype=dtype)
    values_minus = trial_function(walkers_minus, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype = dtype)
    # reshape the values into shape (rep, nwalkers)
    values_plus = np.reshape(values_plus, newshape = (rep, n_walkers))
    values_minus = np.reshape(values_minus, newshape = (rep, n_walkers))
    values = trial_function(walkers, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype=dtype)
    # calculate the gradient, it should have shape (nwalkers, rep)
    gradients = (values_plus + values_minus - 2*values) / (epsilon**2)
    # transpose the values, those should have shape (nwalkers, rep)
    gradients = np.transpose(gradients)
    # reshape the gradients back to (nwalkers, n_molecules, n_atoms, n_coordinates)
    gradients = np.reshape(gradients, newshape=(n_walkers, n_molecules, n_atoms, n_coordinates))
    return np.array(gradients)


# # input: modes: shape of (nwalkers, n_molecules)
# # output: d_trial_d_molecule_modes: shape of (nwalkers, n_molecules)
def get_d_trial_d_molecule_modes(modes):
    return np.prod(modes, axis=1, keepdims=True) / modes

# # input: OH_modes:  shape of (nwalker, n_molecules, 2)
def get_d_molecule_modes_d_OH_modes(OH_modes, HOH_modes):
    OH_1_modes = OH_modes[:, :, 0]
    OH_2_modes = OH_modes[:, :, 1]
    d1 = OH_2_modes * HOH_modes
    d2 = OH_1_modes * HOH_modes
    return np.stack((d1, d2), axis=-1)

# # input: OH_lengths: shape of (nwalkers, n_molecules, 2)
# # input: OH_modes:   shape of (nwalkers, n_molecules, 2)
def get_d_OH_modes_d_lengths(OH_lengths, OH_modes, kOH, reduced_mass_OH, eq_bond_length):
    return -np.sqrt(kOH * reduced_mass_OH) * (OH_lengths - eq_bond_length) * OH_modes

# # input: OH_1_lengths:       shape of (nwalkers, n_molecules,)
# # input: coords (walkers): shape of (nwalkers, n_molecules, n_atoms, 3)
# # output: coords (walkers):shape of (nwalkers, n_molecules, n_atoms, 3)
def get_d_lengths_1_d_coords(OH_1_lengths, walkers):
    d_lengths_d_coords_O = (walkers[:, :, 0, :] - walkers[:, :, 1, :]) / OH_1_lengths[..., np.newaxis]
    d_lengths_d_coords_H1 = -d_lengths_d_coords_O
    d_lengths_d_coords_H2 = np.zeros((walkers.shape[0], walkers.shape[1], walkers.shape[3]), dtype=walkers.dtype)
    d_lengths_coords = np.stack([d_lengths_d_coords_O, d_lengths_d_coords_H1, d_lengths_d_coords_H2], axis=2)
    return d_lengths_coords

# # input: OH_2_lengths:       shape of (nwalkers, n_molecules,)
# # input: coords (walkers): shape of (nwalkers, n_molecules, n_atoms, 3)
# # output: coords (walkers):shape of (nwalkers, n_molecules, n_atoms, 3)
def get_d_lengths_2_d_coords(OH_2_lengths, walkers):
    d_lengths_d_coords_O = (walkers[:, :, 0, :] - walkers[:, :, 2, :]) / OH_2_lengths[..., np.newaxis]
    d_lengths_d_coords_H1 = np.zeros((walkers.shape[0], walkers.shape[1], walkers.shape[3]), dtype=walkers.dtype)
    d_lengths_d_coords_H2 = -d_lengths_d_coords_O
    d_lengths_coords = np.stack([d_lengths_d_coords_O, d_lengths_d_coords_H1, d_lengths_d_coords_H2], axis=2)
    return d_lengths_coords

# # input: OH_modes:  shape of (nwalkers, n_molecules, 2)
# # output: d_HOH_modes shape of (nwalkers, n_molecules)
def get_d_molecule_modes_d_HOH_modes(OH_modes):
    return np.prod(OH_modes, axis=2)

# # input: angles: shape of (nwalkers, n_molecules)
# # input: HOH mdoes: shape of (nwalkers, n_molecules)
# # output: d_angles: shape of (nwalkers, n_molecules)
def get_d_HOH_modes_d_angles(angles, HOH_modes, kA, reduced_mass_HOH, eq_bond_angle):
    return -np.sqrt(kA * reduced_mass_HOH) * (angles - eq_bond_angle) * HOH_modes

# # input: OH_vectors: shape of (nwalkers, n_molecules, 2, 3)
# # input: OH_lengths: shape of (nwalkers, n_molecules, 2)
# # input: walkers:    shape of (nwalkers, n_molecules, natoms, 3)
def get_d_angles_d_coords(OH_vectors, OH_lengths, walkers):
    OH_1_vectors = OH_vectors[:, :, 0, :]
    OH_2_vectors = OH_vectors[:, :, 1, :]
    dot = np.einsum('ijk,ijk->ij', OH_1_vectors, OH_2_vectors)[..., np.newaxis]
    OH_length_1 = OH_lengths[:, :, 0][..., np.newaxis]
    OH_length_2 = OH_lengths[:, :, 1][..., np.newaxis]
    prod_length = OH_length_1 * OH_length_2
    beta = 1 / np.sqrt(1 - dot**2 / (prod_length)**2)
    d_angles_d_coords_O = (- (2 * walkers[:, :, 0, :] - walkers[:, :, 1, :] - walkers[:, :, 2, :]) / (prod_length)
                           + dot / (OH_length_1**3 * OH_length_2) * (walkers[:, :, 0, :] - walkers[:, :, 1, :])
                           + dot / (OH_length_1 * OH_length_2**3) * (walkers[:, :, 0, :] - walkers[:, :, 2, :]))
    d_angles_d_coords_H1 = ((walkers[:, :, 0, :] - walkers[:, :, 2, :]) / (prod_length) - dot / (OH_length_1**3 * OH_length_2) * (walkers[:, :, 0, :] - walkers[:, :, 1, :]))
    d_angles_d_coords_H2 = ((walkers[:, :, 0, :] - walkers[:, :, 1, :]) / (prod_length) - dot / (OH_length_1 * OH_length_2**3) * (walkers[:, :, 0, :] - walkers[:, :, 2, :]))
    d_angles_d_coords = beta[..., np.newaxis] * np.stack([d_angles_d_coords_O, d_angles_d_coords_H1, d_angles_d_coords_H2], axis=2)
    return d_angles_d_coords


def trial_function_analytical_d1(walkers, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype=np.float32):
    '''forward pass on trial function'''
    walkers = np.array(walkers, dtype=dtype)
    OH_vectors = walkers[:, :, 1:] - walkers[:, :, np.newaxis, 0]
    OH_lengths = np.linalg.norm(OH_vectors, axis=3)
    normalized_OH_vectors = OH_vectors / np.expand_dims(OH_lengths, -1)
    angles = np.arccos(np.sum(normalized_OH_vectors[:, :, 0] * normalized_OH_vectors[:, :, 1], axis=2))
    # Calculate OH_modes and HOH_modes
    OH_modes = np.exp(-0.5 * np.sqrt(kOH * reduced_mass_OH) * (OH_lengths - eq_bond_length)**2)
    HOH_modes = np.exp(-0.5 * np.sqrt(kA * reduced_mass_HOH) * (angles - eq_bond_angle)**2)
    # Molecule modes
    molecule_modes = np.prod(OH_modes, axis=2) * HOH_modes
    d_trial_d_molecule_modes = get_d_trial_d_molecule_modes(molecule_modes)[..., np.newaxis, np.newaxis]
    
    '''bond length backprop'''
    d_molecule_modes_d_OH_modes = get_d_molecule_modes_d_OH_modes(OH_modes, HOH_modes)[..., np.newaxis]
    d_OH_modes_d_lengths = get_d_OH_modes_d_lengths(OH_lengths, OH_modes, kOH, reduced_mass_OH, eq_bond_length)[..., np.newaxis]
    d_lengths_1_d_coords = get_d_lengths_1_d_coords(OH_lengths[:, :, 0], walkers)
    d_lengths_2_d_coords = get_d_lengths_2_d_coords(OH_lengths[:, :, 1], walkers)
    d_molecule_modes_d_length = d_molecule_modes_d_OH_modes * d_OH_modes_d_lengths
    # combind the gradients for bond length flows shape (nWalkers, mMolecules, natoms, 3)
    OH_d_molecule_modes_d_coords = d_molecule_modes_d_length[:, :, 0:1] * d_lengths_1_d_coords + d_molecule_modes_d_length[:, :, 1:] * d_lengths_2_d_coords
    
    '''bond angle backprop'''
    d_molecule_modes_d_HOH_modes = get_d_molecule_modes_d_HOH_modes(OH_modes)[..., np.newaxis, np.newaxis]
    d_HOH_modes_d_angles = get_d_HOH_modes_d_angles(angles, HOH_modes, kA, reduced_mass_HOH, eq_bond_angle)[..., np.newaxis, np.newaxis]
    d_angles_d_coords = get_d_angles_d_coords(OH_vectors, OH_lengths, walkers)
    # combine the gradients for bond angle flows shape (nWalkers, nMolecules, natoms, 3)
    HOH_d_molecule_modes_d_coords = d_molecule_modes_d_HOH_modes * d_HOH_modes_d_angles * d_angles_d_coords
    
    '''sum up all modes with shape, return gradients with shape (nWalkers, nMolecules, natoms, 3)'''
    d_molecule_modes_d_coords = OH_d_molecule_modes_d_coords + HOH_d_molecule_modes_d_coords
    d_trial_d_coords = d_trial_d_molecule_modes * d_molecule_modes_d_coords
    return d_trial_d_coords



# chain rule of d/dx (a*b*c*d), given d/dx (a) = 0
def chain_rule(a, b, c, m, d_b, d_c, d_m):
    return a * (d_b * c * m + b * d_c * m + b * c * d_m)

# d/dx (d_molecule_d_OH_mode_1) for OH bond 1
def get_d_d_molecule_d_OH_mode_1(OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords):
    return OH_mode_2 * d_HOH_mode_d_angle * d_angle_d_coords + HOH_mode[..., np.newaxis, np.newaxis] * d_OH_mode_2_d_length_2 * d_length_2_d_coords

# d/dx (d_OH_mode_1_d_length_1) for OH bond 1
def get_d_d_OH_mode_1_d_length_1(OH_length_1, OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, kOH = kOH, reduced_mass_OH = reduced_mass_OH):
    return -np.sqrt(kOH*reduced_mass_OH) * (d_length_1_d_coords * OH_mode_1 + d_OH_mode_1_d_length_1 * d_length_1_d_coords * (OH_length_1 - eq_bond_length))

# d/dx (d_length_1_d_coords) for OH bond 1
def get_d_d_length_1_d_coords(walkers, OH_length_1):
    OH_length_1 = OH_length_1[:,:,:,0]
    d_coords = np.zeros_like(walkers, dtype=np.float64)
    d = -(walkers[:,:,0,:] - walkers[:,:,1,:])**2/OH_length_1**3 + 1/OH_length_1
    d_coords[:,:,0,:] = d
    d_coords[:,:,1,:] = d
    return d_coords

# d/dx (d_molecule_d_OH_mode_2) for OH bond 2    
def get_d_d_molecule_d_OH_mode_2(OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords):
    return OH_mode_1 * d_HOH_mode_d_angle * d_angle_d_coords + HOH_mode[..., np.newaxis, np.newaxis] * d_OH_mode_1_d_length_1 * d_length_1_d_coords

# d/dx (d_OH_mode_2_d_length_2) for OH bond 2
def get_d_d_OH_mode_2_d_length_2(OH_length_2, OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, kOH = kOH, reduced_mass_OH = reduced_mass_OH):
    return -np.sqrt(kOH*reduced_mass_OH) * (d_length_2_d_coords * OH_mode_2 + d_OH_mode_2_d_length_2 * d_length_2_d_coords * (OH_length_2 - eq_bond_length))

# d/dx (d_length_2_d_coords) for OH bond 2
def get_d_d_length_2_d_coords(walkers, OH_length_2):
    OH_length_2 = OH_length_2[:,:,:,0]
    d_coords = np.zeros_like(walkers, dtype=np.float64)
    d = -(walkers[:,:,0,:] - walkers[:,:,2,:])**2/OH_length_2**3 + 1/OH_length_2
    d_coords[:,:,0,:] = d
    d_coords[:,:,2,:] = d
    return d_coords

# d/dx (d_molecule_d_HOH_mode) for HOH mode
def get_d_d_molecule_d_HOH_mode(OH_mode_1,d_OH_mode_1_d_length_1, d_length_1_d_coords, OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords):
    return OH_mode_1 * d_OH_mode_2_d_length_2 * d_length_2_d_coords + OH_mode_2 * d_OH_mode_1_d_length_1 * d_length_1_d_coords

# d/dx (d_HOH_mode_d_angle)
def get_d_d_HOH_mode_d_angle(angles, HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords, kA = kA, reduced_mass_HOH = reduced_mass_HOH):
    return -np.sqrt(kA*reduced_mass_HOH) * (d_angle_d_coords * HOH_mode[..., np.newaxis, np.newaxis] + d_HOH_mode_d_angle * d_angle_d_coords * (angles[..., np.newaxis, np.newaxis] - eq_bond_angle))

# d/dx (d_angle_d_coords)
def get_d_d_angle_d_coords(walkers, OH_vectors, OH_lengths):
    OH_lengths = OH_lengths[:,:,:,0]
    OH_1_vectors = OH_vectors[:, :, 0, :]
    OH_2_vectors = OH_vectors[:, :, 1, :]
    dot = np.einsum('ijk,ijk->ij', OH_1_vectors, OH_2_vectors)[..., np.newaxis]
    l1 = OH_lengths[:, :, 0][..., np.newaxis]
    l2 = OH_lengths[:, :, 1][..., np.newaxis]
    prod_length = l1 * l2
    beta1 = 1 / np.sqrt(1 - dot**2 / prod_length**2)
    beta2 = beta1**3
    # x is just a chosen symbol, the same operation is broadcast to x,y,z coords
    x0, x1, x2 = walkers[:,:,0,:], walkers[:,:,1,:], walkers[:,:,2,:]
    
    d_angles_d_coords_O =  -(   beta1*( ((-3*x0+3*x1)*(-x0+x1)*dot/(l1**5*l2)) + ((-3*x0+3*x2)*(-x0+x2)*dot/(l1*l2**5)) +
                                      (2*(-x0+x1)*(-x0+x2)*dot)/(prod_length**3) + (2*(-x0+x1)*(2*x0-x1-x2)/(l1**3*l2)) +
                                      (2*(-x0+x2)*(2*x0-x1-x2)/(l1*l2**3)) - (dot/(l1*l2**3)) - (dot/(l1**3*l2)) + (2/(prod_length))
                                    ) +
                                beta2*( ((-2*x0+2*x1)*dot**2/(2*l1**4*l2**2) + (-2*x0+2*x2)*dot**2/(2*l1**2*l2**4) + (4*x0-2*x1-2*x2)*dot/(2*l1**2*l2**2))*
                                      ((-x0+x1)*dot/(l1**3*l2) + (-x0+x2)*dot/(l1*l2**3) + (2*x0-x1-x2)/(l1*l2))
                                    )
                            )
    d_angles_d_coords_H1 = -(   beta1*( (2*(-x0+x2)*(x0-x1)/(l1**3*l2) + (x0-x1)*(3*x0-3*x1)*dot/(l1**5*l2) - dot/(l1**3*l2))    
                                    ) +
                                beta2*( ((-x0+x2)/(prod_length) + (x0-x1)*dot/(l1**3*l2)) * ((-2*x0+2*x2)*dot/(2*l1**2*l2**2) + (2*x0-2*x1)*dot**2/(2*l1**4*l2**2))
                              )
                            )
    d_angles_d_coords_H2 = -(   beta1*( (2*(-x0+x1)*(x0-x2)/(l1*l2**3) + (x0-x2)*(3*x0-3*x2)*dot/(l1*l2**5) - dot/(l1*l2**3))    
                                    ) +
                                beta2*( ((-x0+x1)/(prod_length) + (x0-x2)*dot/(l1*l2**3)) * ((-2*x0+2*x1)*dot/(2*l1**2*l2**2) + (2*x0-2*x2)*dot**2/(2*l1**2*l2**4))
                              )
                            )
    d_angles_d_coords = np.stack([d_angles_d_coords_O, d_angles_d_coords_H1, d_angles_d_coords_H2], axis=2)
    return d_angles_d_coords

# OH 1 chain rule
def OH_flow_1_d2(d_trial_d_molecule, d_molecule_d_OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, d_d_molecule_d_OH_mode_1, d_d_OH_mode_1_d_length_1, d_d_length_1_d_coords):
    return chain_rule(d_trial_d_molecule, d_molecule_d_OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, d_d_molecule_d_OH_mode_1, d_d_OH_mode_1_d_length_1, d_d_length_1_d_coords)

# OH 2 chain rule
def OH_flow_2_d2(d_trial_d_molecule, d_molecule_d_OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, d_d_molecule_d_OH_mode_2, d_d_OH_mode_2_d_length_2, d_d_length_2_d_coords):
    return chain_rule(d_trial_d_molecule, d_molecule_d_OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, d_d_molecule_d_OH_mode_2, d_d_OH_mode_2_d_length_2, d_d_length_2_d_coords)

# HOH chain rule
def HOH_flow_d2(d_trial_d_molecule, d_molecule_d_HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords, d_d_molecule_d_HOH_mode, d_d_HOH_mode_d_angle, d_d_angle_d_coords):
    return chain_rule(d_trial_d_molecule, d_molecule_d_HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords, d_d_molecule_d_HOH_mode, d_d_HOH_mode_d_angle, d_d_angle_d_coords)

# assemble all second order gradients
def trial_d2(d_trial_d_molecule, d_molecule_d_OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, d_d_molecule_d_OH_mode_1, d_d_OH_mode_1_d_length_1, d_d_length_1_d_coords, d_molecule_d_OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, d_d_molecule_d_OH_mode_2, d_d_OH_mode_2_d_length_2, d_d_length_2_d_coords, d_molecule_d_HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords, d_d_molecule_d_HOH_mode, d_d_HOH_mode_d_angle, d_d_angle_d_coords):
    OH_flow_1 = OH_flow_1_d2(d_trial_d_molecule, d_molecule_d_OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, d_d_molecule_d_OH_mode_1, d_d_OH_mode_1_d_length_1, d_d_length_1_d_coords)
    OH_flow_2 = OH_flow_2_d2(d_trial_d_molecule, d_molecule_d_OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, d_d_molecule_d_OH_mode_2, d_d_OH_mode_2_d_length_2, d_d_length_2_d_coords)
    HOH_flow = HOH_flow_d2(d_trial_d_molecule, d_molecule_d_HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords, d_d_molecule_d_HOH_mode, d_d_HOH_mode_d_angle, d_d_angle_d_coords)
    return OH_flow_1 + OH_flow_2 + HOH_flow 
    
# analytical second order gradients
def trial_function_analytical_d2(walkers, kOH = kOH, reduced_mass_OH = reduced_mass_OH, eq_bond_length = eq_bond_length, kA = kA, reduced_mass_HOH = reduced_mass_HOH, eq_bond_angle=eq_bond_angle, dtype = np.float32):
    '''forward pass on trial function'''
    walkers = np.array(walkers, dtype=dtype)
    OH_vectors = walkers[:, :, 1:] - walkers[:, :, np.newaxis, 0]
    OH_lengths = np.linalg.norm(OH_vectors, axis=3)
    normalized_OH_vectors = OH_vectors / np.expand_dims(OH_lengths, -1)
    angles = np.arccos(np.sum(normalized_OH_vectors[:, :, 0] * normalized_OH_vectors[:, :, 1], axis=2))
    # calculate OH_modes and HOH_modes
    OH_modes = np.exp(-0.5 * np.sqrt(kOH * reduced_mass_OH) * (OH_lengths - eq_bond_length)**2)
    HOH_modes = np.exp(-0.5 * np.sqrt(kA * reduced_mass_HOH) * (angles - eq_bond_angle)**2)
    # molecule modes
    molecule_modes = np.prod(OH_modes, axis=2) * HOH_modes
    d_trial_d_molecule = get_d_trial_d_molecule_modes(molecule_modes)[..., np.newaxis, np.newaxis]
    
    '''bond length first order backprop'''
    d_molecule_d_OH_modes = get_d_molecule_modes_d_OH_modes(OH_modes, HOH_modes)[..., np.newaxis]
    d_molecule_d_OH_mode_1 = d_molecule_d_OH_modes[:,:,0:1,:]
    d_molecule_d_OH_mode_2 = d_molecule_d_OH_modes[:,:,1:,:]
    
    d_OH_modes_d_lengths = get_d_OH_modes_d_lengths(OH_lengths, OH_modes, kOH, reduced_mass_OH, eq_bond_length)[..., np.newaxis]
    d_OH_mode_1_d_length_1 = d_OH_modes_d_lengths[:,:,0:1,:]
    d_OH_mode_2_d_length_2 = d_OH_modes_d_lengths[:,:,1:,:]
    
    d_length_1_d_coords = get_d_lengths_1_d_coords(OH_lengths[:, :, 0], walkers)
    d_length_2_d_coords = get_d_lengths_2_d_coords(OH_lengths[:, :, 1], walkers)

    '''bond angle first order backprop'''
    d_molecule_d_HOH_mode = get_d_molecule_modes_d_HOH_modes(OH_modes)[..., np.newaxis, np.newaxis]
    d_HOH_mode_d_angle = get_d_HOH_modes_d_angles(angles, HOH_modes, kA, reduced_mass_HOH, eq_bond_angle)[..., np.newaxis, np.newaxis]
    d_angle_d_coords = get_d_angles_d_coords(OH_vectors, OH_lengths, walkers)

    '''d2 for OH modes'''
    OH_lengths = OH_lengths[..., np.newaxis]
    OH_length_1 = OH_lengths[:,:,0:1,:]
    OH_length_2 = OH_lengths[:,:,1:,:]
    OH_modes = OH_modes[..., np.newaxis]
    OH_mode_1 = OH_modes[:,:,0:1,:]
    OH_mode_2 = OH_modes[:,:,1:,:]
    
    d_d_molecule_d_OH_mode_1 = get_d_d_molecule_d_OH_mode_1(OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, HOH_modes, d_HOH_mode_d_angle, d_angle_d_coords)
    d_d_molecule_d_OH_mode_2 = get_d_d_molecule_d_OH_mode_2(OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, HOH_modes, d_HOH_mode_d_angle, d_angle_d_coords)
    d_d_OH_mode_1_d_length_1 = get_d_d_OH_mode_1_d_length_1(OH_length_1, OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, kOH, reduced_mass_OH)
    d_d_OH_mode_2_d_length_2 = get_d_d_OH_mode_2_d_length_2(OH_length_2, OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, kOH, reduced_mass_OH)
    d_d_length_1_d_coords = get_d_d_length_1_d_coords(walkers, OH_length_1)
    d_d_length_2_d_coords = get_d_d_length_2_d_coords(walkers, OH_length_2)
    
    '''d2 for HOH mode'''
    d_d_molecule_d_HOH_mode = get_d_d_molecule_d_HOH_mode(OH_mode_1,d_OH_mode_1_d_length_1, d_length_1_d_coords, OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords)
    d_d_HOH_mode_d_angle = get_d_d_HOH_mode_d_angle(angles, HOH_modes, d_HOH_mode_d_angle, d_angle_d_coords, kA, reduced_mass_HOH)
    d_d_angle_d_coords = get_d_d_angle_d_coords(walkers, OH_vectors, OH_lengths)

    '''combine the gradients via chain rule'''
    return trial_d2(d_trial_d_molecule, d_molecule_d_OH_mode_1, d_OH_mode_1_d_length_1, d_length_1_d_coords, d_d_molecule_d_OH_mode_1, d_d_OH_mode_1_d_length_1, d_d_length_1_d_coords, d_molecule_d_OH_mode_2, d_OH_mode_2_d_length_2, d_length_2_d_coords, d_d_molecule_d_OH_mode_2, d_d_OH_mode_2_d_length_2, d_d_length_2_d_coords, 
                    d_molecule_d_HOH_mode, d_HOH_mode_d_angle, d_angle_d_coords, d_d_molecule_d_HOH_mode, d_d_HOH_mode_d_angle, d_d_angle_d_coords)
    
    
# Input: 4D array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: 1D array of the local energy at each walker position
def local_energy(x, num_molecules, trial_function_d2, potentials, atomic_masses = atomic_masses, coord_const = coord_const, dtype = np.float32):
    # shape of (nwalkers, nmolecules, natoms, 1)
    masses = np.transpose(np.tile(atomic_masses, (x.shape[walker_axis], num_molecules, coord_const, 1)), (walker_axis, molecule_axis, coord_axis, atom_axis))
    return (np.sum( -hbar**2 / (2*masses) * trial_function_d2, axis = (1,2,3)) ) / trial_function(x, dtype=dtype) + potentials 


# Input: 4D array of walkers
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
#       n_atoms and coordinates are always both 3
# Output: 4D array of the quantum force at each walker position and each coordinate
# Dimensions of (n_walkers, n_molecules, n_atoms, coordinates)
def Fq(trial_values, d1):
    return 2 * d1 / trial_values[:, np.newaxis, np.newaxis, np.newaxis]

'''---------the drift term of given walker-------------'''
def drift_term(trial_values, d1, num_molecules, timeStep, atomic_masses = atomic_masses, coord_const = coord_const):
    masses = np.transpose(np.tile(atomic_masses, (trial_values.shape[walker_axis], num_molecules, coord_const, 1)), \
                          (walker_axis, molecule_axis, coord_axis, atom_axis))
    return 0.5 * timeStep / masses * Fq(trial_values, d1)

def diffusion_term(x, timeStep, num_molecules, coord_const, atomic_masses = atomic_masses):
    masses = np.transpose(np.tile(atomic_masses, (x.shape[walker_axis], num_molecules, coord_const, 1)), (walker_axis, molecule_axis, coord_axis, atom_axis))
    propagations = np.random.normal(0, np.sqrt(timeStep/masses))
    return propagations

'''--------natural log of green's function---------'''
def ln_Green(x, x_prime, timeStep, num_molecules, trial_values, d1, atomic_masses = atomic_masses, coord_const = coord_const):
    masses = np.transpose(np.tile(atomic_masses, (x.shape[walker_axis], num_molecules, coord_const, 1)), (walker_axis, molecule_axis, coord_axis, atom_axis))
    ln_green = -1 * np.sum((x_prime-x-hbar**2/(2*masses)*timeStep*Fq(trial_values, d1))**2 / (4*(hbar**2 / (2*masses) * timeStep)), axis = (1,2,3))
    return ln_green



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
def sim_loop(walkers,sim_length,dt,wf_save,equilibration_phase, dtype = np.float64, free_steps = 20, fast_steps = 2500):
    start_time = time.time()  # Record the start time of the simulation
    walkers = np.copy(walkers)
    # Extract initial size constants from walkers array
    n_walkers, num_molecules, n_atoms, coord_const = walkers.shape 
    wave_func_snapshots = []
    populations = np.zeros(shape = sim_length)
    E0_estimations = np.zeros(shape = sim_length)
    E0_intras = np.zeros(shape = sim_length)
    E0_inters = np.zeros(shape = sim_length)

   ### simulation loop
    for i in range(sim_length):
        if i < fast_steps:
            dt = 10
        else:
            dt = 1
        # print out runtime and progress
        if i % (sim_length // 10) == 0 and i > 0:
            elapsed_time = time.time() - start_time
            progress = i / sim_length
            projected_total_time = elapsed_time / progress
            time_left = projected_total_time - elapsed_time

            current_time = datetime.now().strftime("%b %d %H:%M")
            estimated_completion_time = datetime.now() + timedelta(seconds=time_left)
            estimated_completion_str = estimated_completion_time.strftime("%b %d %H:%M")

            print(f"Current time: {current_time} | Simulation progress: {progress * 100:.0f}% | Estimated completion time: {estimated_completion_str}")
        
        x = walkers
        trial_values = trial_function(x, dtype=dtype)
        d1 = trial_function_analytical_d1(x, dtype=dtype)
        
        # calculate diffusion and drift
        diffusion, drift = diffusion_term(walkers, dt, num_molecules, coord_const), drift_term(trial_values, d1, num_molecules, dt)
        movement = diffusion + drift
        x_prime = walkers + movement
        
        trial_values_prime = trial_function(x_prime, dtype=dtype)
        d1_prime = trial_function_analytical_d1(x_prime, dtype=dtype)
        # local energy of walkers before and after movement (if all movement is accepted)
        # local_energy_old = local_energy_restore
        # calculate the metropolis term (the probability of moving from x to x')
        ln_w = 2*np.log(trial_values_prime) + ln_Green(x_prime, x, dt, num_molecules, trial_values_prime, d1_prime) - 2*np.log(trial_values) - ln_Green(x, x_prime, dt, num_molecules, trial_values, d1)
        # decide the acceptance of movement
        ln_rand = np.log(np.random.uniform(0, 1, size = walkers.shape[0]))
        booleans = ln_w > ln_rand    # if w is greater than 1 the movement will be definetly accepted, otherwise the probability is w
        # diffuse and drift the walkers with movement accepted
        walkers = x + booleans[:, np.newaxis, np.newaxis, np.newaxis] * movement
        # walkers = newPositions
        
        # Save snapshots if needed
        if wf_save > 0 and ((i+1) % wf_save) == 0 and i > equilibration_phase:
            wave_func_snapshots.append(walkers)
        
        # calcualte the local energy of walkers after the movement (with some movement rejected)
        trial_function_d2 = trial_function_analytical_d2(walkers)
        intras = intra_pe(walkers)
        inters, c, c = inter_pe(walkers)
        potentials = intras + inters
        local_energy_new = local_energy(walkers, num_molecules, trial_function_d2, potentials, dtype=dtype)
        # update ZPE estimation in current iteration
        E0_estimation = np.average(local_energy_new)
        # negative feedback for population control
        penalty = (1.0 - (walkers.shape[walker_axis] / n_walkers) ) / (dt)
        # penalty = - np.log(walkers.shape[walker_axis] / n_walkers)
        E0_estimation = E0_estimation + penalty
        # update data
        E0_estimations[i] = E0_estimation
        E0_inters[i] = np.average(inters) + 0.5*penalty
        E0_intras[i] = np.average(local_energy(walkers, num_molecules, trial_function_d2, intras, dtype=dtype)) + 0.5*penalty
        populations[i] = walkers.shape[walker_axis]
        
        if i >= free_steps:
            # calcualte branching term
            B = np.exp( -dt* ( (local_energy_new) - E0_estimation) )
            # replication and deletion
            thresholds = np.random.uniform(0, 1, size = walkers.shape[0])
            prob_replicate = B-1
            prob_remain = B
            walkers_to_replicate = walkers[prob_replicate > thresholds]
            walkers_to_remain = walkers[prob_remain > thresholds]
            walkers = np.concatenate([walkers_to_remain, walkers_to_replicate])    

        if walkers.shape[walker_axis] == 0:
            print("Ouch! walkers all dead!")
            break

    # All possible returns
    # To access a particular output: sim_loop(...)['w|r|n|s|a']
    return {"E0": E0_estimations, "E0_inter": E0_inters, "E0_intra": E0_intras, "walkers": wave_func_snapshots, "final_walkers": walkers, "populations": populations}