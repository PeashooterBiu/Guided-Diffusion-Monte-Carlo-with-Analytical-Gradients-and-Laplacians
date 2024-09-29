# Sijing Zhu (Paul) Summer 2024
# qSPC/Fw PES analytical gradients + numerical gradients

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import itertools as it
import time as time

# Scientific Constants
# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23
# Chemistry constants for intermolecular energy
sigma = 3.165492 / 0.529177
epsilon = 0.1554252 * (4.184 / 2625.5)
# Coulombic charges
q_oxygen = -0.84
q_hydrogen = 0.42
# Coulomb's Constant
coulomb_const = 1.0 / (4.0 * np.pi)
# Normalization constant
N = 4.033938699359097
# Number of coordinates
coord_const = 3

# Simulation Loop Constants
walker_axis = 0
molecule_axis = 1
atom_axis = 2
coord_axis = 3

# Molecule Model Constants
oxygen_mass = 15.99491461957
hydrogen_mass = 1.007825
HOH_bond_angle = 112.0
eq_bond_length = 1.0 / 0.529177
eq_bond_angle = HOH_bond_angle * np.pi / 180
bohrToAng = 0.529177
kOH = 1059.162 * (bohrToAng ** 2) * (4.184 / 2625.5)
kA = 75.90 * (4.184 / 2625.5)
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)
reduced_mass = ((atomic_masses[0] + atomic_masses[1]) * atomic_masses[2]) / np.sum(atomic_masses)
atomic_charges = np.array([q_oxygen, q_hydrogen, q_hydrogen])
coulombic_charges = (np.transpose(atomic_charges[np.newaxis]) @ atomic_charges[np.newaxis])



'''helper functions for suppressing division by zero'''
inf_to_zero = lambda dist: np.where(np.abs(dist) == np.inf, 0, dist)
'''helper function for getting the molecule pairs, with shape (2, n_pairs), where the first dimension contains pairs_a and pairs_b indices'''
molecule_index = lambda n_mol: list(zip(*it.combinations(range(n_mol),2)))


'''backpropagation for intra_pe gradients'''
def intra_pe(x):
    # Return the two OH vectors
    OH_vectors = x[:,:,1:] - x[:,:,np.newaxis,0]
    # Returns the lengths of each OH bond vector
    lengths = np.sqrt(np.sum(OH_vectors**2, axis=3))
    # Calculates the bond angle in the HOH bond
    normalized_OH_vectors = OH_vectors / lengths[:, :, :, np.newaxis]
    angle = np.arccos(np.sum(normalized_OH_vectors[:, :, 0] * normalized_OH_vectors[:, :, 1], axis=2))
    # Calculates the harmonic oscillator potential energies for bond streching and angle bending
    pe_bond_lengths = .5 * kOH * (lengths - eq_bond_length)**2
    pe_bond_angle = .5 * kA * (angle - eq_bond_angle)**2
    # Sums the potential energy of the bond lengths with the bond angle to get potential energy
    return np.sum(np.sum(pe_bond_lengths, axis = 2) + pe_bond_angle, axis=1)

# Gradient of bond lengths with respect to coordinates
def intra_pe_get_d_lengths_1_d_coords(OH_1_lengths, walkers):
    d_lengths_d_coords_O = (walkers[:, :, 0, :] - walkers[:, :, 1, :]) / OH_1_lengths[..., np.newaxis]
    d_lengths_d_coords_H1 = -d_lengths_d_coords_O
    d_lengths_d_coords_H2 = np.zeros((walkers.shape[0], walkers.shape[1], walkers.shape[3]), dtype=walkers.dtype)
    d_lengths_coords = np.stack([d_lengths_d_coords_O, d_lengths_d_coords_H1, d_lengths_d_coords_H2], axis=2)
    return d_lengths_coords

# Gradient of bond lengths with respect to coordinates
def intra_pe_get_d_lengths_2_d_coords(OH_2_lengths, walkers):
    d_lengths_d_coords_O = (walkers[:, :, 0, :] - walkers[:, :, 2, :]) / OH_2_lengths[..., np.newaxis]
    d_lengths_d_coords_H1 = np.zeros((walkers.shape[0], walkers.shape[1], walkers.shape[3]), dtype=walkers.dtype)
    d_lengths_d_coords_H2 = -d_lengths_d_coords_O
    d_lengths_coords = np.stack([d_lengths_d_coords_O, d_lengths_d_coords_H1, d_lengths_d_coords_H2], axis=2)
    return d_lengths_coords

# Gradient of bond angles with respect to coordinates
def intra_pe_get_d_angles_d_coords(OH_vectors, OH_lengths, walkers):
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

def intra_pe_get_d_OH_modes_d_lengths(OH_lengths, kOH, eq_bond_length):
    return kOH * (OH_lengths - eq_bond_length)

def intra_pe_get_d_HOH_modes_d_angles(angles, kA, eq_bond_angle):
    return kA * (angles - eq_bond_angle)

def intra_pe_get_d_molecule_modes_d_OH_modes(OH_modes, HOH_modes):
    return 1

def intra_pe_get_d_molecule_modes_d_HOH_modes(OH_modes):
    return 1

def intra_pe_get_d_trial_d_molecule_modes(modes):
    return 1

def intra_pe_analytical_d1(walkers):
    '''forward pass on trial function'''
    OH_vectors = walkers[:, :, 1:] - walkers[:, :, np.newaxis, 0]
    OH_lengths = np.linalg.norm(OH_vectors, axis=3)
    normalized_OH_vectors = OH_vectors / np.expand_dims(OH_lengths, -1)
    angles = np.arccos(np.sum(normalized_OH_vectors[:, :, 0] * normalized_OH_vectors[:, :, 1], axis=2))
    # Calculate OH_modes and HOH_modes
    OH_modes = 0.5 * kOH * (OH_lengths - eq_bond_length)**2
    HOH_modes = 0.5 * kA * (angles - eq_bond_angle)**2
    # Molecule modes
    molecule_modes = np.prod(OH_modes, axis=2) * HOH_modes
    d_trial_d_molecule_modes = intra_pe_get_d_trial_d_molecule_modes(molecule_modes)
    
    '''bond length backprop'''
    d_molecule_modes_d_OH_modes = intra_pe_get_d_molecule_modes_d_OH_modes(OH_modes, HOH_modes)
    d_OH_modes_d_lengths = intra_pe_get_d_OH_modes_d_lengths(OH_lengths, kOH, eq_bond_length)[..., np.newaxis]
    d_lengths_1_d_coords = intra_pe_get_d_lengths_1_d_coords(OH_lengths[:, :, 0], walkers)
    d_lengths_2_d_coords = intra_pe_get_d_lengths_2_d_coords(OH_lengths[:, :, 1], walkers)
    d_molecule_modes_d_length = d_molecule_modes_d_OH_modes * d_OH_modes_d_lengths
    # combind the gradients for bond length flows shape (nWalkers, mMolecules, natoms, 3)
    OH_d_molecule_modes_d_coords = d_molecule_modes_d_length[:, :, 0:1] * d_lengths_1_d_coords + d_molecule_modes_d_length[:, :, 1:] * d_lengths_2_d_coords
    
    '''bond angle backprop'''
    d_molecule_modes_d_HOH_modes = intra_pe_get_d_molecule_modes_d_HOH_modes(OH_modes)
    d_HOH_modes_d_angles = intra_pe_get_d_HOH_modes_d_angles(angles, kA, eq_bond_angle)[..., np.newaxis, np.newaxis]
    d_angles_d_coords = intra_pe_get_d_angles_d_coords(OH_vectors, OH_lengths, walkers)
    # combine the gradients for bond angle flows shape (nWalkers, nMolecules, natoms, 3)
    HOH_d_molecule_modes_d_coords = d_molecule_modes_d_HOH_modes * d_HOH_modes_d_angles * d_angles_d_coords
    
    '''sum up all modes with shape, return gradients with shape (nWalkers, nMolecules, natoms, 3)'''
    d_molecule_modes_d_coords = OH_d_molecule_modes_d_coords + HOH_d_molecule_modes_d_coords
    d_trial_d_coords = d_trial_d_molecule_modes * d_molecule_modes_d_coords
    return d_trial_d_coords



'''take the analytical gradients of intermolecular potential energy using backpropagation'''
def LJ_pe(x):
    molecule_index = lambda n_mol: list(zip(*it.combinations(range(n_mol),2)))
    molecule_index_a, molecule_index_b = molecule_index(x.shape[1]) 
    pairs_a = x[:,molecule_index_a]
    pairs_b = x[:,molecule_index_b]
    distances = np.sqrt( np.sum( (pairs_a[...,None] \
            - pairs_b[:,:,np.newaxis,...].transpose(0,1,2,4,3) )**2, axis=3) )
    sigma_dist = inf_to_zero( sigma / distances[:,:,0,0] )
    lennard_jones_energy = np.sum( 4*epsilon*(sigma_dist**12 - sigma_dist**6), axis = 1)
    return lennard_jones_energy


def LJ_pe_analytical_d1(walkers):
    '''forward pass'''
    oxygen_coords = walkers[:, :, 0, :]  # shape: (n_walkers, n_molecules, 3)   
    mol_idx_a, mol_idx_b = molecule_index(oxygen_coords.shape[1])
    oxygen_pairs_a = oxygen_coords[:, mol_idx_a]
    oxygen_pairs_b = oxygen_coords[:, mol_idx_b]
    pair_distances = np.sqrt(np.sum((oxygen_pairs_a - oxygen_pairs_b) ** 2, axis=2))
    sigma_over_dist = inf_to_zero(sigma / pair_distances)
    '''backprop'''
    d_LJ_potential_d_distance = 4 * epsilon * (-12 * sigma_over_dist**12 + 6 * sigma_over_dist**6) / pair_distances
    # Gradient of distances with respect to coordinates
    d_distance_d_coords = (oxygen_pairs_a - oxygen_pairs_b) / pair_distances[..., np.newaxis]
    # Zero out the gradients for hydrogen atoms
    gradients = np.zeros_like(walkers)
    # Vectorized accumulation of gradients
    d_potential_d_coords = d_LJ_potential_d_distance[..., np.newaxis] * d_distance_d_coords
    np.add.at(gradients, (slice(None), mol_idx_a, 0, slice(None)), d_potential_d_coords)
    np.add.at(gradients, (slice(None), mol_idx_b, 0, slice(None)), -d_potential_d_coords)
    return gradients


def coulombic_pe(x):
    molecule_index_a, molecule_index_b = molecule_index(x.shape[1]) 
    pairs_a = x[:,molecule_index_a]
    pairs_b = x[:,molecule_index_b]
    distances = np.sqrt( np.sum( (pairs_a[...,None] \
            - pairs_b[:,:,np.newaxis,...].transpose(0,1,2,4,3) )**2, axis=3) )
    
    coulombic_energy = np.sum( inf_to_zero(coulombic_charges / distances), axis=(1,2,3))
    return coulombic_energy


def coulombic_pe_analytical_d1(x):
    molecule_index_a, molecule_index_b = molecule_index(x.shape[1]) 
    ''' Forward pass '''
    # Extract the positions of the atoms for each pair
    # shape of (n_walkers, n_pairs, n_atoms, n_coords)
    pairs_a = x[:, molecule_index_a]
    pairs_b = x[:, molecule_index_b]
    # Calculate the distances between the pairs of molecules
    # shape of (n_walkers, n_pairs, n_atoms, n_atoms)
    dists = pairs_a[..., None] - pairs_b[:, :, np.newaxis, ...].transpose(0, 1, 2, 4, 3)
    distances = np.sqrt(np.sum((dists)**2, axis=3))
    ''' Backpropagation '''
    # Calculate the derivative of the Coulombic potential with respect to distances
    # shape of (n_walkers, n_pairs, n_atoms, n_atoms)
    d_coulombic_potential_d_distance = -coulombic_charges / distances**2
    # shape of (n_walkers, n_pairs, n_atoms, n_atoms, n_coords)
    d_distance_d_coords = (dists).transpose(0, 1, 2, 4, 3) / distances[..., np.newaxis]
    # shape of (n_walkers, n_pairs, n_atoms, n_atoms, n_coords)
    d_potential_d_coords = d_coulombic_potential_d_distance[..., np.newaxis] * d_distance_d_coords
    # Initialize the gradients array wit hshape (n_walkers, nmolecules, natoms, ncoords)
    gradients = np.zeros_like(x)
    # Accumulate gradients onto the gradients array into each atom pair
    '''unvectorzied version (debugged)'''
    # for i, (a, b) in enumerate(zip(molecule_index_a, molecule_index_b)):
    #     for j in range(pairs_a.shape[2]):  # iterate over atoms in molecule a
    #         for k in range(pairs_b.shape[2]):  # iterate over atoms in molecule b
    #             gradients[:, a, j, :] += d_potential_d_coords[:, i, j, k, :]
    #             gradients[:, b, k, :] -= d_potential_d_coords[:, i, j, k, :]
    '''partly vectorized version (debugged)'''
    # for i, (a, b) in enumerate(zip(molecule_index_a, molecule_index_b)):
    #     np.add.at(gradients, (slice(None), a, slice(None), slice(None)), d_potential_d_coords[:, i, :, :, :].sum(axis=2))
    #     np.add.at(gradients, (slice(None), b, slice(None), slice(None)), -d_potential_d_coords[:, i, :, :, :].sum(axis=1))
    '''fully vectorized version'''
    np.add.at(gradients, 
              (slice(None), molecule_index_a, slice(None), slice(None)), 
              d_potential_d_coords.sum(axis=3))
    np.add.at(gradients, 
              (slice(None), molecule_index_b, slice(None), slice(None)), 
              -d_potential_d_coords.sum(axis=2))
    return gradients

def total_pe(walkers):
    return intra_pe(walkers) + LJ_pe(walkers) + coulombic_pe(walkers)

'''sum the analytical gradients for intramolecular pe and intermolecular pe'''
def total_pe_analytical_d1(walkers):
    return intra_pe_analytical_d1(walkers) + LJ_pe_analytical_d1(walkers) + coulombic_pe_analytical_d1(walkers)





'''numerical gradients for total_pe'''
def repeat_elements(arr, n_coord_prime):
    # Reshape the input array to have shape (n_walkers, n_coord, 1)
    reshaped_arr = arr[:, :, np.newaxis]
    # Repeat the elements along the n_coord axis by n_coord' times
    repeated_arr = np.repeat(reshaped_arr, n_coord_prime, axis=2)
    return repeated_arr

def total_pe_numerical_d1(walkers, epsilon=1e-3):
    func = total_pe
    n_walkers, n_molecules, n_atoms, n_coordinates = walkers.shape
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
    # reshape the arrays inorder to plug in intro func()
    walkers_plus = np.reshape(walkers_plus, newshape = (rep * n_walkers, n_molecules, n_atoms, n_coordinates))
    walkers_minus = np.reshape(walkers_minus, newshape = (rep * n_walkers, n_molecules, n_atoms, n_coordinates))
    # calculate the values, those should have shape (rep*nwalkers)
    values_plus = func(walkers_plus)
    values_minus = func(walkers_minus)
    # reshape the values into shape (rep, nwalkers)
    values_plus = np.reshape(values_plus, newshape = (rep, n_walkers))
    values_minus = np.reshape(values_minus, newshape = (rep, n_walkers))
    # calculate the gradient, it should have shape (nwalkers, rep)
    gradients = (values_plus - values_minus) / (2 * epsilon)
    # transpose the values, those should have shape (nwalkers, rep)
    gradients = np.transpose(gradients)
    # reshape the gradients back to (nwalkers, n_molecules, n_atoms, n_coordinates)
    gradients = np.reshape(gradients, newshape=(n_walkers, n_molecules, n_atoms, n_coordinates))
    return gradients