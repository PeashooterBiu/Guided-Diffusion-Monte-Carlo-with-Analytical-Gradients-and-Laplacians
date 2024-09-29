# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 03/01/20

# Function for printing XYZ output of a given walker array
# This only works for walkers with homogeneous molecules
# Also had to hard-code the sig digs (it's 8 right now)
# To change that, change the 8 in the ".8f" on line 27

# Added descendent weighting output
# Comment line now shows ancestor

#added to prevent 64 threads from being launched
import os
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np



def print_xyz(walkers, ancestors = None, atoms = ['O','H','H'], \
        comment = '### Comment ###', coord_cnst = 3):
    '''
    Input: walkers: ndarray shape=(nWalkers,nMolecules,nAtoms,coord_cnst)
           atoms: list of strings --> order of atoms in a particular molecule
           comment: string --> comment for that walker
           coord_cnst: int --> number of coordinate dimensions (usually 3 for XYZ)
    Output: string --> for each walker:
    nAtoms*nMol
    ### Comment ###
    Atom0    X0   Y0   Z0
    Atom1    X1   Y1   Z1
    ...
    AtomN    XN   YN   ZN
    [Blank Line]
    '''
    tb = '\t'
    nl = '\n'
    return '\n\n'.join( [ (
        f'{walkers.shape[2]*walkers.shape[1]}\n{ancestors[i] if ancestors is not None else comment}\n'
        f'''{nl.join( [atoms[c % walkers.shape[2]] + tb 
            + tb.join( [f"{el:.8f}" for el in row] ) 
            for c,row in enumerate( walkers[i,...].reshape((-1,coord_cnst)) )] )}'''
        ) for i in range(walkers.shape[0]) ] )

'''need to a function to mark the O, H molecules for the first molecule'''


def print_csv(walkers):
    '''
    Could be implemented if desired
    '''
    pass


def print_arr(walkers,ext,**kwargs):
    '''
    Input: walkers: ndarray shape=(nWalkers,nMolecules,nAtoms,coord_const)
           ext: string --> file extension "xyz" or "csv"
    Output: string --> xyz or csv formatted output for walkers
    '''
    if ext == 'xyz':
        return print_xyz(walkers,**kwargs)
    elif ext == 'csv':
        return print_csv(walkers)
    else:
        print('Error: Invalid Extension for Printing')
        exit()

def write_array(filename,ext='xyz',**kwargs):
    '''
    Input: filename: str --> name of input file to load (no extension needed)
           ext: str --> output file type 'xyz' or 'csv'
    Output: None
           filename.ext written with proper formatted output of walker array loaded from input
    '''
    # Format filename -- remove ext
    filename = filename.strip('./').split('.')[0]
    with open(filename+'.'+ext,'w') as fl:
        wlk = np.load(filename+'.npy')
        fl.write(print_arr(wlk,ext,**kwargs))


def write_xyz(filename, data):
    with open(filename, 'w') as f:
        xyz_string = print_xyz(data)
        f.write(xyz_string)




#########################################################################################
# Read XYZ Methods
#TODO: COMMENT THE SHIT OUT OF THESE

def unpack(l):
    try:
        return int(l[0])
    except:
        try:
            return float(l[0])
        except:
            return l[0]

def tokenize_xyz(filename):
    '''
    Output: [[n_atoms],[comment],[Atom,xpos,ypos,zpos],...,[Atom,xpos,ypos,zpos]]
    for each walker
    '''
    filename = filename.strip('./').split('.')[0]
    with open(filename+'.xyz','r') as fi:
        tokens = fi.read().strip().split('\n\n')
        wlk_proto = [t.strip().split('\n') for t in tokens]
        wlk_atoms = [[s.split() for s in w] for w in wlk_proto]
        return wlk_atoms

def read_xyz(filename,nMol = 3):
    wlk = tokenize_xyz(filename)
    walkers_out = []
    comments_out = []
    for w in wlk:
        n_atoms = unpack(w[0])//nMol
        comment = unpack(w[1])
        atoms = [a[1:] for a in w[2:]]
        z = [atoms[i::n_atoms] for i in range(n_atoms)]
        walker = list(zip(*z))
        walkers_out.append(walker)
        comments_out.append(comment)
    return {'w':np.array(walkers_out).astype(np.float64),'c':comments_out}


# Input filename of xyz file
# Broadcasts to n_walkers by num_molecules by num_atoms by 3
# Used to make initialization easier given it is a difficult process
def gen_walker_array(filename, n_walkers, prop_amount, n_molecules = 3):
    # Read in the xyz file. Returns a 1 by num_molecules by num atoms by 3
    walk = read_xyz(filename,n_molecules)['w']
    _, num_molecules, num_atoms, _, = walk.shape

    # Broadcasts walkers to the shape given by the number of walkers
    walkers = np.broadcast_to(walk[0], (n_walkers,num_molecules,num_atoms,3)) + \
        np.random.uniform(-prop_amount, prop_amount, (n_walkers,num_molecules,num_atoms,3))
    
    return walkers, num_molecules


def read_xyz_2(filename, nMol=3):
    wlk = tokenize_xyz(filename)
    walkers_out = []
    comments_out = []
    for w in wlk:
        n_atoms = unpack(w[0]) // nMol
        comment = unpack(w[1])
        atoms = [a[1:] for a in w[2:]]
        z = [atoms[i::n_atoms] for i in range(n_atoms)]
        walker = list(zip(*z))
        walkers_out.append(walker)
        comments_out.append(comment)
    return {'w': np.array(walkers_out).astype(np.float64), 'c': comments_out}

def load_xyz(filename, n_molecules=3):
    # Read in the xyz file. Returns an array of walkers
    walkers_data = read_xyz(filename, n_molecules)
    walkers = walkers_data['w']

    return walkers


if __name__ == '__main__':

    write_array('tetramer.npy')
 
