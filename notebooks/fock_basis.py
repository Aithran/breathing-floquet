from itertools import combinations
import functools
import numpy as np


def particles_in_sites(n, m):
    """Generate combinations of n particles in m sites

    Inputs:
    n: Integer, total number of particles
    m: Integer, total number of sites
    
    Outputs:
    Generator producing all of those combinations
    """
    for c in combinations(range(n + m - 1), m - 1):
        yield tuple(b - a - 1 for a, b in zip((-1,) + c, c + (n + m - 1,)))

@functools.lru_cache(maxsize=8)
def unit_filling(n):
    """Generates states with n particles on n sites
    
    Inputs:
    n: Integer, number of particles and sites
    
    Outputs:
    states: Read-only numpy array, s x n, all possible states"""
    states = np.array(list(particles_in_sites(n, n)))
    states.flags.writeable = False
    return states

def n_i_op(i, states):
    """Generates the operator n_i acting among states
    
    Inputs:
    i: site index n will act on
    
    Outputs:
    n_i_op
    """
    return np.diag(states[:, i])

def find_state(ref_state, state_array):
    """Find which row of state_array is equivalent to ref_state
    
    Inputs:
    ref_state:   state we're looking for
    state_array: basis search for that state
    
    Outputs:
    index of the row in state_array if found,
    otherwise -1
    """
    spot = np.where((state_array == tuple(ref_state)).all(axis=1))[0]
    if len(spot) == 0:
        spot = -1
    else:
        spot = np.asscalar(spot)
    return spot

def ai_aj_dag(i, j, states):
    """a_i a_j^dagger acting among states
    
    Inputs:
    i, j:   Indices for those operators
    states: Basis we're acting in
    
    Outputs:
    a_i a_j^dagger
    """
    amps = (np.sqrt(states[:, i]) * np.sqrt(states[:, j] + 1)).flatten()
    change = np.zeros_like(states[0])
    change[i] -= 1
    change[j] += 1
    idx = np.apply_along_axis(find_state, 1, states+change, state_array=states)
    op = np.zeros((states.shape[0], states.shape[0]))
    for row, col in enumerate(idx):
        if col >= 0:
            op[row, col] = amps[row]
    return op

def ai_dag_aj(i, j, states):
    """a_i^dagger a_j acting among states
    
    Inputs:
    i, j:   Indices for those operators
    states: Basis we're acting in
    
    Outputs:
    a_i^dagger a_j
    """
    amps = np.sqrt(states[:, i] + 1) * np.sqrt(states[:, j])
    change = np.zeros_like(states[0])
    change[i] += 1
    change[j] -= 1
    idx = np.apply_along_axis(find_state, 1, states+change, state_array=states)
    op = np.zeros((states.shape[0], states.shape[0]))
    for row, col in enumerate(idx):
        if col >= 0:
            op[row, col] = amps[row]
    return op

def sym_hopping(i, j, states):
    """Symmetric combination (a_i a_j^dagger + a_i^dagger a_j)"""
    return ai_aj_dag(i, j, states) + ai_dag_aj(i, j, states)

def nearest_neighbor_hopping(states):
    """Generates all nearest-neighbor hoppings amoung states"""
    n_sites = states.shape[1]
    nn_hopping = np.zeros((states.shape[0], states.shape[0]))
    for i in np.arange(n_sites-1):
        nn_hopping += sym_hopping(i, i+1, states)
    return nn_hopping

def onsite_interactions(states):
    """Generates sum_i (n_i*(n_i - 1)) acting among states"""
    u_op = np.zeros((states.shape[0], states.shape[0]))
    for i in np.arange(states.shape[1]):
        u_op += n_i_op(i, states) * (n_i_op(i,states) - 1)
    return u_op.astype(np.float)

def onsite_offset(offsets, states):
    """Generates sum_i offsets[i] * n_i acting among states"""
    if states.shape[1] != len(offsets):
        raise ValueError("states and offset must have same length")
    epsilon_op = np.zeros((states.shape[0], states.shape[0]))
    for idx, offset in enumerate(offsets):
        epsilon_op += offsets[idx] * n_i_op(idx, states)
    return epsilon_op

def jsq_offset(n_sites):
    """Generates offsets for j^2, without prefactor
    
    E.g { ((N-1)/2)^2, ..., 4, 1, 0, 1, 4, ..., ((N-1)/2)^2 }
    """
    return np.square(np.arange(-(n_sites//2), n_sites//2 + 1))

def j_offset(n_sites):
    """Generates offsets for j, without prefactor

    E.g. { -(N-1)/2, ..., -2, -1, 0, 1, 2, ..., (N-1)/2 }
    """
    return np.arange(-(n_sites//2), n_sites//2 + 1)

def build_sys(u_over_j, drive_over_j, k_over_drive, basis_states, floquet_radius):
    m_list = np.arange(-floquet_radius, floquet_radius+1)
    raw_int = onsite_interactions(basis_states)
    raw_hop = -nearest_neighbor_hopping(basis_states)
    floquet_interactions = 0.5*u_over_j * np.kron(np.diag(np.ones(2*floquet_radius+1)), raw_int)
    floquet_diag = np.kron(np.diag(m_list), np.diag(np.full(basis_states.shape[0], drive_over_j)))
    tun_blocks = scaling_blocks(floquet_radius, k_over_drive, s_matrix(basis_states, raw_hop), raw_hop)
    return floquet_interactions + floquet_diag + tun_blocks


