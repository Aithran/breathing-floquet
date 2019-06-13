import drive_coefficients
import mathieu
import numpy as np
import logging

logger = logging.getLogger(__name__)

def positive_rightward(state, idx):
    """Set states so total to right of idx is positive
    
    Inputs:
    state: 1D numpy array
    idx:   Index to start the sum from
    """
    if (np.sum(state[idx:])) < 0.0:
        return np.negative(state)
    else:
        return state

def find_min_idx(vec):
    """Find the smallest index that preserves the norm of the input vector
    
    Inputs:
    vec: Vector to trim

    idx: Index meeting the above
    """
    vec_magsq = np.square(np.abs(vec))
    vec_norm = np.sum(vec_magsq)
    
    max_idx = np.argmax(vec_magsq)
    # 1e-35 is a magic value, chosen because (1e-16)^2 is 1000 * 1e-35
    # That is, having magnitude 1e-16 means you contribute less that 1000x
    # The floating point round-off error to the total magnitude
    big_count = np.sum(np.greater(vec_magsq[max_idx:], 1e-35))
    idx = max_idx + big_count + 1
    if (idx >= vec.shape[0]):
        logger.warn('H0 basis too small, population at edges')
        idx = vec.shape[0]
    return idx

def lattice_plus_trap_sitespace(n_basis, n_states, tunneling_J, harm_param, site_rad=None):
    """Generates site-occupation basis solutions for tight-binding with trap
    
    Output states will be trimmed to smallest width that returns a norm of 1.
    
    Inputs:
    n_basis:     size of basis to use for computing mathieu equation, must be > n_states/2
    n_states:    number of states to include.  Must be even
    tunneling_J: J coefficient from the hamiltonian
    harm_param:  harmonic trapping term energy scale,
                 the energy offset of adjacent sites at the bottom of the trap
                 see paper for details
    Optional Inputs:
    site_rad [None]:   number of site to include to the left and right of center
                       default chooses automatically to preserve the norm of the
                       largest kept state
    
    Outputs:
    (energies, states):
        energies: Energy levels, given in same units as harm_param, numpy array of shape (n_states)
        states:   Corresponding states in the site-occupation basis, numpy array of shape (n_states, *)
    """
    if (n_states % 2) or (n_states < 2):
        raise ValueError('n_states ({}) must be an even positive integer'.format(n_states))
    if (n_basis % 2) or (n_basis < 2):
        raise ValueError('n_basis ({}) must be an even positive integer'.format(n_basis))
    if (n_basis < n_states/2):
        raise ValueError('2*n_basis 2*({}) must be greater than n_states ({})'.format(n_basis, n_states))
    # Calculate some convienient parameters first
    n_half = n_states//2
    q_param = 4.0*tunneling_J/harm_param
    logger.info('q_param = {}'.format(q_param))
    # Eigenvectors and values are propoportional to state populations and energies
    avals, avecs = mathieu.a_even_sys(n_basis, -q_param)
    bvals, bvecs = mathieu.b_even_sys(n_basis, -q_param)
    # Combine and scale eigenvalues
    energies = np.empty(n_states)
    energies[0::2] = avals[:n_half]
    energies[1::2] = bvals[:n_half]
    energies *= harm_param/4.0
    if site_rad is None:
        # Find space needed to hold largest state (spatial extent)
        site_rada = find_min_idx(avecs[:, n_half-1])
        site_radb = find_min_idx(bvecs[:, n_half-1])
        site_rad = max(site_rada, site_radb) + 2
        logger.info('Automatically determined site_rad = {}'.format(site_rad))
    n_sites = 2*site_rad - 1
    center_idx = site_rad - 1
    # Fill in the states, normed to 1
    states = np.empty((n_states, n_sites))
    # Even states, same pop left and right of zero
    states[0::2, center_idx:] = avecs[:site_rad, :n_half].T
    states[0::2, :center_idx] = np.flipud(avecs[1:site_rad, :n_half]).T
    # Double value at center, because integral give twice the sum coefficient
    states[0::2, center_idx] *= 2.0 # double pop at center
    # Fill in zeros for the odd state's center site
    states[1::2, center_idx] = 0.0
    # Odd states, sign change left vs right side of zero
    states[1::2, center_idx+1:] = bvecs[:site_rad-1, :n_half].T
    states[1::2, :center_idx] = -np.flipud(bvecs[:site_rad-1, :n_half]).T
    # Normalize so sum(|states|^2) = 1
    states *= np.sqrt(0.5) # Norm properly
    # Fix so right side doesn't randomly flop sign
    states = np.apply_along_axis(positive_rightward, 1, states, center_idx)
    return (energies, states)

def j_op_statespace(states):
    """
    Compute j operator from list of states in site space
    
    Input:
    states:  Array of states, expected that each row is a state
    
    Output:
    j_op:  j operator acting on these states, with no prefactors
    """
    j_mat = np.diag(np.arange(states.shape[1]) - states.shape[1]//2)
    return (states @ j_mat @ states.T)

def jsq_op_statespace(states):
    """
    Compute j^2 operator from list of states in site space
    
    Input:
    states:  Array of states, expected that each row is a state
    
    Output:
    jsq_op:  j^2 operator acting on these states, with no prefactors
    """
    jsq_mat = np.diag(np.square(np.arange(states.shape[1]) - states.shape[1]//2))
    return (states @ jsq_mat @ states.T)
