import harm_lattice as h0
import drive_coefficients as dc
import numpy as np
from scipy import linalg as LA
from scipy.constants import h,hbar
from scipy.constants import atomic_mass as amu
import logging

logger = logging.getLogger(__name__)

# Setup some constants
m_rb87 = 86.909180527 * amu
um = 1e-6
kHz = 1e3
Hz = 1
um2 = 1e-12

# Assume Rb87, can be changed later if necessary
sim_mass = m_rb87

def tunneling_J(V_depth, E_recoil):
    """Compute tunneling term magnitude

    limited to V > a few ER
    
    Inputs:
    V_depth:  lattice depth in any units
    E_recoil: recoil energy in same units as V0
    
    Returns:
    J: in the same units as V_depth
    """
    V_ER = V_depth/E_recoil
    return 1.397*np.power(V_ER, 1.051)*np.exp(-2.121*np.sqrt(V_ER))*E_recoil

# Double check this...
def lattice_recoil_energy(spacing_um):
    """Recoil energy of a lattice
    
    Inputs:
    spacing_um: the lattice site spacing, in microns
    
    Returns:
    recoil energy in naturnal frequency (E_R/h)
    """
    return h/(8.0 * sim_mass * spacing_um * spacing_um * um2)

def diagonalize_floquet_system(trap_freq, lattice_depth, site_spacing_um,
                               drive_freq, gamma,
                               n_mathieu_basis, n_h0_states, floquet_m_radius,
                               sort=True):
    """Build and diagonalize the Floquet problem for breathing lattice
    
    Inputs:
    trap_freq: Harmonic trapping frequency, E/h (Hz) units
    lattice_depth: Lattice depth parameter V_0 before a cos (not cos^2), E/h (Hz)
    site_spacing_um: Lattice site spacing, in microns
    drive_freq: Natural drive frequency, in (Hz)
    gamma: Drive amplitude parameter, 0<gamma<1
    n_mathieu_basis: Number of basis vectors to use in underlying mathieu function computation
    n_h0_states: Number of states to keep in time-averaged Hamiltonian, this is the size of the H-space in the Floquet problem
    floquet_m_radius: Number of Floquet couplings to use, generates 2m+1 blocks
    
    Optional Inputs:
    sort [True]: If True, output eigenvalues and eigenvectors are return in order of increasing eigenvalue
    
    Globals Accessed: sim_mass?
    
    Returns 3 tuple:
    h0_states: eigenstates of time-averaged Hamiltonian/basis states for floquet Hamiltonian
    evals:     eigenvalues of floquet Hamiltonian (possibly sorted, not modulus)
    evecs:     eigenvectors of floquet Hamiltonian (possibly sorted)
    """
    # Compute needed coefficients
    n_floquet = 2*floquet_m_radius + 1
    lattice_tunneling = tunneling_J(lattice_depth, lattice_recoil_energy(site_spacing_um))
    raw_jsq_scale = gen_jsq_coeffs(site_spacing_um, drive_freq, trap_freq, gamma, n_floquet)
    h0_jsq_scale = raw_jsq_scale[0]
    h0_scale_factors = dc.all_chis(dc.y(gamma))
    adj_jsq_scale = adjust_jsq_coefs(raw_jsq_scale, h0_scale_factors)
    # Find solution to time-averaged part
    h0_en, h0_states = h0.lattice_plus_trap_sitespace(n_mathieu_basis, n_h0_states, lattice_tunneling, h0_jsq_scale)
    # Build up floquet problem
    jsq_op = h0.jsq_op_statespace(h0_states)
    floquet_h = build_floquet_jsq(jsq_op, floquet_m_radius, adj_jsq_scale)
    floquet_h = add_h0_diags(floquet_h, floquet_m_radius, h0_en, h0_scale_factors)
    floquet_h = add_floquet_diag(floquet_h, floquet_m_radius, drive_freq)
    # Some consistency checks, warn but do not stop.
    if not np.allclose(floquet_h.T, floquet_h):
        # More stringent, requires only real entries
        logger.warn("Symmetry test failed for this gamma {}".format(gamma))
    if not np.allclose(np.conj(floquet_h.T), floquet_h):
        logger.warn("Conjugate transpose test failed for this gamma {}".format(gamma))
    # Diagonalize, return results, copies because the inner workings of h5py aren't clear to me
    evals, evecs = LA.eigh(floquet_h) # Uses faster algo than eig(), guaranteed real evals too!
    if sort:
        sort_order = np.argsort(evals)
        return h0_states.T, evals[sort_order], evecs[:, sort_order]
    else:
        return h0_states.T, evals, evecs

def gen_jsq_coeffs(site_spacing_um, drive_freq, trap_freq, gamma, n_floquet):
    """Generate the j^2 operators prefactors for the driven system
    
    Inputs:
    site_spacing_um: Site spacing, in microns
    drive_freq:      Drive frequency, in Hz (Natural f, not omega)
    trap_freq:       Harmonic trap frequency in Hz (Natural f, not omega)
    gamma:           Drive amplitude
    n_floquet:       number of Floquet couplings required"""
    # In paper, prefactor is 1/2 m * a^2 * (ang freq ^2), since we're requiring nat. freq, need 4 pi^2
    prefactor = 2.0 * np.pi * np.pi * sim_mass * um2 * site_spacing_um * site_spacing_um/h
    big_xis, small_xis = dc.n_both_xis(n_floquet, dc.y(gamma))
    return prefactor * ((trap_freq * trap_freq * small_xis) - (drive_freq*drive_freq*big_xis))

def adjust_jsq_coefs(raw_jsq_coefs, chis):
    """Do the subtraction of the j^2 coefficients to allow the magical factoring
    
    Inputs:
    raw_jsq_coefs: The orignal j^2 operator coefficients
    chis:          The scale factors for the H0 part
    
    Returns:
    adj_jsq_coefs: Coefficient adjusted properly"""
    adj_jsq_coefs = raw_jsq_coefs.copy()
    adj_jsq_coefs[:3] = adj_jsq_coefs[:3] - chis * adj_jsq_coefs[0]
    if adj_jsq_coefs[0] != 0:
        logger.warn("jsq[0] did not cancel")
    return adj_jsq_coefs

def build_floquet_jsq(jsq_op, floquet_radius, residual_coefs):
    """Starts building floquet hamiltonian by tiling scaled j^2 operator
    
    Inputs:
    jsq_op:         j^2 operator, unscaled
    floquet_radius: Number of floquet blocks on either side of 0
    residual_coefs: 1D Array of scale factors for j^2 differing by i = |m'-m|
    """
    jsq_amp_idx = np.abs(np.arange(-floquet_radius, floquet_radius+1)[:,np.newaxis]
                         - np.arange(-floquet_radius, floquet_radius+1)[np.newaxis,:])
    return np.kron(residual_coefs[jsq_amp_idx], jsq_op)

def add_h0_diags(floquet_h, floquet_radius, h0_en, h0_scale_factors):
    """Adds diagonal terms to floquet matrix along every set of m'-m blocks
    
    Inputs:
    floquet_h:        The matrix we're adding these terms to
    floquet_radius:   The number of m-couplings on one side of zero
    h0_en:            The energies associated with the eigenstates of h0
    h0_scale_factors: The prefactor to H0-like terms for blocks |m'-m|
    
    Outputs:
    with_h0: floquet_h with the desired couplings added
    """
    n_en = len(h0_en)
    h0_couplings = np.diag(np.tile(h0_scale_factors[0] * h0_en, 2*floquet_radius +1))
    for idx, this_factor in enumerate(h0_scale_factors[1:], start=1):
        h0_couplings += np.diag(np.tile(this_factor * h0_en, 2*floquet_radius + 1 - idx), k=idx*n_en)
        h0_couplings += np.diag(np.tile(this_factor * h0_en, 2*floquet_radius + 1 - idx), k=-idx*n_en)
    return (floquet_h + h0_couplings)

def add_floquet_diag(floquet_h, floquet_radius, drive_freq):
    """Add the m*hbar*Omega floquet terms on main diagonal
    
    but we're working in E/h, so m*f
    
    Inputs:
    floquet_h:      The floquet hamiltonian we're adding this to
    floquet_radius: The number of floquet blocks either side of zero
    drive_freq:     The drive frequency, in Hz (natural, not angular)
    
    Outputs:
    with_mf: floquet_h with the desired terms added
    """
    n_states = floquet_h.shape[0]//(2*floquet_radius + 1)
    m_couplings = np.kron(np.arange(-floquet_radius, floquet_radius+1), np.tile(drive_freq, n_states))
    return (floquet_h + np.diag(m_couplings))
