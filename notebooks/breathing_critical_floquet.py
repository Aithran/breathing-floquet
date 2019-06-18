import drive_coefficients as dc
import numpy as np
from scipy import linalg as LA
from scipy.constants import h,hbar
from scipy.constants import atomic_mass as amu
import logging

logger = logging.getLogger('floquet.'+__name__)

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

def diagonalize_critical_floquet_system(trap_freq, lattice_depth, site_spacing_um,
                               gamma,
                               n_sites, floquet_m_radius,
                               sort=True):
    """Build and diagonalize the Floquet problem for breathing lattice
    
    Inputs:
    trap_freq: Harmonic trapping frequency, E/h (Hz) units
    lattice_depth: Lattice depth parameter V_0 before a cos (not cos^2), E/h (Hz)
    site_spacing_um: Lattice site spacing, in microns
    gamma: Drive amplitude parameter, 0<gamma<1
    n_sites: Number of sites to include in the lattice, should be odd
    floquet_m_radius: Number of Floquet couplings to use, generates 2m+1 blocks
    
    Optional Inputs:
    sort [True]: If True, output eigenvalues and eigenvectors are return in order of increasing eigenvalue
    
    Globals Accessed: sim_mass?
    
    Returns 3 tuple:
    drive:     critical drive frequency used
    evals:     eigenvalues of floquet Hamiltonian (possibly sorted, not modulus)
    evecs:     eigenvectors of floquet Hamiltonian (possibly sorted)
    """
    # Compute needed coefficients
    n_floquet = 2*floquet_m_radius + 1
    lattice_tunneling = tunneling_J(lattice_depth, lattice_recoil_energy(site_spacing_um))
    jsq_scale = gen_jsq_coeffs(site_spacing_um, trap_freq, gamma, n_floquet)
    h0_scale = dc.all_chis(dc.y(gamma))
    crit_drive = dc.crit_drive_freq(dc.y(gamma), trap_freq)
    # Build up floquet problem
    jsq_block = on_site_jsq(n_sites)
    tun_block = tunneling_block(n_sites, lattice_tunneling)
    floquet_h = build_floquet_jsq(jsq_block, floquet_m_radius, jsq_scale)
    floquet_h = add_tunneling_blocks(floquet_h, floquet_m_radius, tun_block, h0_scale)
    floquet_h = add_floquet_diag(floquet_h, floquet_m_radius, crit_drive)
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
        return crit_drive, evals[sort_order], evecs[:, sort_order]
    else:
        return crit_drive, evals, evecs

def gen_jsq_coeffs(site_spacing_um, trap_freq, gamma, n_floquet):
    """Generate the j^2 operators prefactors for the driven system
    
    Inputs:
    site_spacing_um: Site spacing, in microns
    trap_freq:       Harmonic trap frequency in Hz (Natural f, not omega)
    gamma:           Drive amplitude
    n_floquet:       number of Floquet couplings required"""
    # In paper, prefactor is 1/2 m * a^2 * (ang freq ^2), since we're requiring nat. freq, need 4 pi^2
    prefactor = 2.0 * np.pi * np.pi * sim_mass * um2 * site_spacing_um * site_spacing_um/h
    crit_xis = dc.n_crit_xis(n_floquet, dc.y(gamma))
    return (prefactor * trap_freq * trap_freq * crit_xis)

def build_floquet_jsq(jsq_op, floquet_radius, coefs):
    """Starts building floquet hamiltonian by tiling scaled j^2 operator
    
    Inputs:
    jsq_op:         j^2 operator, unscaled
    floquet_radius: Number of floquet blocks on either side of 0
    coefs: 1D Array of scale factors for j^2 differing by i = |m'-m|
    """
    jsq_amp_idx = np.abs(np.arange(-floquet_radius, floquet_radius + 1)[:, np.newaxis]
                       - np.arange(-floquet_radius, floquet_radius + 1)[np.newaxis, :])
    return np.kron(coefs[jsq_amp_idx], jsq_op)

def add_tunneling_blocks(floquet_h, floquet_radius, tunneling_block, scales):
    """Adds diagonal terms to floquet matrix along every set of m'-m blocks
    
    Inputs:
    floquet_h:        The matrix we're adding these terms to
    floquet_radius:   The number of m-couplings on one side of zero
    tunneling_block:  The tunneling operator, without floquet scaling
    scales:           The prefactor to tunneling terms for blocks |m'-m|
    
    Outputs:
    with_tun: floquet_h with the desired couplings added
    """
    padded_scales = np.zeros(2*floquet_radius + 1)
    padded_scales[:len(scales)] = scales
    scale_idx = np.abs(np.arange(-floquet_radius, floquet_radius + 1)[:, np.newaxis]
                     - np.arange(-floquet_radius, floquet_radius + 1)[np.newaxis, :])
    return (floquet_h + np.kron(padded_scales[scale_idx], tunneling_block))

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

def tunneling_block(n_sites, tunneling_J0):
    """Generates tunneling matrix for n_sites lattice sites
    
    Inputs:
    n_sites:      Odd integer, gives number of sites we're keeping
    tunneling_J0: Magnitude of tunneling term.
    
    Outputs:
    tunnel_mat: n_sites x n_sites with nearest neighbor tunneling elements"""
    j_list = np.full(n_sites - 1, -np.abs(tunneling_J0))
    tunnel_mat = np.diag(j_list, k=1) + np.diag(j_list, k=-1)
    return tunnel_mat

def on_site_jsq(n_sites):
    """Generate j^2 on the main diagonal
    
    Inputs:
    n_sites: Odd integer, number of sites in basis
    
    Outputs:
    on_site_mat: j^2 on main diagonal, centered
    """
    return np.diag(np.square(np.arange(-(n_sites//2), n_sites//2 + 1)))
