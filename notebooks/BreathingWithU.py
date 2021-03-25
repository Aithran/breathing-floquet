#!/usr/bin/env python
# coding: utf-8

# In[6]:


from itertools import combinations
import functools

import numpy as np

import fock_basis, drive_coefficients

from matplotlib import pyplot as plt
from zss_progbar import log_progress as progbar


# In[2]:


get_ipython().run_cell_magic('time', '', 'state_count = 5\nbasis_states = fock_basis.unit_filling(state_count).copy()\ntestnn = fock_basis.nearest_neighbor_hopping(basis_states)\nuop = fock_basis.onsite_interactions(basis_states)\njsq = fock_basis.onsite_offset(fock_basis.jsq_offset(state_count), basis_states)\nprint(basis_states.shape)')


# In[3]:


plt.pcolormesh(testnn, cmap='inferno')
plt.colorbar()


# In[4]:


plt.pcolormesh(jsq, cmap='inferno')
plt.colorbar()


# In[5]:


plt.pcolormesh(uop, cmap='inferno')
plt.colorbar()


# In[6]:


np.array_equal(testnn, testnn.T)


# In[7]:


np.array_equal(jsq, jsq.T)


# In[8]:


np.array_equal(uop, uop.T)


# In[9]:


126*9


# In[10]:


fock_basis.jsq_offset(5)


# In[11]:


# Try to reproduce arXiv:cond-mat/0601020


# In[12]:


n_scan = 65
n_sites = 5
j_sc = 20
u_sc = np.linspace(0.0, 10.0, n_scan, endpoint=True)
basis = fock_basis.unit_filling(n_sites).copy()
hopping_mi = -j_sc*fock_basis.nearest_neighbor_hopping(basis)
unscaled_interactions = fock_basis.onsite_interactions(basis)

print(basis.shape, basis.shape[0]*9)


# In[13]:


get_ipython().run_cell_magic('time', '', 'mi_energies = np.empty((n_scan, basis.shape[0]))\nfor idx, this_u_sc in progbar(enumerate(u_sc), every=1, size=n_scan):\n    ham = hopping_mi + 0.5 * this_u_sc * j_sc * unscaled_interactions\n    mi_energies[idx] = np.linalg.eigvalsh(ham)')


# In[14]:


for idx, this_u_sc in enumerate(u_sc):
    plt.plot(np.full(basis.shape[0], this_u_sc), mi_energies[idx]/j_sc, ',', c='k')
plt.ylim(-15, 15)
plt.xlim(0, 10)


# In[15]:


floquet_interactions_un = np.kron(np.diag(np.ones(9)), unscaled_interactions)
unscaled_floquet_diag = np.kron(np.diag(np.arange(-4,5)), np.ones(basis.shape[0]))
jnj_unscaled = fock_basis.onsite_offset(np.arange(-(n_sites//2), n_sites//2 + 1), basis)


# In[16]:


plt.pcolormesh(unscaled_interactions)


# In[17]:


def s_matrix(states, raw_interactions):
    s_mat = np.empty((states.shape[0], states.shape[0]))
    j_vec = np.arange(-(states.shape[1]//2), states.shape[1]//2 + 1)
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            s_mat[i, j] = np.sum((state_j - state_i)*j_vec) if raw_interactions[i,j] != 0 else 0
    return s_mat


# In[18]:


this_s = s_matrix(basis, fock_basis.nearest_neighbor_hopping(basis))
plt.pcolormesh(this_s, cmap='RdBu')


# In[19]:


plt.pcolormesh(fock_basis.nearest_neighbor_hopping(basis)*np.power(this_s, -3, where=(this_s!=0)), cmap='RdBu', vmin=-5, vmax=5)


# In[20]:


from scipy.special import jv as bessel_j


# In[21]:


def scaling_blocks(m_rad, kscaled, s_matrix, raw_tunneling):
    mprime = np.arange(-m_rad, m_rad+1)[:, np.newaxis]
    m = np.arange(-m_rad, m_rad+1)[np.newaxis, :]
    tunneling_part = np.kron(bessel_j(mprime-m, kscaled), raw_tunneling)
    s_powers = np.power(np.tile(s_matrix, (2*m_rad+1, 2*m_rad+1)), np.kron(np.abs(mprime-m), np.ones_like(raw_tunneling)))
    return tunneling_part * s_powers


# In[22]:


full_unscaled_hopping = scaling_blocks(4, 2.4, this_s, fock_basis.nearest_neighbor_hopping(basis))
maxmag = np.abs(full_unscaled_hopping).max()
plt.pcolormesh(full_unscaled_hopping, cmap='RdBu', vmin=-maxmag, vmax=maxmag)
plt.colorbar()


# In[ ]:


import drive_coefficients


# In[112]:


m_rb = 1.4447e-25 #kg
h = 6.62607004e10-34 #SI
tpf = 0.5*m_rb*1.3e-6*1.3e-6 * 4.0 * np.pi * np.pi / h
def build_sys(u_over_j, j_tunn, trap_freq,  drive_freq, basis_states, floquet_radius,
             gamma):
    drive_y = drive_coefficients.y(gamma)
    drive_freq = drive_coefficients.crit_drive_freq(drive_y, trap_freq)
    m_list = np.arange(-floquet_radius, floquet_radius+1)
    raw_int = fock_basis.onsite_interactions(basis_states)
    raw_hop = -j_tunn*fock_basis.nearest_neighbor_hopping(basis_states)
    raw_jsq_onsite = fock_basis.onsite_offset(fock_basis.jsq_offset(basis_states.shape[1]), basis_states)
    scale_idxs = np.abs(np.arange((2*floquet_radius + 1))[np.newaxis, :]
                        - np.arange((2*floquet_radius + 1))[:, np.newaxis])
    small_xi_blocks = drive_coefficients.n_small_xis(2*floquet_radius + 1, drive_y)[scale_idxs]
    big_xi_blocks = drive_coefficients.n_big_xis(2*floquet_radius + 1, drive_y)[scale_idxs]
    jsq_prescale = tpf * (np.square(trap_freq*small_xi_blocks) - np.square(drive_freq*big_xi_blocks))
    
    chi = drive_coefficients.all_chis(drive_y)
    chi_blocks = (np.diag(np.full(2*floquet_radius+1, chi[0]))
                  + np.diag(np.full(2*floquet_radius, chi[1]), k=1)
                  + np.diag(np.full(2*floquet_radius, chi[1]), k=-1)
                  + np.diag(np.full(2*floquet_radius-1, chi[2]), k=2)
                  + np.diag(np.full(2*floquet_radius-1, chi[2]), k=-2) )
    floquet_interactions = 0.5*u_over_j*j_tunn * np.kron(chi_blocks, raw_int)
    floquet_diag = np.kron(np.diag(m_list), np.diag(np.full(basis_states.shape[0], drive_freq)))
    floquet_tunneling = np.kron(chi_blocks, raw_hop)
    floquet_offsets = np.kron(jsq_prescale, raw_jsq_onsite)
    return floquet_interactions + floquet_diag + floquet_tunneling + floquet_offsets


# In[115]:


best_basis = fock_basis.unit_filling(5).copy()
test = build_sys(3.0, 20., 20., 1000.0, best_basis, 3, 0.01)


# In[116]:


absmax = np.abs(test).max()
plt.pcolormesh(np.log(np.abs(test)), cmap='inferno', vmin=-20)
plt.colorbar()


# In[153]:


get_ipython().run_cell_magic('time', '', 'best_basis = fock_basis.unit_filling(7).copy()\nfham = build_sys(3., 14., 2.4, best_basis, 5)\nvals = np.linalg.eigvalsh(fham)')


# In[136]:


get_ipython().run_cell_magic('time', '', 'n_scan = 257\nf_rad = 5\ngamma_vals = np.linspace(1e-4, 0.5, n_scan)\nbest_basis = fock_basis.unit_filling(5).copy()\nresults = np.empty((n_scan, (2*f_rad+1)*best_basis.shape[0]))\nfor idx, this_gamma in progbar(enumerate(gamma_vals), size=len(gamma_vals), every=1):\n    fham = build_sys(1, 20., 25., 1000.0, best_basis, 5, this_gamma)\n    results[idx] = np.linalg.eigvalsh(fham)')


# In[133]:


for idx, gamma in enumerate(gamma_vals):
    plt.plot(np.full(len(results[idx]), gamma), np.mod(results[idx] + 200., 1000.) - 200., ',k')
plt.ylim(-200, 800)
plt.xlim(0,0.5)
plt.savefig('/home/zachsmith/Desktop/lines.png', dpi=300)


# In[152]:


f_drive = np.empty(len(gamma_vals))
for idx, gamma in enumerate(gamma_vals):
    plt.plot(np.full(len(results[idx]), gamma),results[idx], ',k')
    f_drive[idx] = drive_coefficients.crit_drive_freq(drive_coefficients.y(gamma), 25.)
plt.plot(gamma_vals,  0.5*f_drive + 60, 'r')
plt.plot(gamma_vals, -0.5*f_drive + 60, 'r')
plt.plot(gamma_vals, np.full(len(gamma_vals), 60), '--r')
plt.ylim(-0, 100)
plt.xlim(0,0.5)
plt.tight_layout()
plt.savefig('/home/zachsmith/Desktop/lines.png', dpi=300)


# In[92]:


for idx, gamma in enumerate(gamma_vals):
    plt.plot(np.full(len(results[idx]), gamma),results[idx], ',k')
plt.ylim(-100, 200)
plt.xlim(0,0.5)
plt.tight_layout()
plt.savefig('/home/zachsmith/Desktop/lines.png', dpi=300)


# In[ ]:




