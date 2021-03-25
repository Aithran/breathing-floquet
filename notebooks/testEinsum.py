#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# In[2]:


m_rad=5
n_ms = 2*m_rad + 1
n_sites=33
n_states= 1
n_ts = 10*1024
print(10*2*(2*m_rad+1))
two_pi_i = np.complex(0, np.pi*2.0)
state_matrix = np.random.randn(n_ms*n_sites, n_states)*np.exp(two_pi_i * np.random.rand(n_ms*n_sites, n_states))


# In[14]:


start = np.array([[0,100],[1,101],[10,110],[11,111],[20,120],[21,121]])
print(start[:,0])
rs = start.reshape(3,2,-1)
print(rs[0,:,1], rs[:,0,1])


# In[3]:


get_ipython().run_cell_magic('time', '', 'c = np.transpose(state_matrix.reshape(2*m_rad+1, n_sites, -1), axes=(2,1,0))\nc_adj = np.transpose(c, axes=(0,2,1))\nc_nm = np.matmul(c_adj, c)\nprint(c.shape, c_adj.shape, c_nm.shape)\naccum = - 2.0 * np.sum(np.square(np.abs(state_matrix)), axis=0, dtype=np.complex)\nprint(accum)\nprint(np.diagonal(c_nm, 1, 1, 2).shape, (np.diagonal(c_nm, 1, 1, 2)[:,None,:] * np.diagonal(c_nm, -1, 1, 2)[:,:,None]).shape)\naccum += np.sum(np.diagonal(c_nm, 0, 1, 2)[:,None,:] * np.diagonal(c_nm, 0, 1, 2)[:,:,None], axis=(1,2))\nfor k in np.arange(1, 2*m_rad + 1):\n    accum += 2.0 * np.sum(np.diagonal(c_nm, k, 1, 2)[:,None,:] * np.diagonal(c_nm, -k, 1, 2)[:,:,None], axis=(1,2))\naccum = np.sqrt(np.abs(accum + 1.0))\nprint(c_nm.shape, accum.shape, (np.diagonal(c_nm, 1, 1, 2)[:,None,:] * np.diagonal(c_nm, -1, 1, 2)[:,:,None]).shape)')


# In[4]:


np.diagonal(c_nm, 200, 1, 2).shape


# In[5]:


get_ipython().run_cell_magic('time', '', "reshaped_states = state_matrix.reshape(2*m_rad+1, n_sites, -1)\ntwo_pi_i = np.complex(0, np.pi*2.0)\nts = np.linspace(0.0, 1.0, n_ts)[np.newaxis,:]\nm_block = two_pi_i * np.arange(-m_rad, m_rad + 1)[:,np.newaxis]\nmt_mat = np.exp(m_block * ts)\nprint(mt_mat.shape)\n%time test = np.einsum('mt,mns->tns', mt_mat, reshaped_states)\nprint(test.shape)\n%time pops = np.einsum('tns,tns->ts', np.conjugate(test), test)\nprint(pops.shape)")


# In[6]:


get_ipython().run_cell_magic('time', '', "reshaped_states = state_matrix.reshape(2*m_rad+1, n_sites, -1)\ntwo_pi_i = np.complex(0, np.pi*2.0)\nts = np.linspace(0.0, 1.0, n_ts)[np.newaxis,:]\nm_block = -two_pi_i * np.arange(-m_rad, m_rad + 1)[:,np.newaxis]\nmt_mat = np.exp(m_block * ts)\n%time test = np.einsum('mt,mns->tns', mt_mat, reshaped_states)\nprint(test.shape)\n%time pops = np.einsum('tns,tns->ts', np.conjugate(test), test)\nprint(pops.shape)\n%time done = np.sqrt(np.sum(np.square(1.0-np.abs(pops)),axis=0)/n_ts)\nprint(done)")


# In[7]:


done.shape


# In[8]:


get_ipython().run_cell_magic('time', '', 'reshaped_states = state_matrix.reshape(2*m_rad+1, n_sites, -1)\ntwo_pi_i = np.complex(0, np.pi*2.0)\nts = np.linspace(0.0, 1.0, n_ts)\npops = np.zeros((n_states, n_ts))\n##\nm_block = (np.arange(-m_rad, m_rad+1) * -two_pi_i).reshape(-1,1,1)\nsite_amps = np.empty((n_sites, n_states), dtype=np.complex)\nexpand_states = np.empty((2*m_rad + 1, n_sites, n_states), dtype=np.complex)\nresid_pops = np.zeros((n_states))\n#print(reshaped_states.shape, np.exp(m_block * t).shape)\n#print(np.sum(expand_states, axis=0).shape)\nfor idx, t in enumerate(ts):\n        phasor = np.exp(m_block * t)\n        np.multiply(phasor, reshaped_states, out=expand_states)\n        np.sum(expand_states, axis=0, out=site_amps)\n        resid_pops[:] += np.square(np.abs(1.0-np.sum(np.real(np.conjugate(site_amps) * site_amps), axis=0)))\nresid_pops = np.sqrt(resid_pops/n_ts)')


# In[9]:


resid_pops.shape


# In[10]:


np.allclose(done, resid_pops), np.allclose(accum, resid_pops), np.allclose(accum, done)


# In[11]:


plt.plot(done)
plt.plot(resid_pops)


# In[12]:


plt.plot(np.abs(accum))


# In[ ]:





# In[ ]:




