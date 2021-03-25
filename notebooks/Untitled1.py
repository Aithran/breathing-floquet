#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt


# In[2]:


import numpy as np


# In[5]:


levels = np.array([0, 4873.852, 5674.807, 20261.561, 21952.404])
labels = ['$^2S_{1/2}$', '$2^D_{3/2}$', '$2^D_{5/2}$', '$2^P_{1/2}$', '$2^P_{3/2}$']
nist_string = """
"6s"	"2S"	"1/2"	" 0.000"	""	""	"L11975"
"5d"	"2D"	"3/2"	" 4873.852"	""	"0.79"	"L9224"
"5d"	"2D"	"5/2"	" 5674.807"	""	"1.12"	"L9224"
"6p"	"2P*"	"1/2"	" 20261.561"	""	""	"L9224"
"6p"	"2P*"	"3/2"	" 21952.404"	""	"1.32"	"L9224"
"""
def wavelength(ground, excited):
    return 1e7/(excited-ground)


# In[12]:


linepairs = [(0,2), (0,3), (0,4), (1,3), (1,4), (2,4)]
for g, e in linepairs:
    print('{}<->{}: {0.0d}')
[wavelength(levels[0], levels[2]),
 wavelength(levels[0], levels[3]),
 wavelength(levels[0], levels[4]),
 wavelength(levels[1], levels[3]),
 wavelength(levels[1], levels[4]),
 wavelength(levels[2], levels[4])]


# In[ ]:




