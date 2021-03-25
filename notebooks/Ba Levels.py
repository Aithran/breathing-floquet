#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt


# In[2]:


import numpy as np


# In[44]:


levels = np.array([0, 4873.852, 5674.807, 20261.561, 21952.404])
s_list = ['2','2','2','2','2']
L_list = ['S','D','D','P','P']
j_list = ['1/2','3/2','5/2','1/2','3/2']
def build_label(s,l,j):
    return '$^{}{}_{{{}}}$'.format(s,l,j)
labels = [build_label(s,l,j) for s,l,j in zip(s_list, L_list, j_list)]
sublevels = [(1,0),(2,1),(2,3),(1,0),(2,1)]
max_sublevels = 2*np.array(sublevels).flatten().max() +1
print(max_sublevels)
nist_string = """
"6s"	"2S"	"1/2"	" 0.000"	""	""	"L11975"
"5d"	"2D"	"3/2"	" 4873.852"	""	"0.79"	"L9224"
"5d"	"2D"	"5/2"	" 5674.807"	""	"1.12"	"L9224"
"6p"	"2P*"	"1/2"	" 20261.561"	""	""	"L9224"
"6p"	"2P*"	"3/2"	" 21952.404"	""	"1.32"	"L9224"
"""
def wavelength(ground, excited):
    return 1.0e7/(excited-ground)


# In[45]:


linepairs = [(0,2), (0,3), (0,4), (1,3), (1,4), (2,4)]
for g, e in linepairs:
    print('{}<->{}: {:0.0f} nm'.format(labels[g], labels[e], wavelength(levels[g], levels[e])))


# In[53]:


fig = plt.figure(figsize=(3,5))
ax = plt.subplot(111)
sub_offset = 500
sub_width = 0.15
sub_gap = 0.025
for i in np.arange(len(sublevels)):
    col = 2 if L_list[i]=='D' else 0
    print(col)
    for sub_i, f in enumerate(sublevels[i]):
        level_offset = -10000 if L_list[i]=='P' else 0
        print(levels[i]+sub_i*sub_offset)
        print(col + np.arange(-f,f)*0.20 + 0.05)
        print(col + (1+np.arange(-f,f))*0.20 - 0.05)
        ax.hlines(np.full(2*f+1, levels[i]+sub_i*sub_offset+level_offset), col + np.arange(-f,f+1)*sub_width + sub_gap, col + (1+np.arange(-f,f+1))*sub_width - sub_gap)
    ax.axis('off')


# In[ ]:




