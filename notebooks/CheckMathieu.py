#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import mathieu


# In[13]:


vals, vecs = mathieu.b_even_sys(2048, 5)


# In[18]:


vecs[:11,[0,4]]


# In[16]:


vals[[0,1,4]]


# In[19]:


defaults_dict = {'period':1.3, 'depth_kHz':1.4}
new_dict = {'depth_kHz':9.1}
print(defaults_dict, new_dict, {**defaults_dict, **new_dict})


# In[95]:


import copy
required_params = frozenset({'unchanged', 'scan_me', 'missing'})
def test_scan_merge(default_dict, scan_param, scan_array):
    if not (required_params <= set(default_dict)):
        missing_params = list(required_params - set(default_dict))
        raise KeyError('Missing required entries from default_dict: {}'.format(missing_params))
    if scan_param not in default_dict:
        raise KeyError('Unknown scan_param provided: {}'.format(scan_param))
    # Make a copy so default
    default_copy = copy.deepcopy(default_dict)
    for scan_value in scan_array:
        these_settings = {**default_copy, scan_param:scan_value}
        print(these_settings)


# In[94]:


test_scan_merge({'unchanged':1, 'scan_me':0, 'missing':1}, 'scan_me', [1,2,3,4,5])
test_scan_merge({'unchanged':1, 'scan_me':0}, 'missing', [1,2,3,4,5])


# In[92]:





# In[ ]:





# In[ ]:




