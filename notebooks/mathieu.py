import numpy as np
import scipy.linalg as LA

# ## Mathieu's Equation Solutions ##
# 
# Method based on:
# 
# Mathieu functions revisited: matrix evaluation and generating functions  
# Chaos-Cador, L. & Ley-Koo, E  
# Revista mexicana de fisica 48, 67-78 (2002)  
# http://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S0035-001X2002000100013&nrm=iso%7D  
# http://ref.scielo.org/xr2wm4

# In[7]:


## Clean up docstrings, give actual form of series solution.
## NOTE A0's have extra sqrt(2) weighting!
def a_even_sys(n, q):
    """Gives even-order, even-parity pi-periodic solutions to Mathieu's equation with parameter q
    
    $\left[ \frac{d^2}{dV^2} - 2 q \cos(2 v) \right] V = -a V$ 
    Returns the even fourier coefficients for the even pi-periodic solutions
    
    Inputs:
    n: size of basis to use
    q: q-parameter
    
    Outputs:
    (evals, evecs): eigenvalues and eigenvectors that corresponding to a and V
    NOTE: sqrt(2) factor for A0 is removed, unlike paper
    """
    M = (np.diag(np.square(2*np.arange(n)))
         + np.diag(q * np.ones(n-1), k=-1)
         + np.diag(q * np.ones(n-1), k=1)
        )
    M[0, 1] *= np.sqrt(2)
    M[1, 0] *= np.sqrt(2)
    evals, evecs = LA.eigh(M)
    evecs[0] /= np.sqrt(2)
    return evals, evecs

def a_odd_sys(n, q):
    M = (np.diag(np.square(2*np.arange(n) + 1))
            + np.diag(q * np.ones(n-1), k=-1)
            + np.diag(q * np.ones(n-1), k=1)
            )
    M[0, 0] += q
    return LA.eigh(M)

def b_even_sys(n, q):
    """Gives even-order, odd-parity pi-periodic solutions to Mathieu's equation with parameter q
    
    $\left[ \frac{d^2}{dV^2} - 2 q \cos(2 v) \right] V = -a V$ 
    Returns the odd fourier coefficients for the odd pi-periodic solutions
    
    Inputs:
    n: size of basis to use
    q: q-parameter
    
    Outputs:
    (evals, evecs): eigenvalues and eigenvectors that corresponding to a and V
    """
    M = (np.diag(np.square(2 + 2*np.arange(n)))
         + np.diag(q * np.ones(n-1), k=-1)
         + np.diag(q * np.ones(n-1), k=1)
        )
    return LA.eigh(M)

def b_odd_sys(n, q):
    M = (np.diag(np.square(2*np.arange(n) + 1))
            + np.diag(q * np.ones(n-1), k=-1)
            + np.diag(q * np.ones(n-1), k=1)
            )
    M[0, 0] -= q
    return LA.eigh(M)

