from mpmath import mp
import numpy as np
import functools
mp.dps = 400 # use precision of 00 decimal places throughtout, should be -log(gamma)*n + 64 ish, this lets us go as small as 10^-5
return_dtype = np.float

@functools.lru_cache(maxsize=1024)
def y(gamma):
    return mp.sqrt(- mp.powm1(mp.mpmathify(gamma), 2))

@functools.lru_cache(maxsize=2*1024)
def keff_factor(y, cancel_keff=False):
    if cancel_keff:
        return mp.mpf(1)
    else:
        return (2 + one_m_ysq(y))

@functools.lru_cache(maxsize=1024)
def sqrt_minus_over_plus_factor(y):
    return mp.sqrt(-mp.powm1(y, 1)/(1+y))

@functools.lru_cache(maxsize=128)
def neg_tothe_n(n):
    return mp.power(-1, n)

@functools.lru_cache(maxsize=1024)
def one_m_ysq(y):
    return (-mp.powm1(y, 2))

def big_xi(n, y, cancel_keff=False):
    poly_part = 3 + 3*n*y + (3*(n*n-1))*mp.power(y,2) + (2*n*(n*n - 1))*mp.power(y,3)
    return neg_tothe_n(n)*keff_factor(y, cancel_keff=cancel_keff)/(mp.mpf(12)*mp.power(y, 5)) * mp.power(sqrt_minus_over_plus_factor(y), n) * poly_part

def small_xi(n, y, cancel_keff=False):
    return neg_tothe_n(n)*keff_factor(y, cancel_keff=cancel_keff)*(1 + n*y)/(mp.mpf(2)*mp.power(y, 3)) * (mp.power(sqrt_minus_over_plus_factor(y), n))

def chi(n, y):
    if (n == 0):
        return mp.mpf(1)
    elif (n == 1):
        return 2*mp.sqrt(-mp.powm1(y, 2))/keff_factor(y, cancel_keff=False)
    elif (n == 2):
        return -mp.powm1(y, 2)/(2*keff_factor(y, cancel_keff=False))
    else:
        return 0

def all_chis(y):
    return np.array((1.0, chi(1, y), chi(2, y)), dtype=return_dtype)

def mp_settings_str():
    return('mpmath set to {} decimal digits (equiv. {} binary digits)'.format(str(mp.dps), str(mp.prec)))

def mp_settings_tuple():
    return mp.dps, mp.prec

def both_xis(n, y, cancel_keff=False):
    return np.array((big_xi(n, y, cancel_keff=cancel_keff),
                     small_xi(n, y, cancel_keff=cancel_keff)), dtype=return_dtype)

def n_big_xis(n_count, y, cancel_keff=False):
    xi_list = [big_xi(n, y, cancel_keff=cancel_keff) for n in mp.arange(n_count+1)]
    return np.array(xi_list, dtype=return_dtype)

def n_small_xis(n_count, y, cancel_keff=False):
    xi_list = [small_xi(n, y, cancel_keff=cancel_keff) for n in mp.arange(n_count+1)]
    return np.array(xi_list, dtype=return_dtype)

def n_both_xis(n_max, y, cancel_keff=False):
    big_xi_list = [big_xi(n, y, cancel_keff=cancel_keff) for n in mp.arange(n_max+1)]
    small_xi_list = [small_xi(n, y, cancel_keff=cancel_keff) for n in mp.arange(n_max+1)]
    return  np.array(big_xi_list, dtype=return_dtype),  np.array(small_xi_list, dtype=return_dtype)

def n_both_xis_for_gammas_range(n_max, gamma_min, gamma_max, n_gammas, cancel_keff=False):
    gamma_list = mp.linspace(gamma_min, gamma_max, n_gammas)
    y_list = [y(this_gamma) for this_gamma in gamma_list]
    big_xis = np.empty((n_gammas, n_max+1), dtype=return_dtype)
    small_xis = np.empty((n_gammas, n_max+1), dtype=return_dtype)
    for idx, this_y in enumerate(y_list):
        big_xis[idx], small_xis[idx] = n_both_xis(n_max, this_y, cancel_keff=cancel_keff)
    return big_xis, small_xis, np.array(gamma_list, dtype=return_dtype)

def n_both_xis_from_gamma_list(n_max, gamma_list, cancel_keff=False):
    y_list = [y(this_gamma) for this_gamma in gamma_list]
    big_xis = np.empty((len(y_list), n_max+1), dtype=return_dtype)
    small_xis = np.empty((len(y_list), n_max+1), dtype=return_dtype)
    for idx, this_y in enumerate(y_list):
        big_xis[idx], small_xis[idx] = n_both_xis(n_max, this_y, cancel_keff=cancel_keff)
    return big_xis, small_xis
