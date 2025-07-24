
from scipy.interpolate import CubicSpline
import numpy as np
from constants import *
import plot_funcs as plf



# Prepend (0.0, 0.0) to ensure warp goes to zero at origin

def extend_warp_profile(r_warp, dinc, dpa, plot=False, dinc_ext_lower=None, dpa_ext_lower=None, r_ext_lower=None):
    
    extend_lower = False
    if not r_ext_lower is None:            
        r_ext = np.insert(r_warp, 0, r_ext_lower[1])
        
        r_ext = np.insert(r_ext, 0, r_ext_lower[0])
        extend_lower = True

    if not dinc_ext_lower is None and extend_lower:
        dinc1 = dinc_ext_lower[1]
        dinc2 = dinc_ext_lower[0]
        dinc_ext = np.insert(dinc, 0, dinc1)
        dinc_ext = np.insert(dinc_ext, 0, dinc2)
    elif extend_lower:
        raise ValueError("dinc_ext_lower must be provided if extending lower warp profile.")
    else:
        dinc_ext = dinc
    
    if not dpa_ext_lower is None:
        dpa1 = dpa_ext_lower[1]
        dpa2 = dpa_ext_lower[0]
        dpa_ext = np.insert(dpa, 0, dpa1)
        dpa_ext = np.insert(dpa_ext, 0, dpa2)
    elif extend_lower:
        raise ValueError("dpa_ext_lower must be provided if extending lower warp profile.")
    else:
        dpa_ext = dpa

    # Create cubic spline interpolators (with extrapolation)
    f_inc = CubicSpline(r_ext, dinc_ext, extrapolate=False)
    f_pa  = CubicSpline(r_ext, dpa_ext, extrapolate=False)


    # Optional: plot to verify behavior
    r_test = np.linspace(0, 280, 1000)*au
    # Compute which r_ext points are not in r_warp (assuming 1D arrays)
    mask_new_points = ~np.isin(r_ext, r_warp)

    if plot:
        plf.plot_warp_profile(r_warp, dinc, dpa, r_ext, dinc_ext, dpa_ext, r_test, f_inc, f_pa, mask_new_points)

    return f_inc, f_pa

