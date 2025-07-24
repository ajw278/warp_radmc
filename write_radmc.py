import numpy as np

def write_amr_grid(xi, yi, zi, fname='amr_grid.inp'):
	with open(fname, 'w') as f:
		f.write("1\n")             # format
		f.write("0\n")             # regular grid
		f.write("0\n")             # cartesian
		f.write("0\n")             # no gridinfo
		f.write("1 1 1\n")         # include x, y, z
		f.write(f"{len(xi)-1} {len(yi)-1} {len(zi)-1}\n")
		for arr in (xi, yi, zi):
			for v in np.ravel(arr):
				f.write(f"{float(v):13.6e}\n")

def write_co_number_density(rho_sph, abundance=1e-5):
	mu = 2.3  # mean molecular weight
	mH = 1.6737e-24  # hydrogen mass in grams

	nCO = rho_sph * (abundance*100.0  / (mu * mH))# gas-to-dust ratio

	with open('numberdens_co.inp', 'w') as f:
		f.write("1\n")
		f.write(f"{nCO.size}\n")
		f.write("1\n")  # number of species
		nCO.ravel(order='F').tofile(f, sep='\n', format='%13.6e')
		f.write('\n')

def write_density(rhod):
	with open('dust_density.inp', 'w') as f:
		f.write('1\n')
		f.write(f"{nx * ny * nz}\n1\n")
		rhod.ravel(order='F').tofile(f, sep='\n', format="%13.6e")
		f.write('\n')

def write_wavelength_grid():
	lam1, lam2, lam3, lam4 = 0.1, 7.0, 25., 1e4
	n12, n23, n34 = 20, 100, 30
	lam = np.concatenate([
		np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False),
		np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False),
		np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
	])
	with open('wavelength_micron.inp', 'w') as f:
		f.write(f"{len(lam)}\n")
		f.writelines(f"{l:13.6e}\n" for l in lam)
	return lam

def write_stars(lam, mstar, rstar, tstar, pstar):
	with open('stars.inp', 'w') as f:
		f.write('2\n')
		f.write(f"1 {len(lam)}\n\n")
		f.write(f"{rstar:13.6e} {mstar:13.6e} {pstar[0]:13.6e} {pstar[1]:13.6e} {pstar[2]:13.6e}\n\n")
		f.writelines(f"{l:13.6e}\n" for l in lam)
		f.write(f"\n{-tstar:13.6e}\n")

def write_opacity_control():
	with open('dustopac.inp', 'w') as f:
		f.write('2\n1\n============================================================================\n')
		f.write('10\n0\nsilicate\n----------------------------------------------------------------------------\n')

def write_line_input():
	with open('lines.inp', 'w') as f:
		f.write('2\n1\n')
		f.write('co   leiden   0   0   0')

def write_radmc3d_inp(nphot):
	with open('radmc3d.inp', 'w') as f:
		f.write(f"nphot = {nphot}\niranfreqmode = 1\n")
		#Optical depth 5 for speedup
		f.write("mc_scat_maxtauabs = 5.d0\n")
		f.write("tgas_eq_tdust=1")



def write_amr_grid_spherical(r_edges, theta_edges, phi_edges, fname='amr_grid.inp'):
    with open(fname, 'w') as f:
        f.write("1\n")               # format
        f.write("0\n")               # regular grid
        f.write("100\n")             # spherical coordinates
        f.write("0\n")               # no grid info
        f.write("1 1 1\n")           # all directions active
        f.write(f"{len(r_edges)-1} {len(theta_edges)-1} {len(phi_edges)-1}\n")
        for arr in (r_edges, theta_edges, phi_edges):
            for val in arr:
                f.write(f"{val:.8e}\n")
                
def make_cell_edges(xc):
    dx = np.diff(xc)
    xe = np.zeros(len(xc) + 1)
    xe[1:-1] = 0.5 * (xc[:-1] + xc[1:])
    xe[0] = xc[0] - dx[0] / 2
    xe[-1] = xc[-1] + dx[-1] / 2
    return xe

def write_density_spherical(rho):
    with open('dust_density.inp', 'w') as f:
        f.write("1\n")
        f.write(f"{rho.size}\n")
        f.write("1\n")
        rho.ravel(order='F').tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
		
def write_gas_velocity(vxyz, fname="gas_velocity.inp"):
	"""
	Writes the gas velocity field to gas_velocity.inp for RADMC-3D.

	Parameters
	----------
	vxyz : ndarray
		Shape (nr, ntheta, nphi, 3), velocity in cm/s (Cartesian components).
	fname : str
		Output filename (default: 'gas_velocity.inp').
	"""
	nr, ntheta, nphi, _ = vxyz.shape
	nrcells = nr * ntheta * nphi

	# Reshape to (nrcells, 3) and then flatten to (nrcells * 3,)
	vflat = vxyz.reshape(-1, 3).T.flatten()

	with open(fname, 'w') as f:
		f.write('1\n')  # ASCII format
		f.write(f'{nrcells}\n')
		np.savetxt(f, vflat, fmt="%.9e")